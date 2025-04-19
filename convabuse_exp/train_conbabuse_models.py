# train_convabuse_models.py

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

# Suppress warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

print("--- ConvAbuse Model Training Script ---")

# --- Configuration ---
FULL_DATA_FILE = '/home/alessandro/Documents/convabuse/ConvAbuseEMNLPfull.csv'  # Path to the FULL dataset
OUTPUT_DIR = '/home/alessandro/Documents/convabuse'  # Directory for models/split info/processed data
# This is the FINAL aggregated data file used by the evaluation script
AGGREGATED_DATA_PATH = os.path.join(OUTPUT_DIR, 'convabuse_aggregated.pkl')
MODEL_A_PATH = os.path.join(OUTPUT_DIR, 'model_A_bert_noctx.pth')
MODEL_B_PATH = os.path.join(OUTPUT_DIR, 'model_B_bert_ctx.pth')
SPLIT_INFO_PATH = os.path.join(OUTPUT_DIR, 'train_val_test_split_ids.pkl')

# Verify these are the correct keys to identify unique annotated utterances
# Using conv_id + user utterance seems appropriate to define a unique annotation target
GROUPING_KEYS = ['conv_id', 'user']
MIN_ANNOTATORS = 2  # Minimum *valid* annotations per unique utterance

# Data Splitting (relative to the aggregated data)
TEST_SPLIT_SIZE = 0.15
VALID_SPLIT_SIZE = 0.10  # Validation size out of the non-test data

# Model & Training
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
HIDDEN_DIM = 256
K = 5 # Number of severity classes
OUTPUT_DIM = K
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 4
EARLY_STOPPING_PATIENCE = 1

# Misc
EPSILON = 1e-9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# Ensure order matches severity levels (-3=most severe, 1=least severe for ConvAbuse?)
# Check the dataset description carefully for the exact mapping.
# Assuming -3 -> class 0, -2 -> class 1, -1 -> class 2, 0 -> class 3, 1 -> class 4
SEVERITY_LEVELS_STR = ['is_abuse.-3', 'is_abuse.-2', 'is_abuse.-1', 'is_abuse.0', 'is_abuse.1']
if len(SEVERITY_LEVELS_STR) != K:
    raise ValueError(f"Mismatch between K ({K}) and number of severity level columns ({len(SEVERITY_LEVELS_STR)})")

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Data Loading and Aggregation ---
def load_and_aggregate_convabuse(filepath, save_aggregated_path, grouping_keys, min_annotators):
    """
    Loads the full ConvAbuse dataset, aggregates annotations per unique utterance,
    filters based on minimum annotators, and saves the processed data.
    """
    if os.path.exists(save_aggregated_path):
        print(f"Loading aggregated data from cache: {save_aggregated_path}")
        try:
            with open(save_aggregated_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cached file {save_aggregated_path}: {e}. Re-processing...")

    print(f"Loading and aggregating full dataset from: {filepath}")
    try:
        # Define dtypes for robustness, especially for IDs and text
        dtypes = {'conv_id': str, 'example_no': object, 'annotator_id': str, # example_no can be non-numeric
                  'prev_user': str, 'prev_agent': str, 'agent': str, 'user': str}
        for col in SEVERITY_LEVELS_STR: dtypes[col] = object # Load as object first

        df_full = pd.read_csv(filepath, dtype=dtypes, low_memory=False) # Use low_memory=False for mixed types

        # Fill NA in text fields used for grouping or context BEFORE grouping
        text_cols_to_fill = ['prev_user', 'prev_agent', 'agent', 'user'] + grouping_keys
        for col in text_cols_to_fill:
            if col in df_full.columns:
                df_full[col] = df_full[col].fillna('').astype(str) # Ensure string type after fillna
            else:
                print(f"Warning: Expected text column '{col}' not found.")


        # Convert severity columns to numeric robustly AFTER loading
        for col in SEVERITY_LEVELS_STR:
            if col in df_full.columns:
                # Attempt conversion, coerce errors to NaN, then fill NaN with 0, finally cast to int
                df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0).astype(int)
            else:
                print(f"Warning: Severity column '{col}' not found. Creating with zeros.")
                df_full[col] = 0 # Create column if missing

        # Check if grouping keys exist
        missing_keys = [key for key in grouping_keys if key not in df_full.columns]
        if missing_keys:
            print(f"Error: Grouping keys {missing_keys} not found in columns: {df_full.columns.tolist()}")
            exit()

        print(f"Loaded {len(df_full)} annotations.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}"); exit()
    except Exception as e:
        print(f"Error loading or processing CSV: {e}"); exit()

    print(f"Aggregating annotations per example using keys: {grouping_keys}...")
    # Ensure grouping keys are strings to prevent issues with mixed types if 'user' contained numbers
    for key in grouping_keys:
         df_full[key] = df_full[key].astype(str)

    grouped = df_full.groupby(grouping_keys)
    aggregated_data = []
    unique_instance_ids = set()

    for name_tuple, group in tqdm(grouped, total=len(grouped), desc="Aggregating"):
        # Ensure name_tuple elements are strings for reliable ID generation
        instance_id = "_".join(map(str, name_tuple))

        # Handle potential duplicate instance IDs if joining isn't unique enough
        # (though conv_id + user should be fairly unique)
        original_instance_id = instance_id
        counter = 0
        while instance_id in unique_instance_ids:
             counter += 1
             instance_id = f"{original_instance_id}_dup{counter}"
        unique_instance_ids.add(instance_id)


        first_row = group.iloc[0]
        # Extract context safely using .get with default empty string
        user_text = str(first_row.get('user', ''))
        agent_text = str(first_row.get('agent', ''))
        prev_user_text = str(first_row.get('prev_user', ''))
        prev_agent_text = str(first_row.get('prev_agent', ''))

        # Deduplicate annotations by annotator_id within the group first
        group_unique_annotators = group.drop_duplicates(subset=['annotator_id'])
        annotator_ids = group_unique_annotators['annotator_id'].unique()
        n_total_annotators = len(annotator_ids)

        counts = np.zeros(K, dtype=int)
        actual_labels_found_count = 0 # Count how many annotators provided a valid label (==1 in exactly one severity column)

        # Iterate through unique annotators' first annotation row for this group
        for _, annotator_row in group_unique_annotators.iterrows():
            found_label_for_annotator = False
            label_sum = 0
            label_idx = -1
            for idx, col_name in enumerate(SEVERITY_LEVELS_STR):
                label_val = annotator_row.get(col_name, 0) # Use .get for safety
                if label_val == 1:
                    label_sum += 1
                    label_idx = idx

            # Only count if exactly one label column is 1
            if label_sum == 1:
                counts[label_idx] += 1
                found_label_for_annotator = True

            if found_label_for_annotator:
                actual_labels_found_count += 1

        # Filter based on the number of *valid* annotations found meeting the minimum
        if actual_labels_found_count < min_annotators:
            continue # Skip this instance if not enough valid annotations

        aggregated_data.append({
            'instance_id': instance_id,  # Use the generated unique ID
            'user': user_text, 'agent': agent_text,
            'prev_user': prev_user_text, 'prev_agent': prev_agent_text,
            'counts': counts,          # Counts derived from valid annotations
            'n': actual_labels_found_count # Number of valid annotations
        })

    df_processed = pd.DataFrame(aggregated_data)
    if df_processed.empty:
        print(f"Error: No instances found with >= {min_annotators} valid annotations. Check data and MIN_ANNOTATORS setting.")
        exit()

    print(f"Aggregation complete. Found {len(df_processed)} unique instances with >= {min_annotators} valid annotations.")

    # Save the final processed data to the specified path
    try:
        with open(save_aggregated_path, 'wb') as f:
            pickle.dump(df_processed, f)
        print(f"Processed aggregated data saved to {save_aggregated_path}")
    except Exception as e:
        print(f"Error saving processed data to {save_aggregated_path}: {e}")
        exit()

    return df_processed


# --- PyTorch Dataset Classes ---
class ConvAbuseDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, use_context):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.use_context = use_context

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        user_text = str(row['user']) # Ensure text is string
        target_counts = row['counts']
        n = row['n']
        # Calculate empirical probability distribution
        target_p_emp = target_counts / (n + EPSILON) # Add epsilon for stability if n=0 (shouldn't happen with filtering)
        target_p_emp = target_p_emp / (target_p_emp.sum() + EPSILON) # Normalize

        if self.use_context:
            # Safely get context parts, ensuring they are strings
            pa = str(row.get('prev_agent', ''))
            pu = str(row.get('prev_user', ''))
            ag = str(row.get('agent', ''))
            # Include only non-empty parts
            context_parts = [part for part in [pa, pu, ag, user_text] if part]
            # Join with SEP token
            full_text = f" {self.tokenizer.sep_token} ".join(context_parts)
        else:
            full_text = user_text

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            full_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target_p_emp, dtype=torch.float) # Target is the empirical distribution
        }


# --- Model Architecture ---
class BertClassificationHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.bert_output_dim = 768 # Standard for bert-base-*
        self.l1 = nn.Linear(self.bert_output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, bert_output):
        # Use the [CLS] token representation
        cls_token_state = bert_output.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token_state)
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.l2(x)
        return logits

class ConvAbuseBertModel(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, output_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.head = BertClassificationHead(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(bert_output)
        return logits


# --- Loss Function ---
# KL Divergence: Measures difference between predicted log-probabilities and target probabilities
# reduction='batchmean' averages the loss over the batch
loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False) # Input is log-probs, target is probs


# --- Training and Validation Loops ---
def train_epoch(model, dataloader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device) # Target is empirical probability dist p_emp

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Model outputs logits. Convert to log-probabilities for KLDivLoss.
        log_probs = torch.log_softmax(outputs, dim=1)

        # Ensure targets sum to 1 (already done in Dataset, but double check)
        targets = targets / (targets.sum(dim=1, keepdim=True) + EPSILON)

        loss = loss_fn(log_probs, targets)

        # Handle potential NaN loss
        if torch.isnan(loss):
            print("Warning: NaN loss detected during training. Skipping batch.")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()
        scheduler.step() # Update learning rate

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            log_probs = torch.log_softmax(outputs, dim=1)

            # Ensure targets sum to 1
            targets = targets / (targets.sum(dim=1, keepdim=True) + EPSILON)

            loss = loss_fn(log_probs, targets)

            if not torch.isnan(loss):
                total_loss += loss.item()
            else:
                 print("Warning: NaN loss detected during validation. Skipping batch calculation.")


    # Avoid division by zero if dataloader is empty
    if not dataloader:
         return 0.0
    num_batches = len(dataloader)
    if num_batches == 0: return 0.0 # Handle case where validation set might be tiny / empty after filtering


    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0


    return avg_loss # Return average loss


# --- Function to Run Training ---
def run_training(model, train_df, valid_df, tokenizer, max_len, use_context, model_save_path):
    print(f"  Preparing datasets (Use Context: {use_context})...")
    train_dataset = ConvAbuseDataset(train_df, tokenizer, max_len, use_context)
    valid_dataset = ConvAbuseDataset(valid_df, tokenizer, max_len, use_context)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE) # No shuffle for validation

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

    # Scheduler
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_valid_loss = float('inf')
    epochs_no_improve = 0

    print(f"  Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        print(f"  Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, DEVICE, scheduler)
        valid_loss = validate_epoch(model, valid_dataloader, loss_fn, DEVICE)

        print(f"    Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        # Early stopping and model saving logic
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"    Validation loss improved. Best model saved to {model_save_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"    Validation loss did not improve for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping triggered after {epoch + 1} epochs.")
                break
    print("-" * 20)


# --- Main Execution ---

# 1. Load/Aggregate Data
# This function now saves the result directly to AGGREGATED_DATA_PATH if not cached
df_processed = load_and_aggregate_convabuse(
    FULL_DATA_FILE, AGGREGATED_DATA_PATH, GROUPING_KEYS, MIN_ANNOTATORS
)
print('done loading')
if df_processed is not None and not df_processed.empty:
    print(f"Successfully loaded/processed {len(df_processed)} aggregated instances.")

    # 2. Split Data based on instance_id
    print("Splitting aggregated data into Train/Validation/Test sets...")
    instance_ids = df_processed['instance_id'].unique()

    # Split instance IDs first
    train_val_ids, test_ids = train_test_split(
        instance_ids, test_size=TEST_SPLIT_SIZE, random_state=42
    )
    # Calculate validation split size relative to the remaining data
    relative_valid_size = VALID_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
    train_ids, valid_ids = train_test_split(
        train_val_ids, test_size=relative_valid_size, random_state=42
    )

    # Create dataframes based on the split IDs
    df_train = df_processed[df_processed['instance_id'].isin(train_ids)].reset_index(drop=True)
    df_valid = df_processed[df_processed['instance_id'].isin(valid_ids)].reset_index(drop=True)
    df_test = df_processed[df_processed['instance_id'].isin(test_ids)].reset_index(drop=True) # Test set not used in training

    print(f"Split sizes: Train={len(df_train)}, Validation={len(df_valid)}, Test={len(df_test)}")

    # Save the split instance IDs for the evaluation script
    split_info = {
        'train_ids': list(train_ids),
        'valid_ids': list(valid_ids),
        'test_ids': list(test_ids)
    }
    try:
        with open(SPLIT_INFO_PATH, 'wb') as f:
            pickle.dump(split_info, f)
        print(f"Split instance IDs saved to {SPLIT_INFO_PATH}")
    except Exception as e:
        print(f"Error saving split info to {SPLIT_INFO_PATH}: {e}")


    # 3. Initialize Tokenizer
    print(f"Initializing tokenizer: {BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 4. Train Model A (No Context)
    print("\n--- Training Model A (BERT - No Context) ---")
    model_A = ConvAbuseBertModel(BERT_MODEL_NAME, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    run_training(model_A, df_train, df_valid, tokenizer, MAX_LEN, use_context=False, model_save_path=MODEL_A_PATH)

    # 5. Train Model B (With Context)
    print("\n--- Training Model B (BERT - With Context) ---")
    model_B = ConvAbuseBertModel(BERT_MODEL_NAME, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    run_training(model_B, df_train, df_valid, tokenizer, MAX_LEN, use_context=True, model_save_path=MODEL_B_PATH)

    print("\n--- Training Complete ---")
    print(f"Models saved in: {OUTPUT_DIR}")
    print(f"Aggregated data saved to: {AGGREGATED_DATA_PATH}")
    print(f"Split info saved to: {SPLIT_INFO_PATH}")

else:
    print("Failed to load or process data. Training script exiting.")