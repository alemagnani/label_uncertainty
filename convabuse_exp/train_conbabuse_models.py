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
PROCESSED_DATA_CACHE = os.path.join(OUTPUT_DIR, 'convabuse_aggregated_processed.pkl')  # Cache processed data
MODEL_A_PATH = os.path.join(OUTPUT_DIR, 'model_A_bert_noctx.pth')
MODEL_B_PATH = os.path.join(OUTPUT_DIR, 'model_B_bert_ctx.pth')
SPLIT_INFO_PATH = os.path.join(OUTPUT_DIR, 'train_val_test_split_ids.pkl')

# !! Verify these are the correct keys to identify unique annotated utterances !!
GROUPING_KEYS = ['conv_id', 'user']
MIN_ANNOTATORS = 2  # Minimum annotators per unique utterance

# Data Splitting (relative to the aggregated data)
TEST_SPLIT_SIZE = 0.15
VALID_SPLIT_SIZE = 0.10  # Validation size out of the non-test data

# Model & Training
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 5  # K classes for severity
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 4
EARLY_STOPPING_PATIENCE = 1

# Misc
EPSILON = 1e-9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
K = 5
SEVERITY_LEVELS_STR = ['is_abuse.-3', 'is_abuse.-2', 'is_abuse.-1', 'is_abuse.0', 'is_abuse.1']

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Data Loading and Aggregation ---
def load_and_aggregate_convabuse(filepath, cache_path, grouping_keys, min_annotators):
    if os.path.exists(cache_path):
        print(f"Loading aggregated data from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"Loading and aggregating full dataset from: {filepath}")
    try:
        # Define dtypes for robustness
        dtypes = {'conv_id': str, 'example_no': int, 'annotator_id': str,
                  'prev_user': str, 'prev_agent': str, 'agent': str, 'user': str}
        for col in SEVERITY_LEVELS_STR: dtypes[col] = object

        df_full = pd.read_csv(filepath, dtype=dtypes, low_memory=False)

        # Fill NA in text fields used for grouping or context
        text_cols_to_fill = ['prev_user', 'prev_agent', 'agent', 'user']
        for col in text_cols_to_fill:
            if col in df_full.columns: df_full[col].fillna('', inplace=True)

        # Convert severity columns to numeric robustly
        for col in SEVERITY_LEVELS_STR:
            if col in df_full.columns:
                df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0).astype(int)
            else:
                print(f"Warning: Severity column '{col}' not found. Creating with zeros.")
                df_full[col] = 0

        # Check if grouping keys exist
        if not all(key in df_full.columns for key in grouping_keys):
            print(f"Error: One or more grouping keys {grouping_keys} not found in columns: {df_full.columns}")
            exit()

        print(f"Loaded {len(df_full)} annotations.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}");
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}");
        exit()

    print(f"Aggregating annotations per example using keys: {grouping_keys}...")
    grouped = df_full.groupby(grouping_keys)
    aggregated_data = []

    for name_tuple, group in tqdm(grouped, total=len(grouped), desc="Aggregating"):
        first_row = group.iloc[0]
        user_text = str(first_row['user'])
        agent_text = str(first_row['agent'])
        prev_user_text = str(first_row['prev_user'])
        prev_agent_text = str(first_row['prev_agent'])

        # Generate a unique ID from the grouping keys
        instance_id = "_".join(map(str, name_tuple))  # Simple join for ID

        annotator_ids = group['annotator_id'].unique()
        n = len(annotator_ids)

        if n < min_annotators: continue

        counts = np.zeros(K, dtype=int)
        actual_labels_found = 0
        for annotator in annotator_ids:
            annotator_rows = group[group['annotator_id'] == annotator]
            if annotator_rows.empty: continue
            annotator_row = annotator_rows.iloc[0]

            found_label_for_annotator = False
            for idx, col_name in enumerate(SEVERITY_LEVELS_STR):
                if annotator_row[col_name] == 1:
                    counts[idx] += 1
                    found_label_for_annotator = True
                    break
            if found_label_for_annotator: actual_labels_found += 1

        # Filter based on actual labels found meeting the minimum
        if actual_labels_found < min_annotators: continue

        aggregated_data.append({
            'instance_id': instance_id,  # Use the generated unique ID
            'user': user_text, 'agent': agent_text, 'prev_user': prev_user_text, 'prev_agent': prev_agent_text,
            'counts': counts, 'n': actual_labels_found
        })

    df_processed = pd.DataFrame(aggregated_data)
    print(
        f"Aggregation complete. Found {len(df_processed)} unique instances with >= {min_annotators} valid annotations.")

    # Save the processed data
    with open(cache_path, 'wb') as f:
        pickle.dump(df_processed, f)
    print(f"Processed data cached to {cache_path}")
    return df_processed


# --- PyTorch Dataset Classes ---
# (Identical to previous script - code omitted for brevity)
class ConvAbuseDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, use_context):
        self.tokenizer = tokenizer;
        self.data = dataframe;
        self.max_len = max_len;
        self.use_context = use_context

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index];
        user_text = str(row['user'])
        target_counts = row['counts'];
        n = row['n'];
        target_p_emp = target_counts / (n + EPSILON)
        if self.use_context:
            pa = str(row.get('prev_agent', ''));
            pu = str(row.get('prev_user', ''));
            ag = str(row.get('agent', ''))
            tp = [part for part in [pa, pu, ag, user_text] if part];
            ft = f" {self.tokenizer.sep_token} ".join(tp)
        else:
            ft = user_text
        enc = self.tokenizer.encode_plus(ft, add_special_tokens=True, max_length=self.max_len,
                                         return_token_type_ids=False, padding='max_length', truncation=True,
                                         return_attention_mask=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(),
                'targets': torch.tensor(target_p_emp, dtype=torch.float)}


# --- Model Architecture ---
# (Identical to previous script - code omitted for brevity)
class BertClassificationHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__();
        self.l1 = nn.Linear(768, hidden_dim);
        self.relu = nn.ReLU()
        self.d = nn.Dropout(0.1);
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, bo):
        cls = bo.last_hidden_state[:, 0, :];
        x = self.d(cls);
        x = self.l1(x);
        x = self.relu(x);
        x = self.d(x);
        return self.l2(x)


class ConvAbuseBertModel(nn.Module):
    def __init__(self, bmn, hhd, od):
        super().__init__();
        self.bert = BertModel.from_pretrained(bmn);
        self.head = BertClassificationHead(hhd, od)

    def forward(self, ii, am): return self.head(self.bert(input_ids=ii, attention_mask=am))


# --- Loss Function ---
loss_fn = nn.KLDivLoss(reduction='batchmean')


# --- Training and Validation Loops ---
# (Identical to previous script - code omitted for brevity)
def train_epoch(model, dl, lf, opt, dv, sched):
    model.train();
    tl = 0.0
    for batch in tqdm(dl, desc="Training", leave=False):
        ii = batch['input_ids'].to(dv);
        am = batch['attention_mask'].to(dv);
        tg = batch['targets'].to(dv)
        opt.zero_grad();
        outputs = model(ii, am);
        lp = torch.log_softmax(outputs, dim=1)
        tg = tg / (tg.sum(dim=1, keepdim=True) + EPSILON)
        loss = lf(lp, tg)
        if torch.isnan(loss): continue
        loss.backward();
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step();
        sched.step();
        tl += loss.item()
    return tl / len(dl)


def validate_epoch(model, dl, lf, dv):
    model.eval();
    tl = 0.0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Validating", leave=False):
            ii = batch['input_ids'].to(dv);
            am = batch['attention_mask'].to(dv);
            tg = batch['targets'].to(dv)
            outputs = model(ii, am);
            lp = torch.log_softmax(outputs, dim=1)
            tg = tg / (tg.sum(dim=1, keepdim=True) + EPSILON)
            loss = lf(lp, tg)
            if not torch.isnan(loss): tl += loss.item()
    return tl / len(dl)


# --- Function to Run Training ---
# (Identical to previous script - code omitted for brevity)
def run_training(model, train_df, valid_df, tokenizer, max_len, use_context, model_save_path):
    train_ds = ConvAbuseDataset(train_df, tokenizer, max_len, use_context)
    valid_ds = ConvAbuseDataset(valid_df, tokenizer, max_len, use_context)
    train_ldr = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_ldr = DataLoader(valid_ds, batch_size=BATCH_SIZE)
    opt = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_ldr) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=total_steps)
    best_vl = float('inf');
    epochs_ni = 0
    for epoch in range(EPOCHS):
        print(f"  Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_ldr, loss_fn, opt, DEVICE, sched)
        valid_loss = validate_epoch(model, valid_ldr, loss_fn, DEVICE)
        print(f"    Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        if valid_loss < best_vl:
            best_vl = valid_loss;
            torch.save(model.state_dict(), model_save_path);
            print(f"    Best model saved.");
            epochs_ni = 0
        else:
            epochs_ni += 1;
            if epochs_ni >= EARLY_STOPPING_PATIENCE: print(f"    Early stopping."); break
    print("-" * 20)


# --- Main Execution ---

# 1. Load/Aggregate Data
df_processed = load_and_aggregate_convabuse(
    FULL_DATA_FILE, PROCESSED_DATA_CACHE, GROUPING_KEYS, MIN_ANNOTATORS
)

if not df_processed.empty:
    # 2. Split Data
    print("Splitting aggregated data...")
    df_train_val, df_test = train_test_split(df_processed, test_size=TEST_SPLIT_SIZE, random_state=42)
    relative_valid_size = VALID_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
    df_train, df_valid = train_test_split(df_train_val, test_size=relative_valid_size, random_state=42)
    print(f"Split sizes: Train={len(df_train)}, Validation={len(df_valid)}, Test={len(df_test)}")

    split_ids = {  # Use the generated instance_id
        'train_ids': df_train['instance_id'].tolist(),
        'valid_ids': df_valid['instance_id'].tolist(),
        'test_ids': df_test['instance_id'].tolist()
    }
    with open(SPLIT_INFO_PATH, 'wb') as f:
        pickle.dump(split_ids, f)
    print(f"Split IDs saved to {SPLIT_INFO_PATH}")

    # 3. Initialize Tokenizer
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
    print(f"Split info saved to: {SPLIT_INFO_PATH}")

else:
    print("Failed to load or process data. Training script exiting.")