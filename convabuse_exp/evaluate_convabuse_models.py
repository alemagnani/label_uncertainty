# evaluate_convabuse_models.py

import pandas as pd
import numpy as np
from scipy.special import digamma
from scipy.stats import wasserstein_distance
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import joblib # joblib isn't used here, maybe remove? Kept for now.
import warnings
from collections import defaultdict
import json # For saving results

# ML Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel # Need BertModel for architecture definition

# Suppress warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

print("--- ConvAbuse Model Evaluation Script ---")

# --- Configuration ---
# Directories and paths (must match training script)
OUTPUT_DIR = '/home/alessandro/Documents/convabuse' # Base directory for models and results
PROCESSED_DATA_DIR = OUTPUT_DIR # Data is in the output dir now
MODEL_DIR = OUTPUT_DIR

# This path MUST match AGGREGATED_DATA_PATH in the training script
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'convabuse_aggregated.pkl')
MODEL_A_PATH = os.path.join(MODEL_DIR, 'model_A_bert_noctx.pth')
MODEL_B_PATH = os.path.join(MODEL_DIR, 'model_B_bert_ctx.pth')
SPLIT_INFO_PATH = os.path.join(MODEL_DIR, 'train_val_test_split_ids.pkl')

# JSON Results Paths
OVERALL_RESULTS_JSON_PATH = os.path.join(MODEL_DIR, 'overall_ranking_convabuse.json')
NBIN_RESULTS_JSON_PATH = os.path.join(MODEL_DIR, 'n_bin_analysis_convabuse.json')
RAW_RESULTS_JSON_PATH = os.path.join(MODEL_DIR, 'raw_results_convabuse.json')

PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')  # Directory to save plots
os.makedirs(PLOT_DIR, exist_ok=True)

# Model parameters (must match training script)
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
HIDDEN_DIM = 256
K = 5 # Number of severity classes
OUTPUT_DIM = K
BATCH_SIZE = 32 # Can be larger for evaluation if memory allows

# Metric Parameters
N_MONTE_CARLO_EMD = 1000
EPSILON = 1e-9
ALPHA = np.ones(K); ALPHA_0 = ALPHA.sum() # Dirichlet prior parameters for expected metrics
ORDINAL_INDICES = np.arange(K) # Indices 0, 1, 2, 3, 4
DISTANCE_MATRIX = np.abs(ORDINAL_INDICES[:, None] - ORDINAL_INDICES[None, :]) / (K - 1) # Normalized distance matrix for EMD

# n-bin Analysis Configuration
# Ensure MIN_ANNOTATORS_EVAL matches MIN_ANNOTATORS in training (or is >=)
MIN_ANNOTATORS_EVAL = 2  # From training script pre-processing
N_BINS_CONFIG = [
    (MIN_ANNOTATORS_EVAL, 3), (4, 5), (6, 10), (11, float('inf'))
]

# Misc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Check for Required Files ---
required_files = [MODEL_A_PATH, MODEL_B_PATH, PROCESSED_DATA_PATH, SPLIT_INFO_PATH]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print("Error: Missing required files:")
    for f in missing_files: print(f" - {f}")
    print("Please ensure preprocess_convabuse.py (if separate) and train_convabuse_models.py finished successfully.")
    exit()

# --- Model Architecture Definition (Must match training) ---
class BertClassificationHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.bert_output_dim = 768 # Standard for bert-base-*
        self.l1 = nn.Linear(self.bert_output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1) # Dropout is typically active during eval too if part of trained model
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, bert_output):
        cls_token_state = bert_output.last_hidden_state[:, 0, :]
        # Don't apply dropout explicitly here if it's within the layers; model.eval() handles it.
        x = self.l1(cls_token_state)
        x = self.relu(x)
        x = self.dropout(x) # Keep dropout if it was trained with it
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

# --- PyTorch Dataset for Evaluation ---
class ConvAbuseEvalDataset(Dataset):
    # Note: Needs 'instance_id' column in the dataframe
    def __init__(self, dataframe, tokenizer, max_len, use_context):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.use_context = use_context
        # Store instance IDs to return them with batches
        self.instance_ids = dataframe['instance_id'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        instance_id = self.instance_ids[index] # Get the instance_id
        user_text = str(row['user']) # Ensure text is string

        if self.use_context:
            pa = str(row.get('prev_agent', ''))
            pu = str(row.get('prev_user', ''))
            ag = str(row.get('agent', ''))
            context_parts = [part for part in [pa, pu, ag, user_text] if part]
            full_text = f" {self.tokenizer.sep_token} ".join(context_parts)
        else:
            full_text = user_text

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
            'instance_id': instance_id, # Return instance_id
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
            # No targets needed for evaluation predictions
        }


# --- Metric Implementations ---
def normalize_dist(p):
    p_norm = np.maximum(0, p) # Ensure non-negative
    p_norm = p_norm / (p_norm.sum() + EPSILON)
    return p_norm

def calculate_emd_1d(p, q):
    """Calculates 1D Wasserstein distance (Earth Mover's Distance) between two probability distributions."""
    p_norm = normalize_dist(p)
    q_norm = normalize_dist(q)
    try:
        # wasserstein_distance expects 1D arrays of values and corresponding weights
        # Here, values are the class indices, weights are the probabilities
        # Ensure weights sum to 1 (handled by normalize_dist)
        return wasserstein_distance(ORDINAL_INDICES, ORDINAL_INDICES, u_weights=p_norm, v_weights=q_norm)
    except Exception as e:
        # Catch potential errors, e.g., if distributions are degenerate after normalization
        # print(f"Warning: EMD calculation failed for p={p}, q={q}. Error: {e}. Returning NaN.")
        return np.nan

def calculate_ce(p, q):
    """Calculates Cross-Entropy H(p, q) = - sum(p_i * log(q_i))."""
    p_norm = normalize_dist(p)
    q_norm = normalize_dist(q) # Ensure q is normalized too
    # Add epsilon to q to prevent log(0)
    log_q = np.log(q_norm + EPSILON)
    # Check if log_q resulted in NaNs or Infs (e.g., if q_norm was 0 everywhere)
    if not np.all(np.isfinite(log_q)):
        # print(f"Warning: Non-finite values in log(q) for p={p}, q={q}. Returning NaN.")
        return np.nan
    ce = -np.sum(p_norm * log_q)
    return ce if np.isfinite(ce) else np.nan # Final check for the sum


def calculate_expected_ce(counts, n, q):
    """Calculates Expected Cross-Entropy E_{p~Dir(alpha+counts)}[H(p, q)] using analytical form."""
    if n == 0: return np.nan # Avoid division by zero if n=0
    q_norm = normalize_dist(q)
    log_q = np.log(q_norm + EPSILON)
    if not np.all(np.isfinite(log_q)): return np.nan

    posterior_alpha = ALPHA + counts
    posterior_alpha_0 = ALPHA_0 + n

    # E[p_i] = (alpha_i + counts_i) / (alpha_0 + n)
    # expected_p = posterior_alpha / (posterior_alpha_0 + EPSILON) # Not needed directly for E[CE] formula

    # Using the formula E[CE] = - sum( E[p_i] * log(q_i) )
    # E[CE] = - sum{ [(alpha_i + count_i) / (alpha_0 + n)] * log(q_i) } # This is CE(E[p], q)
    # The paper likely refers to: E[CE] = - sum{ E[p_i] * log(q_i) } + H(E[p]) ??? No.
    # Let's use the alternative calculation: E[CE] = sum( (psi(alpha_0+n) - psi(alpha_i+counts_i)) * E[p_i] ) + H(E[p], q) -> seems overly complex
    # Let's stick to the simpler E[CE] = CE(E[p], q) as used in some implementations
    # This represents the cross-entropy between the *expected* posterior distribution and the prediction.

    expected_p = posterior_alpha / (posterior_alpha_0 + EPSILON)
    return calculate_ce(expected_p, q_norm)


def calculate_expected_emd(counts, n, q, n_samples=N_MONTE_CARLO_EMD):
    """Calculates Expected EMD E_{p~Dir(alpha+counts)}[EMD(p, q)] via Monte Carlo sampling."""
    if n == 0: return np.nan
    q_norm = normalize_dist(q)
    posterior_alpha = ALPHA + counts

    # Handle cases where posterior_alpha might have non-positive values due to counts issues (shouldn't happen)
    if np.any(posterior_alpha <= 0):
        # Fallback: calculate EMD using the expected distribution E[p]
        # print(f"Warning: Non-positive values in posterior alpha {posterior_alpha}. Falling back to EMD(E[p], q).")
        posterior_alpha_0 = ALPHA_0 + n
        expected_p = posterior_alpha / (posterior_alpha_0 + EPSILON)
        return calculate_emd_1d(expected_p, q_norm)

    try:
        # Sample multiple distributions 'p' from the posterior Dirichlet
        sampled_ps = np.random.dirichlet(posterior_alpha, n_samples)
    except ValueError as e:
        # Catch potential error if posterior_alpha is invalid for Dirichlet
        # print(f"Warning: Dirichlet sampling failed for alpha={posterior_alpha}. Error: {e}. Falling back to EMD(E[p], q).")
        posterior_alpha_0 = ALPHA_0 + n
        expected_p = posterior_alpha / (posterior_alpha_0 + EPSILON)
        return calculate_emd_1d(expected_p, q_norm) # Fallback

    # Calculate EMD for each sampled 'p' against the fixed prediction 'q'
    emds = [calculate_emd_1d(p_s, q_norm) for p_s in sampled_ps]

    # Filter out potential NaN results from EMD calculations
    valid_emds = [e for e in emds if not np.isnan(e)]

    # Return the average of the valid EMDs
    return np.mean(valid_emds) if valid_emds else np.nan


# --- Main Evaluation Execution ---

# 1. Load Processed Data and Split Info
print(f"Loading processed aggregated data from: {PROCESSED_DATA_PATH}")
try:
    df_processed = pd.read_pickle(PROCESSED_DATA_PATH)
except Exception as e:
    print(f"Error loading processed data file {PROCESSED_DATA_PATH}: {e}")
    exit()

print(f"Loading train/val/test split info from: {SPLIT_INFO_PATH}")
try:
    with open(SPLIT_INFO_PATH, 'rb') as f:
        split_ids = pickle.load(f)
    test_ids = set(split_ids['test_ids']) # Use set for faster lookups
except Exception as e:
    print(f"Error loading split info file {SPLIT_INFO_PATH}: {e}")
    exit()

# Filter the processed dataframe to get only the test set instances
df_test = df_processed[df_processed['instance_id'].isin(test_ids)].reset_index(drop=True)
if df_test.empty:
    print("Error: No test instances found after filtering based on split IDs. Check split info file.")
    exit()
print(f"Loaded {len(df_test)} test examples based on split IDs.")

print("\nAnnotation count distribution in test set:")
annotation_counts = df_test['n'].value_counts().sort_index()
print(annotation_counts)
print(f"Mean annotations per instance: {df_test['n'].mean():.2f}")
print(f"Median annotations per instance: {df_test['n'].median()}")
print(f"Min annotations per instance: {df_test['n'].min()}")
print(f"Max annotations per instance: {df_test['n'].max()}")


# Create a dictionary mapping instance_id to its ground truth data (counts, n) for easy lookup
test_labels_dict = df_test.set_index('instance_id')[['counts', 'n']].to_dict('index')
print(f"Created ground truth lookup for {len(test_labels_dict)} test instances.")

# 2. Load Tokenizer
print(f"Loading tokenizer: {BERT_MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# 3. Load Trained Models
print("Loading trained models...")
try:
    model_A = ConvAbuseBertModel(BERT_MODEL_NAME, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    model_B = ConvAbuseBertModel(BERT_MODEL_NAME, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    model_A.load_state_dict(torch.load(MODEL_A_PATH, map_location=DEVICE))
    model_B.load_state_dict(torch.load(MODEL_B_PATH, map_location=DEVICE))
    model_A.eval() # Set model to evaluation mode
    model_B.eval() # Set model to evaluation mode
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading model state dictionaries: {e}")
    exit()


# 4. Create Test Datasets and DataLoaders
print("Creating test datasets and dataloaders...")
test_dataset_A = ConvAbuseEvalDataset(df_test, tokenizer, MAX_LEN, use_context=False)
test_dataset_B = ConvAbuseEvalDataset(df_test, tokenizer, MAX_LEN, use_context=True)
test_loader_A = DataLoader(test_dataset_A, batch_size=BATCH_SIZE) # No shuffle needed
test_loader_B = DataLoader(test_dataset_B, batch_size=BATCH_SIZE)


# 5. Generate Predictions
# Store predictions: {instance_id: {'q_a': pred_dist_A, 'q_b': pred_dist_B}}
predictions = defaultdict(dict)
print("Generating predictions for Model A (No Context)...")
with torch.no_grad(): # Disable gradient calculations for inference
    for batch in tqdm(test_loader_A, desc="Predicting Model A"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        instance_ids_batch = batch['instance_id'] # Get instance_ids from batch

        outputs = model_A(input_ids=input_ids, attention_mask=attention_mask)
        # Convert logits to probabilities
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        # Store predictions mapped by instance_id
        for i, inst_id in enumerate(instance_ids_batch):
             # Check if instance_id exists in the ground truth dict before storing
             if inst_id in test_labels_dict:
                 predictions[inst_id]['q_a'] = probabilities[i]

print("Generating predictions for Model B (With Context)...")
with torch.no_grad():
    for batch in tqdm(test_loader_B, desc="Predicting Model B"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        instance_ids_batch = batch['instance_id']

        outputs = model_B(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        for i, inst_id in enumerate(instance_ids_batch):
             if inst_id in test_labels_dict: # Check again for model B
                predictions[inst_id]['q_b'] = probabilities[i]

# Verify predictions were generated for the expected number of test instances
print(f"Generated predictions for {len(predictions)} test instances with valid ground truth.")


# 6. Calculate Metrics
print("Calculating evaluation metrics for each test instance...")
results_list = []
processed_ids = set()

# Iterate through the instance_ids from the test set ground truth to ensure all are processed
# for inst_id, ground_truth in tqdm(test_labels_dict.items(), desc="Calculating Metrics"):
# Iterate through predictions dict to ensure we only calculate for instances we have preds for
for inst_id, preds_dict in tqdm(predictions.items(), desc="Calculating Metrics"):

    if inst_id in processed_ids: continue # Skip if already processed (shouldn't happen with dict)
    if inst_id not in test_labels_dict: continue # Skip if no ground truth (shouldn't happen with prediction logic)

    ground_truth = test_labels_dict[inst_id]
    counts = ground_truth['counts']
    n = ground_truth['n']

    # Ensure n is valid (>= min_annotators)
    if n < MIN_ANNOTATORS_EVAL:
        # print(f"Skipping instance {inst_id} due to insufficient annotations (n={n} < {MIN_ANNOTATORS_EVAL}).")
        continue

    # Calculate empirical distribution p_emp
    p_emp = counts / (n + EPSILON)
    p_emp = normalize_dist(p_emp) # Ensure it sums to 1

    # Get model predictions (predicted distributions q_a, q_b)
    if 'q_a' not in preds_dict or 'q_b' not in preds_dict:
        # print(f"Warning: Missing prediction for instance {inst_id}. Skipping.")
        continue # Ensure both predictions exist for a fair comparison row

    q_a = preds_dict['q_a']
    q_b = preds_dict['q_b']
    # Ensure predictions are valid distributions (normalize just in case)
    q_a = normalize_dist(q_a)
    q_b = normalize_dist(q_b)

    instance_results = {'instance_id': inst_id, 'n': n}

    # Calculate metrics for Model A (No Context)
    instance_results['CE_emp_A'] = calculate_ce(p_emp, q_a)
    instance_results['EMD_emp_A'] = calculate_emd_1d(p_emp, q_a)
    instance_results['E_CE_A'] = calculate_expected_ce(counts, n, q_a)
    instance_results['E_EMD_A'] = calculate_expected_emd(counts, n, q_a)

    # Calculate metrics for Model B (With Context)
    instance_results['CE_emp_B'] = calculate_ce(p_emp, q_b)
    instance_results['EMD_emp_B'] = calculate_emd_1d(p_emp, q_b)
    instance_results['E_CE_B'] = calculate_expected_ce(counts, n, q_b)
    instance_results['E_EMD_B'] = calculate_expected_emd(counts, n, q_b)

    results_list.append(instance_results)
    processed_ids.add(inst_id)

# Convert results to DataFrame
df_results = pd.DataFrame(results_list)
print(f"Calculated metrics for {len(df_results)} instances.")

# Drop rows where any metric calculation resulted in NaN
initial_count = len(df_results)
df_results.dropna(inplace=True)
dropped_count = initial_count - len(df_results)
if dropped_count > 0:
    print(f"Dropped {dropped_count} rows due to NaN metric values.")

# Check if any results remain after dropping NaNs
if df_results.empty:
    print("\nError: No valid results remaining after metric calculation and NaN filtering.")
else:
    print(f"Proceeding with analysis on {len(df_results)} valid result instances.")

    # 7. Overall Analysis & Ranking
    print(f"\nCalculating overall average scores...")
    avg_scores = df_results.mean(numeric_only=True) # Calculate mean for numeric columns only

    # Create a DataFrame for easy ranking visualization
    ranking_data = {
        'Metric': ['CE_emp', 'E[CE]', 'EMD_emp', 'E[EMD]'],
        'Model A (No Context)': [
            avg_scores.get('CE_emp_A', np.nan),
            avg_scores.get('E_CE_A', np.nan),
            avg_scores.get('EMD_emp_A', np.nan),
            avg_scores.get('E_EMD_A', np.nan)
        ],
        'Model B (With Context)': [
            avg_scores.get('CE_emp_B', np.nan),
            avg_scores.get('E_CE_B', np.nan),
            avg_scores.get('EMD_emp_B', np.nan),
            avg_scores.get('E_EMD_B', np.nan)
        ]
    }
    df_ranking = pd.DataFrame(ranking_data).set_index('Metric')

    # Determine the winner (lower score is better)
    df_ranking['Winner'] = df_ranking.idxmin(axis=1, skipna=True)

    print("\n--- Overall Model Ranking (ConvAbuse - Test Set Evaluation) ---")
    print(df_ranking.round(4).to_markdown(numalign="right", stralign="right"))

    # Save overall ranking to JSON
    try:
        df_ranking.reset_index().to_json(OVERALL_RESULTS_JSON_PATH, orient='records', indent=4)
        print(f"Overall ranking saved to: {OVERALL_RESULTS_JSON_PATH}")
    except Exception as e:
        print(f"Error saving overall ranking JSON: {e}")


    # Plot Overall Comparison
    df_plot_overall = df_ranking.drop(columns=['Winner']).reset_index().melt(
        id_vars='Metric', var_name='Model', value_name='Average Score'
    )
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_plot_overall, x='Metric', y='Average Score', hue='Model', palette='viridis')
    plt.title('Overall Comparison - ConvAbuse Severity\n(Test Set Evaluation - Lower is Better)')
    plt.ylabel('Average Metric Score')
    plt.xlabel('Evaluation Metric')
    plt.xticks(rotation=0)
    # Add bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    plt.ylim(bottom=0) # Start y-axis at 0
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'overall_comparison.png'), dpi=300, bbox_inches='tight')

    plt.show()

    # 8. n-bin Analysis
    print("\nPerforming analysis based on number of annotations (n)...")

    # Define bin labels dynamically from config
    bin_labels = [f"{low}-{high if high != float('inf') else '+'}" for low, high in N_BINS_CONFIG]



    def assign_bin(n):
            for i, (low, high) in enumerate(N_BINS_CONFIG):
                if low <= n <= high:
                    return bin_labels[i]
            return "Other"




    df_results['n_bin'] = df_results['n'].apply(assign_bin)
    # Convert n_bin to a categorical type with the defined order
    df_results['n_bin'] = pd.Categorical(df_results['n_bin'], categories=bin_labels, ordered=True)

    # Group by the ordered bins and calculate mean scores and counts
    bin_analysis = df_results.groupby('n_bin', observed=False).agg(
        Count=('instance_id', 'count'),
        # Calculate mean for each metric and model
        **{f'{m}_{p}_mean': (f'{m}_{p}', 'mean') for m in ['CE_emp', 'E_CE', 'EMD_emp', 'E_EMD'] for p in ['A', 'B']}
    ).reset_index()

    print("\n--- Average Scores per Annotation Count (n) Bin ---")
    print(bin_analysis.round(4).to_markdown(index=False, numalign="right", stralign="right"))

    # Save n-bin analysis to JSON
    try:
        bin_analysis.to_json(NBIN_RESULTS_JSON_PATH, orient='records', indent=4)
        print(f"N-bin analysis saved to: {NBIN_RESULTS_JSON_PATH}")
    except Exception as e:
        print(f"Error saving n-bin analysis JSON: {e}")

    # Save raw results to JSON (optional, can be large)
    try:
        # Convert n_bin category back to string for JSON compatibility if needed
        df_results_json = df_results.copy()
        df_results_json['n_bin'] = df_results_json['n_bin'].astype(str)
        df_results_json.to_json(RAW_RESULTS_JSON_PATH, orient='records', indent=4)
        print(f"Raw instance-level results saved to: {RAW_RESULTS_JSON_PATH}")
    except Exception as e:
        print(f"Error saving raw results JSON: {e}")


    # --- Visualize n-bin Analysis ---

    # Prepare data for plotting
    df_plot_n = bin_analysis.melt(id_vars=['n_bin', 'Count'], var_name='Metric_Model', value_name='Average Score')
    # Filter only the mean score columns
    df_plot_n = df_plot_n[df_plot_n['Metric_Model'].str.contains('_mean')]
    # Extract Metric Type and Model from the combined column name
    df_plot_n[['Metric_Type', 'Model', '_']] = df_plot_n['Metric_Model'].str.rsplit('_', n=2, expand=True) # Split from right
    # Map model letter to full name
    df_plot_n['Model_Name'] = df_plot_n['Model'].map({'A': 'Model A (No Ctx)', 'B': 'Model B (Ctx)'})
    # Define which metrics to plot and their display names
    metric_types_plot = ['CE_emp', 'E_CE', 'EMD_emp', 'E_EMD']
    metric_names_map = {'CE_emp': 'CE Empirical', 'E_CE': 'E[CE]', 'EMD_emp': 'EMD Empirical', 'E_EMD': 'E[EMD]'}
    df_plot_n['Metric_Name'] = df_plot_n['Metric_Type'].map(metric_names_map)

    # Plot 1: Model Comparison Across Bins
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    fig.suptitle('Model Comparison Across Annotation Count Bins - ConvAbuse Abuse Severity', fontsize=16, y=1.02)
    #fig.suptitle('Model Comparison Across Annotation Count Bins (n) - ConvAbuse', fontsize=16, y=1.02)

    counts_map = bin_analysis.set_index('n_bin')['Count'] # For annotating counts

    for i, metric_key in enumerate(metric_types_plot):
        metric_display_name = metric_names_map[metric_key]
        df_metric_bin = df_plot_n[df_plot_n['Metric_Type'] == metric_key]

        if df_metric_bin.empty:
             print(f"No data to plot for metric: {metric_display_name}")
             axes[i].set_title(f'Metric: {metric_display_name} (No Data)')
             axes[i].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
             continue

        sns.pointplot(data=df_metric_bin, x='n_bin', y='Average Score', hue='Model_Name',
                      palette='viridis', ax=axes[i], markers=["o", "s"], linestyles=["--", "-"], errorbar=None) # Use pointplot

        axes[i].set_title(f'Metric: {metric_display_name}')
        axes[i].set_ylabel('Avg Score (Lower is Better)')
        axes[i].set_xlabel('') # Remove x-label from top plots
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, axis='y', linestyle=':', alpha=0.7)

        if i >= 2: # Add x-label only to bottom plots
            axes[i].set_xlabel('Number of Annotations (n)')

        # Annotate counts on x-axis ticks
        try:
            current_labels = []
            for tick_label in axes[i].get_xticklabels():
                 bin_name = tick_label.get_text()
                 count = counts_map.get(bin_name, 0)
                 current_labels.append(f"{bin_name}\n(N={count})")
            axes[i].set_xticklabels(current_labels)
            axes[i].tick_params(axis='x', which='major', labelsize=9) # Adjust label size if needed
        except Exception as e:
            print(f"Plot 1 Count Annotation Error for metric {metric_display_name}: {e}")

        axes[i].legend(title='Model', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(PLOT_DIR, 'model_comparison_bins.png'), dpi=300, bbox_inches='tight')
    print(f"Model comparison across bins plot saved to: {os.path.join(PLOT_DIR, 'model_comparison_bins.png')}")

    plt.savefig(os.path.join(PLOT_DIR, 'empirical_vs_expected_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"Empirical vs expected metrics plot saved to: {os.path.join(PLOT_DIR, 'empirical_vs_expected_metrics.png')}")
    plt.show()


    # Plot 2: Empirical vs. Expected Metrics Across Bins
    # Prepare data further: Add Metric_Base (CE/EMD) and Metric_Kind (Empirical/Expected)
    df_plot_n['Metric_Base'] = df_plot_n['Metric_Type'].apply(lambda x: 'CE' if 'CE' in x else 'EMD')
    df_plot_n['Metric_Kind'] = df_plot_n['Metric_Type'].apply(lambda x: 'Expected' if x.startswith('E_') else 'Empirical')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    fig.suptitle('Empirical vs. Expected Metrics Across Annotation Count Bins - ConvAbuse Abuse Severity', fontsize=16,
                 y=1.03)
    #fig.suptitle('Empirical vs. Expected Metrics Across Annotation Count Bins (n) - ConvAbuse', fontsize=16, y=1.03)

    # Plot for CE metrics
    df_ce_bin = df_plot_n[df_plot_n['Metric_Base'] == 'CE']
    if not df_ce_bin.empty:
        sns.pointplot(data=df_ce_bin, x='n_bin', y='Average Score', hue='Metric_Kind',
                    palette='rocket', ax=axes[0], markers=["o", "s"], linestyles=["--", "-"], errorbar=('ci', 95)) # Add confidence intervals
        axes[0].set_title('Cross-Entropy (CE vs E[CE])')
        axes[0].set_ylabel('Avg Score (Lower is Better)')
        axes[0].set_xlabel('Number of Annotations (n)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, axis='y', linestyle=':', alpha=0.7)
        axes[0].legend(title='Metric Kind', fontsize=9)
        # Annotate counts on x-axis ticks for CE plot
        try:
            current_labels = []
            for tick_label in axes[0].get_xticklabels():
                 bin_name = tick_label.get_text().split('\n')[0] # Get original bin name if already annotated
                 count = counts_map.get(bin_name, 0)
                 current_labels.append(f"{bin_name}\n(N={count})")
            axes[0].set_xticklabels(current_labels)
            axes[0].tick_params(axis='x', which='major', labelsize=9)
        except Exception as e:
             print(f"Plot 2 CE Count Annotation Error: {e}")

    else:
        axes[0].set_title('Cross-Entropy (CE vs E[CE]) - No Data')
        axes[0].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)


    # Plot for EMD metrics
    df_emd_bin = df_plot_n[df_plot_n['Metric_Base'] == 'EMD']
    if not df_emd_bin.empty:
        sns.pointplot(data=df_emd_bin, x='n_bin', y='Average Score', hue='Metric_Kind',
                    palette='mako', ax=axes[1], markers=["o", "s"], linestyles=["--", "-"], errorbar=('ci', 95)) # Add confidence intervals
        axes[1].set_title('Earth Mover\'s Distance (EMD vs E[EMD])')
        axes[1].set_ylabel('Avg Score (Lower is Better)')
        axes[1].set_xlabel('Number of Annotations (n)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, axis='y', linestyle=':', alpha=0.7)
        axes[1].legend(title='Metric Kind', fontsize=9)
        # Annotate counts on x-axis ticks for EMD plot
        try:
             current_labels = []
             for tick_label in axes[1].get_xticklabels():
                  bin_name = tick_label.get_text().split('\n')[0] # Get original bin name
                  count = counts_map.get(bin_name, 0)
                  current_labels.append(f"{bin_name}\n(N={count})")
             axes[1].set_xticklabels(current_labels)
             axes[1].tick_params(axis='x', which='major', labelsize=9)
        except Exception as e:
            print(f"Plot 2 EMD Count Annotation Error: {e}")

    else:
        axes[1].set_title('Earth Mover\'s Distance (EMD vs E[EMD]) - No Data')
        axes[1].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show()


    print("\n--- Evaluation Finished ---")