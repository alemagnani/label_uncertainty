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
import joblib
import warnings
from collections import defaultdict

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
PROCESSED_DATA_DIR = '/home/alessandro/Documents/convabuse'
MODEL_DIR = '/home/alessandro/Documents/convabuse'
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'convabuse_aggregated.pkl')
MODEL_A_PATH = os.path.join(MODEL_DIR, 'model_A_bert_noctx.pth')
MODEL_B_PATH = os.path.join(MODEL_DIR, 'model_B_bert_ctx.pth')
SPLIT_INFO_PATH = os.path.join(MODEL_DIR, 'train_val_test_split_ids.pkl')

# Model parameters (must match training script)
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 5
BATCH_SIZE = 32

# Metric Parameters
N_MONTE_CARLO_EMD = 1000
EPSILON = 1e-9

# n-bin Analysis Configuration
MIN_ANNOTATORS_EVAL = 2 # Use the min annotators from preprocessing
N_BINS_CONFIG = [
    (MIN_ANNOTATORS_EVAL, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))
]

# Misc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
K = 5 # Number of severity classes
ORDINAL_INDICES = np.arange(K)

# --- Check for Required Files ---
required_files = [MODEL_A_PATH, MODEL_B_PATH, PROCESSED_DATA_PATH, SPLIT_INFO_PATH]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print("Error: Missing required files:")
    for f in missing_files: print(f" - {f}")
    print("Please run preprocess_convabuse.py and train_convabuse_models.py first.")
    exit()

# --- Model Architecture Definition (Must match training) ---
# (Code omitted for brevity - keep MLP & ConvAbuseBertModel classes)
class BertClassificationHead(nn.Module):
    def __init__(self,hidden_dim,output_dim):
        super().__init__(); self.l1=nn.Linear(768,hidden_dim); self.relu=nn.ReLU()
        self.d=nn.Dropout(0.1); self.l2=nn.Linear(hidden_dim,output_dim)
    def forward(self,bo):
        cls=bo.last_hidden_state[:,0,:]; x=self.d(cls); x=self.l1(x); x=self.relu(x); x=self.d(x); return self.l2(x)
class ConvAbuseBertModel(nn.Module):
    def __init__(self, bmn, hhd, od):
        super().__init__(); self.bert=BertModel.from_pretrained(bmn); self.head=BertClassificationHead(hhd,od)
    def forward(self, ii, am): return self.head(self.bert(input_ids=ii, attention_mask=am))

# --- PyTorch Dataset for Evaluation ---
# (Code omitted for brevity - keep ConvAbuseEvalDataset class, uses instance_id)
class ConvAbuseEvalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, use_context):
        self.tokenizer=tokenizer; self.data=dataframe; self.max_len=max_len; self.use_context=use_context
        self.instance_ids = dataframe['instance_id'].tolist() # Use instance_id
    def __len__(self): return len(self.data)
    def __getitem__(self, index):
        row = self.data.iloc[index]; user_text = str(row['user']); inst_id = self.instance_ids[index]
        if self.use_context:
            pa=str(row.get('prev_agent','')); pu=str(row.get('prev_user','')); ag=str(row.get('agent',''))
            tp=[part for part in [pa, pu, ag, user_text] if part]; ft=f" {self.tokenizer.sep_token} ".join(tp)
        else: ft = user_text
        enc = self.tokenizer.encode_plus(ft, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt')
        return {'instance_id': inst_id, 'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten()}


# --- Metric Implementations ---
# (Code omitted for brevity - keep metric functions)
DISTANCE_MATRIX = np.abs(ORDINAL_INDICES[:, None] - ORDINAL_INDICES[None, :]) / (K - 1)
def calculate_emd_1d(p, q):
    p = p / (p.sum() + EPSILON); q = q / (q.sum() + EPSILON)
    try:
        uw = np.maximum(0, p); uw /= uw.sum() + EPSILON
        vw = np.maximum(0, q); vw /= vw.sum() + EPSILON
        return wasserstein_distance(ORDINAL_INDICES, ORDINAL_INDICES, uw, vw)
    except Exception: return np.nan
def calculate_ce(p, q):
    p = p / (p.sum() + EPSILON); q = q / (q.sum() + EPSILON)
    log_q = np.log(q + EPSILON)
    if not np.all(np.isfinite(log_q)): return np.nan
    ce = -np.sum(p * log_q)
    return ce if np.isfinite(ce) else np.nan
ALPHA = np.ones(K); ALPHA_0 = ALPHA.sum()
def calculate_expected_ce(counts, n, q):
    pa = ALPHA + counts; pa0 = ALPHA_0 + n
    ep = pa / (pa0 + EPSILON)
    return calculate_ce(ep, q)
def calculate_expected_emd(counts, n, q, n_samples=N_MONTE_CARLO_EMD):
    pa = ALPHA + counts
    if np.any(pa <= 0): pa = np.maximum(pa, EPSILON)
    try: sps = np.random.dirichlet(pa, n_samples)
    except ValueError:
        pa0 = ALPHA_0 + n; ep = pa / (pa0 + EPSILON)
        return calculate_emd_1d(ep, q) # Fallback
    emds = [calculate_emd_1d(ps, q) for ps in sps]
    vemds = [s for s in emds if not np.isnan(s)]
    return np.mean(vemds) if vemds else np.nan


# --- Main Evaluation Execution ---

# 1. Load Processed Data and Split Info
print(f"Loading processed data from: {PROCESSED_DATA_PATH}")
df_processed = pd.read_pickle(PROCESSED_DATA_PATH)
print(f"Loading split info from: {SPLIT_INFO_PATH}")
with open(SPLIT_INFO_PATH, 'rb') as f: split_ids = pickle.load(f)
test_ids = split_ids['test_ids']
# Use instance_id from split info to filter df_processed
df_test = df_processed[df_processed['instance_id'].isin(test_ids)].reset_index(drop=True)
print(f"Loaded {len(df_test)} test examples.")
test_labels_dict = df_test.set_index('instance_id').to_dict('index') # Use instance_id

# 2. Load Tokenizer
print(f"Loading tokenizer: {BERT_MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# 3. Load Trained Models
print("Loading trained models...")
model_A = ConvAbuseBertModel(BERT_MODEL_NAME, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
model_B = ConvAbuseBertModel(BERT_MODEL_NAME, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
model_A.load_state_dict(torch.load(MODEL_A_PATH, map_location=DEVICE))
model_B.load_state_dict(torch.load(MODEL_B_PATH, map_location=DEVICE))
model_A.eval(); model_B.eval()
print("Models loaded.")

# 4. Create Test Datasets and DataLoaders
test_dataset_A = ConvAbuseEvalDataset(df_test, tokenizer, MAX_LEN, use_context=False)
test_dataset_B = ConvAbuseEvalDataset(df_test, tokenizer, MAX_LEN, use_context=True)
test_loader_A = DataLoader(test_dataset_A, batch_size=BATCH_SIZE)
test_loader_B = DataLoader(test_dataset_B, batch_size=BATCH_SIZE)

# 5. Generate Predictions
predictions = defaultdict(dict) # Store predictions: {instance_id: {'q_a': ..., 'q_b': ...}}
print("Generating predictions for Model A (No Context)...")
with torch.no_grad():
    for batch in tqdm(test_loader_A, desc="Predicting Model A"):
        ids=batch['input_ids'].to(DEVICE); mask=batch['attention_mask'].to(DEVICE)
        inst_ids = batch['instance_id'] # Get instance_id
        outputs = model_A(ids, mask); probs = torch.softmax(outputs, dim=1).cpu().numpy()
        for i, inst_id in enumerate(inst_ids): predictions[inst_id]['q_a'] = probs[i]

print("Generating predictions for Model B (With Context)...")
with torch.no_grad():
    for batch in tqdm(test_loader_B, desc="Predicting Model B"):
        ids=batch['input_ids'].to(DEVICE); mask=batch['attention_mask'].to(DEVICE)
        inst_ids = batch['instance_id'] # Get instance_id
        outputs = model_B(ids, mask); probs = torch.softmax(outputs, dim=1).cpu().numpy()
        for i, inst_id in enumerate(inst_ids): predictions[inst_id]['q_b'] = probs[i]
print(f"Generated predictions for {len(predictions)} test examples.")


# 6. Calculate Metrics
print("Calculating evaluation metrics...")
# (Calculation loop and subsequent analysis/plotting code is identical to the
# previous evaluate_amazon_models.py script, just ensure it uses instance_id
# and the correct model names in labels/titles)
# ... [Paste the metric calculation loop, overall analysis, n-bin analysis,
#      and plotting code from the previous evaluate_amazon_models.py here,
#      making sure to use 'instance_id' instead of 'product_id' or 'example_no'
#      and changing plot titles/legends accordingly] ...

# --- Example of pasting and adapting the analysis part ---
results = []
for inst_id, preds in tqdm(predictions.items(), desc="Calculating Metrics"):
    if inst_id not in test_labels_dict: continue

    counts = test_labels_dict[inst_id]['counts']
    n = test_labels_dict[inst_id]['n']
    p_emp = counts / (n + EPSILON)

    if 'q_a' not in preds or 'q_b' not in preds: continue # Ensure both preds exist
    q_a = preds['q_a']; q_b = preds['q_b']
    q_a = q_a / (q_a.sum() + EPSILON); q_b = q_b / (q_b.sum() + EPSILON); p_emp = p_emp / (p_emp.sum() + EPSILON)

    instance_results = {'instance_id': inst_id, 'n': n} # Use instance_id
    # Metrics for Model A
    instance_results['CE_emp_A'] = calculate_ce(p_emp, q_a)
    instance_results['EMD_emp_A'] = calculate_emd_1d(p_emp, q_a)
    instance_results['E_CE_A'] = calculate_expected_ce(counts, n, q_a)
    instance_results['E_EMD_A'] = calculate_expected_emd(counts, n, q_a)
    # Metrics for Model B
    instance_results['CE_emp_B'] = calculate_ce(p_emp, q_b)
    instance_results['EMD_emp_B'] = calculate_emd_1d(p_emp, q_b)
    instance_results['E_CE_B'] = calculate_expected_ce(counts, n, q_b)
    instance_results['E_EMD_B'] = calculate_expected_emd(counts, n, q_b)
    results.append(instance_results)

df_results = pd.DataFrame(results); df_results.dropna(inplace=True)

if df_results.empty: print("\nNo valid results.")
else:
    # Overall Analysis
    print(f"\nCalculating overall average scores...")
    avg_scores = df_results.mean(numeric_only=True)
    ranking_data = {
        'Metric': ['CE_emp', 'E[CE]', 'EMD_emp', 'E[EMD]'],
        'Model A (No Context)': [ avg_scores.get(k, np.nan) for k in ['CE_emp_A', 'E_CE_A', 'EMD_emp_A', 'E_EMD_A']],
        'Model B (With Context)': [ avg_scores.get(k, np.nan) for k in ['CE_emp_B', 'E_CE_B', 'EMD_emp_B', 'E_EMD_B']]
    }
    df_ranking = pd.DataFrame(ranking_data).set_index('Metric')
    df_ranking['Winner'] = df_ranking.idxmin(axis=1, skipna=True)
    print("\n--- Overall Model Ranking (ConvAbuse - Test Set Evaluation) ---") # Changed title
    print(df_ranking.round(4).to_markdown(numalign="right", stralign="right"))
    df_plot_overall = df_ranking.drop(columns=['Winner']).reset_index().melt(id_vars='Metric', var_name='Model', value_name='Average Score')
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_plot_overall, x='Metric', y='Average Score', hue='Model', palette='viridis')
    plt.title('Overall Comparison - ConvAbuse Severity\n(Test Set Evaluation)') # Changed title
    plt.ylabel('Average Metric Score (Lower is Better)'); plt.xlabel('Evaluation Metric'); plt.xticks(rotation=0)
    for container in ax.containers: ax.bar_label(container, fmt='%.3f', padding=3)
    plt.ylim(bottom=0); plt.tight_layout(); plt.show()

    # n-bin Analysis
    print("\nPerforming analysis based on number of annotations (n)...")
    bin_labels=[f"{low}-{high if high!=float('inf') else '+'}" for low,high in N_BINS_CONFIG]
    def assign_bin(n):
        for i, (low, high) in enumerate(N_BINS_CONFIG):
            if low <= n <= high: return bin_labels[i]
        return "Other"
    df_results['n_bin'] = df_results['n'].apply(assign_bin)
    df_results['n_bin'] = pd.Categorical(df_results['n_bin'], categories=bin_labels, ordered=True)
    bin_analysis = df_results.groupby('n_bin', observed=False).agg(
        Count=('instance_id', 'count'), # Use instance_id
        **{f'{m}_{p}_mean': (f'{m}_{p}', 'mean') for m in ['CE_emp', 'E_CE', 'EMD_emp', 'E_EMD'] for p in ['A', 'B']}
    ).reset_index()
    print("\n--- Average Scores per Annotation Count (n) Bin ---") # Changed title
    print(bin_analysis.round(4).to_markdown(index=False, numalign="right", stralign="right"))

    # Visualize n-bin (Plot 1)
    df_plot_n = bin_analysis.melt(id_vars=['n_bin', 'Count'], var_name='Metric_Model', value_name='Average Score')
    df_plot_n = df_plot_n[df_plot_n['Metric_Model'].str.contains('_A_mean|_B_mean')]
    df_plot_n[['Metric_Type', 'Model', '_1', '_2']] = df_plot_n['Metric_Model'].str.split('_', expand=True)
    df_plot_n['Model_Name'] = df_plot_n['Model'].map({'A': 'Model A (No Ctx)', 'B': 'Model B (Ctx)'}) # Changed legend
    metric_types_plot=['CE_emp', 'E_CE', 'EMD_emp', 'E_EMD']; metric_names_map={'CE_emp':'CE Empirical', 'E_CE': 'E[CE]', 'EMD_emp': 'EMD Empirical', 'E_EMD': 'E[EMD]'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True); axes = axes.flatten()
    fig.suptitle('Model Comparison Across Annotation Count Bins (n) - ConvAbuse', fontsize=16, y=1.03) # Changed title
    for i, metric_key in enumerate(metric_types_plot):
        mfname = metric_names_map[metric_key]; df_met_bin = df_plot_n[df_plot_n['Metric_Model'].str.startswith(metric_key)]
        sns.pointplot(data=df_met_bin, x='n_bin', y='Average Score', hue='Model_Name', palette='viridis', ax=axes[i], markers=["o", "s"], linestyles=["--", "-"])
        axes[i].set_title(f'Metric: {mfname}'); axes[i].set_ylabel('Avg Score (Lower is Better)'); axes[i].set_xlabel(''); axes[i].tick_params(axis='x', rotation=45); axes[i].grid(True, axis='y', linestyle=':', alpha=0.7)
        if i >= 2 : axes[i].set_xlabel('Number of Annotations (n)')
        # Annotate counts
        try:
            counts_map = bin_analysis.set_index('n_bin')['Count']
            for tick_label in axes[i].get_xticklabels():
                 b_name = tick_label.get_text()
                 count = counts_map.get(b_name, 0)
                 tick_label.set_text(f"{b_name}\n(N={count})")
            axes[i].tick_params(axis='x', which='major', labelsize=9)
        except Exception as e: print(f"Plot 1 Count Annotation Error: {e}")
        axes[i].legend(title='Model', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()

    # Visualize n-bin (Plot 2)
    df_plot_n['Metric_Base'] = df_plot_n['Metric_Type'].replace({'CEemp': 'CE', 'ECE': 'CE', 'EMDemp': 'EMD', 'EEMD': 'EMD'})
    df_plot_n['Metric_Kind'] = df_plot_n['Metric_Type'].apply(lambda x: 'Expected' if x.startswith('E') else 'Empirical')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    fig.suptitle('Empirical vs. Expected Metrics Across Annotation Count Bins (n) - ConvAbuse', fontsize=16, y=1.03) # Changed title
    df_ce_bin = df_plot_n[df_plot_n['Metric_Base'] == 'CE']
    sns.pointplot(data=df_ce_bin, x='n_bin', y='Average Score', hue='Metric_Kind', palette='rocket', ax=axes[0], markers=["o", "s"], linestyles=["--", "-"], errorbar=('ci', 95))
    axes[0].set_title('Cross-Entropy (CE vs E[CE])'); axes[0].set_ylabel('Avg Score (Lower is Better)'); axes[0].set_xlabel('Number of Annotations (n)'); axes[0].tick_params(axis='x', rotation=45); axes[0].grid(True, axis='y', linestyle=':', alpha=0.7); axes[0].legend(title='Metric Kind', fontsize=9)
    df_emd_bin = df_plot_n[df_plot_n['Metric_Base'] == 'EMD']
    sns.pointplot(data=df_emd_bin, x='n_bin', y='Average Score', hue='Metric_Kind', palette='mako', ax=axes[1], markers=["o", "s"], linestyles=["--", "-"], errorbar=('ci', 95))
    axes[1].set_title('Earth Mover\'s Distance (EMD vs E[EMD])'); axes[1].set_ylabel('Avg Score (Lower is Better)'); axes[1].set_xlabel('Number of Annotations (n)'); axes[1].tick_params(axis='x', rotation=45); axes[1].grid(True, axis='y', linestyle=':', alpha=0.7); axes[1].legend(title='Metric Kind', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()


    print("\n--- Evaluation Finished ---")