# train_amazon_models_enhanced.py

import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import random
from tqdm import tqdm
import os
import joblib
import warnings
import matplotlib as mpl

# ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import torch.distributions as D

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None

print("--- Enhanced Model Training Script ---")

# --- Configuration ---
# Replace with the path to your FULL Electronics TSV file
DATA_FILE = 'amazon_reviews_us_Electronics_v1_00.tsv'
DATA_PATH = '/home/alessandro/Documents/amazon_review/' + DATA_FILE

OUTPUT_DIR = '/home/alessandro/Documents/amazon_review/'
MODEL_A_PATH = os.path.join(OUTPUT_DIR, 'model_A.pth')
MODEL_B_PATH = os.path.join(OUTPUT_DIR, 'model_B.pth')
MODEL_C_PATH = os.path.join(OUTPUT_DIR, 'model_C.pth')
MODEL_D_PATH = os.path.join(OUTPUT_DIR, 'model_D.pth')
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.joblib')
AGG_DATA_CACHE = os.path.join(OUTPUT_DIR, 'aggregated_sampled_data.pkl')

# Data Processing & Sampling
SAMPLE_SIZE = 20000  # Increase sample size for training
MIN_REVIEWS = 5  # Increase minimum reviews for more stable targets
TEST_SPLIT_SIZE = 0.15  # Proportion for TEST set
VALID_SPLIT_SIZE = 0.10  # Proportion for VALIDATION set

# TF-IDF
MAX_FEATURES = 10000

# Model & Training
INPUT_DIM = MAX_FEATURES  # Should match TF-IDF output
HIDDEN_DIM = 128
OUTPUT_DIM = 5  # 1-5 stars
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10  # Adjust as needed
SMOOTHING_SIGMA = 0.5  # Sigma for Gaussian smoothing for Model B target
DROPOUT_RATE = 0.2  # For BONN and TORP
TRANSFORMER_NHEAD = 4  # Number of attention heads for TORP
TRANSFORMER_LAYERS = 2  # Number of transformer layers for TORP

# Misc
EPSILON = 1e-9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ordinal mapping and K
STAR_RATINGS = [1, 2, 3, 4, 5]
K = len(STAR_RATINGS)


# --- Data Loading and Aggregation ---
def load_aggregate_and_cache(filepath, cache_path, min_reviews=5):
    if os.path.exists(cache_path):
        print(f"Loading aggregated data from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"Aggregating reviews and titles from {filepath}...")
    product_data = defaultdict(lambda: {'counts': np.zeros(K, dtype=int), 'n': 0, 'titles': []})

    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return pd.DataFrame()

    try:
        chunksize = 100000
        required_cols = ['product_id', 'star_rating', 'product_title']
        with pd.read_csv(filepath, sep='\t', header=0, on_bad_lines='skip',
                         usecols=required_cols, chunksize=chunksize, low_memory=False,
                         dtype={'star_rating': 'Int64', 'product_title': 'str'}) as reader:

            for chunk in tqdm(reader, desc="Processing Chunks"):
                chunk.dropna(subset=required_cols, inplace=True)
                for _, row in chunk.iterrows():
                    product_id = row['product_id']
                    rating = int(row['star_rating'])
                    title = str(row['product_title'])  # Ensure title is string
                    if 1 <= rating <= K:
                        rating_index = rating - 1
                        product_data[product_id]['counts'][rating_index] += 1
                        product_data[product_id]['n'] += 1
                        # Store titles, could pick the most common later if needed
                        if title not in product_data[product_id]['titles']:
                            product_data[product_id]['titles'].append(title)

    except Exception as e:
        print(f"Error reading or processing file {filepath}: {e}")
        return pd.DataFrame()

    print(f"Finished aggregation. Total unique products found: {len(product_data)}")

    aggregated_list = []
    for pid, data in product_data.items():
        if data['n'] >= min_reviews:
            # Use the first title found for simplicity
            title = data['titles'][0] if data['titles'] else ""
            aggregated_list.append({
                'product_id': pid,
                'counts': data['counts'],
                'n': data['n'],
                'title': title
            })

    df_aggregated = pd.DataFrame(aggregated_list)
    print(f"Filtered products: {len(df_aggregated)} products with >= {min_reviews} reviews.")

    # Cache the result
    df_aggregated.to_pickle(cache_path)
    print(f"Aggregated data cached to {cache_path}")
    return df_aggregated


# --- Sampling and Splitting ---
def sample_and_split_data(df_aggregated, sample_size, test_size, valid_size):
    """Samples products and splits into train, validation, test."""
    if len(df_aggregated) <= sample_size:
        print("Sample size >= total products. Using all filtered products.")
        df_sampled = df_aggregated
    else:
        print(f"Sampling {sample_size} products randomly...")
        df_sampled = df_aggregated.sample(n=sample_size, random_state=42)

    # Split: First separate test set, then split remaining into train/validation
    df_temp, df_test = train_test_split(
        df_sampled, test_size=test_size, random_state=42
    )
    # Adjust validation size relative to the remaining temp data
    relative_valid_size = valid_size / (1.0 - test_size)
    df_train, df_valid = train_test_split(
        df_temp, test_size=relative_valid_size, random_state=42
    )

    print(f"Data split: Train={len(df_train)}, Validation={len(df_valid)}, Test={len(df_test)}")
    return df_train, df_valid, df_test


# --- PyTorch Dataset ---
class AmazonReviewDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert sparse features to dense tensor for models
        feature_tensor = torch.tensor(self.features[idx].todense()).float().squeeze(0)
        # Label tensor (empirical probability distribution)
        label_tensor = torch.tensor(self.labels[idx]).float()
        return feature_tensor, label_tensor


# --- Model Architectures ---

# Model A & B: Simple MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Output raw logits; softmax will be applied in loss or post-processing

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Model C: Bayesian Ordinal Neural Network
class BONN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # Use a single output for ordinal prediction (cumulative approach)
        self.theta = nn.Parameter(torch.zeros(output_dim - 1))
        self.beta = nn.Linear(hidden_dim // 2, 1)
        self.output_dim = output_dim

    def forward(self, x):
        # Feature extraction
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)

        # Projection to scalar
        f = self.beta(x).squeeze(-1)

        # Ensure thresholds are ordered (cumulative approach)
        theta = torch.cat([
            torch.tensor([-float('inf')], device=self.theta.device),
            torch.cumsum(F.softplus(self.theta), dim=0),
            torch.tensor([float('inf')], device=self.theta.device)
        ])

        # Compute probabilities for each ordinal class
        probs = []
        for i in range(self.output_dim):
            probs.append(torch.sigmoid(theta[i + 1] - f) - torch.sigmoid(theta[i] - f))

        return torch.stack(probs, dim=1)


# Model D: Transformer-Based Ordinal Rating Predictor
class TORP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        # Initial projection from sparse to dense
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Global attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Create dense embedding from sparse input
        x = self.input_proj(x)
        x = F.relu(x)

        # Reshape for transformer: [batch, seq_len=1, hidden_dim]
        x = x.unsqueeze(1)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Global attention pooling
        attn_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(x * attn_weights, dim=1)

        # Output layer
        logits = self.fc_out(x)

        return F.softmax(logits, dim=1)  # Direct probability output


# --- Smoothing Function ---
def smooth_distribution(p, sigma=SMOOTHING_SIGMA):
    """Applies Gaussian smoothing to a probability distribution."""
    smoothed_p = gaussian_filter1d(p, sigma=sigma, mode='nearest')
    # Renormalize after smoothing
    smoothed_p = np.maximum(0, smoothed_p)  # Ensure non-negative
    smoothed_p /= (smoothed_p.sum() + EPSILON)
    return smoothed_p


# --- Loss Functions ---
def cross_entropy_loss_manual(model_output_logits, target_probs):
    """ Manual CE loss calculation: -sum(target * log_softmax(logits)) """
    log_probs = F.log_softmax(model_output_logits, dim=1)
    loss = -torch.sum(target_probs * log_probs, dim=1)
    return loss.mean()


def kl_divergence_loss_manual(model_output_logits, target_probs):
    """ Manual KL loss calculation: sum(target * (log(target) - log_softmax(logits))) """
    model_log_probs = F.log_softmax(model_output_logits, dim=1)
    target_log_probs = torch.log(target_probs + EPSILON)  # Add epsilon for stability
    # Ensure target_log_probs doesn't have -inf where target_probs is 0
    target_log_probs = torch.where(target_probs > EPSILON, target_log_probs, torch.zeros_like(target_log_probs))

    kl_div = torch.sum(target_probs * (target_log_probs - model_log_probs), dim=1)
    return kl_div.mean()


def ordinal_nll_loss(pred, target):
    # For numerical stability
    eps = 1e-9
    # Ensure predictions sum to 1
    pred = pred / (pred.sum(dim=1, keepdim=True) + eps)
    # Negative log likelihood
    nll = -torch.sum(target * torch.log(pred + eps), dim=1)
    return nll.mean()


def emd_loss(pred, target, device=DEVICE):
    batch_size = pred.size(0)
    K = pred.size(1)

    # Create distance matrix (can be precomputed)
    ord_indices = torch.arange(K, device=device).float()
    dist_matrix = torch.abs(ord_indices.unsqueeze(1) - ord_indices.unsqueeze(0)) / (K - 1)

    total_loss = 0
    for i in range(batch_size):
        # Compute 1D Wasserstein distance (much faster for 1D)
        pred_cdf = torch.cumsum(pred[i], dim=0)
        target_cdf = torch.cumsum(target[i], dim=0)
        w_dist = torch.sum(torch.abs(pred_cdf - target_cdf))
        total_loss += w_dist

    return total_loss / batch_size


# --- Training and Validation Loops ---
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = loss_fn(outputs, labels)

        # Check for NaN loss
        if torch.isnan(loss):
            print("NaN loss detected! Skipping batch.")
            continue

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            if not torch.isnan(loss):  # Ignore NaN losses in validation avg
                total_loss += loss.item()

    return total_loss / len(dataloader)


# --- Main Execution ---

# 1. Load/Aggregate Data
df_aggregated = load_aggregate_and_cache(DATA_PATH, AGG_DATA_CACHE, MIN_REVIEWS)

if not df_aggregated.empty:
    # 2. Sample and Split
    df_train, df_valid, df_test = sample_and_split_data(
        df_aggregated, SAMPLE_SIZE, TEST_SPLIT_SIZE, VALID_SPLIT_SIZE
    )

    # Separate data needed for training/validation
    train_titles = df_train['title'].tolist()
    valid_titles = df_valid['title'].tolist()

    # Prepare labels (empirical distributions)
    train_labels_counts = np.stack(df_train['counts'].values)
    train_labels_n = df_train['n'].values
    train_p_emp = train_labels_counts / (train_labels_n[:, np.newaxis] + EPSILON)

    valid_labels_counts = np.stack(df_valid['counts'].values)
    valid_labels_n = df_valid['n'].values
    valid_p_emp = valid_labels_counts / (valid_labels_n[:, np.newaxis] + EPSILON)

    # Prepare smoothed labels for Model B validation (training labels smoothed in dataloader)
    valid_p_emp_smooth = np.array([smooth_distribution(p) for p in valid_p_emp])

    # 3. TF-IDF Vectorization
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
    train_features = vectorizer.fit_transform(train_titles)
    valid_features = vectorizer.transform(valid_titles)

    print(f"Feature matrix shape (Train): {train_features.shape}")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"TF-IDF Vectorizer saved to {VECTORIZER_PATH}")

    # 4. Create Datasets and DataLoaders
    # Model A uses p_emp directly
    train_dataset_A = AmazonReviewDataset(train_features, train_p_emp)
    valid_dataset_A = AmazonReviewDataset(valid_features, valid_p_emp)
    train_loader_A = DataLoader(train_dataset_A, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_A = DataLoader(valid_dataset_A, batch_size=BATCH_SIZE)

    # Model B needs smoothed targets during training
    train_p_emp_smooth = np.array([smooth_distribution(p) for p in train_p_emp])
    train_dataset_B = AmazonReviewDataset(train_features, train_p_emp_smooth)
    # For validation, compare against the smoothed target as that's what it optimizes for
    valid_dataset_B = AmazonReviewDataset(valid_features, valid_p_emp_smooth)
    train_loader_B = DataLoader(train_dataset_B, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_B = DataLoader(valid_dataset_B, batch_size=BATCH_SIZE)

    # Model C uses p_emp directly, like Model A
    train_dataset_C = AmazonReviewDataset(train_features, train_p_emp)
    valid_dataset_C = AmazonReviewDataset(valid_features, valid_p_emp)
    train_loader_C = DataLoader(train_dataset_C, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_C = DataLoader(valid_dataset_C, batch_size=BATCH_SIZE)

    # Model D uses p_emp directly as well
    train_dataset_D = AmazonReviewDataset(train_features, train_p_emp)
    valid_dataset_D = AmazonReviewDataset(valid_features, valid_p_emp)
    train_loader_D = DataLoader(train_dataset_D, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader_D = DataLoader(valid_dataset_D, batch_size=BATCH_SIZE)

    # 5. Initialize Models, Losses, Optimizers
    # Original Models (A & B)
    model_A = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    model_B = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)

    # Advanced Models (C & D)
    model_C = BONN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, dropout_rate=DROPOUT_RATE).to(DEVICE)
    model_D = TORP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                   nhead=TRANSFORMER_NHEAD,
                   num_layers=TRANSFORMER_LAYERS,
                   dropout=DROPOUT_RATE).to(DEVICE)

    # Define loss functions
    loss_fn_A = cross_entropy_loss_manual
    loss_fn_B = kl_divergence_loss_manual
    loss_fn_C = ordinal_nll_loss
    loss_fn_D = emd_loss

    # Define optimizers
    optimizer_A = optim.Adam(model_A.parameters(), lr=LEARNING_RATE)
    optimizer_B = optim.Adam(model_B.parameters(), lr=LEARNING_RATE)
    optimizer_C = optim.Adam(model_C.parameters(), lr=LEARNING_RATE)
    optimizer_D = optim.Adam(model_D.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop - Model A (Original MLP with CE Loss)
    print("\n--- Training Model A (MLP with Categorical CE Loss) ---")
    best_valid_loss_A = float('inf')
    epochs_no_improve_A = 0
    patience = 2  # Early stopping patience

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model_A, train_loader_A, loss_fn_A, optimizer_A, DEVICE)
        valid_loss = validate_epoch(model_A, valid_loader_A, loss_fn_A, DEVICE)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss_A:
            best_valid_loss_A = valid_loss
            torch.save(model_A.state_dict(), MODEL_A_PATH)
            print(f"  Best model A saved to {MODEL_A_PATH}")
            epochs_no_improve_A = 0
        else:
            epochs_no_improve_A += 1
            if epochs_no_improve_A >= patience:
                print(f"  Early stopping triggered for Model A after {epoch + 1} epochs.")
                break

    # 7. Training Loop - Model B (Original MLP with KL Loss vs Smoothed Target)
    print("\n--- Training Model B (MLP with KL Div Loss vs Smoothed Target) ---")
    best_valid_loss_B = float('inf')
    epochs_no_improve_B = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model_B, train_loader_B, loss_fn_B, optimizer_B, DEVICE)
        # Validate using KL loss against smoothed validation targets
        valid_loss = validate_epoch(model_B, valid_loader_B, loss_fn_B, DEVICE)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss_B:
            best_valid_loss_B = valid_loss
            torch.save(model_B.state_dict(), MODEL_B_PATH)
            print(f"  Best model B saved to {MODEL_B_PATH}")
            epochs_no_improve_B = 0
        else:
            epochs_no_improve_B += 1
            if epochs_no_improve_B >= patience:
                print(f"  Early stopping triggered for Model B after {epoch + 1} epochs.")
                break

    # 8. Training Loop - Model C (Bayesian Ordinal Neural Network)
    print("\n--- Training Model C (Bayesian Ordinal Neural Network) ---")
    best_valid_loss_C = float('inf')
    epochs_no_improve_C = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model_C, train_loader_C, loss_fn_C, optimizer_C, DEVICE)
        valid_loss = validate_epoch(model_C, valid_loader_C, loss_fn_C, DEVICE)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss_C:
            best_valid_loss_C = valid_loss
            torch.save(model_C.state_dict(), MODEL_C_PATH)
            print(f"  Best model C saved to {MODEL_C_PATH}")
            epochs_no_improve_C = 0
        else:
            epochs_no_improve_C += 1
            if epochs_no_improve_C >= patience:
                print(f"  Early stopping triggered for Model C after {epoch + 1} epochs.")
                break

    # 9. Training Loop - Model D (Transformer-Based Ordinal Rating Predictor)
    print("\n--- Training Model D (Transformer-Based Ordinal Rating Predictor) ---")
    best_valid_loss_D = float('inf')
    epochs_no_improve_D = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model_D, train_loader_D, loss_fn_D, optimizer_D, DEVICE)
        valid_loss = validate_epoch(model_D, valid_loader_D, loss_fn_D, DEVICE)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss_D:
            best_valid_loss_D = valid_loss
            torch.save(model_D.state_dict(), MODEL_D_PATH)
            print(f"  Best model D saved to {MODEL_D_PATH}")
            epochs_no_improve_D = 0
        else:
            epochs_no_improve_D += 1
            if epochs_no_improve_D >= patience:
                print(f"  Early stopping triggered for Model D after {epoch + 1} epochs.")
                break

    print("\nTraining complete. All models and vectorizer saved.")
    print(f"Model A saved to: {MODEL_A_PATH}")
    print(f"Model B saved to: {MODEL_B_PATH}")
    print(f"Model C saved to: {MODEL_C_PATH}")
    print(f"Model D saved to: {MODEL_D_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print("You can now run the evaluation script using these saved artifacts.")

else:
    print("Failed to load or aggregate data. Training script exiting.")