import pandas as pd
import numpy as np
from scipy.special import digamma
from scipy.stats import wasserstein_distance
from collections import defaultdict
import random
from tqdm.notebook import tqdm # Use tqdm.notebook for Jupyter/Colab
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress minor warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Pandas SettingWithCopyWarning suppression (use with caution)
pd.options.mode.chained_assignment = None

# --- Configuration ---
# !!! IMPORTANT: Replace with the path to your FULL Electronics TSV file !!!
# The sample file is too small for meaningful results.
DATA_FILE = 'amazon_reviews_us_Electronics_v1_00.tsv' 
DATA_PATH = './' + DATA_FILE # Assumes file in current directory

SAMPLE_SIZE = 10000      # Number of products to sample after aggregation
MIN_REVIEWS = 2         # Minimum reviews per product to include
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VALID_RATIO

N_MONTE_CARLO_EMD = 1000  # Number of samples for E[EMD] approximation
EPSILON = 1e-9            # Small value for numerical stability

# Ordinal mapping: 1* -> 0, 2* -> 1, 3* -> 2, 4* -> 3, 5* -> 4
STAR_RATINGS = [1, 2, 3, 4, 5]
K = len(STAR_RATINGS)
ORDINAL_INDICES = np.arange(K) # [0, 1, 2, 3, 4]

# --- Data Loading and Aggregation ---
def load_and_aggregate_reviews(filepath, min_reviews=2):
    """
    Loads review data line-by-line, aggregates ratings per product,
    and filters based on min_reviews.
    """
    print(f"Aggregating reviews from {filepath}...")
    product_ratings = defaultdict(lambda: {'counts': np.zeros(K, dtype=int), 'n': 0})
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return []

    try:
        # Use chunking for potentially large files, adjust chunksize as needed
        chunksize = 100000 
        # Get total lines for tqdm progress bar (optional, can be slow for huge files)
        # try:
        #     total_lines = sum(1 for line in open(filepath, 'r', encoding='utf-8')) -1 # Subtract header
        # except Exception:
        #     total_lines = None # Fallback if getting line count fails

        # Using context manager for file handling
        with pd.read_csv(filepath, sep='\t', header=0, on_bad_lines='skip', 
                         usecols=['product_id', 'star_rating'], 
                         chunksize=chunksize, low_memory=False, # low_memory=False sometimes helps parsing
                         dtype={'star_rating': 'Int64'}) as reader: # Use nullable Int64

            # with tqdm(total=total_lines) as pbar: # Optional progress bar
            for chunk in reader:
                chunk.dropna(subset=['product_id', 'star_rating'], inplace=True) # Drop rows where key info is missing
                for _, row in chunk.iterrows():
                    product_id = row['product_id']
                    rating = int(row['star_rating']) # Already checked for NA
                    if 1 <= rating <= K:
                        rating_index = rating - 1 # Map 1-5 stars to 0-4 index
                        product_ratings[product_id]['counts'][rating_index] += 1
                        product_ratings[product_id]['n'] += 1
                # if total_lines: pbar.update(len(chunk)) # Optional progress update

    except Exception as e:
        print(f"Error reading or processing file {filepath}: {e}")
        return [] # Return empty list on error

    print(f"Finished aggregation. Total unique products found: {len(product_ratings)}")

    # Convert to list and filter
    aggregated_list = []
    for pid, data in product_ratings.items():
        if data['n'] >= min_reviews:
            aggregated_list.append({
                'product_id': pid,
                'counts': data['counts'],
                'n': data['n']
            })
            
    print(f"Filtered products: {len(aggregated_list)} products with >= {min_reviews} reviews.")
    return aggregated_list

# --- Sampling ---
def sample_products(aggregated_data, sample_size):
    """Randomly samples products."""
    if len(aggregated_data) <= sample_size:
        print("Sample size >= total products. Using all filtered products.")
        return aggregated_data
    
    print(f"Sampling {sample_size} products randomly...")
    random.seed(42) # for reproducibility
    sampled_data = random.sample(aggregated_data, sample_size)
    return sampled_data

# --- Data Splitting ---
def split_data(sampled_products, train_ratio, valid_ratio):
    """Splits sampled products into train, validation, and test sets."""
    random.seed(42) # ensure consistent shuffling
    shuffled_products = random.sample(sampled_products, len(sampled_products)) # Shuffle the list

    n_total = len(shuffled_products)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    n_test = n_total - n_train - n_valid # Ensure all products are used

    train_set = shuffled_products[:n_train]
    valid_set = shuffled_products[n_train : n_train + n_valid]
    test_set = shuffled_products[n_train + n_valid :]

    print(f"Data split: Train={len(train_set)}, Validation={len(valid_set)}, Test={len(test_set)}")
    return train_set, valid_set, test_set

# --- Simulate Model Predictions ---
def simulate_predictions(counts, n):
    """Simulates predictions for Model A and Model B based on instance counts."""
    p_emp = counts / (n + EPSILON) # Avoid division by zero if n was somehow 0

    # Model A: Closer to empirical, maybe more noise/categorical focus
    noise_a = np.random.normal(0, 0.1, K) # Gaussian noise
    q_a_unnorm = np.clip(p_emp + noise_a, 0, 1)
    if q_a_unnorm.sum() < EPSILON: q_a_unnorm = np.ones(K) # Handle all zeros
    q_a = q_a_unnorm / (q_a_unnorm.sum() + EPSILON)

    # Model B: Smoother, penalizes distant errors implicitly (ordinal focus)
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1]) # 5-point smoothing kernel centered
    if K == 5: # Ensure kernel matches K
       p_emp_padded = np.pad(p_emp, (K//2, K//2), 'edge')
       q_b_unnorm = np.convolve(p_emp_padded, kernel, mode='valid')
    else: # Fallback to just adding less noise if K is not 5
        noise_b = np.random.normal(0, 0.05, K)
        q_b_unnorm = np.clip(p_emp + noise_b, 0, 1)
        
    if q_b_unnorm.sum() < EPSILON: q_b_unnorm = np.ones(K) # Handle all zeros
    q_b = q_b_unnorm / (q_b_unnorm.sum() + EPSILON)

    # Ensure predictions are valid
    q_a = np.clip(q_a, 0, 1) / (np.clip(q_a, 0, 1).sum() + EPSILON)
    q_b = np.clip(q_b, 0, 1) / (np.clip(q_b, 0, 1).sum() + EPSILON)
    
    return q_a, q_b


# --- Metric Implementations ---

# Ground distance for EMD (Normalized |i-j| / (K-1))
DISTANCE_MATRIX = np.abs(ORDINAL_INDICES[:, None] - ORDINAL_INDICES[None, :]) / (K - 1)

def calculate_emd_1d(p, q):
    """Calculates 1D EMD (Wasserstein distance) using SciPy."""
    p = p / (p.sum() + EPSILON)
    q = q / (q.sum() + EPSILON)
    try:
        # Ensure distributions are 1D and probabilities are non-negative
        u_values, v_values = ORDINAL_INDICES, ORDINAL_INDICES
        u_weights = np.maximum(0, p)
        v_weights = np.maximum(0, q)
        u_weights /= u_weights.sum() + EPSILON
        v_weights /= v_weights.sum() + EPSILON
        return wasserstein_distance(u_values, v_values, u_weights, v_weights)
    except Exception as e:
        print(f"Error in wasserstein_distance: p={p}, q={q}, Error: {e}")
        return np.nan # Return NaN on error

def calculate_ce(p, q):
    """Calculates Cross-Entropy H(p, q)."""
    p = p / (p.sum() + EPSILON)
    q = q / (q.sum() + EPSILON)
    # Check for non-finite values that might arise from log(0)
    log_q = np.log(q + EPSILON)
    if not np.all(np.isfinite(log_q)):
        print(f"Warning: Non-finite values in log(q+eps). q={q}")
        # Handle or return NaN
        return np.nan 
    ce = -np.sum(p * log_q)
    return ce if np.isfinite(ce) else np.nan # Ensure result is finite


# Expected Metrics (Bayesian)
ALPHA = np.ones(K)  # Uniform prior
ALPHA_0 = ALPHA.sum()

def calculate_expected_ce(counts, n, q):
    """Calculates Expected Cross-Entropy E[CE(p, q)]."""
    posterior_alpha = ALPHA + counts
    posterior_alpha_0 = ALPHA_0 + n
    expected_p = posterior_alpha / (posterior_alpha_0 + EPSILON)
    return calculate_ce(expected_p, q) # Reuse CE calculation

def calculate_expected_emd(counts, n, q, n_samples=N_MONTE_CARLO_EMD):
    """Calculates Expected EMD E[EMD(p, q)] using Monte Carlo."""
    posterior_alpha = ALPHA + counts

    if np.any(posterior_alpha <= 0):
        print(f"Warning: Non-positive values in posterior_alpha: {posterior_alpha}. Falling back.")
        posterior_alpha = np.maximum(posterior_alpha, EPSILON) # Ensure positive

    try:
        sampled_ps = np.random.dirichlet(posterior_alpha, n_samples)
    except ValueError as e:
        print(f"Error sampling Dirichlet with alpha={posterior_alpha}: {e}")
        # Fallback: calculate EMD using the expected probability E[p]
        posterior_alpha_0 = ALPHA_0 + n
        expected_p = posterior_alpha / (posterior_alpha_0 + EPSILON)
        return calculate_emd_1d(expected_p, q)

    emd_samples = [calculate_emd_1d(p_sample, q) for p_sample in sampled_ps]
    
    # Filter out potential NaNs from EMD calculation errors
    valid_emd_samples = [s for s in emd_samples if not np.isnan(s)]
    if not valid_emd_samples:
        print(f"Warning: All EMD samples were NaN for counts={counts}, n={n}, q={q}")
        return np.nan 
        
    return np.mean(valid_emd_samples)


# --- Main Experiment ---
print("--- Experiment 2: Model Ranking Evaluation (Amazon Electronics Data) ---")
print("NOTE: Processing potentially large data file. This may take time.")
print("NOTE: Models are SIMULATED based on test set data. Evaluation is performed on the TEST set.")

# 1. Aggregate Data
aggregated_data = load_and_aggregate_reviews(DATA_PATH, min_reviews=MIN_REVIEWS)

if not aggregated_data:
    print("No data aggregated. Exiting.")
else:
    # 2. Sample Products
    sampled_data = sample_products(aggregated_data, SAMPLE_SIZE)

    # 3. Split Data
    train_products, valid_products, test_products = split_data(sampled_data, TRAIN_RATIO, VALID_RATIO)

    # 4. Evaluate on Test Set
    results = []
    print(f"\nCalculating metrics for simulated models on the test set ({len(test_products)} products)...")
    for product_data in tqdm(test_products):
        counts = product_data['counts']
        n = product_data['n']

        # Simulate predictions based on this product's observed counts
        q_a, q_b = simulate_predictions(counts, n)

        # Calculate empirical distribution (target for empirical metrics)
        p_emp = counts / (n + EPSILON)

        instance_results = {'product_id': product_data['product_id'], 'n': n}

        # Calculate metrics for Model A
        instance_results['CE_emp_A'] = calculate_ce(p_emp, q_a)
        instance_results['EMD_emp_A'] = calculate_emd_1d(p_emp, q_a)
        instance_results['E_CE_A'] = calculate_expected_ce(counts, n, q_a)
        instance_results['E_EMD_A'] = calculate_expected_emd(counts, n, q_a)

        # Calculate metrics for Model B
        instance_results['CE_emp_B'] = calculate_ce(p_emp, q_b)
        instance_results['EMD_emp_B'] = calculate_emd_1d(p_emp, q_b)
        instance_results['E_CE_B'] = calculate_expected_ce(counts, n, q_b)
        instance_results['E_EMD_B'] = calculate_expected_emd(counts, n, q_b)

        results.append(instance_results)

    df_results = pd.DataFrame(results)
    df_results.dropna(inplace=True) # Drop rows where metric calculation might have failed

    if df_results.empty:
        print("\nNo valid results after metric calculation. Cannot proceed with analysis.")
    else:
        # 5. Analysis and Visualization
        print(f"\nCalculating average scores over {len(df_results)} test set products...")
        avg_scores = df_results.mean(numeric_only=True)

        # Create ranking table
        ranking_data = {
            'Metric': ['CE_emp', 'E[CE]', 'EMD_emp', 'E[EMD]'],
            'Model A (Simulated)': [
                avg_scores.get('CE_emp_A', np.nan), avg_scores.get('E_CE_A', np.nan),
                avg_scores.get('EMD_emp_A', np.nan), avg_scores.get('E_EMD_A', np.nan)
            ],
            'Model B (Simulated)': [
                avg_scores.get('CE_emp_B', np.nan), avg_scores.get('E_CE_B', np.nan),
                avg_scores.get('EMD_emp_B', np.nan), avg_scores.get('E_EMD_B', np.nan)
            ]
        }
        df_ranking = pd.DataFrame(ranking_data).set_index('Metric')
        df_ranking['Winner'] = df_ranking.idxmin(axis=1, skipna=True)

        print("\n--- Model Ranking (Amazon Electronics - Test Set Evaluation) ---")
        print(df_ranking.round(4).to_markdown(numalign="right", stralign="right"))
        print("\n(Lower scores are better)")

        # Plotting
        df_plot = df_ranking.drop(columns=['Winner']).reset_index().melt(id_vars='Metric', var_name='Model', value_name='Average Score')

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df_plot, x='Metric', y='Average Score', hue='Model', palette='viridis')
        plt.title('Comparison of Simulated Model Performance - Amazon Electronics Ratings\n(Test Set Evaluation)')
        plt.ylabel('Average Metric Score (Lower is Better)')
        plt.xlabel('Evaluation Metric')
        plt.xticks(rotation=0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3) # Add padding
        plt.ylim(bottom=0) # Ensure y-axis starts at 0
        plt.tight_layout()
        plt.show()

        print("\n--- Interpretation Notes ---")
        print("*   Compare the 'Winner' column and bar heights.")
        print("*   Key Question: Does the ranking from EMD_emp/E[EMD] differ from CE_emp/E[CE]?")
        print("*   This would suggest that considering the ordinal nature (via EMD) changes our assessment of which model ('A' or 'B') performs better.")
        print("*   Also compare empirical vs. expected metrics (e.g., CE_emp vs E[CE]). Differences highlight the impact of the Bayesian approach, especially with varying 'n'.")
        print("\nLIMITATION: Predictions are SIMULATED based on test data counts. For definitive results, train real models on product features using the train/validation sets.")
