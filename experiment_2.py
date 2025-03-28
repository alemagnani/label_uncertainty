import pandas as pd
import numpy as np
from scipy.special import digamma
from scipy.stats import wasserstein_distance
from tqdm.notebook import tqdm  # Use tqdm.notebook in Jupyter/Colab
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os  # Import os module

# --- Configuration ---
# Define paths for your train, validation, and test files
BASE_PATH = '/Users/alemagnani/Downloads/ConvAbuse'  # Assuming files are in the same directory as the script
TRAIN_FILE = 'ConvAbuseEMNLPtrain.csv'
VALID_FILE = 'ConvAbuseEMNLPvalid.csv'  # Or your specific validation filename
TEST_FILE = 'ConvAbuseEMNLPtest.csv'

TRAIN_DATA_PATH = os.path.join(BASE_PATH, TRAIN_FILE)
VALID_DATA_PATH = os.path.join(BASE_PATH, VALID_FILE)
TEST_DATA_PATH = os.path.join(BASE_PATH, TEST_FILE)  # <<< We will evaluate on this

N_MONTE_CARLO_EMD = 1000  # Number of samples for E[EMD] approximation
MIN_ANNOTATIONS = 2  # Minimum number of annotations per instance
EPSILON = 1e-9  # Small value for numerical stability

# Ordinal mapping: -3 -> 0, -2 -> 1, -1 -> 2, 0 -> 3, 1 -> 4
SEVERITY_LEVELS = [-3, -2, -1, 0, 1]
K = len(SEVERITY_LEVELS)
ORDINAL_INDICES = np.arange(K)  # [0, 1, 2, 3, 4]


# --- Data Loading and Preprocessing ---
def load_and_process_convabuse(filepath):
    """Loads ConvAbuse data from a specific file and extracts severity counts."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()  # Return empty DataFrame

    df = pd.read_csv(filepath)
    processed_data = []

    print(f"Processing ConvAbuse data from {os.path.basename(filepath)}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        counts = np.zeros(K, dtype=int)
        n = 0
        # Check annotations from up to 8 annotators
        for annotator_idx in range(1, 9):
            annotated = False
            # Check if any severity label exists for this annotator
            has_annotator_data = any(f'Annotator{annotator_idx}_is_abuse.{sev}' in row for sev in SEVERITY_LEVELS)

            if has_annotator_data:
                found_label = False
                for severity_idx, severity_val in enumerate(SEVERITY_LEVELS):
                    col_name = f'Annotator{annotator_idx}_is_abuse.{severity_val}'
                    # Ensure column exists and value is 1 (not NaN or 0)
                    if col_name in row and pd.notna(row[col_name]) and int(row[col_name]) == 1:
                        counts[severity_idx] += 1
                        found_label = True
                        break  # Each annotator provides one severity label per example
                if found_label:
                    n += 1  # Only count annotators who provided a valid label

        if n >= MIN_ANNOTATIONS:
            processed_data.append({
                'example_id': row['example_id'],
                'text': row['user'],  # Using 'user' utterance as input text
                'counts': counts,
                'n': n
            })
        # Optional: Add handling for rows with n < MIN_ANNOTATIONS if needed
        # else:
        #     print(f"Skipping example_id {row['example_id']} with {n} annotations.")

    print(
        f"Finished processing {os.path.basename(filepath)}. Found {len(processed_data)} instances with >= {MIN_ANNOTATIONS} annotations.")
    return pd.DataFrame(processed_data)


# --- Simulate Model Predictions ---
def simulate_predictions(counts, n):
    """Simulates predictions for Model A and Model B based on instance counts."""
    p_emp = counts / n

    # Model A: Closer to empirical, maybe more noise/categorical focus
    noise_a = np.random.uniform(-0.1, 0.1, K)
    q_a_unnorm = np.clip(p_emp + noise_a, 0, 1)
    # Handle cases where all might clip to 0
    if q_a_unnorm.sum() == 0:
        q_a_unnorm = np.ones(K)
    q_a = q_a_unnorm / (q_a_unnorm.sum() + EPSILON)

    # Model B: Smoother, penalizes distant errors implicitly (ordinal focus)
    # Simple smoothing kernel (Gaussian-like)
    kernel = np.array([0.1, 0.8, 0.1])  # Weights center more
    p_emp_padded = np.pad(p_emp, (len(kernel) // 2, len(kernel) // 2), 'edge')  # Pad for convolution
    q_b_unnorm = np.convolve(p_emp_padded, kernel, mode='valid')
    # Handle potential all-zero case after convolution if p_emp was all zero (unlikely with MIN_ANNOTATIONS>=1)
    if q_b_unnorm.sum() == 0:
        q_b_unnorm = np.ones(K)
    q_b = q_b_unnorm / (q_b_unnorm.sum() + EPSILON)

    # Ensure q_a and q_b are slightly different but plausible
    # If they are too similar, add a bit more difference to Model B
    if np.allclose(q_a, q_b, atol=1e-2):
        noise_b_extra = np.random.normal(0, 0.05, K)  # Use normal noise
        q_b_unnorm_new = np.clip(q_b + noise_b_extra, 0, 1)
        if q_b_unnorm_new.sum() == 0:
            q_b_unnorm_new = np.ones(K)
        q_b = q_b_unnorm_new / (q_b_unnorm_new.sum() + EPSILON)

    # Final check for valid probability distributions
    q_a = np.clip(q_a, 0, 1)
    q_a /= (q_a.sum() + EPSILON)
    q_b = np.clip(q_b, 0, 1)
    q_b /= (q_b.sum() + EPSILON)

    return q_a, q_b


# --- Metric Implementations ---

# Ground distance for EMD (Normalized |i-j| / (K-1))
def emd_ground_distance(k_classes):
    """Creates the ground distance matrix d(i,j) = |i-j| / (K-1)."""
    indices = np.arange(k_classes)
    return np.abs(indices[:, None] - indices[None, :]) / (k_classes - 1)


DISTANCE_MATRIX = emd_ground_distance(K)  # Precompute


def calculate_emd_1d(p, q):
    """Calculates 1D EMD (Wasserstein distance) using SciPy."""
    # Ensure p and q are valid probability distributions
    p = p / (p.sum() + EPSILON)
    q = q / (q.sum() + EPSILON)
    return wasserstein_distance(ORDINAL_INDICES, ORDINAL_INDICES, p, q)


def calculate_ce(p, q):
    """Calculates Cross-Entropy H(p, q) = - sum(p_i * log(q_i))."""
    # Ensure p and q are valid distributions and handle log(0)
    p = p / (p.sum() + EPSILON)
    q = q / (q.sum() + EPSILON)
    return -np.sum(p * np.log(q + EPSILON))


# Expected Metrics (Bayesian)
ALPHA = np.ones(K)  # Uniform prior
ALPHA_0 = ALPHA.sum()


def calculate_expected_ce(counts, n, q):
    """Calculates Expected Cross-Entropy E[CE(p, q)] using Formula 5."""
    posterior_alpha = ALPHA + counts
    posterior_alpha_0 = ALPHA_0 + n
    expected_p = posterior_alpha / posterior_alpha_0
    return calculate_ce(expected_p, q)  # Reuse CE calculation


def calculate_expected_emd(counts, n, q, n_samples=N_MONTE_CARLO_EMD):
    """Calculates Expected EMD E[EMD(p, q)] using Monte Carlo (Formula 10)."""
    posterior_alpha = ALPHA + counts

    # Handle potential issue if posterior_alpha has non-positive values (shouldn't happen with ALPHA=1 and counts>=0)
    if np.any(posterior_alpha <= 0):
        print(f"Warning: Non-positive values in posterior_alpha: {posterior_alpha}. Using prior.")
        posterior_alpha = ALPHA.copy()  # Fallback to prior for sampling

    try:
        sampled_ps = np.random.dirichlet(posterior_alpha, n_samples)
    except ValueError as e:
        print(f"Error sampling Dirichlet with alpha={posterior_alpha}: {e}")
        # Fallback: calculate EMD using the expected probability E[p]
        posterior_alpha_0 = ALPHA_0 + n
        expected_p = posterior_alpha / posterior_alpha_0
        return calculate_emd_1d(expected_p, q)

    emd_samples = []
    for p_sample in sampled_ps:
        # Ensure sample is valid before calculating EMD
        p_sample = p_sample / (p_sample.sum() + EPSILON)
        q_norm = q / (q.sum() + EPSILON)
        emd_samples.append(calculate_emd_1d(p_sample, q_norm))

    return np.mean(emd_samples)


# --- Main Experiment ---
print("--- Experiment 2: Model Ranking Evaluation ---")
print("NOTE: Models are SIMULATED. Evaluation is performed on the TEST set.")
print(f"Train file: {TRAIN_FILE} (conceptual source of model training)")
print(f"Validation file: {VALID_FILE} (conceptual source of model tuning)")
print(f"Test file: {TEST_FILE} (used for evaluation)")

print("\nLoading TEST data...")
df_test = load_and_process_convabuse(TEST_DATA_PATH)

if df_test.empty:
    print("\nTest data could not be loaded or processed. Exiting.")
else:
    print(f"\nLoaded {len(df_test)} instances from the test set for evaluation.")

    results = []

    print("Calculating metrics for simulated models on the test set...")
    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        counts = row['counts']
        n = row['n']

        # Simulate predictions for this instance based on its *observed* counts
        # In a real scenario, models trained on TRAIN/VALID would predict here.
        q_a, q_b = simulate_predictions(counts, n)

        # Calculate empirical distribution (target for empirical metrics)
        p_emp = counts / n

        instance_results = {'example_id': row['example_id'], 'n': n}  # Store n for potential analysis

        # Ensure predictions are valid distributions before metric calculation
        q_a = q_a / (q_a.sum() + EPSILON)
        q_b = q_b / (q_b.sum() + EPSILON)
        p_emp = p_emp / (p_emp.sum() + EPSILON)

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

    # --- Analysis and Visualization ---
    print("\nCalculating average scores over the test set...")
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

    # Determine winner for each metric (lower is better)
    # Handle potential NaN values if calculation failed for some reason
    df_ranking['Winner'] = df_ranking.idxmin(axis=1, skipna=True)

    print("\n--- Model Ranking (Test Set Evaluation) ---")
    # Format for better readability
    print(df_ranking.round(4).to_markdown(numalign="right", stralign="right"))
    print("\n(Lower scores are better)")

    # Plotting
    df_plot = df_ranking.drop(columns=['Winner']).reset_index().melt(id_vars='Metric', var_name='Model',
                                                                     value_name='Average Score')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_plot, x='Metric', y='Average Score', hue='Model', palette='viridis')
    plt.title(
        'Comparison of Simulated Model Performance using Different Metrics\n(ConvAbuse Severity - Test Set Evaluation)')
    plt.ylabel('Average Metric Score (Lower is Better)')
    plt.xlabel('Evaluation Metric')
    plt.xticks(rotation=0)
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    plt.tight_layout()
    plt.show()

    print("\n--- Interpretation Notes ---")
    print("Compare the 'Winner' column in the table and the bar heights in the plot.")
    print(" - Do CE_emp and E[CE] rank the models the same way?")
    print(" - Do EMD_emp and E[EMD] rank the models the same way?")
    print(" - Most importantly: Does the ranking based on EMD metrics differ from the ranking based on CE metrics?")
    print("   (e.g., Model A wins on CE, but Model B wins on EMD?)")
    print(
        " - Observe the difference between empirical and expected metrics. Expected metrics incorporate prior belief and might yield different scores, especially if the test set has instances with low 'n'.")
    print(
        "\nReminder: Model predictions are SIMULATED based on test set observations to illustrate metric behavior. Real models would be trained on the train/validation sets.")