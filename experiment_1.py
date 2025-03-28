# experiment1_synthetic_behavior.py

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.special import digamma, loggamma
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm # Use tqdm.notebook for Jupyter/Colab
import warnings

# Suppress minor warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration ---
K = 5  # Number of ordinal classes (e.g., 1-5 stars mapped to 0-4)
ORDINAL_INDICES = np.arange(K)

# True distribution (Example: slightly skewed towards higher ratings)
P_TRUE = np.array([0.05, 0.15, 0.3, 0.35, 0.15])
assert np.isclose(P_TRUE.sum(), 1.0), "P_TRUE must sum to 1"

# Prediction distributions
Q_GOOD = np.array([0.06, 0.17, 0.31, 0.33, 0.13]) # Close to P_TRUE
Q_BAD_ADJACENT = np.array([0.06, 0.31, 0.33, 0.17, 0.13]) # Confuses adjacent (2*/3* vs 3*/4*)
Q_BAD_DISTANT = np.array([0.33, 0.17, 0.1, 0.17, 0.23]) # Confuses distant (shifts mass to 1* and 5*)

Q_LIST = {
    "Good": Q_GOOD / Q_GOOD.sum(),
    "Bad (Adjacent Err)": Q_BAD_ADJACENT / Q_BAD_ADJACENT.sum(),
    "Bad (Distant Err)": Q_BAD_DISTANT / Q_BAD_DISTANT.sum(),
}

# Bayesian Prior (Uniform Dirichlet)
ALPHA = np.ones(K)
ALPHA_0 = ALPHA.sum()

# Experiment Parameters
N_VALUES = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000] # Sample sizes to test
N_REPETITIONS = 50       # Number of simulations per sample size for stability
N_MC_EMD = 500           # Monte Carlo samples for E[EMD]
EPSILON = 1e-9           # Small value for numerical stability

# --- Metric Implementations ---

def calculate_ce(p, q):
    """Calculates Cross-Entropy H(p, q)."""
    p = p / (p.sum() + EPSILON)
    q = q / (q.sum() + EPSILON)
    log_q = np.log(q + EPSILON)
    if not np.all(np.isfinite(log_q)): return np.nan
    ce = -np.sum(p * log_q)
    return ce if np.isfinite(ce) else np.nan

def calculate_kl(p, q):
    """Calculates KL Divergence D_KL(p || q)."""
    p = p / (p.sum() + EPSILON)
    q = q / (q.sum() + EPSILON)
    log_p_q = np.log((p + EPSILON) / (q + EPSILON))
    if not np.all(np.isfinite(log_p_q)): return np.nan
    # Handle 0 * log(0) cases - should be 0
    kl_terms = np.where(p > EPSILON, p * log_p_q, 0.0)
    kl = np.sum(kl_terms)
    return kl if np.isfinite(kl) else np.nan

def calculate_emd_1d(p, q):
    """Calculates 1D EMD (Wasserstein distance) using SciPy."""
    p = p / (p.sum() + EPSILON)
    q = q / (q.sum() + EPSILON)
    try:
        u_weights = np.maximum(0, p); u_weights /= u_weights.sum() + EPSILON
        v_weights = np.maximum(0, q); v_weights /= v_weights.sum() + EPSILON
        return wasserstein_distance(ORDINAL_INDICES, ORDINAL_INDICES, u_weights, v_weights)
    except Exception:
        return np.nan

# Expected Metrics (Bayesian)
def calculate_expected_ce(counts, n, q):
    """Calculates Expected Cross-Entropy E[CE(p, q)] using Formula 5."""
    posterior_alpha = ALPHA + counts
    posterior_alpha_0 = ALPHA_0 + n
    expected_p = posterior_alpha / (posterior_alpha_0 + EPSILON)
    return calculate_ce(expected_p, q) # Reuse CE

def calculate_expected_kl(counts, n, q):
    """Calculates Expected KL Divergence E[KL(p || q)] using Formula 6."""
    posterior_alpha = ALPHA + counts
    posterior_alpha_0 = ALPHA_0 + n

    term1 = digamma(posterior_alpha + EPSILON) # Add EPSILON for stability near 0
    term2 = digamma(posterior_alpha_0 + EPSILON)
    log_q = np.log(q + EPSILON)

    if not np.all(np.isfinite(term1)) or not np.isfinite(term2) or not np.all(np.isfinite(log_q)):
        return np.nan

    expected_p = posterior_alpha / (posterior_alpha_0 + EPSILON)
    expected_kl = np.sum(expected_p * (term1 - term2 - log_q))

    return expected_kl if np.isfinite(expected_kl) else np.nan


def calculate_expected_emd(counts, n, q, n_samples=N_MC_EMD):
    """Calculates Expected EMD E[EMD(p, q)] using Monte Carlo (Formula 10)."""
    posterior_alpha = ALPHA + counts
    if np.any(posterior_alpha <= 0): posterior_alpha = np.maximum(posterior_alpha, EPSILON)

    try:
        sampled_ps = np.random.dirichlet(posterior_alpha, n_samples)
    except ValueError:
        posterior_alpha_0 = ALPHA_0 + n
        expected_p = posterior_alpha / (posterior_alpha_0 + EPSILON)
        return calculate_emd_1d(expected_p, q) # Fallback

    emd_samples = [calculate_emd_1d(p_sample, q) for p_sample in sampled_ps]
    valid_emd_samples = [s for s in emd_samples if not np.isnan(s)]
    return np.mean(valid_emd_samples) if valid_emd_samples else np.nan

# --- Simulation Loop ---
print("Running simulations...")
results_list = []

for q_name, q_pred in Q_LIST.items():
    print(f"  Simulating for Prediction: {q_name}")
    for n in tqdm(N_VALUES, desc=f"   Sample Size (n)"):
        rep_results = []
        for _ in range(N_REPETITIONS):
            # Simulate observation counts
            counts = np.random.multinomial(n, P_TRUE)

            # Calculate empirical distribution
            p_emp = counts / (n + EPSILON)

            # Calculate all metrics
            ce_emp = calculate_ce(p_emp, q_pred)
            kl_emp = calculate_kl(p_emp, q_pred)
            emd_emp = calculate_emd_1d(p_emp, q_pred)
            e_ce = calculate_expected_ce(counts, n, q_pred)
            e_kl = calculate_expected_kl(counts, n, q_pred)
            e_emd = calculate_expected_emd(counts, n, q_pred)

            rep_results.append({
                'CE_emp': ce_emp, 'KL_emp': kl_emp, 'EMD_emp': emd_emp,
                'E_CE': e_ce, 'E_KL': e_kl, 'E_EMD': e_emd,
            })

        # Aggregate results for this n and q
        df_rep = pd.DataFrame(rep_results).dropna() # Drop reps where calc failed
        if not df_rep.empty:
             mean_metrics = df_rep.mean()
             std_metrics = df_rep.std()
             results_list.append({
                 'n': n,
                 'q_name': q_name,
                 **{f'{k}_mean': v for k, v in mean_metrics.items()},
                 **{f'{k}_std': v for k, v in std_metrics.items()}
             })

df_results = pd.DataFrame(results_list)

# --- Visualization ---
print("\nGenerating plots...")
sns.set_theme(style="whitegrid")

metric_pairs = [
    ('CE', 'Cross-Entropy'),
    ('KL', 'KL Divergence'),
    ('EMD', 'Earth Mover\'s Distance')
]

for q_name in Q_LIST.keys():
    df_q = df_results[df_results['q_name'] == q_name]

    fig, axes = plt.subplots(len(metric_pairs), 2, figsize=(14, 5 * len(metric_pairs)), sharex=True)
    fig.suptitle(f"Metric Behavior and Stability vs. Sample Size (n)\nPrediction: {q_name}", fontsize=16, y=1.02)

    for i, (metric_abbr, metric_full_name) in enumerate(metric_pairs):
        emp_mean_col = f'{metric_abbr}_emp_mean'
        exp_mean_col = f'E_{metric_abbr}_mean'
        emp_std_col = f'{metric_abbr}_emp_std'
        exp_std_col = f'E_{metric_abbr}_std'

        # Mean Plot
        ax_mean = axes[i, 0]
        ax_mean.plot(df_q['n'], df_q[emp_mean_col], marker='o', linestyle='--', label=f'Empirical {metric_abbr}')
        ax_mean.plot(df_q['n'], df_q[exp_mean_col], marker='s', linestyle='-', label=f'Expected {metric_abbr} (E[{metric_abbr}])')
        ax_mean.set_ylabel(f'Mean {metric_full_name}')
        ax_mean.set_xscale('log')
        ax_mean.legend()
        ax_mean.grid(True, which="both", ls="-", alpha=0.5)


        # Std Dev Plot
        ax_std = axes[i, 1]
        ax_std.plot(df_q['n'], df_q[emp_std_col], marker='o', linestyle='--', label=f'Empirical {metric_abbr}')
        ax_std.plot(df_q['n'], df_q[exp_std_col], marker='s', linestyle='-', label=f'Expected {metric_abbr} (E[{metric_abbr}])')
        ax_std.set_ylabel(f'Std Dev {metric_full_name}')
        ax_std.set_xscale('log')
        ax_std.legend()
        ax_std.grid(True, which="both", ls="-", alpha=0.5)


    axes[-1, 0].set_xlabel('Sample Size (n) [Log Scale]')
    axes[-1, 1].set_xlabel('Sample Size (n) [Log Scale]')
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()

    # Ordinal vs Categorical Comparison for Bad Predictions
    if 'Bad' in q_name:
        plt.figure(figsize=(8, 5))
        plt.plot(df_q['n'], df_q['E_CE_mean'], marker='^', linestyle=':', label='E[CE]', color='tab:orange')
        plt.plot(df_q['n'], df_q['E_EMD_mean'], marker='s', linestyle='-', label='E[EMD]', color='tab:purple')
        plt.title(f'Comparison of E[CE] and E[EMD]\nPrediction: {q_name}')
        plt.xlabel('Sample Size (n) [Log Scale]')
        plt.ylabel('Mean Expected Metric Value')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.show()


print("\n--- Interpretation Notes ---")
print("Observe the plots:")
print("1. Behavior (Mean Plots - Left Column):")
print("   - Do the Expected metrics (solid lines) start higher than Empirical (dashed) at low 'n'?")
print("   - Do Expected metrics show smoother convergence towards a stable value as 'n' increases?")
print("   - Does the gap between good/bad predictions widen more clearly/smoothly with Expected metrics?")
print("2. Stability (Std Dev Plots - Right Column):")
print("   - Are the standard deviations for Expected metrics (solid lines) lower than Empirical (dashed), especially at low 'n'?")
print("3. Ordinality (E[CE] vs E[EMD] plot for Bad predictions):")
print("   - Compare the E[EMD] values between 'Bad (Adjacent Err)' and 'Bad (Distant Err)'. Is E[EMD] significantly higher for the distant error, even if E[CE] might be similar?")
print("   - This highlights EMD's sensitivity to the *magnitude* of ordinal error.")
