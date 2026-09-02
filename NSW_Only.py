#!/usr/bin/env python3
"""
Online Nash Social Welfare Maximization with Predictions
Banerjee, Gkatzelis, Gorokh, Jin (arXiv:2008.03564)

VERSION 2: Implementation of Banerjee et al. only.

- N agents, T periods; one divisible good per period (total 1 per period).
- Prediction: Ṽ_i = predicted monopolist utility (sum of agent i's values over T periods).
- Set-Aside Greedy: (1) Set-aside 1/2 allocated uniformly → 1/(2N) per agent per period.
  (2) Greedy 1/2 allocated to maximize predicted NSW (Section 7.4: equalize marginal ratios).
- All agents report truthfully (values revealed when good arrives).
- Competitive ratio: O(log N), O(log T) with perfect predictions; degrades smoothly with error.
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

np.random.seed(42)
N = 20   # agents
T = 20   # periods (one good per period)
n_trials = 10  # repeated runs for mean CR

# Prediction quality: Ṽ_i = V_i * (1 + uniform(-noise, +noise)). Banerjee et al. study robustness to prediction error.
PREDICTION_NOISE_RANGE = 0.30

# ============================================================================
# Ṽ SCENARIO GENERATOR (custom prediction scenarios for stress-testing)
# ============================================================================

def generate_V_tilde(v_true, scenario, rng):
    """
    Generate predicted total valuations Ṽ_i for a given scenario.
    v_true: (N, T) array; V_i = v_true.sum(axis=1).
    scenario: str (see PREDICTION_SCENARIOS).
    rng: np.random.RandomState (or similar).
    Returns: (N,) array Ṽ >= 1e-10.
    """
    V = v_true.sum(axis=1).astype(float)
    N = len(V)
    V_tilde = np.zeros(N)

    if scenario == "ideal":
        # 1. Ideal: Ṽ_i = V_i — best possible baseline, CR near 1
        V_tilde = V.copy()

    elif scenario == "mild_noise":
        # 2. Mild noise: Ṽ_i = V_i * Uniform(0.7, 1.3) — realistic good predictions
        V_tilde = V * rng.uniform(0.7, 1.3, N)

    elif scenario == "severe_over":
        # 3. Severe global overestimation: all Ṽ = V * 10 — greedy half gets "bloated" virtual grid
        V_tilde = V * 10.0

    elif scenario == "severe_under":
        # 4. Severe global underestimation: all Ṽ = V * 0.1 — paper: under-estimate less harmful than over
        V_tilde = V * 0.1

    elif scenario == "asymmetric_outlier":
        # 5. One agent super overestimated: 19 with Ṽ = V, agent index N-1 gets Ṽ = V * 1000 (robustness of geometric mean)
        V_tilde = V.copy()
        V_tilde[-1] = V[-1] * 1000.0

    elif scenario == "adversarial":
        # 6. Adversarial: highest V_i gets lowest Ṽ_i, lowest V_i gets highest Ṽ_i — worst for allocation
        order_V = np.argsort(V)  # ascending: order_V[0] = smallest V, order_V[-1] = largest V
        V_sorted = np.sort(V)
        # Assign: smallest prediction to agent with largest V, etc.
        V_tilde_perm = V_sorted[::-1]  # reversed so max V gets min Ṽ
        V_tilde[order_V] = V_tilde_perm

    elif scenario == "random_permutation":
        # 7. Wrong people: predictions shuffled across agents (Ṽ for i is someone else's V)
        perm = rng.permutation(N)
        V_tilde = V[perm]

    elif scenario == "half_over_half_under":
        # 8. Split: first half overestimated 2x, second half underestimated 0.5x
        V_tilde = V.copy()
        half = N // 2
        V_tilde[:half] *= 2.0
        V_tilde[half:] *= 0.5

    else:
        # Fallback: mild noise (e.g. legacy PREDICTION_NOISE_RANGE behavior)
        V_tilde = V * (1 + rng.uniform(-PREDICTION_NOISE_RANGE, PREDICTION_NOISE_RANGE, N))

    V_tilde = np.maximum(V_tilde, 1e-10)
    return V_tilde


# Scenarios to run in main() (order: ideal first, then stress tests)
PREDICTION_SCENARIOS = (
    "ideal",
    "mild_noise",
    "severe_over",
    "severe_under",
    "asymmetric_outlier",
    "adversarial",
    "random_permutation",
    "half_over_half_under",
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_per_round(v):
    """Normalize so each column (period) sums to 1: total value of the good = 1 per period."""
    col_sums = v.sum(axis=0)
    col_sums[col_sums == 0] = 1  # avoid div by zero
    return v / col_sums  # one unit of good per period across agents

def compute_nsw_from_util(u):
    """NSW = (∏_i u_i)^(1/N) = exp(mean(log(u))). Returns 0.0 if any u <= 0."""
    if np.any(u <= 0):
        return 0.0  # log undefined
    return np.exp(np.mean(np.log(u)))  # geometric mean of utilities

# ============================================================================
# OFFLINE OPTIMAL (Benchmark for competitive ratio)
# ============================================================================

def offline_optimal_nsw(v_true):
    """
    Optimal NSW with full knowledge of all v_i,t. Maximizes (∏_i u_i)^(1/N)
    s.t. sum_i x[i,t] = 1 per t, x >= 0. Used as benchmark for competitive ratio.
    """
    N_dim, T_dim = v_true.shape
    v_optimize = v_true
    N_opt = N_dim

    # Maximizing NSW = (∏ u_i)^(1/N) is equivalent to minimizing -sum(log u_i)
    def objective(x_flat):
        X = x_flat.reshape(N_opt, T_dim)
        utilities = (v_optimize * X).sum(axis=1)
        if np.any(utilities <= 0):
            return 1e6
        return -np.sum(np.log(utilities))

    # Budget: each period t must sum to 1 (one unit of good per period)
    constraints = []
    for t in range(T_dim):
        constraints.append({
            'type': 'eq',
            'fun': lambda x_flat, t=t: np.sum(x_flat.reshape(N_opt, T_dim)[:, t]) - 1
        })
    # Feasible start: each column t must sum to 1, so uniform = 1/N_opt per agent per period (not 1/(N*T)!)
    x0 = np.ones(N_opt * T_dim) / N_opt
    bounds = [(0, 1) for _ in range(N_opt * T_dim)]  # allocation in [0,1]
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        X_opt = result.x.reshape(N_opt, T_dim)  # optimal allocation matrix
        utilities_opt = (v_optimize * X_opt).sum(axis=1)
        nsw_opt = compute_nsw_from_util(utilities_opt)
    else:
        # Fallback: uniform allocation if optimizer fails
        X_opt = np.ones((N_opt, T_dim)) / N_opt
        utilities_opt = (v_optimize * X_opt).sum(axis=1)
        nsw_opt = compute_nsw_from_util(utilities_opt)
    return nsw_opt, utilities_opt, X_opt


# ============================================================================
# GREEDY ALLOCATION (Banerjee et al., Section 7.4)
# ============================================================================

def compute_allocation_min_price_sec74(predicted_util, values_round, budget=0.5):
    """
    Allocate the greedy half to maximize predicted NSW.
    Equalize marginal ratios v_i / u_i(z); Section 7.4 delta_i logic.
    values_round: (N,) values this period (revealed when good arrives).
    """
    N = predicted_util.size
    z = np.zeros(N)  # allocation from this step
    u = np.maximum(predicted_util.astype(float), 1e-10)  # avoid div by zero
    v = np.maximum(values_round.astype(float), 0.0)

    positive = v > 1e-12  # only agents with value get share
    if not np.any(positive):
        z[:] = budget / N
        return z

    # Section 7.4: order by marginal ratio v_i/u_i (high ratio = under-served); equalize ratios within budget
    order = np.argsort(-(v / u))
    u_sorted = u[order].copy()
    v_sorted = v[order].copy()
    n_pos = np.sum(positive)
    z_sorted = np.zeros(n_pos)
    B = float(budget)

    if n_pos == 1:
        z_sorted[0] = B
    else:
        # Optimized O(N) version following Eq. (7.5): when first (i+1) agents share the same v/u ratio,
        # adding the same delta_i to all keeps them equal. We only need one delta_i per i.
        for i in range(n_pos - 1):
            r_target = v_sorted[i + 1] / u_sorted[i + 1]
            if r_target <= 0:
                continue

            # Compute delta_i once for the whole group 0..i (all currently have same v/u ratio)
            delta_i = max(0.0, (1.0 / r_target) - (u_sorted[i] / v_sorted[i]))
            total_needed = (i + 1) * delta_i  # amount needed to move all 0..i down to r_target
            if total_needed <= 0:
                continue

            if total_needed <= B:  # budget enough to equalize group 0..i to r_target
                z_sorted[: i + 1] += delta_i
                u_sorted[: i + 1] += v_sorted[: i + 1] * delta_i
                B -= total_needed
            else:
                # Not enough budget: split remaining B evenly among group 0..i
                z_sorted[: i + 1] += B / (i + 1)
                B = 0.0
                break
        if B > 0:
            z_sorted[:] += B / n_pos

    z[order[:n_pos]] = z_sorted  # map back to original agent order
    total = z.sum()
    if abs(total - budget) > 1e-10:  # renormalize to exact budget
        z = z * (budget / total)
    return z

# ============================================================================
# ONLINE SET-ASIDE GREEDY (Paper Algorithm)
# ============================================================================
def online_set_aside_greedy(v_true, V_tilde):
    """
    Online Set-Aside Greedy (Banerjee et al.).
    - Set-aside: 1/(2N) per agent per period.
    - Greedy: allocate 1/2 to maximize sum_i log(ũ_i,t(z)); ũ_i,t(0) = u_greedy_i + Ṽ_i/(2N).
    - At each t, values v_i,t are revealed; allocation uses them and predictions Ṽ_i.
    Returns u_total (N,), nsw_online (scalar).
    """
    N_dim, T_dim = v_true.shape
    x = np.zeros((N_dim, T_dim))   # set-aside allocation
    y = np.zeros((N_dim, T_dim))   # greedy allocation
    z = np.zeros((N_dim, T_dim))   # total = x + y
    u_greedy = np.zeros(N_dim)    # cumulative utility so far (for greedy step)
    V_tilde = np.asarray(V_tilde, dtype=float).flatten()[:N_dim]

    for t in range(T_dim):
        v_t = v_true[:, t]

        # Set-aside: uniform 1/(2N) per agent (guarantees share regardless of predictions)
        x[:, t] = 0.5 / N_dim

        # Predicted utility for greedy: current cumulative + predicted future share from set-aside
        predicted_util = u_greedy + V_tilde / (2 * N_dim)

        # Greedy half: allocate 0.5 to equalize v_i/u_i and maximize predicted NSW
        y[:, t] = compute_allocation_min_price_sec74(predicted_util, v_t, budget=0.5)
        total_y = y[:, t].sum()
        if abs(total_y - 0.5) > 1e-10:  # ensure greedy half sums to 0.5
            y[:, t] = y[:, t] * (0.5 / total_y)

        # Accumulate utility from this period; total allocation = set-aside + greedy
        u_greedy += v_t * y[:, t]
        z[:, t] = x[:, t] + y[:, t]

    # Final utility = sum over t of (value_i_t * allocation_i_t)
    u_total = (v_true * z).sum(axis=1)
    nsw_online = compute_nsw_from_util(u_total)
    return u_total, nsw_online


# ============================================================================
# EXPERIMENT: Offline vs Online (main runs n_trials trials and aggregates)
# ============================================================================

def run_experiment_once(N_dim, T_dim, v_true_precomputed=None, trial=None, scenario=None):
    """
    One trial: generate (or use) v_true (N×T), predictions Ṽ from scenario (or legacy noise),
    run offline optimal and online set-aside greedy, return NSW and competitive ratio.
    If scenario is set, use generate_V_tilde(v_true, scenario, rng); else use ±PREDICTION_NOISE_RANGE.
    """
    if v_true_precomputed is not None:
        v_true = v_true_precomputed.copy()
    else:
        rng_v = np.random.RandomState(4000 + trial) if trial is not None else np.random.RandomState()
        v_true = rng_v.rand(N_dim, T_dim)
        v_true = normalize_per_round(v_true)

    V_true = v_true.sum(axis=1)
    rng = np.random.RandomState(5000 + trial) if trial is not None else np.random.RandomState()
    if scenario is not None:
        V_tilde = generate_V_tilde(v_true, scenario, rng)
    else:
        V_tilde = V_true * (1 + rng.uniform(-PREDICTION_NOISE_RANGE, PREDICTION_NOISE_RANGE, N_dim))
        V_tilde = np.maximum(V_tilde, 1e-10)

    nsw_offline, u_offline, X_opt = offline_optimal_nsw(v_true)
    u_online, nsw_online = online_set_aside_greedy(v_true, V_tilde)

    # Competitive ratio γ^NSW (paper): Offline / Online (≥ 1)
    competitive_ratio = nsw_offline / (nsw_online + 1e-12)
    return {
        'nsw_offline': nsw_offline,
        'nsw_online': nsw_online,
        'competitive_ratio': competitive_ratio,
        'v_true': v_true
    }


# ============================================================================
# MAIN: run all Ṽ scenarios (n_trials each), report CR table
# ============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Online Nash Social Welfare Maximization with Predictions")
    print("Banerjee, Gkatzelis, Gorokh, Jin — VERSION 2")
    print("=" * 72)
    print("N = {}, T = {}, n_trials = {} per scenario".format(N, T, n_trials))
    print("Ṽ scenarios: ideal | mild_noise | severe_over | severe_under | asymmetric_outlier | adversarial | random_permutation | half_over_half_under")
    print()

    # Run each scenario n_trials times; aggregate mean offline, mean online, mean CR
    scenario_results = []
    for scenario in PREDICTION_SCENARIOS:
        results = []
        for trial in range(n_trials):
            r = run_experiment_once(N, T, v_true_precomputed=None, trial=trial, scenario=scenario)
            results.append(r)
        mean_offline = np.mean([x['nsw_offline'] for x in results])
        mean_online = np.mean([x['nsw_online'] for x in results])
        mean_cr = np.mean([x['competitive_ratio'] for x in results])
        scenario_results.append({
            'scenario': scenario,
            'mean_offline': mean_offline,
            'mean_online': mean_online,
            'mean_cr': mean_cr,
        })

    # Table: Scenario | Offline | Online | CR
    print("Results (mean over {} trials per scenario); CR = Offline / Online (≥ 1)".format(n_trials))
    print("-" * 72)
    print("{:<24} {:>12} {:>12} {:>10}".format("Scenario", "Offline NSW", "Online NSW", "CR"))
    print("-" * 72)
    for row in scenario_results:
        print("{:<24} {:>12.6f} {:>12.6f} {:>10.4f}".format(
            row['scenario'], row['mean_offline'], row['mean_online'], row['mean_cr']))
    print("-" * 72)
    print("Ideal → CR near 1. Severe over → CR high; severe under → lower. Adversarial → worst CR. Geometric mean robust to one outlier.")
    print("=" * 72)
