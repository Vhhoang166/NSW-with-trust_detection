#!/usr/bin/env python3
"""
Nash Social Welfare (NSW) Maximization with Trust-Based Malicious Agent Detection
VERSION 1: Set-Aside Greedy (Algorithm 1, Paper 1) + Center trust (Paper 2)

- Set-Aside Greedy: Greedy half = argmax sum log(u_{i,t}(z_t)) via sorting (Section 7.4), not proportional.
- Fixed prediction: V_tilde_i set once at start (Paper 1); not updated each round.
- Trust: Center observes each agent alpha_j(t); beta_j(t) = sum alpha_j (cumulative). Center detects (no NxN voting).
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

import numpy as np
from scipy.optimize import minimize
# SciPy optimize: Provides optimization algorithms
# minimize() uses Sequential Least Squares Programming (SLSQP) to solve constrained optimization
import math
import warnings # Warnings: Suppress numerical warnings from optimization (e.g., convergence issues)

import matplotlib
matplotlib.use('Agg')
# 'Agg' backend: Non-interactive backend for saving plots to files (no GUI required)
# This is necessary for running on servers or in environments without display

import matplotlib.pyplot as plt
# Pyplot: Matplotlib's plotting interface, provides MATLAB-like plotting functions

warnings.filterwarnings("ignore")
# Suppress all warnings to keep output clean (optimization may produce convergence warnings)

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

np.random.seed(42) # Random seed: Ensures reproducibility
N_trust = 10 # Number of trustworthy agents
N_mal = 4 # Number of malicious agents
T = 10 # Number of time periods (rounds)

c_mal = 5.0
# Malicious distortion multiplier
# Malicious agents report: v_reported = v_true * factor
# where factor ~ Uniform(1/c_mal, c_mal) = Uniform(0.2, 5.0)
# This means malicious agents can under-report (0.2x) or over-report (5x) their true values
# c_mal=5.0 is a realistic distortion level (not too extreme, but detectable)

n_trials = 20
# Number of independent trials to run
# Each trial generates new random valuations and runs the full experiment
# Averaging across trials reduces variance and provides statistical confidence

xi0_default = 0.01
# Default initial detection threshold
# Lower values = stricter detection (more likely to flag agents as malicious)
# Higher values = looser detection (more lenient)
# This is the default when not doing sensitivity analysis

gamma = 0.1
# (Legacy: was used for xi_t = xi0*(t+1)^gamma; now Paper 2 formula used, gamma kept for API compat.)

# Paper 2 Assumption 4: xi_t = xi0 * sqrt((1+epsilon)*(t+1)*ln(t+2))
epsilon_default = 0.1

# Trust detection uses alpha/beta (Paper 2); alpha can be dynamic (from distortion) or precomputed.
# The constants below are used for the α stream (good vs malicious).
# (Legacy: alpha_pairwise was used in the old report-similarity beta; no longer used.)

# ============================================================================
# SENSITIVITY ANALYSIS PARAMETERS
# ============================================================================

RUN_SENSITIVITY_ANALYSIS = True
# Flag: If True, run parameter sweep over xi0
# If False, run single experiment with default parameters
# Sensitivity analysis tests robustness by varying parameters

XI0_MIN = 0.05
# Minimum xi0 value to test in sensitivity analysis
# Starting from 0.05 avoids very strict thresholds that kill all agents

XI0_MAX = 0.50
# Maximum xi0 value to test
# Extended range captures full sensitivity curve: dead zone → peak → degradation

XI0_STEP = 0.01
# Step size for xi0 sweep
# Smaller step = finer resolution but more computation
# Step of 0.01 gives ~45 different xi0 values to test

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_per_round(v):
    """
    Normalize valuation matrix so each column sums to 1.
    
    Formula: v_normalized[i,t] = v[i,t] / sum_j(v[j,t])
    
    Purpose: Ensures total value per round is normalized to 1
    This represents the constraint that total resources per round = 1
    
    Parameters:
        v: numpy array of shape (N, T)
           v[i,t] = agent i's valuation for resource at time t
    
    Returns:
        Normalized matrix where each column sums to 1
    
    Example:
        Input: [[0.5, 0.3], [0.3, 0.7]]
        Column sums: [0.8, 1.0]
        Output: [[0.625, 0.3], [0.375, 0.7]]
    """
    col_sums = v.sum(axis=0)
    # Sum along axis=0 (rows) → sum of each column
    # Result: array of length T, where col_sums[t] = sum of column t
    
    col_sums[col_sums == 0] = 1
    # Avoid division by zero: if a column sums to 0, set it to 1
    # This prevents NaN errors (though shouldn't happen with random data)
    
    return v / col_sums
    # Element-wise division: each element divided by its column sum
    # Broadcasting: col_sums is (T,), v is (N, T) → automatically broadcasts


def compute_nsw_from_util(u):
    """
    Compute Nash Social Welfare (NSW) from utility vector.
    
    Formula: NSW = (∏_i u_i)^(1/N) = exp((1/N) * Σ_i log(u_i))
    
    Mathematical derivation:
        NSW = (u_1 * u_2 * ... * u_N)^(1/N)
        log(NSW) = (1/N) * log(u_1 * u_2 * ... * u_N)
                 = (1/N) * (log(u_1) + log(u_2) + ... + log(u_N))
                 = (1/N) * Σ_i log(u_i)
        NSW = exp((1/N) * Σ_i log(u_i))
    
    Why geometric mean?
        - Balances efficiency (total utility) with fairness (distribution)
        - If one agent gets 0 utility, NSW = 0 (punishes extreme inequality)
        - More fair than arithmetic mean (which ignores distribution)
    
    Parameters:
        u: numpy array of shape (N,)
           u[i] = total utility of agent i across all rounds
    
    Returns:
        NSW: scalar, geometric mean of utilities
    
    Example:
        u = [1.0, 2.0, 4.0]
        log(u) = [0.0, 0.693, 1.386]
        mean(log(u)) = 0.693
        NSW = exp(0.693) ≈ 2.0
        (Geometric mean: (1*2*4)^(1/3) = 8^(1/3) = 2.0)
    """
    if np.any(u <= 0):
        # Zero or negative utilities → return NaN to indicate crash
        return np.nan
    
    return np.exp(np.mean(np.log(u)))
    # Step 1: np.log(u) → log of each utility
    # Step 2: np.mean(...) → average of logs = (1/N) * Σ log(u_i)
    # Step 3: np.exp(...) → exponentiate to get geometric mean
    # If u contains zeros, this will return NaN (visible in plots as missing data)


# ============================================================================
# OFFLINE OPTIMAL ALLOCATION (Benchmark)
# ============================================================================

def offline_optimal_nsw(v_true, agent_mask=None):
    """
    Compute optimal Nash Social Welfare with full knowledge of future.
    
    This is the "ideal" benchmark: what we could achieve if we knew all
    valuations in advance. Uses constrained optimization to maximize NSW.
    
    Optimization Problem:
        Maximize: NSW = (∏_i u_i)^(1/N)
        Subject to:
            - Σ_i x[i,t] = 1 for all t (total allocation per round = 1)
            - x[i,t] ≥ 0 for all i, t (non-negative allocations)
    
    Since maximizing NSW is equivalent to maximizing log(NSW):
        Maximize: Σ_i log(u_i)
        where u_i = Σ_t (v_true[i,t] * x[i,t])
    
    We minimize the negative (standard optimization convention):
        Minimize: -Σ_i log(u_i)
    
    Parameters:
        v_true: numpy array of shape (N, T)
                v_true[i,t] = agent i's true valuation for resource at time t
        agent_mask: numpy array of shape (N,), boolean
                    If provided, only optimize for agents where mask[i] = True
                    Used to compute NSW only for trustworthy agents (fair comparison)
    
    Returns:
        nsw_opt: scalar, optimal Nash Social Welfare
        u_full: numpy array of shape (N,), utilities for all agents
                (zeros for masked agents)
        X_opt: numpy array of shape (N_opt, T), optimal allocation matrix
               X_opt[i,t] = fraction of resource allocated to agent i at time t
    
    Algorithm:
        1. Filter to trustworthy agents if mask provided
        2. Define objective function (negative sum of logs)
        3. Define constraints (sum to 1 per round)
        4. Solve using SLSQP optimizer
        5. Compute NSW from optimal utilities
    """
    N, T = v_true.shape
    # Extract dimensions: N agents, T time periods
    
    # Filter to trustworthy agents if mask provided
    if agent_mask is not None:
        trust_indices = np.where(agent_mask)[0]
        # np.where(agent_mask)[0] returns indices where mask is True
        # Example: mask = [True, True, False, False] → indices = [0, 1]
        
        v_optimize = v_true[trust_indices, :]
        # Extract rows corresponding to trustworthy agents
        # Shape: (N_trust, T)
        
        N_opt = len(trust_indices)
        # Number of agents to optimize for
    else:
        v_optimize = v_true
        # Optimize for all agents
        N_opt = N
        trust_indices = np.arange(N)
        # All agents are trustworthy
    
    # Define objective function
    def objective(x_flat):
        """
        Objective: Minimize negative sum of log utilities.
        
        Parameters:
            x_flat: numpy array of shape (N_opt * T,)
                    Flattened allocation matrix (optimizer works with 1D arrays)
        
        Returns:
            Objective value: -Σ_i log(u_i)
        """
        X = x_flat.reshape(N_opt, T)
        # Reshape flattened array back to matrix
        # x_flat = [x[0,0], x[0,1], ..., x[0,T-1], x[1,0], ..., x[N_opt-1,T-1]]
        # Reshape to (N_opt, T)
        
        utilities = (v_optimize * X).sum(axis=1)
        # Element-wise multiplication: v_optimize[i,t] * X[i,t]
        # Sum along axis=1 (columns) → sum across time for each agent
        # Result: utilities[i] = Σ_t (v_optimize[i,t] * X[i,t])
        # This is each agent's total utility
        
        if np.any(utilities <= 0):
            return 1e6
            # Penalty for invalid solution (utilities must be > 0 for log)
            # Large penalty guides optimizer away from invalid regions
        
        return -np.sum(np.log(utilities))
        # Negative sum of logs (we minimize, so negate to maximize)
        # Equivalent to maximizing: Σ_i log(u_i)
        # Which is equivalent to maximizing: (∏_i u_i)^(1/N) = NSW
    
    # Define constraints: total allocation per round = 1
    constraints = []
    for t in range(T):
        constraints.append({
            'type': 'eq',  # Equality constraint
            'fun': lambda x_flat, t=t: np.sum(x_flat.reshape(N_opt, T)[:, t]) - 1
            # Constraint function: Σ_i X[i,t] - 1 = 0
            # t=t captures the current value of t in the lambda closure
            # Without t=t, all lambdas would use the final value of t (closure issue)
        })
    
    # Initial guess: uniform allocation
    x0 = np.ones(N_opt * T) / (N_opt * T)
    # Start with equal allocation: each agent gets 1/(N_opt*T) per round
    # This is a feasible starting point
    
    # Bounds: allocations must be non-negative (upper bound of 1 is implied by constraints)
    bounds = [(0, 1) for _ in range(N_opt * T)]
    # Each allocation x[i,t] must be in [0, 1]
    
    # Solve optimization problem
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    # SLSQP: Sequential Least Squares Programming
    # Handles constrained optimization with bounds and equality constraints
    
    if result.success:
        # Optimization converged successfully
        X_opt = result.x.reshape(N_opt, T)
        # Extract optimal allocation matrix
        
        utilities_opt = (v_optimize * X_opt).sum(axis=1)
        # Compute optimal utilities
        
        nsw_opt = compute_nsw_from_util(utilities_opt)
        # Compute optimal NSW
    else:
        # Optimization failed (shouldn't happen, but handle gracefully)
        X_opt = np.ones((N_opt, T)) / N_opt
        # Fallback: uniform allocation (each agent gets 1/N_opt per round)
        
        utilities_opt = (v_optimize * X_opt).sum(axis=1)
        nsw_opt = compute_nsw_from_util(utilities_opt)
    
    # Create full utility vector (zeros for masked agents)
    u_full = np.zeros(N)
    # Initialize with zeros
    
    u_full[trust_indices] = utilities_opt
    # Fill in utilities for trustworthy agents
    
    return nsw_opt, u_full, X_opt


# ============================================================================
# GREEDY ALLOCATION (Paper 1, Section 7.4) - Minimize predicted price, balance marginal ratios
# ============================================================================

def compute_greedy_allocation_paper1_sec74(predicted_util, reported_round, eligible_mask, budget=0.5):
    """
    Greedy-half: find z_t that minimizes predicted price (Paper 1 Section 7.4).
    Equivalent to max sum_i log(u_pred_i + v_i*z_i). Sort by v_{i,t}/u_pred_{i,t}(0) descending;
    allocate to balance marginal ratios (water-filling).

    Parameters:
        predicted_util: (N,) predicted utility u_pred_i (e.g. u_greedy + V_tilde/(2N))
        reported_round: (N,) reported valuations v_i this round
        eligible_mask: (N,) boolean; only eligible agents can receive positive allocation
        budget: total budget to allocate (default 0.5)

    Returns:
        z: (N,) allocation; sum(z) = budget, z[~eligible] = 0
    """
    N = predicted_util.size
    z = np.zeros(N)
    eligible_idx = np.where(eligible_mask)[0]
    if len(eligible_idx) == 0:
        return z
    u = np.maximum(predicted_util[eligible_idx].astype(float), 1e-10)
    v = np.maximum(reported_round[eligible_idx].astype(float), 0.0)

    # Only agents with v > 0 contribute; others get 0
    positive = v > 1e-12
    if not np.any(positive):
        # All reported 0: split budget evenly among eligible
        n_el = len(eligible_idx)
        for i, idx in enumerate(eligible_idx):
            z[idx] = budget / n_el
        return z

    # Restrict to eligible agents with v > 0
    idx_pos = eligible_idx[positive]
    u_pos = u[positive]
    v_pos = v[positive]
    n_pos = len(idx_pos)
    if n_pos == 1:
        z[idx_pos[0]] = budget
        return z

    # Sort by r_i = v_i / u_pred_i descending (highest ratio first)
    ratio = v_pos / u_pos
    order = np.argsort(-ratio)
    u_sorted = u_pos[order]
    v_sorted = v_pos[order]
    idx_sorted = idx_pos[order]

    # Find largest k such that w = (budget + S_k) / k >= u_sorted[k-1]/v_sorted[k-1]
    S = 0.0
    k_star = 0
    for k in range(1, n_pos + 1):
        i = k - 1
        S += u_sorted[i] / v_sorted[i]
        w = (budget + S) / k
        if w >= u_sorted[i] / v_sorted[i]:
            k_star = k
        else:
            break

    if k_star <= 0:
        # Fallback: give all budget to first agent
        z[idx_sorted[0]] = budget
        return z

    S_star = sum(u_sorted[j] / v_sorted[j] for j in range(k_star))
    w_final = (budget + S_star) / k_star
    for j in range(k_star):
        z[idx_sorted[j]] = max(0.0, w_final - u_sorted[j] / v_sorted[j])
    # Remaining eligible (j >= k_star) stay 0; sum(z) = budget by construction
    total = z.sum()
    if abs(total - budget) > 1e-10:
        z[eligible_mask] = z[eligible_mask] * (budget / total)
    return z


# ============================================================================
# ONLINE SET-ASIDE GREEDY ALGORITHM
# ============================================================================

def online_set_aside_with_trust(v_true, reports_func_for_malicious, detect_fn, 
                                remove_detected=True, trust_indices=None, 
                                xi0_param=None, gamma_param=None, epsilon_param=None, predicted_V_fixed=None):
    """
    Online resource allocation using Set-Aside Greedy (Algorithm 1, Paper 1) with trust detection.
    
    Greedy half: Set-Aside Greedy = argmax Σ_i log(ũ_{i,t}(z_t)) via sorting (Section 7.4), not proportional.
    predicted_V_fixed: if provided, Ṽ_i is fixed for all t (Paper 1); else not used (caller must pass).
    
    Trust: Center-only vector α_j(t), β_j(t); Center detects, no N×N voting.
    
    Parameters:
        v_true: (N_all, T) true valuations
        reports_func_for_malicious: reporting function (or None)
        detect_fn: (reports_history, xi0, gamma, epsilon) -> (detected, beta, xi_t); beta (N,).
        remove_detected, trust_indices, xi0_param, gamma_param, epsilon_param: as before
        predicted_V_fixed: (N_all,) optional. Fixed V_tilde_i for all t (Paper 1); required for correct setup.
    
    Returns:
        u_total, nsw_online, detected
    """
    # Use parameter values if provided, otherwise use defaults
    if xi0_param is None:
        xi0_param = xi0_default
    if gamma_param is None:
        gamma_param = gamma
    if epsilon_param is None:
        epsilon_param = epsilon_default
    
    N_all = v_true.shape[0]
    # Total number of agents (trustworthy + malicious)
    
    # Initialize allocation matrices
    x = np.zeros((N_all, T))  # Uniform half allocation
    y = np.zeros((N_all, T))  # Greedy half allocation
    z = np.zeros((N_all, T))  # Total allocation (x + y)
    
    u_greedy = np.zeros(N_all)
    # Cumulative utility from greedy allocations
    # Updated each round: u_greedy[i] += v_true[i,t] * y[i,t]
    
    reported_cum = np.zeros((N_all, 0))
    # Matrix to store cumulative reported values
    # Shape grows: (N_all, 1), (N_all, 2), ..., (N_all, T)
    # reported_cum[i, t] = cumulative reported value of agent i up to round t

    detected = np.zeros(N_all, dtype=bool)
    # Boolean mask: detected[i] = True if agent i is flagged as malicious

    any_detected_ever = False
    # Flag: True if we've detected any malicious agents at any point
    # Used for conditional soft weighting (only apply if detection is working)
    
    # Main loop: process each time period
    for t in range(T):
        # Step 1: Collect reported valuations
        reported_round = np.zeros(N_all)
        # reported_round[i] = agent i's reported valuation for round t
        
        for i in range(N_all):
            if reports_func_for_malicious is None:
                # No malicious function → all agents report truthfully
                reported_round[i] = v_true[i, t]
            else:
                # Call reporting function (may distort if malicious)
                reported_round[i] = reports_func_for_malicious(i, t, v_true)

        # Step 2: Update cumulative reported history
        if reported_cum.size == 0:
            # First round: initialize matrix
            reported_cum = reported_round.reshape((N_all, 1))
            # Reshape to column vector: (N_all, 1)
        else:
            # Append new column to existing matrix
            reported_cum = np.hstack([reported_cum, reported_round.reshape((N_all, 1))])
            # hstack: horizontal stack (concatenate columns)
            # Result: matrix grows from (N_all, t) to (N_all, t+1)
        
        # Step 3: Run detection algorithm
        if detect_fn is not None:
            detected_mask, beta, xi_t = detect_fn(reported_cum, xi0=xi0_param, gamma=gamma_param, epsilon=epsilon_param)
            # detected_mask: boolean array, True for agents flagged as malicious
            # beta: similarity matrix, beta[i,j] = similarity between agents i and j
            # xi_t: current detection threshold
            
            detected = detected_mask.copy()
            
            if detected.any():
                any_detected_ever = True
                # Mark that detection has found something (for conditional weighting)
        else:
            detected = np.zeros(N_all, dtype=bool)
            beta = np.ones(N_all)
            xi_t = 0.0

        # Step 4: Uniform half allocation (Paper 1 Set-Aside + Paper 2: cut off detected)
        # Promise 1/(2N) per agent; detected agents get 0 (no resource to malicious). Option 1: do not redistribute.
        x[:, t] = 0.5 / N_all
        x[detected, t] = 0.0   # Cut off detected agents completely (uniform = 0, greedy already 0 via eligible)
        eligible = ~detected if remove_detected else np.ones(N_all, dtype=bool)
        n_eligible = max(1, eligible.sum())

        # Step 5: Predicted utility (Paper 1: V_tilde is fixed input at t=0, never updated from reports)
        if predicted_V_fixed is not None:
            predicted_Vi = np.asarray(predicted_V_fixed, dtype=float).flatten()
        else:
            predicted_Vi = np.ones(N_all) / N_all
        predicted_util = u_greedy + predicted_Vi / (2 * N_all)

        # Step 6: (Marginal utilities not needed: solve_eisenberg_gale_greedy uses u_pred and v directly.)

        # Step 7: Set-Aside Greedy (Paper 1 Section 7.4): minimize predicted price, balance marginal ratios
        y[:, t] = compute_greedy_allocation_paper1_sec74(predicted_util, reported_round, eligible, budget=0.5)
        total_allocated = y[:, t].sum()
        if abs(total_allocated - 0.5) > 1e-10:
            y[:, t] = y[:, t] * (0.5 / total_allocated)
        # Step 8: Update cumulative greedy utility (algorithm uses REPORTS only, Paper 1 online)
        u_greedy += reported_round * y[:, t]
        # Center does not know v_true; it uses reported values for allocation state.
        # Final evaluation u_total = (v_true * z).sum(axis=1) uses true valuations for NSW metric only.

        # Step 11: Compute total allocation
        z[:, t] = x[:, t] + y[:, t]
        # Total allocation = uniform + greedy

    # Compute final total utilities
    u_total = (v_true * z).sum(axis=1)
    # Element-wise: v_true[i,t] * z[i,t]
    # Sum along axis=1 (columns) → sum across time
    # Result: u_total[i] = Σ_t (v_true[i,t] * z[i,t])
    
    # Compute NSW (only for trustworthy agents if specified)
    if trust_indices is not None:
        # Fair comparison: only compute NSW for trustworthy agents
        u_trustonly = u_total[trust_indices]
        # Extract utilities of trustworthy agents only
        nsw_online = compute_nsw_from_util(u_trustonly)
        # Compute NSW for trustworthy agents
    else:
        # Compute NSW for all agents
        nsw_online = compute_nsw_from_util(u_total)
    
    return u_total, nsw_online, detected


# ============================================================================
# TRUST-BASED DETECTION (Paper 2: Exogenous alpha, CUMULATIVE beta, Paper 2 threshold)
# ============================================================================
#
# Paper 2 Eq. (9): beta_i(t) = sum_{k=0}^t alpha_i(k). CUMULATIVE SUM from t=0, no reset.
# Alpha is EXOGENOUS (side information): Center does not know v_true; trust from separate channel.
# In simulation: good agents alpha in [0.7,1.0], malicious in [0.0,0.4] (pre-generated or by type).
# Threshold (Paper 2 Assumption 4): xi_t = xi0 * sqrt((1+epsilon)*(t+1)*ln(t+2)).
# Detection: RELATIVE. detected_j = (max_beta - beta_j > xi_t).
#

ALPHA_GOOD_MIN, ALPHA_GOOD_MAX = 0.7, 1.0
ALPHA_MAL_MIN, ALPHA_MAL_MAX = 0.0, 0.4


def threshold_paper2(t, xi0, epsilon=0.1):
    """Paper 2 Assumption 4: xi_t = xi0 * sqrt((1+epsilon)*(t+1)*ln(t+2)). ln(t+2) avoids zero at t=0."""
    return xi0 * math.sqrt((1.0 + epsilon) * (t + 1) * math.log(t + 2))


def generate_alpha_true(N_all, T, mal_indices, seed=None):
    """
    Pre-generate exogenous alpha matrix (N_all x T). Good agents high alpha, malicious low.
    Pass to make_detect_fn_trust_observations(..., alpha_precomputed=alpha_true).
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    alpha_true = np.zeros((N_all, T))
    mal_set = set(mal_indices) if mal_indices is not None else set()
    for j in range(N_all):
        for t in range(T):
            if j in mal_set:
                alpha_true[j, t] = rng.uniform(ALPHA_MAL_MIN, ALPHA_MAL_MAX)
            else:
                alpha_true[j, t] = rng.uniform(ALPHA_GOOD_MIN, ALPHA_GOOD_MAX)
    return alpha_true


def make_detect_fn_trust_observations(mal_indices, alpha_precomputed=None, seed=None):
    """
    Center-only trust. beta = CUMULATIVE SUM of alpha from k=0 to t (Paper 2 Eq. 9), no reset.
    Alpha is exogenous (Center does not use v_true or reports to compute trust).
    If alpha_precomputed is provided: use pre-generated (N x T) alpha matrix.
    Else: fallback static random alpha by agent type (good high, mal low). Paper 2 threshold.
    """
    if alpha_precomputed is not None:
        alpha_true = np.asarray(alpha_precomputed)
        N_all, T_max = alpha_true.shape

        def detect_fn(reports_history, xi0=0.1, gamma=0.7, epsilon=0.1):
            t_current = reports_history.shape[1] if reports_history.ndim > 1 else 1
            t = t_current - 1
            # beta(t) = sum_{k=0}^t alpha(k); strictly cumulative, no reset
            beta = alpha_true[:, : t + 1].sum(axis=1)
            xi_t = threshold_paper2(t, xi0, epsilon)
            max_beta = beta.max()
            detected_malicious = (max_beta - beta > xi_t)
            return detected_malicious, beta, xi_t

        return detect_fn

    # Fallback: static random alpha by agent type (exogenous); cumulative beta; Paper 2 threshold
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    state = {'cum_prev': None, 't_done': -1}

    def _alpha_vec(N_all, mal_indices, s, rng):
        a = np.zeros(N_all)
        mal_set = set(mal_indices) if mal_indices is not None else set()
        for j in range(N_all):
            a[j] = rng.uniform(ALPHA_MAL_MIN, ALPHA_MAL_MAX) if j in mal_set else rng.uniform(ALPHA_GOOD_MIN, ALPHA_GOOD_MAX)
        return a

    def detect_fn(reports_history, xi0=0.1, gamma=0.7, epsilon=0.1):
        N_all = reports_history.shape[0]
        t = (reports_history.shape[1] if reports_history.ndim > 1 else 1) - 1
        if state['cum_prev'] is None:
            state['cum_prev'] = np.zeros(N_all)
        # Cumulative sum from k=0 to t (Paper 2 Eq. 9); no reset
        for s in range(state['t_done'] + 1, t + 1):
            state['cum_prev'] += _alpha_vec(N_all, mal_indices, s, rng)
        state['t_done'] = t
        beta = state['cum_prev'].copy()
        xi_t = threshold_paper2(t, xi0, epsilon)
        detected_malicious = (beta.max() - beta > xi_t)
        return detected_malicious, beta, xi_t

    return detect_fn


def detect_malicious_from_reports(reports_history, xi0=0.1, gamma=0.7, epsilon=0.1):
    """Legacy: no detection when called directly."""
    N_all = reports_history.shape[0]
    t = reports_history.shape[1] - 1 if reports_history.ndim > 1 and reports_history.shape[1] > 0 else 0
    return np.zeros(N_all, dtype=bool), np.ones(N_all), threshold_paper2(t, xi0, epsilon)


# ============================================================================
# MALICIOUS REPORTING FUNCTION
# ============================================================================

def malicious_report_func_factory(v_true, c_mal, mal_indices, seed=None):
    """
    Factory function that creates a malicious reporting function.
    
    Returns a function that:
        - Trustworthy agents: report true values
        - Malicious agents: report distorted values
    
    Distortion Model:
        For malicious agent i at round t:
            factor ~ Uniform(1/c_mal, c_mal)
            reported_value = v_true[i,t] * factor
    
    Example with c_mal=5.0:
        - factor in [0.2, 5.0]
        - If v_true=1.0:
          * Under-report: reported = 0.2 (20% of true value)
          * Over-report: reported = 5.0 (500% of true value)
    
    Parameters:
        v_true: numpy array of shape (N_all, T)
                True valuations (used as reference)
        c_mal: float
               Distortion multiplier (higher = more extreme distortion)
        mal_indices: list or array
                     Indices of malicious agents
        seed: int, optional
              If provided, fixes the random stream so version 1 and 2 see the same reports (fair comparison).
    
    Returns:
        reports_func: function(i, t, v_true_inner) -> reported value
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    # Seeded RNG when seed given → same dataset across version 1 and 2

    def reports_func(i, t, v_true_inner):
        """
        Reporting function for agent i at round t.
        
        Parameters:
            i: int, agent index
            t: int, time period
            v_true_inner: numpy array, true valuations
        
        Returns:
            reported_value: float, what agent i reports
        """
        if i in mal_indices:
            # Malicious agent: distort the value
            factor = rng.uniform(1.0 / c_mal, c_mal)
            # Random distortion factor: Uniform(1/c_mal, c_mal)
            # Example: c_mal=5.0 → factor ∈ [0.2, 5.0]
            
            return v_true_inner[i, t] * factor
            # Distorted report: true value multiplied by random factor
        else:
            # Trustworthy agent: report truthfully
            return v_true_inner[i, t]

    return reports_func


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_experiment_once(N_trust, N_mal, T, c_mal, detection_fn, 
                        xi0_param=None, gamma_param=None, v_true_precomputed=None, trial=None):
    """
    Run one complete experiment trial.
    
    Experiment Design:
        Scenario A: Online Clean (only trustworthy agents)
        Scenario B: Online Polluted WITHOUT Detection (baseline)
        Scenario C: Online Polluted WITH Detection (test detection effectiveness)
        Benchmark: Offline Optimal (ideal performance)
    
    All scenarios compute NSW only for trustworthy agents (fair comparison).
    
    Parameters:
        N_trust: int, number of trustworthy agents
        N_mal: int, number of malicious agents
        T: int, number of time periods
        c_mal: float, malicious distortion multiplier
        detection_fn: function, detection algorithm
        xi0_param: float, optional, detection threshold
        gamma_param: float, optional, threshold growth rate
        v_true_precomputed: numpy array, optional, pre-generated valuation matrix
                           If provided, uses this instead of generating new random data
                           This ensures fair comparison across different xi0 values
        trial: int, optional
               Trial index. If provided, seeds the report factory so version 1 and 2 use the same dataset.
    
    Returns:
        Dictionary with:
            - nsw_offline: NSW for offline optimal (trustworthy agents only)
            - nsw_online_trustonly: NSW for online clean scenario
            - nsw_online_all_no_detect: NSW for polluted scenario without detection
            - nsw_online_all_with_detect: NSW for polluted scenario with detection
            - detected_mask: boolean array of detected agents
            - v_true: true valuation matrix
    """
    # Use parameter values if provided, otherwise use defaults
    if xi0_param is None:
        xi0_param = xi0_default
    if gamma_param is None:
        gamma_param = gamma
    
    N_all = N_trust + N_mal
    # Total number of agents
    
    # Use precomputed dataset if provided, otherwise generate new random data
    if v_true_precomputed is not None:
        v_true = v_true_precomputed.copy()
        # Use the fixed dataset (ensures fair comparison across xi0 values)
    else:
        # Generate random true valuations (for single-run mode)
        v_true = np.random.rand(N_all, T)
        # Random values in [0, 1)
        v_true = normalize_per_round(v_true)
        # Normalize so each column sums to 1

    # Define agent indices
    trust_idx = np.arange(N_trust)
    # Indices 0 to N_trust-1 are trustworthy
    
    mal_idx = list(range(N_trust, N_all))
    # Indices N_trust to N_all-1 are malicious
    
    # Fixed predicted total valuation V_tilde_i (Paper 1): input only, independent of reports
    V_true_total = v_true.sum(axis=1)
    rng_pred = np.random.RandomState(5000 + trial) if trial is not None else np.random.RandomState()
    predicted_V_fixed = V_true_total * (1 + rng_pred.uniform(-0.15, 0.15, N_all))

    # Trust/detection: exogenous alpha (pre-generated by agent type; Center does not use v_true).
    alpha_true = generate_alpha_true(N_all, T, mal_idx, seed=(4000 + trial) if trial is not None else None)

    # Scenario: Offline Optimal (benchmark)
    nsw_offline, u_offline_full, Xopt = offline_optimal_nsw(
        v_true,
        agent_mask=np.array([True] * N_trust + [False] * N_mal)
    )
    # Compute optimal NSW knowing all future valuations
    # Only optimize for trustworthy agents (fair comparison)
    
    # Scenario A: Online Clean (only trustworthy agents)
    v_trustonly = v_true[:N_trust, :]
    # Extract valuations for trustworthy agents only
    
    u_online_trustonly, nsw_online_trustonly, detected_dummy = \
        online_set_aside_with_trust(
            v_trustonly, None, detect_fn=None, remove_detected=False,
            trust_indices=None, xi0_param=xi0_param, gamma_param=gamma_param,
            predicted_V_fixed=predicted_V_fixed[:N_trust]
        )
    # Run online algorithm with only trustworthy agents
    # No malicious agents → no detection needed
    
    # Scenario B: Online Polluted WITHOUT Detection (baseline)
    report_seed_no_detect = (1000 + trial) if trial is not None else None
    report_seed_with_detect = (2000 + trial) if trial is not None else None
    reports_func = malicious_report_func_factory(v_true, c_mal, mal_idx, seed=report_seed_no_detect)
    # Create reporting function with malicious distortion (seeded for fair comparison across v1/v2)
    
    u_online_all_no_detect, nsw_online_all_no_detect, detected_dummy2 = \
        online_set_aside_with_trust(
            v_true, reports_func, detect_fn=None, remove_detected=False,
            trust_indices=trust_idx, xi0_param=xi0_param, gamma_param=gamma_param,
            predicted_V_fixed=predicted_V_fixed
        )
    # Run online algorithm with malicious agents but NO detection
    # This is the baseline: how bad is it without detection?
    # trust_indices=trust_idx: only compute NSW for trustworthy agents
    
    # Scenario C: Online Polluted WITH Detection
    # Detection: exogenous alpha (pre-generated), cumulative beta (Paper 2 Eq. 9), Paper 2 threshold.
    detect_fn_trust = make_detect_fn_trust_observations(
        mal_idx, alpha_precomputed=alpha_true, seed=(3000 + trial) if trial is not None else None
    )
    reports_func2 = malicious_report_func_factory(v_true, c_mal, mal_idx, seed=report_seed_with_detect)
    # Seeded so version 1 and 2 see same report stream (fair comparison)
    
    u_online_all_with_detect, nsw_online_all_with_detect, detected_mask = \
        online_set_aside_with_trust(
            v_true, reports_func2, detect_fn=detect_fn_trust, remove_detected=True,
            trust_indices=trust_idx, xi0_param=xi0_param, gamma_param=gamma_param,
            predicted_V_fixed=predicted_V_fixed
        )
    # Run online algorithm with malicious agents AND detection
    # remove_detected=True: detected agents get zero uniform allocation
    # This tests: does detection improve performance?
    
    return {
        'nsw_offline': nsw_offline,
        'nsw_online_trustonly': nsw_online_trustonly,
        'nsw_online_all_no_detect': nsw_online_all_no_detect,
        'nsw_online_all_with_detect': nsw_online_all_with_detect,
        'detected_mask': detected_mask,
        'v_true': v_true
    }


# ============================================================================
# SENSITIVITY ANALYSIS MODE
# ============================================================================

if RUN_SENSITIVITY_ANALYSIS:
    # Parameter sweep: test algorithm across range of xi0 values
    # Fixed dataset approach: Generate all datasets upfront, reuse for all xi0 values
    # This ensures fair comparison - all xi0 values tested on same datasets
    
    print("=" * 70)
    print("SENSITIVITY ANALYSIS MODE (FIXED DATASET APPROACH)")
    print("=" * 70)
    print("Sweeping xi0 from {:.2f} to {:.2f} with step {:.2f}".format(XI0_MIN, XI0_MAX, XI0_STEP))
    print("Fixed parameters: c_mal = {:.1f}, gamma = {:.1f}, n_trials = {}".format(c_mal, gamma, n_trials))
    print()
    print("Generating {} fixed datasets (one per trial)...".format(n_trials))
    
    # Step 1: Generate all datasets upfront with fixed seeds
    # Each trial gets a unique seed, but same seed used for all xi0 values
    fixed_datasets = []
    for trial in range(n_trials):
        # Use trial number as seed offset to ensure reproducibility
        np.random.seed(42 + trial)
        # Generate dataset for this trial
        N_all = N_trust + N_mal
        v_true = np.random.rand(N_all, T)
        v_true = normalize_per_round(v_true)
        fixed_datasets.append(v_true)
    
    print("✓ Generated {} fixed datasets".format(len(fixed_datasets)))
    print("All xi0 values will be tested on the same datasets (fair comparison)")
    print()
    
    # Generate xi0 values to test
    xi0_values = np.arange(XI0_MIN, XI0_MAX + XI0_STEP/2, XI0_STEP)
    # Create array: [XI0_MIN, XI0_MIN+STEP, ..., XI0_MAX]
    # +XI0_STEP/2 ensures we include XI0_MAX in the range
    
    n_xi0 = len(xi0_values)
    # Number of xi0 values to test
    
    sensitivity_results = []
    # List to store results for each xi0 value
    
    # Loop through each xi0 value
    for idx, xi0_val in enumerate(xi0_values):
        print("Progress: {}/{} - Testing xi0 = {:.3f}...".format(idx + 1, n_xi0, xi0_val), end=' ')
        
        # Run multiple trials for this xi0 value, using FIXED datasets
        results = []
        for trial in range(n_trials):
            try:
                # Use the pre-generated dataset for this trial
                v_true_fixed = fixed_datasets[trial]
                result = run_experiment_once(
                    N_trust, N_mal, T, c_mal, detect_malicious_from_reports,
                    xi0_param=xi0_val, gamma_param=gamma,
                    v_true_precomputed=v_true_fixed, trial=trial
                )
                results.append(result)
            except Exception as e:
                print(f"\nError in trial {trial + 1} for xi0={xi0_val:.3f}: {e}")
                continue
        
        if len(results) == 0:
            print("SKIPPED (no successful trials)")
            continue
        
        # Compute statistics across trials
        mean_offline = np.mean([r['nsw_offline'] for r in results])
        mean_online_trustonly = np.mean([r['nsw_online_trustonly'] for r in results])
        mean_online_all_no_detect = np.mean([r['nsw_online_all_no_detect'] for r in results])
        mean_online_all_with_detect = np.mean([r['nsw_online_all_with_detect'] for r in results])
        
        # Option 1: Compare vs Offline (Ideal Benchmark)
        # Ratio = Online Performance / Offline Performance
        # Measures: How close does online algorithm get to ideal?
        option1_no_detect = mean_online_all_no_detect / (mean_offline + 1e-12)
        # Without detection: polluted environment vs ideal
        option1_with_detect = mean_online_all_with_detect / (mean_offline + 1e-12)
        # With detection: polluted environment with detection vs ideal
        # +1e-12 prevents division by zero
        
        # Option 2: Compare vs Online Clean (Pollution Impact)
        # Ratio = Online Polluted / Online Clean
        # Measures: How much does pollution reduce performance?
        option2_no_detect = mean_online_all_no_detect / (mean_online_trustonly + 1e-12)
        # Without detection: polluted vs clean
        option2_with_detect = mean_online_all_with_detect / (mean_online_trustonly + 1e-12)
        # With detection: polluted with detection vs clean
        # This shows: does detection mitigate pollution impact?
        
        # Compute detection statistics
        all_detected = []
        for r in results:
            detected_mask = r['detected_mask']
            mal_indices = np.arange(N_trust, N_trust + N_mal)
            mal_detected = detected_mask[mal_indices].sum()
            # Count how many malicious agents were detected
            all_detected.append(mal_detected)
        
        avg_mal_detected = np.mean(all_detected)
        detection_rate = 100 * avg_mal_detected / N_mal
        # Detection rate: percentage of malicious agents detected
        # Formula: (average detected) / (total malicious) * 100%
        
        # Store results
        sensitivity_results.append({
            'xi0': xi0_val,
            'option1_no_detect': option1_no_detect,
            'option1_with_detect': option1_with_detect,
            'option2_no_detect': option2_no_detect,
            'option2_with_detect': option2_with_detect,
            'detection_rate': detection_rate,
            'nsw_offline': mean_offline,
            'nsw_with_detect': mean_online_all_with_detect,
            'nsw_no_detect': mean_online_all_no_detect,
            'nsw_clean': mean_online_trustonly
        })
        
        print("Option1: NoDet={:.3f}, WithDet={:.3f} | Option2: NoDet={:.3f}, WithDet={:.3f} | DetRate={:.1f}%".format(
            option1_no_detect, option1_with_detect, option2_no_detect, option2_with_detect, detection_rate))
    
    print()
    print("=" * 70)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("=" * 70)
    
    # Extract data for plotting
    xi0_vals = [r['xi0'] for r in sensitivity_results]
    option1_no_detect = [r['option1_no_detect'] for r in sensitivity_results]
    option1_with_detect = [r['option1_with_detect'] for r in sensitivity_results]
    option2_no_detect = [r['option2_no_detect'] for r in sensitivity_results]
    option2_with_detect = [r['option2_with_detect'] for r in sensitivity_results]
    detection_rates = [r['detection_rate'] for r in sensitivity_results]
    
    # Compute baseline (without detection should be constant, but average across xi0 for robustness)
    # Filter out NaN values (crashes) when computing baseline
    baseline_option1_no_detect = np.nanmean(option1_no_detect)
    baseline_option2_no_detect = np.nanmean(option2_no_detect)
    
    # Find optimal xi0 separately for Option 1 and Option 2 (independent optimization)
    # Filter out NaN values (crashes) when finding optimal - same logic for both options
    if len(option2_with_detect) > 0 and len(option1_with_detect) > 0:
        # Option 1: Find optimal based on Option 1 metric (vs Offline)
        # Ignore NaN values (crashes) when finding maximum - same as Option 2
        valid_option1 = np.array(option1_with_detect)
        valid_mask1 = ~np.isnan(valid_option1)
        if np.any(valid_mask1):
            optimal_idx_option1 = np.nanargmax(option1_with_detect)
            optimal_xi0_option1 = xi0_vals[optimal_idx_option1]
            optimal_option1_value = option1_with_detect[optimal_idx_option1]
        else:
            optimal_idx_option1 = 0
            optimal_xi0_option1 = xi0_vals[0]
            optimal_option1_value = np.nan
        
        # Option 2: Find optimal based on Option 2 metric (vs Clean)
        # Ignore NaN values (crashes) when finding maximum - same logic as Option 1
        valid_option2 = np.array(option2_with_detect)
        valid_mask2 = ~np.isnan(valid_option2)
        if np.any(valid_mask2):
            optimal_idx_option2 = np.nanargmax(option2_with_detect)
            optimal_xi0_option2 = xi0_vals[optimal_idx_option2]
            optimal_option2_value = option2_with_detect[optimal_idx_option2]
        else:
            optimal_idx_option2 = 0
            optimal_xi0_option2 = xi0_vals[0]
            optimal_option2_value = np.nan
        
        print("\n" + "=" * 70)
        print("OPTIMAL THRESHOLD ANALYSIS (INDEPENDENT FOR EACH OPTION)")
        print("=" * 70)
        print("NOTE: Optimal points are found from NON-CRASHED regions only")
        if not np.isnan(optimal_option1_value):
            print("Option 1 Optimal xi0: {:.3f} (Ratio vs Offline: {:.3f}) [VALID - Non-Crashed]".format(optimal_xi0_option1, optimal_option1_value))
        else:
            print("Option 1 Optimal xi0: {:.3f} (Ratio vs Offline: NaN) [ALL VALUES CRASHED]".format(optimal_xi0_option1))
        if not np.isnan(optimal_option2_value):
            print("Option 2 Optimal xi0: {:.3f} (Ratio vs Clean: {:.3f}) [VALID - Non-Crashed]".format(optimal_xi0_option2, optimal_option2_value))
        else:
            print("Option 2 Optimal xi0: {:.3f} (Ratio vs Clean: NaN) [ALL VALUES CRASHED]".format(optimal_xi0_option2))
        print()
        print("Baseline (Without Detection):")
        print("  Option 1 (vs Offline): {:.3f}".format(baseline_option1_no_detect))
        print("  Option 2 (vs Clean):   {:.3f}".format(baseline_option2_no_detect))
        
        # Find optimal range for Option 1 (within 5% of peak)
        threshold_option1 = optimal_option1_value * 0.95
        good_indices_option1 = [i for i, ratio in enumerate(option1_with_detect) if ratio >= threshold_option1]
        if good_indices_option1:
            good_xi0_min_option1 = min([xi0_vals[i] for i in good_indices_option1])
            good_xi0_max_option1 = max([xi0_vals[i] for i in good_indices_option1])
            print("\nOption 1 Optimal range (within 5% of peak): [{:.3f}, {:.3f}]".format(good_xi0_min_option1, good_xi0_max_option1))
        
        # Find optimal range for Option 2 (within 5% of peak)
        threshold_option2 = optimal_option2_value * 0.95
        good_indices_option2 = [i for i, ratio in enumerate(option2_with_detect) if ratio >= threshold_option2]
        if good_indices_option2:
            good_xi0_min_option2 = min([xi0_vals[i] for i in good_indices_option2])
            good_xi0_max_option2 = max([xi0_vals[i] for i in good_indices_option2])
            print("Option 2 Optimal range (within 5% of peak): [{:.3f}, {:.3f}]".format(good_xi0_min_option2, good_xi0_max_option2))
        
        # Identify three performance zones for Option 1
        dead_zone_threshold_option1 = optimal_option1_value * 0.3
        dead_indices_option1 = [i for i, ratio in enumerate(option1_with_detect) if ratio <= dead_zone_threshold_option1]
        if dead_indices_option1:
            dead_zone_max_option1 = max([xi0_vals[i] for i in dead_indices_option1])
            print("\nOption 1 - Zone 1 (Dead Zone): xi0 <= {:.3f} - Threshold too strict".format(dead_zone_max_option1))
        
        peak_zone_threshold_option1 = optimal_option1_value * 0.9
        peak_indices_option1 = [i for i, ratio in enumerate(option1_with_detect) if ratio >= peak_zone_threshold_option1]
        if peak_indices_option1:
            peak_zone_min_option1 = min([xi0_vals[i] for i in peak_indices_option1])
            peak_zone_max_option1 = max([xi0_vals[i] for i in peak_indices_option1])
            print("Option 1 - Zone 2 (Peak Zone): xi0 in [{:.3f}, {:.3f}]".format(peak_zone_min_option1, peak_zone_max_option1))
        
        degradation_threshold_option1 = optimal_option1_value * 0.7
        degradation_indices_option1 = [i for i, ratio in enumerate(option1_with_detect) 
                                       if ratio <= degradation_threshold_option1 and xi0_vals[i] > optimal_xi0_option1]
        if degradation_indices_option1:
            degradation_zone_min_option1 = min([xi0_vals[i] for i in degradation_indices_option1])
            print("Option 1 - Zone 3 (Degradation Zone): xi0 >= {:.3f}".format(degradation_zone_min_option1))
        
        # Identify three performance zones for Option 2
        dead_zone_threshold_option2 = optimal_option2_value * 0.3
        dead_indices_option2 = [i for i, ratio in enumerate(option2_with_detect) if ratio <= dead_zone_threshold_option2]
        if dead_indices_option2:
            dead_zone_max_option2 = max([xi0_vals[i] for i in dead_indices_option2])
            print("\nOption 2 - Zone 1 (Dead Zone): xi0 <= {:.3f} - Threshold too strict".format(dead_zone_max_option2))
        
        peak_zone_threshold_option2 = optimal_option2_value * 0.9
        peak_indices_option2 = [i for i, ratio in enumerate(option2_with_detect) if ratio >= peak_zone_threshold_option2]
        if peak_indices_option2:
            peak_zone_min_option2 = min([xi0_vals[i] for i in peak_indices_option2])
            peak_zone_max_option2 = max([xi0_vals[i] for i in peak_indices_option2])
            print("Option 2 - Zone 2 (Peak Zone): xi0 in [{:.3f}, {:.3f}]".format(peak_zone_min_option2, peak_zone_max_option2))
        
        degradation_threshold_option2 = optimal_option2_value * 0.7
        degradation_indices_option2 = [i for i, ratio in enumerate(option2_with_detect) 
                                       if ratio <= degradation_threshold_option2 and xi0_vals[i] > optimal_xi0_option2]
        if degradation_indices_option2:
            degradation_zone_min_option2 = min([xi0_vals[i] for i in degradation_indices_option2])
            print("Option 2 - Zone 3 (Degradation Zone): xi0 >= {:.3f}".format(degradation_zone_min_option2))
    
    # Create visualizations
    if len(option2_with_detect) > 0 and len(option1_with_detect) > 0:
        # Plot 1: Option 1 Comparison (vs Offline Ideal) - Uses Option 1's own optimal point
        fig1, ax1 = plt.subplots(1, 1, figsize=(11, 7))
        # Create figure and axis: 1 row, 1 column, size 11x7 inches
        
        # Mark crash points (NaN values) with red X markers - same as Option 2
        crash_mask_option1 = np.isnan(option1_with_detect)
        if np.any(crash_mask_option1):
            crash_xi0_option1 = np.array(xi0_vals)[crash_mask_option1]
            ax1.scatter(crash_xi0_option1, [baseline_option1_no_detect] * len(crash_xi0_option1), 
                       marker='x', color='red', s=200, linewidths=3, zorder=10,
                       label='Crash Points (NaN)', alpha=0.8)
        
        # Plot valid data points only (filter out NaN values) - same as Option 2
        valid_mask_option1 = ~np.isnan(option1_with_detect)
        if np.any(valid_mask_option1):
            ax1.plot(np.array(xi0_vals)[valid_mask_option1], 
                    np.array(option1_with_detect)[valid_mask_option1], 
                    'b-o', markersize=4, linewidth=2, 
                    label='With Detection', zorder=3)
        # Plot: blue line with circles, shows Option 1 ratio with detection vs xi0
        # Red X markers show where code crashed (NaN values)
        # zorder=3: draw on top
        
        ax1.axhline(y=baseline_option1_no_detect, color='r', linestyle='--', linewidth=2.5, 
                    label='Without Detection (Baseline)', zorder=2)
        # Horizontal line: baseline performance without detection
        # Red dashed line, zorder=2: draw below main line
        
        ax1.set_xlabel('xi0 (Detection Threshold)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Ratio (vs Offline Ideal)', fontsize=13, fontweight='bold')
        ax1.set_title('Option 1: Detection Effectiveness vs Offline Ideal', fontsize=15, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, zorder=0)
        # Grid: light gray, zorder=0: draw at bottom
        
        ax1.legend(fontsize=11, loc='upper right', framealpha=0.95)
        # Legend: upper right, semi-transparent background
        
        # Mark optimal point for Option 1 (independent optimization)
        # Red circle marking Option 1's optimal point - make it very visible
        # Only plot if optimal value is not NaN (no crash)
        if not np.isnan(optimal_option1_value):
            ax1.plot(optimal_xi0_option1, optimal_option1_value, 'ro', markersize=14, 
                    markeredgewidth=2, markeredgecolor='darkred', zorder=10, label='Optimal Point')
            
            # Yellow annotation box - position at (0.5, 0.7) - right below the legend (same as Option 2)
            ax1.annotate(f'Optimal Point (Option 1)\nxi0={optimal_xi0_option1:.3f}\nRatio={optimal_option1_value:.3f}', 
                         xy=(optimal_xi0_option1, optimal_option1_value),
                         xytext=(0.5, 0.7),
                         textcoords='axes fraction',  # Use axes coordinates for xytext
                         fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.95, edgecolor='black', linewidth=2),
                         ha='center', va='top',
                         zorder=11)  # Ensure annotation is on top
        # Annotation: yellow box with Option 1's optimal point info
        # Positioned at (0.5, 0.7) in axes coordinates (fraction of plot width/height)
        # va='top' ensures the box is positioned below the specified y-coordinate
        
        # Shade region where detection helps
        # Filter out NaN values for fill_between - same as Option 2
        valid_mask_fill1 = ~np.isnan(option1_with_detect)
        if np.any(valid_mask_fill1):
            valid_xi0_1 = np.array(xi0_vals)[valid_mask_fill1]
            valid_option1_1 = np.array(option1_with_detect)[valid_mask_fill1]
            ax1.fill_between(valid_xi0_1, baseline_option1_no_detect, valid_option1_1, 
                             where=(valid_option1_1 > baseline_option1_no_detect),
                             interpolate=True, alpha=0.15, color='green', label='Detection Helps Region')
        # Green shading: region where detection improves performance
        # interpolate=True: calculates intersections between lines for smooth filling
        
        plt.tight_layout()
        plt.savefig('option1_detection_effectiveness_version1.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Generated: option1_detection_effectiveness_version1.png")
        
        # Plot 2: Option 2 Comparison (vs Online Clean) - Uses Option 2's own optimal point
        fig2, ax2 = plt.subplots(1, 1, figsize=(11, 7))
        # Mark crash points (NaN values) with red X markers
        crash_mask_option2 = np.isnan(option2_with_detect)
        if np.any(crash_mask_option2):
            crash_xi0_option2 = np.array(xi0_vals)[crash_mask_option2]
            crash_y_option2 = np.array(option2_with_detect)[crash_mask_option2]
            ax2.scatter(crash_xi0_option2, [baseline_option2_no_detect] * len(crash_xi0_option2), 
                       marker='x', color='red', s=200, linewidths=3, zorder=10,
                       label='Crash Points (NaN)', alpha=0.8)
        
        # Plot valid data points
        valid_mask_option2 = ~np.isnan(option2_with_detect)
        if np.any(valid_mask_option2):
            ax2.plot(np.array(xi0_vals)[valid_mask_option2], 
                    np.array(option2_with_detect)[valid_mask_option2], 
                    'g-s', markersize=4, linewidth=2, 
                    label='With Detection', zorder=3)
        # Green line with squares
        # Red X markers show where code crashed (NaN values)
        
        ax2.axhline(y=baseline_option2_no_detect, color='orange', linestyle='--', linewidth=2.5, 
                    label='Without Detection (Baseline)', zorder=2)
        # Orange dashed baseline
        
        ax2.set_xlabel('xi0 (Detection Threshold)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Ratio (vs Online Clean)', fontsize=13, fontweight='bold')
        ax2.set_title('Option 2: Detection Effectiveness vs Online Clean\nPollution Impact Comparison', 
                     fontsize=15, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, zorder=0)
        
        ax2.legend(fontsize=11, loc='upper right', framealpha=0.95)
        # Legend: upper right, semi-transparent background
        
        # Mark optimal point for Option 2 (independent optimization)
        # Red circle marking Option 2's optimal point - make it very visible
        # Only plot if optimal value is not NaN (no crash)
        if not np.isnan(optimal_option2_value):
            ax2.plot(optimal_xi0_option2, optimal_option2_value, 'ro', markersize=14, 
                    markeredgewidth=2, markeredgecolor='darkred', zorder=10, label='Optimal Point')
        
            # Yellow annotation box - position at (0.5, 0.7) - right below the legend (same as Option 1)
            ax2.annotate(f'Optimal Point (Option 2)\nxi0={optimal_xi0_option2:.3f}\nRatio={optimal_option2_value:.3f}', 
                         xy=(optimal_xi0_option2, optimal_option2_value),
                         xytext=(0.5, 0.7),
                         textcoords='axes fraction',  # Use axes coordinates for xytext
                         fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.95, edgecolor='black', linewidth=2),
                         ha='center', va='top',
                         zorder=11)  # Ensure annotation is on top
        # Annotation: yellow box with Option 2's optimal point info
        # Positioned at (0.1, 0.8) in axes coordinates (fraction of plot width/height)
        
        # Shade region where detection helps
        # Use interpolate=True to fill smoothly even between data points
        # This ensures continuous filling without gaps at intersections
        # Filter out NaN values for fill_between
        valid_mask_fill2 = ~np.isnan(option2_with_detect)
        if np.any(valid_mask_fill2):
            valid_xi0_2 = np.array(xi0_vals)[valid_mask_fill2]
            valid_option2_2 = np.array(option2_with_detect)[valid_mask_fill2]
            ax2.fill_between(valid_xi0_2, 
                            baseline_option2_no_detect, 
                            valid_option2_2, 
                            where=(valid_option2_2 > baseline_option2_no_detect),
                            interpolate=True, alpha=0.15, color='green', label='Detection Helps Region')
        # Green shading: region where detection improves performance
        # interpolate=True: calculates intersections between lines for smooth filling
        
        plt.tight_layout()
        plt.savefig('option2_detection_effectiveness_version1.png', dpi=150, bbox_inches='tight')
        print("Plot 2 saved as 'option2_detection_effectiveness_version1.png'")
        plt.close()
        
        # Plot 3: Detection Rate Reference
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
        ax3.plot(xi0_vals, detection_rates, 'm-^', markersize=4, linewidth=1.5, 
                 label='Detection Rate (%)', zorder=3)
        # Magenta line with triangles: detection rate vs threshold
        
        ax3.set_xlabel('xi0 (Detection Threshold)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Detection Rate (%)', fontsize=13, fontweight='bold')
        ax3.set_title('Detection Rate vs Threshold (Reference)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, zorder=0)
        ax3.legend(fontsize=11)
        ax3.set_ylim([0, 105])
        # Y-axis: 0 to 105% (slightly above 100% for visual clarity)
        
        plt.tight_layout()
        plt.savefig('detection_rate_reference_version1.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Generated: detection_rate_reference_version1.png")
        
        print("\n✓ All 3 plots generated successfully (Version 1):")
        print("  - option1_detection_effectiveness_version1.png")
        print("  - option2_detection_effectiveness_version1.png")
        print("  - detection_rate_reference_version1.png")
    else:
        print("WARNING: No data to plot!")
    
    # Print conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION (INDEPENDENT OPTIMIZATION)")
    print("=" * 70)
    if len(option2_with_detect) > 0 and len(option1_with_detect) > 0:
        print("✓ Found INDEPENDENT optimal thresholds:")
        print("  Option 1 Optimal xi0: {:.3f} (Ratio vs Offline: {:.3f})".format(optimal_xi0_option1, optimal_option1_value))
        print("  Option 2 Optimal xi0: {:.3f} (Ratio vs Clean: {:.3f})".format(optimal_xi0_option2, optimal_option2_value))
        print()
        print("Performance at respective optimal thresholds:")
        print("  Option 1 (vs Offline): {:.3f} (with detection) vs {:.3f} (without detection)".format(
            optimal_option1_value, baseline_option1_no_detect))
        print("  Option 2 (vs Clean):   {:.3f} (with detection) vs {:.3f} (without detection)".format(
            optimal_option2_value, baseline_option2_no_detect))
        print()
        
        # Calculate improvement percentages
        improvement_option1 = 100 * (optimal_option1_value / baseline_option1_no_detect - 1) if baseline_option1_no_detect > 0 else 0
        improvement_option2 = 100 * (optimal_option2_value / baseline_option2_no_detect - 1) if baseline_option2_no_detect > 0 else 0
        
        print("Detection Effectiveness at Respective Optimal xi0:")
        if improvement_option1 > 0:
            print("  ✓ Option 1 (xi0={:.3f}): Detection improves by {:.1f}% vs baseline".format(optimal_xi0_option1, improvement_option1))
        else:
            print("  ✗ Option 1 (xi0={:.3f}): Detection degrades by {:.1f}% vs baseline".format(optimal_xi0_option1, -improvement_option1))
        
        if improvement_option2 > 0:
            print("  ✓ Option 2 (xi0={:.3f}): Detection improves by {:.1f}% vs baseline".format(optimal_xi0_option2, improvement_option2))
        else:
            print("  ✗ Option 2 (xi0={:.3f}): Detection degrades by {:.1f}% vs baseline".format(optimal_xi0_option2, -improvement_option2))
        
        print()
        print("Key Insight: Each comparison metric (Option 1 vs Option 2) may have different optimal thresholds.")
        print("Option 1 focuses on absolute performance vs ideal (offline).")
        print("Option 2 focuses on relative performance vs clean environment (pollution impact).")
        print("The optimal threshold depends on which metric is most important for the application.")
        
        # Export optimal xi0 values for use in utility_trajectories.py
        # Use Option 2's optimal (vs Clean) as default since it's more practical
        # But also provide Option 1's optimal for reference
        optimal_xi0_for_utility = optimal_xi0_option2 if not np.isnan(optimal_option2_value) else optimal_xi0_option1
        print("\n" + "=" * 70)
        print("EXPORTED OPTIMAL XI0 VALUES (for utility_trajectories.py)")
        print("=" * 70)
        print("Recommended xi0 for utility trajectories: {:.3f} (from Option 2 - vs Clean)".format(optimal_xi0_for_utility))
        if not np.isnan(optimal_option1_value):
            print("Alternative xi0 (Option 1 - vs Offline): {:.3f}".format(optimal_xi0_option1))
        print("=" * 70)
        print("\nTo update utility_trajectories.py, set:")
        print("  xi0_optimal = {:.3f}  # From Option 2 (recommended)".format(optimal_xi0_for_utility))
        print("=" * 70)
    print("=" * 70)

else:
    # ============================================================================
    # SINGLE RUN MODE (for quick testing)
    # ============================================================================
    
    print("SINGLE RUN MODE")
    print("(Set RUN_SENSITIVITY_ANALYSIS = True to run parameter sweep)")
    print()
    
    results = []
    for trial in range(n_trials):
        if (trial + 1) % 5 == 0:
            print(f"Running trial {trial + 1}/{n_trials}...")
        
        try:
            result = run_experiment_once(N_trust, N_mal, T, c_mal, detect_malicious_from_reports, trial=trial)
            results.append(result)
        except Exception as e:
            print(f"Error in trial {trial + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("ERROR: No successful trials!")
        exit(1)
    
    # Compute statistics
    mean_offline = np.mean([r['nsw_offline'] for r in results])
    mean_online_trustonly = np.mean([r['nsw_online_trustonly'] for r in results])
    mean_online_all_no_detect = np.mean([r['nsw_online_all_no_detect'] for r in results])
    mean_online_all_with_detect = np.mean([r['nsw_online_all_with_detect'] for r in results])
    
    print("\n=== Summary over {} trials ===".format(n_trials))
    print("PARAMETERS:")
    print("  c_mal = {:.1f} (malicious distortion multiplier)".format(c_mal))
    print("  xi0 = {:.3f} (initial detection threshold)".format(xi0_default))
    print("  gamma = {:.1f} (threshold growth rate)".format(gamma))
    print()
    print("Offline ideal NSW (trustworthy-only)        mean: {:.6f}".format(mean_offline))
    print("Online NSW (trustworthy-only, clean)        mean: {:.6f}".format(mean_online_trustonly))
    print("Online NSW (with malicious, NO detection)  mean: {:.6f}".format(mean_online_all_no_detect))
    print("Online NSW (with malicious, WITH detection) mean: {:.6f}".format(mean_online_all_with_detect))
    print()
    
    print("=== Competitive Ratios (as per Note) ===")
    print()
    print("Option 1: Online vs Offline (Ideal)")
    print("  Online (clean) / Offline:        {:.3f} ({:.1f}%)".format(
        mean_online_trustonly / (mean_offline + 1e-12), 
        100 * mean_online_trustonly / (mean_offline + 1e-12)))
    print("  Online (malicious, no detect) / Offline:  {:.3f} ({:.1f}%)".format(
        mean_online_all_no_detect / (mean_offline + 1e-12),
        100 * mean_online_all_no_detect / (mean_offline + 1e-12)))
    print("  Online (malicious, with detect) / Offline: {:.3f} ({:.1f}%)".format(
        mean_online_all_with_detect / (mean_offline + 1e-12),
        100 * mean_online_all_with_detect / (mean_offline + 1e-12)))
    print()
    
    print("Option 2: Online (with malicious) vs Online (clean) - MEASURE POLLUTION IMPACT")
    print("  Online (malicious, no detect) / Online (clean):  {:.3f} ({:.1f}%)".format(
        mean_online_all_no_detect / (mean_online_trustonly + 1e-12),
        100 * mean_online_all_no_detect / (mean_online_trustonly + 1e-12)))
    print("  Online (malicious, with detect) / Online (clean): {:.3f} ({:.1f}%)".format(
        mean_online_all_with_detect / (mean_online_trustonly + 1e-12),
        100 * mean_online_all_with_detect / (mean_online_trustonly + 1e-12)))
    print()
    
    print("=== Detection Effectiveness ===")
    print("  Detection helps? Compare:")
    print("    Without detection: {:.6f}".format(mean_online_all_no_detect))
    print("    With detection:    {:.6f}".format(mean_online_all_with_detect))
    if mean_online_all_with_detect > mean_online_all_no_detect:
        improvement = 100 * (mean_online_all_with_detect / mean_online_all_no_detect - 1)
        print("    ✓ Detection improves by {:.1f}%".format(improvement))
    else:
        degradation = 100 * (1 - mean_online_all_with_detect / mean_online_all_no_detect)
        print("    ✗ Detection degrades by {:.1f}%".format(degradation))
    
    # Detection statistics
    all_detected = []
    for r in results:
        detected_mask = r['detected_mask']
        trust_indices = np.arange(N_trust)
        mal_indices = np.arange(N_trust, N_trust + N_mal)
        
        mal_detected = detected_mask[mal_indices].sum()
        trust_falsely_detected = detected_mask[trust_indices].sum()
        
        all_detected.append({
            'mal_detected': mal_detected,
            'trust_falsely_detected': trust_falsely_detected
        })
    
    avg_mal_detected = np.mean([d['mal_detected'] for d in all_detected])
    avg_trust_falsely_detected = np.mean([d['trust_falsely_detected'] for d in all_detected])
    
    print()
    print("=== Detection Statistics ===")
    print("  Average malicious agents detected: {:.2f} / {} ({:.1f}%)".format(
        avg_mal_detected, N_mal, 100 * avg_mal_detected / N_mal))
    print("  Average trustworthy agents falsely detected: {:.2f} / {} ({:.1f}%)".format(
        avg_trust_falsely_detected, N_trust, 100 * avg_trust_falsely_detected / N_trust))
    print()
    print("  Detection Rate: {:.1f}%".format(100 * avg_mal_detected / N_mal))
