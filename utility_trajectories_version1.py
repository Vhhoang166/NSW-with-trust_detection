#!/usr/bin/env python3
"""
Utility Trajectories Visualization

This script creates a visualization showing cumulative utility of trustworthy agents
at each iteration t (time step), comparing 4 scenarios:
1. Offline Ideal (optimal allocation with full knowledge)
2. Online Clean (only trustworthy agents)
3. Online Malicious - No Detection (baseline)
4. Online Malicious - With Detection (using optimal xi0)

This addresses the Note requirement: "visualized graph Utility level at each iteration t"

VERSION 1: Uses Proportional Allocation (follows NSW version 1)

This script creates a visualization showing cumulative utility of trustworthy agents
at each iteration t (time step), comparing 4 scenarios:
1. Offline Ideal (optimal allocation with full knowledge)
2. Online Clean (only trustworthy agents)
3. Online Malicious - No Detection (baseline)
4. Online Malicious - With Detection (using optimal xi0)

This addresses the Note requirement: "visualized graph Utility level at each iteration t"

Updated for Version 1: Uses fixed dataset approach and proportional allocation (same as NSW version 1).
"""

import numpy as np
from scipy.optimize import minimize
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Import functions from version 5 (or copy them)
# For simplicity, we'll copy the essential functions here

# Parameters (using optimal values from Version 1 sensitivity analysis)
N_trust = 10
N_mal = 4
T = 10
c_mal = 5.0  # More realistic malicious distortion
n_trials = 20  # Number of trials to average over
# IMPORTANT: This value should match the optimal xi0 from Version 1 sensitivity analysis
# The optimal xi0 should be from the NON-CRASHED region (Option 2 recommended)
# Updated value: 0.29 (from Version 1 sensitivity analysis output - Option 2 optimal point)
xi0_optimal = 0.29  # Optimal xi0 from Version 1 sensitivity analysis (Option 2 - vs Clean, non-crashed region)
gamma = 0.1
alpha_pairwise = 5.0


# ----------------------------------------------------------------------
# HELPER FUNCTIONS (copied from version 4)
# ----------------------------------------------------------------------
def normalize_per_round(v):
    """Normalize columns so that sum_i v[i,t] = 1 each t"""
    col_sums = v.sum(axis=0)
    col_sums[col_sums == 0] = 1
    return v / col_sums


def compute_nsw_from_util(u):
    """Geometric mean (NSW) given utility vector u"""
    if np.any(u <= 0):
        # Zero or negative utilities → return NaN to indicate crash
        return np.nan
    return np.exp(np.mean(np.log(u)))


def offline_optimal_nsw(v_true, agent_mask=None):
    """Compute offline optimal NSW (with full knowledge of future)"""
    N, T = v_true.shape
    
    if agent_mask is not None:
        trust_indices = np.where(agent_mask)[0]
        v_optimize = v_true[trust_indices, :]
        N_opt = len(trust_indices)
    else:
        v_optimize = v_true
        N_opt = N
        trust_indices = np.arange(N)
    
    def objective(x_flat):
        X = x_flat.reshape(N_opt, T)
        utilities = (v_optimize * X).sum(axis=1)
        if np.any(utilities <= 0):
            return 1e6
        return -np.sum(np.log(utilities))
    
    constraints = []
    for t in range(T):
        constraints.append({
            'type': 'eq',
            'fun': lambda x_flat, t=t: np.sum(x_flat.reshape(N_opt, T)[:, t]) - 1
        })
    
    x0 = np.ones(N_opt * T) / (N_opt * T)
    bounds = [(0, 1) for _ in range(N_opt * T)]
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        X_opt = result.x.reshape(N_opt, T)
        utilities_opt = (v_optimize * X_opt).sum(axis=1)
    else:
        X_opt = np.ones((N_opt, T)) / N_opt
        utilities_opt = (v_optimize * X_opt).sum(axis=1)
    
    # Return X_opt (allocation matrix) for trajectory computation
    return X_opt


def detect_malicious_from_reports(reports_history, xi0=0.1, gamma=0.7):
    """
    Detect malicious agents using pairwise similarity analysis.
    
    Algorithm:
        1. Compute pairwise differences in reported values
        2. Convert differences to similarity scores (beta matrix)
        3. For each agent, determine who they "trust" (similar agents)
        4. Count how many agents trust each agent
        5. Flag agents trusted by fewer than half as malicious
    
    Parameters:
        reports_history: numpy array of shape (N_all, t) where t is current time
        xi0: Initial detection threshold
        gamma: Threshold growth exponent
    
    Returns:
        detected_malicious: boolean array, True if agent is detected as malicious
        beta: similarity matrix
        xi_t: current threshold value
    """
    N_all = reports_history.shape[0]
    t_current = reports_history.shape[1] if reports_history.ndim > 1 else 1
    
    # Step 1: Compute pairwise differences
    diffs = np.zeros((N_all, N_all))
    for i in range(N_all):
        for j in range(N_all):
            if i == j:
                diffs[i, j] = 0.0
            else:
                # Mean absolute difference in reported values
                diffs[i, j] = np.mean(np.abs(reports_history[i, :] - reports_history[j, :]))
    
    # Step 2: Convert differences to similarity scores
    # Higher difference → lower similarity
    beta = np.exp(-alpha_pairwise * diffs)
    
    # Step 3: Compute time-varying threshold
    t = t_current - 1
    xi_t = xi0 * ((t + 1) ** gamma)
    
    # Step 4: For each agent i, determine who they trust
    # An agent i trusts agent j if beta[i,j] is close to the maximum similarity
    kept_counts = np.zeros(N_all, dtype=int)
    # kept_counts[k] = number of agents who "keep" (trust) agent k
    
    for i in range(N_all):
        row = beta[i, :]
        # Similarity scores from agent i's perspective
        # row[j] = how similar agent i thinks agent j is
        
        jstar = np.argmax(row)
        # jstar = agent i's most similar agent (could be themselves)
        
        cutoff = row[jstar] - xi_t
        # Trust threshold: trust agents within xi_t of the most similar agent
        # Lower xi_t = stricter (fewer agents trusted)
        # Higher xi_t = looser (more agents trusted)
        
        kept = (row >= cutoff)
        # Boolean mask: kept[j] = True if agent i trusts agent j
        
        kept_counts += kept.astype(int)
        # Count: for each agent j, increment count if agent i trusts j
        # After loop: kept_counts[j] = number of agents who trust agent j
    
    # Step 5: Detect malicious agents
    detected_malicious = (kept_counts < (N_all // 2))
    # Detection rule: if fewer than half of agents trust you, you're malicious
    # Rationale: trustworthy agents should be trusted by majority
    # Malicious agents (reporting distorted values) will be trusted by few
    
    return detected_malicious, beta, xi_t


def malicious_report_func_factory(v_true, c_mal, mal_indices, seed=None):
    """
    Generate malicious reporting function.
    
    Malicious agents report distorted values: v_reported = v_true * factor
    where factor ~ Uniform(1/c_mal, c_mal)
    
    Parameters:
        v_true: True valuation matrix
        c_mal: Malicious distortion multiplier
        mal_indices: List or array of malicious agent indices
        seed: Random seed for reproducibility
    
    Returns:
        reports_func: Function that takes (i, t, v_true_inner) and returns reported value
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Convert to set for faster lookup (O(1) instead of O(n))
    mal_set = set(mal_indices) if isinstance(mal_indices, (list, np.ndarray)) else {mal_indices}
    
    def reports_func(i, t, v_true_inner):
        if i in mal_set:
            # Malicious agent: distort reported value
            factor = rng.uniform(1.0 / c_mal, c_mal)
            return v_true_inner[i, t] * factor
        else:
            # Trustworthy agent: report truthfully
            return v_true_inner[i, t]
    
    return reports_func


# ----------------------------------------------------------------------
# MODIFIED FUNCTION: Returns utility trajectory over time
# ----------------------------------------------------------------------
def online_set_aside_with_trust_trajectory(v_true, reports_func_for_malicious, detect_fn, 
                                           remove_detected=True, trust_indices=None, 
                                           xi0_param=None, gamma_param=None):
    """
    Modified version that returns cumulative utility trajectory at each time step t
    Returns: (u_total_final, nsw_online, detected, utility_trajectory)
    where utility_trajectory[t] = cumulative utility of trustworthy agents up to time t
    """
    if xi0_param is None:
        xi0_param = xi0_optimal
    if gamma_param is None:
        gamma_param = gamma
    
    N_all = v_true.shape[0]
    x = np.zeros((N_all, T))
    y = np.zeros((N_all, T))
    z = np.zeros((N_all, T))
    u_greedy = np.zeros(N_all)
    reported_cum = np.zeros((N_all, 0))
    detected = np.zeros(N_all, dtype=bool)
    any_detected_ever = False
    
    # NEW: Track cumulative utility trajectory for trustworthy agents
    utility_trajectory = np.zeros(T)
    
    for t in range(T):
        # 1) Reports
        reported_round = np.zeros(N_all)
        for i in range(N_all):
            if reports_func_for_malicious is None:
                reported_round[i] = v_true[i, t]
            else:
                reported_round[i] = reports_func_for_malicious(i, t, v_true)
        
        # Append to history
        if reported_cum.size == 0:
            reported_cum = reported_round.reshape((N_all, 1))
        else:
            reported_cum = np.hstack([reported_cum, reported_round.reshape((N_all, 1))])
        
        # 2) Detection
        if detect_fn is not None:
            detected_mask, beta, xi_t = detect_fn(reported_cum, xi0=xi0_param, gamma=gamma_param)
            detected = detected_mask.copy()
            if detected.any():
                any_detected_ever = True
        else:
            detected = np.zeros(N_all, dtype=bool)
            beta = np.ones((N_all, N_all))
            xi_t = 0.0
        
        # 3) Uniform allocation
        eligible = ~detected if remove_detected else np.ones(N_all, dtype=bool)
        n_eligible = max(1, eligible.sum())
        x[eligible, t] = 0.5 / n_eligible
        x[~eligible, t] = 0.0
        
        # 4) Greedy allocation using proportional allocation (VERSION 1 – follows NSW version 1)
        # Proportional allocation: allocate proportionally to reported value / predicted utility
        reported_total_est = reported_cum.sum(axis=1) if reported_cum.ndim > 1 else reported_cum
        predicted_Vi = (reported_total_est / (t + 1)) * T
        predicted_util = u_greedy + predicted_Vi / (2 * N_all)
        
        # Compute priorities for proportional allocation (same as NSW version 1)
        priority = np.zeros(N_all)
        denom = predicted_util.copy()
        # Note: If denom contains zeros, division will produce inf/nan (visible in plots as crashes)
        priority[eligible] = reported_round[eligible] / denom[eligible]
        # Priority formula: priority[i] = reported_value[i] / predicted_util[i]
        
        if detect_fn is not None and any_detected_ever:
            incoming_trust = beta.mean(axis=0)
            priority = priority * incoming_trust
        
        # Proportional allocation: allocate proportionally to priorities (follows NSW v1)
        y[:, t] = 0.0
        if n_eligible > 0:
            eligible_indices = np.where(eligible)[0]
            eligible_priority = priority[eligible_indices]
            
            total_priority = eligible_priority.sum()
            if total_priority > 0:
                for idx in eligible_indices:
                    y[idx, t] = 0.5 * (priority[idx] / total_priority)
            else:
                y[eligible_indices, t] = 0.5 / n_eligible
        
        # Verify total allocation equals 0.5
        total_allocated = y[:, t].sum()
        if abs(total_allocated - 0.5) > 1e-10:
            y[:, t] = y[:, t] * (0.5 / total_allocated)
        
        # 5) Update utilities
        u_greedy += v_true[:, t] * y[:, t]
        z[:, t] = x[:, t] + y[:, t]
        
        # NEW: Compute cumulative utility up to time t for trustworthy agents
        u_total_so_far = (v_true[:, :t+1] * z[:, :t+1]).sum(axis=1)
        if trust_indices is not None:
            u_trustonly_so_far = u_total_so_far[trust_indices]
            utility_trajectory[t] = u_trustonly_so_far.sum()  # Sum of cumulative utilities
        else:
            utility_trajectory[t] = u_total_so_far.sum()
    
    # Final utilities
    u_total = (v_true * z).sum(axis=1)
    if trust_indices is not None:
        u_trustonly = u_total[trust_indices]
        nsw_online = compute_nsw_from_util(u_trustonly)
    else:
        nsw_online = compute_nsw_from_util(u_total)
    
    return u_total, nsw_online, detected, utility_trajectory


# ----------------------------------------------------------------------
# MAIN: Run experiments and collect trajectories
# ----------------------------------------------------------------------
print("=" * 70)
print("UTILITY TRAJECTORIES VISUALIZATION - VERSION 1 (Water Filling Allocation)")
print("=" * 70)
print("Using optimal xi0 = {:.3f} (from Version 1 sensitivity analysis - Option 2 optimal point, non-crashed region)".format(xi0_optimal))
print("Generating {} fixed datasets (one per trial)...".format(n_trials))

# Step 1: Generate all datasets upfront with fixed seeds (consistent with Version 5)
# This ensures reproducibility and fair comparison
fixed_datasets = []
for trial in range(n_trials):
    np.random.seed(42 + trial)  # Fixed seed per trial (same as Version 5)
    N_all = N_trust + N_mal
    v_true = np.random.rand(N_all, T)
    v_true = normalize_per_round(v_true)
    fixed_datasets.append(v_true)

print("✓ Generated {} fixed datasets".format(len(fixed_datasets)))
print("Running {} trials and averaging trajectories...".format(n_trials))
print()

# Storage for trajectories (averaged across trials)
traj_offline = np.zeros(T)
traj_online_clean = np.zeros(T)
traj_online_no_detect = np.zeros(T)
traj_online_with_detect = np.zeros(T)

for trial in range(n_trials):
    if (trial + 1) % 5 == 0:
        print("Running trial {}/{}...".format(trial + 1, n_trials))
    
    try:
        # Use fixed dataset for this trial (consistent with Version 5 methodology)
        N_all = N_trust + N_mal
        v_true = fixed_datasets[trial].copy()  # Use pre-generated fixed dataset
        
        trust_idx = np.arange(N_trust)
        mal_idx = list(range(N_trust, N_all))
        
        # SCENARIO 1: Offline Ideal (cumulative utility trajectory)
        # Compute optimal allocation for trustworthy agents only
        X_opt_full = offline_optimal_nsw(v_true, 
                                         agent_mask=np.array([True] * N_trust + [False] * N_mal))
        # X_opt_full has shape (N_trust, T) - optimal allocation for trustworthy agents
        # Compute cumulative utility trajectory from optimal allocation
        for t_idx in range(T):
            # Cumulative utility up to time t_idx for trustworthy agents
            # v_true[trust_idx, :t_idx+1] has shape (N_trust, t_idx+1)
            # X_opt_full[:, :t_idx+1] has shape (N_trust, t_idx+1)
            cumulative_util = (v_true[trust_idx, :t_idx+1] * X_opt_full[:, :t_idx+1]).sum()
            traj_offline[t_idx] += cumulative_util
        
        # SCENARIO 2: Online Clean (only trustworthy agents)
        v_trustonly = v_true[:N_trust, :]
        _, _, _, traj_clean = online_set_aside_with_trust_trajectory(
            v_trustonly, None, detect_fn=None, remove_detected=False, 
            trust_indices=None, xi0_param=xi0_optimal, gamma_param=gamma)
        traj_online_clean += traj_clean
        
        # SCENARIO 3: Online Malicious - No Detection
        # Use different seed for malicious reports to ensure independent randomness
        reports_func = malicious_report_func_factory(v_true, c_mal, mal_idx, seed=1000 + trial)
        _, _, _, traj_no_detect = online_set_aside_with_trust_trajectory(
            v_true, reports_func, detect_fn=None, remove_detected=False,
            trust_indices=trust_idx, xi0_param=xi0_optimal, gamma_param=gamma)
        traj_online_no_detect += traj_no_detect
        
        # SCENARIO 4: Online Malicious - With Detection
        # Use different seed for malicious reports (independent from scenario 3)
        reports_func2 = malicious_report_func_factory(v_true, c_mal, mal_idx, seed=2000 + trial)
        _, _, _, traj_with_detect = online_set_aside_with_trust_trajectory(
            v_true, reports_func2, detect_fn=detect_malicious_from_reports, 
            remove_detected=True, trust_indices=trust_idx, 
            xi0_param=xi0_optimal, gamma_param=gamma)
        traj_online_with_detect += traj_with_detect
        
    except Exception as e:
        print(f"Error in trial {trial + 1}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Average across trials
traj_offline /= n_trials
traj_online_clean /= n_trials
traj_online_no_detect /= n_trials
traj_online_with_detect /= n_trials

# Time steps (1-indexed for clarity)
time_steps = np.arange(1, T + 1)

# Create visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot all 4 trajectories (filter out NaN values for crashes)
# Mark crash points with red X markers
crash_markers_added = False

# Plot Offline Ideal
valid_offline = ~np.isnan(traj_offline)
if np.any(valid_offline):
    ax.plot(time_steps[valid_offline], traj_offline[valid_offline], 'k-', linewidth=3, marker='o', markersize=8,
            label='Offline Ideal (Full Knowledge)', zorder=4)
if np.any(~valid_offline) and not crash_markers_added:
    crash_times = time_steps[~valid_offline]
    ax.scatter(crash_times, [0] * len(crash_times), marker='x', color='red', s=200, linewidths=3, 
               zorder=10, label='Crash Points (NaN)', alpha=0.8)
    crash_markers_added = True

# Plot Online Clean
valid_clean = ~np.isnan(traj_online_clean)
if np.any(valid_clean):
    ax.plot(time_steps[valid_clean], traj_online_clean[valid_clean], 'b-', linewidth=2.5, marker='s', markersize=6,
            label='Online Clean (Only Trustworthy Agents)', zorder=3)
if np.any(~valid_clean) and not crash_markers_added:
    crash_times = time_steps[~valid_clean]
    ax.scatter(crash_times, [0] * len(crash_times), marker='x', color='red', s=200, linewidths=3, 
               zorder=10, label='Crash Points (NaN)', alpha=0.8)
    crash_markers_added = True

# Plot Online No Detection
valid_no_detect = ~np.isnan(traj_online_no_detect)
if np.any(valid_no_detect):
    ax.plot(time_steps[valid_no_detect], traj_online_no_detect[valid_no_detect], 'r--', linewidth=2.5, marker='^', markersize=6,
            label='Online Malicious - No Detection (Baseline)', zorder=2)
if np.any(~valid_no_detect) and not crash_markers_added:
    crash_times = time_steps[~valid_no_detect]
    ax.scatter(crash_times, [0] * len(crash_times), marker='x', color='red', s=200, linewidths=3, 
               zorder=10, label='Crash Points (NaN)', alpha=0.8)
    crash_markers_added = True

# Plot Online With Detection
valid_with_detect = ~np.isnan(traj_online_with_detect)
if np.any(valid_with_detect):
    ax.plot(time_steps[valid_with_detect], traj_online_with_detect[valid_with_detect], 'g-', linewidth=3, marker='d', markersize=7,
            label='Online Malicious - With Detection (xi0={:.3f})'.format(xi0_optimal), zorder=3, alpha=0.9)
if np.any(~valid_with_detect) and not crash_markers_added:
    crash_times = time_steps[~valid_with_detect]
    ax.scatter(crash_times, [0] * len(crash_times), marker='x', color='red', s=200, linewidths=3, 
               zorder=10, label='Crash Points (NaN)', alpha=0.8)
    crash_markers_added = True

ax.set_xlabel('Time Step t (Round)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Utility of Trustworthy Agents', fontsize=14, fontweight='bold')
ax.set_title('Utility Trajectories Over Time (Version 1 – Proportional Allocation)\n'
             'Cumulative Utility of Trustworthy Agents at Each Iteration t', 
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, zorder=0)
ax.legend(fontsize=12, loc='best', framealpha=0.9)
ax.set_xlim([0.5, T + 0.5])
ax.set_xticks(time_steps)

# Add annotations for key insights
# Find where detection starts helping (trajectory crosses baseline)
# Only check if values are not NaN
for t in range(1, T):
    if (not np.isnan(traj_online_with_detect[t]) and not np.isnan(traj_online_no_detect[t]) and
        not np.isnan(traj_online_with_detect[t-1]) and not np.isnan(traj_online_no_detect[t-1])):
        if traj_online_with_detect[t] > traj_online_no_detect[t] and traj_online_with_detect[t-1] <= traj_online_no_detect[t-1]:
            ax.annotate('Detection\nKicks In', 
                        xy=(t+1, traj_online_with_detect[t]),
                        xytext=(t+1, traj_online_with_detect[t] + 0.05),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
            break

plt.tight_layout()
plt.savefig('utility_trajectories_over_time_version1.png', dpi=150, bbox_inches='tight')
print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print("Plot saved as 'utility_trajectories_over_time_version1.png'")
print()
print("Trajectory Summary:")
# Handle NaN values in summary
final_offline = traj_offline[-1] if not np.isnan(traj_offline[-1]) else np.nan
final_clean = traj_online_clean[-1] if not np.isnan(traj_online_clean[-1]) else np.nan
final_no_detect = traj_online_no_detect[-1] if not np.isnan(traj_online_no_detect[-1]) else np.nan
final_with_detect = traj_online_with_detect[-1] if not np.isnan(traj_online_with_detect[-1]) else np.nan

if not np.isnan(final_offline):
    print("  Offline Ideal (final):        {:.4f}".format(final_offline))
else:
    print("  Offline Ideal (final):        NaN (CRASH)")
if not np.isnan(final_clean):
    print("  Online Clean (final):         {:.4f}".format(final_clean))
else:
    print("  Online Clean (final):         NaN (CRASH)")
if not np.isnan(final_no_detect):
    print("  Online No Detection (final):  {:.4f}".format(final_no_detect))
else:
    print("  Online No Detection (final):  NaN (CRASH)")
if not np.isnan(final_with_detect):
    print("  Online With Detection (final): {:.4f}".format(final_with_detect))
else:
    print("  Online With Detection (final): NaN (CRASH)")
print()
print("Trajectory Values (With Detection - Green Line):")
for t in range(T):
    if not np.isnan(traj_online_with_detect[t]):
        print("  t={}: {:.4f}".format(t+1, traj_online_with_detect[t]))
    else:
        print("  t={}: NaN (CRASH)".format(t+1))
print()
print("Detection Improvement:")
if not np.isnan(final_no_detect) and not np.isnan(final_with_detect) and final_no_detect > 0:
    improvement = 100 * (final_with_detect / final_no_detect - 1)
    print("  With Detection vs No Detection: {:.1f}% improvement".format(improvement))
elif np.isnan(final_no_detect) or np.isnan(final_with_detect):
    print("  Warning: Cannot compute improvement - crash detected (NaN values)")
else:
    print("  Warning: No Detection trajectory is zero!")
print("=" * 70)
plt.close()
