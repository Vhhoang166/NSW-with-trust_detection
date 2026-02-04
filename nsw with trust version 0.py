import numpy as np
from nsw.optimize import minimize
import math
import warnings
import cvxpy as np

warnings.filterwarnings("ignore")

# set up environment: agents, rounds, values, predictions
np.random.seed(42)  # why 42 not 0?

N_trust = 10
N_mal = 4
T = 10
c_mal = 2.0  # malicious distortion range multiplier (reports uniformly in [1/c_mal, c_mal])
n_trials = 20
xi0 = 0.1  # initial threshold for detection
gamma = 0.7  # exponent in threshold schedule xi_t = xi0 * (t+1)^gamma
alpha_pairwise = 5.0  # scaling factor for converting mean absolute diff


# ----------------------------------------------------------------------
# CREATE HELPERS
# ----------------------------------------------------------------------
def normalize_per_round(v):
    """Normalize columns so that sum_i v[i,t] = 1 each t"""
    col_sums = v.sum(axis=0)
    # avoid division by zero
    col_sums[col_sums == 0] = 1
    return v / col_sums


#
def compute_nsw_from_util(u):
    """Geometric mean (NSW) given utility vector u; if some u<=0 return tiny value."""
    if np.any(u <= 0):
        # numeric stability: if zero or negative utilities appear, return a very small NSW
        return 1e-12
    return np.exp(np.mean(np.log(u)))


# -----------------------------------------------------------------------
'''#true value of v_i,t
v_true = np.random.rand(N,T) #create a matrix of true v_i,t values

#each round resource = 1. normalize so the total value per round sums up to 1
v_true = v_true/ v_true.sum(axis=0) #axis = 0, sum per round

#compute total utility per agent (sum across all rounds)
V_true = v_true.sum(axis=1) #horizontal sum?

#prediction error
c = 1.5 #over and underestimation factor
random_factor = np.random.uniform(1/c,c, size = N)
V_pred = V_true * random_factor
'''


# SETTING 1: OFFLINE SETTING (knowing the future)

def offline_optimal_nsw(v_true):
    N, T = v_true.shape
    # objective function: negative sum of logs of utilities (since we minimize?)
    def objective(x_flat):
        X = x_flat.reshape(N,T)
        utilities = (v_true * X).sum(axis = 1) #each agent's total utility
        if np.any(utilities <=0): #avoid log(0)
            return 1e6
        return -np.sum(np.log(utilities)) #maximize product(u_i) = sum(log(u_i))

    #constraints: in each round t, total allocation = 1
    constraints = []
    for t in range(T):
        constraints.append({
            'type':'eq', #equality constraint
            'fun': lambda x_flat, t=t: np.sum(x_flat.reshape(N,T)[:,t])-1
            #lambda returns to 0 only when total allocation in round t sum to exactly 1
            #t=t: freeze the current value of t inside lamba so that every iteration, it uses different t value

        })

# ------------------------------------------------------------------------
# SETTING 2: ALL AGENTS ARE TRUSTWORTHY, MODEL WITHOUT TRUST FILTERING
# initialize state variables
def online_set_aside_with_trust(v_true, reports_func_for_malicious, detect_fn, remove_detected=True):
    N_all = v_true.shape[0]
    # initialize allocations
    x = np.zeros((N_all, T))  # uniform half
    y = np.zeros((N_all, T))  # greedy half
    z = np.zeros((N_all, T))
    u_greedy = np.zeros(N_all)  # cumulative greedy utility from actual received allocations
    # trackers for reported cumulative values (for detection)
    reported_cum = np.zeros((N_all, 0))  # will horizontally stack columns as rounds progress

    detected = np.zeros(N_all, dtype=bool)

    for t in range(T):
        # 1) every agent reports their valuation for the round
        reported_round = np.zeros(N_all)
        for i in range(N_all):
            if detect_fn is None:
                # if no malicious function provided, all report true values
                reported_round[i] = v_true[i, t]
            else:
                reported_round[i] = reports_func_for_malicious(i, t, v_true)

        # append cumulative reports history (cumulative up to current t)
        if reported_cum.size == 0:
            reported_cum = reported_round.reshape((N_all, 1))
        else:
            reported_cum = np.hstack([reported_cum, reported_round.reshape((N_all, 1))])

        # run detection using reported cumulative history
        detected_mask, beta, xi_t = detect_fn(reported_cum, xi0=xi0, gamma=gamma)
        detected = detected_mask.copy()  # detected True => suspect malicious

        # 2) Uniform half allocation among non-detected agents (or among all if not removing)
        eligible = ~detected if remove_detected else np.ones(N_all, dtype=bool)
        n_eligible = max(1, eligible.sum())  # avoid divide by zero
        x[eligible, t] = 0.5 / n_eligible
        # others get zero uniform share if removed
        x[~eligible, t] = 0.0

        # 3) predicted utilities for greedy decision: use predicted Vi = reported_total * (we use reported cumulative as proxy)
        # For online greedy priority we need predicted utility per agent: u_greedy + predicted guarantee from set-aside
        # We'll use the planner's predicted total: use the cumulative reported total scaled to full horizon
        reported_total_est = reported_cum.sum(axis=1) if reported_cum.ndim > 1 else reported_cum
        # estimate of V_i (predicted full total) by scaling mean observed so far to T rounds:
        predicted_Vi = (reported_total_est / (t + 1)) * T  # if t==0 it's just current report * T
        predicted_util = u_greedy + predicted_Vi / (2 * N_all)

        # 4) set priorities but only for eligible agents (others zero)
        priority = np.zeros(N_all)
        # avoid divide by zero
        denom = predicted_util.copy()
        denom[denom <= 0] = 1e-9
        priority[eligible] = (reported_round[eligible] * (1.0 - detected[eligible].astype(float)) + 0.0) / denom[
            eligible]
        # incorporate trust weighting: use incoming trust average (rows of beta averaged)
        if 'beta' in locals():
            incoming_trust = beta.mean(axis=0)
            priority = priority * incoming_trust  # weight by average incoming trust (0..1)
        # normalize weights only across eligible
        if priority[eligible].sum() <= 0:
            weights = np.zeros_like(priority)
            weights[eligible] = 1.0 / n_eligible
        else:
            weights = np.zeros_like(priority)
            weights[eligible] = priority[eligible] / priority[eligible].sum()

        # 5) allocate greedy half
        y[:, t] = 0.5 * weights

        # 6) update u_greedy with actual true received utility (use true v_true, not reported)
        u_greedy += v_true[:, t] * y[:, t]

        # update z
        z[:, t] = x[:, t] + y[:, t]

    # compute total utilities
    u_total = (v_true * z).sum(axis=1)
    nsw_online = compute_nsw_from_util(u_total)
    return u_total, nsw_online, detected


# --------------------------------------------------------------------
# SETTING 3: MIXED AGENTS - DETECTION ALGORITHM
def detect_malicious_from_reports(reports_history, xi0=0.1, gamma=0.7):
    # reports_history: shape (N_all, t_current) cumulative reported values per agent up to round t
    N_all = reports_history.shape[0]
    t_current = reports_history.shape[1] if reports_history.ndim > 1 else 1
    # compute pairwise mean absolute differences across history
    # dissim_ij = mean_abs_diff((report_i, report_j); then beta_ij = exp(-alpha * dissim_ij)
    diffs = np.zeros((N_all, N_all))
    for i in range(N_all):
        for j in range(N_all):
            diffs[i, j] = np.mean(np.abs(reports_history[i, :] - reports_history[j, :]))
    beta = np.exp(-alpha_pairwise * diffs)  # similarity in (0,1], higher = more similar

    # threshold schedule (use last column index as time index)
    t = t_current - 1
    xi_t = xi0 * ((t + 1) ** gamma)

    # picking out the agent j with the highest beta - compute trusted neighbor set
    kept_counts = np.zeros(N_all, dtype=int)
    for i in range(N_all):
        row = beta[i, :]
        jstar = np.argmax(row)
        cutoff = row[jstar] - xi_t
        # neighbors kept (include jstar if meets cutoff)
        kept = (row >= cutoff)
        # count which agents are kept by i
        kept_counts[i] += kept.astype(int)

    # aggregator: if fewer than half of agents keep agent k, mark agent k malicious
    detected_malicious = (kept_counts < (N_all // 2))
    return detected_malicious, beta, xi_t


# -----------------------------
# MALICIOUS REPORT FUNCTION
# -----------------------------
def malicious_report_func_factory(v_true, c_mal, mal_indices):
    """
    returns a function reports_func(i, t, v_true) that returns the reported value for agent i at round t.
    If i is malicious, it returns v_true[i,t] * factor where factor ~ Uniform(1/c_mal, c_mal).
    If trustworthy, returns true value.
    """
    rng = np.random.RandomState()  # local RNG to allow reproducibility per trial if needed

    def reports_func(i, t, v_true_inner):
        if i in mal_indices:
            factor = rng.uniform(1.0 / c_mal, c_mal)
            return v_true_inner[i, t] * factor
        else:
            return v_true_inner[i, t]

    return reports_func


# -----------------------------
# EXPERIMENT LOOP
# -----------------------------
def run_experiment_once(N_trust, N_mal, T, c_mal, detection_fn):
    """
    Runs one trial:
      - generate v_true for total N_all = N_trust + N_mal
      - normalize per round
      - compute offline ideal NSW for only trustworthy agents (full knowledge)
      - run online set-aside greedy in two scenarios:
            A) online_only_trustworthy: only trustworthy agents present (online)
            B) online_with_malicious: all agents present; malicious may be detected & removed
      - return the three NSWs and detection vector
    """
    N_all = N_trust + N_mal
    # generate v_true
    v_true = np.random.rand(N_all, T)
    v_true = normalize_per_round(v_true)

    # indexes
    trust_idx = np.arange(N_trust)
    mal_idx = list(range(N_trust, N_all))

    # OFFLINE OPTIMAL NSW on trustworthy agents only (ideal benchmark)
    nsw_offline, u_offline_full, Xopt = offline_optimal_nsw(v_true,
                                                            agent_mask=np.array([True] * N_trust + [False] * N_mal))

    # ONLINE CASE A: only trustworthy agents online (simulate by removing malicious agents entirely)
    def reports_none(i, t, v_true_inner):
        # agents present are only trustworthy (we will pass v_true restricted)
        return v_true_inner[i, t]

    # Prepare v_true_trustonly (first N_trust rows)
    v_trustonly = v_true[:N_trust, :]

    u_online_trustonly, nsw_online_trustonly, detected_dummy = online_set_aside_with_trust(
        v_trustonly, None, detect_fn=detect_malicious_from_reports, remove_detected=False)

    # ONLINE CASE B: trustworthy + malicious (malicious may be detected and removed)
    reports_func = malicious_report_func_factory(v_true, c_mal, mal_idx)
    u_online_all, nsw_online_all, detected_mask = online_set_aside_with_trust(
        v_true, reports_func, detect_fn=detect_malicious_from_reports, remove_detected=True)

    return nsw_offline, nsw_online_trustonly, nsw_online_all, detected_mask, v_true


# -----------------------------
# RUN MULTIPLE TRIALS
# -----------------------------
results = []
for trial in range(n_trials):
    nsw_offline, nsw_online_trustonly, nsw_online_all, detected_mask, v_true = run_experiment_once(
        N_trust, N_mal, T, c_mal, detect_malicious_from_reports)
    results.append((nsw_offline, nsw_online_trustonly, nsw_online_all, detected_mask))

# summarize
res_arr = np.array([(r[0], r[1], r[2]) for r in results])
mean_offline = res_arr[:, 0].mean()
mean_online_trustonly = res_arr[:, 1].mean()
mean_online_all = res_arr[:, 2].mean()

print("=== Summary over {} trials ===".format(n_trials))
print("Offline ideal NSW (trustworthy-only)  mean: {:.6f}".format(mean_offline))
print("Online NSW (trustworthy-only)         mean: {:.6f}".format(mean_online_trustonly))
print("Online NSW (with malicious + detection) mean: {:.6f}".format(mean_online_all))
print()
print("Competitive ratios (offline / online):")
print("  vs online (trustworthy-only): {:.3f}".format(mean_offline / (mean_online_trustonly + 1e-12)))
print("  vs online (with malicious):      {:.3f}".format(mean_offline / (mean_online_all + 1e-12)))