#!/usr/bin/env python3
"""
NSW Trustworthy Consensus – Professor Revision
===============================================

Changes from previous version (all 9 professor requests):

[R1] Bar plots converted to line plots for all sweep/comparison figures.
     Bar charts remain only where a discrete categorical axis is genuinely
     appropriate (e.g. the detector F1 grouped bar).

[R2] Attack types reduced to 4 focused, academically meaningful types:
       BYZANTINE       — persistent constant-high (classical baseline)
       BURST           — strategic burst (renamed from STRATEGIC_BURST)
       REPUTATION_POISON — warm-up then hidden evasion (redesigned, see R3)
       TRUST_MIMICRY   — adaptive threshold-aware evasion

[R3] Reputation Poison redesign:
     Phase 1 (t < poison_warmup_rounds): ALL agents appear legitimate.
       Malicious agents behave honestly (p_attack ≈ 0) to build maximum
       β reputation among all observers.
     Phase 2 (t >= poison_warmup_rounds): malicious agents switch to
       Trust Mimicry evasion — they monitor their own β gap globally
       (full information, see R5) and attack only when safely below ξ_t.
       This makes them functionally hidden after the warmup.

[R4] Compound / heterogeneous attack model:
     Each malicious agent is assigned its own sub-type at init by sampling
     uniformly from {BYZANTINE, BURST, TRUST_MIMICRY}.
     When a global synchronisation signal fires (every coalition_sync_every
     rounds), ALL malicious agents attack simultaneously in the same round,
     each using its own sub-type's attack rate and distortion range, and the
     coalition also amplifies malicious reports while suppressing one rotating
     legitimate victim's report.
     This creates compound rounds where all attack but with heterogeneous
     intensities, which is harder for majority-vote detection to catch cleanly.
     Used by the COMPOUND attack type.

[R5] Full information / omniscient agents:
     connectivity is always 1.0 (all agents observe all others — no mask).
     Malicious agents can query the GLOBAL β gap for their id across ALL
     legitimate detectors simultaneously (previously averaged, now exposed
     exactly). This makes Trust Mimicry significantly more powerful because
     the attacker knows its exact detection margin.

[R6] Fewer cases, deeper analysis:
     n_trials raised to 25 for the four focused attack types.
     compare_focused_types() replaces compare_all_types().
     Plots now show 95% confidence intervals (mean ± 1.96σ/√n) instead
     of raw std bands, and include per-attack-type β trajectory plots.

[R7] Higher base attack rates to produce visible misclassification:
     base_attack_prob raised to 0.75 for BYZANTINE and BURST.
     REPUTATION_POISON uses 0.80 during Phase 2 mimicry.
     This ensures the misclassification plot shows meaningful dynamics
     rather than near-zero rates throughout.

[R8] Three-part project structure clearly delineated in code:
     PART A — Alpha computation  (trusted_online_allocation, alpha sweep)
     PART B — Detection mechanism  (TrustDetector, Algorithm 1, M1–M4)
     PART C — Algorithm design  (NSWTrustworthySim, run loop, NSW objective)

[R9] Expanded inline documentation throughout.

[R10] Restored Banerjee prediction-quality stress testing as Plot 15 and fixed
      the asymmetric-outlier scenario so it no longer hard-codes the last agent.
"""

import argparse
import os
import tempfile
from pathlib import Path

# Keep Matplotlib's cache in a writable, platform-independent temporary
# directory unless the caller has already selected a location.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "nsw_matplotlib_cache"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, Set, Tuple, List, Optional
import math
from dataclasses import dataclass, replace
from enum import Enum
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.cluster import SpectralClustering
except ImportError:
    SpectralClustering = None


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

try:
    _MODULE_DIR = Path(__file__).resolve().parent
except NameError:
    _MODULE_DIR = Path.cwd()
OUTPUT_DIR = _MODULE_DIR / "outputs"


# =============================================================================
# UTILITY
# =============================================================================

def binary_f1_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Local F1 score — no sklearn dependency required."""
    y_true = np.asarray(true_labels, dtype=int)
    y_pred = np.asarray(pred_labels, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom > 0 else 0.0


def ci95(data: np.ndarray) -> np.ndarray:
    """95% confidence interval half-width: 1.96 * std / sqrt(n)."""
    n = data.shape[0]
    return 1.96 * data.std(axis=0) / math.sqrt(max(n, 1))


def rolling_mean_edge(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean with edge padding to avoid zero-padding artefacts."""
    x = np.asarray(values, dtype=float)
    if window <= 1 or x.size == 0:
        return x.copy()
    window = min(int(window), x.size)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(x, (left, right), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')


# =============================================================================
# [R2] ATTACK TYPES — reduced to 4 focused types + COMPOUND
# =============================================================================

class AttackType(Enum):
    BYZANTINE        = "byzantine"         # persistent constant-high (classical)
    BURST            = "burst"             # strategic burst (renamed)
    REPUTATION_POISON = "reputation_poison" # warm-up then Trust Mimicry evasion
    TRUST_MIMICRY    = "trust_mimicry"     # adaptive threshold-aware evasion
    COMPOUND         = "compound"          # heterogeneous sub-types per agent [R4]


# Sub-type pool for COMPOUND — each malicious agent draws from this
_COMPOUND_SUBTYPES = [
    AttackType.BYZANTINE,
    AttackType.BURST,
    AttackType.TRUST_MIMICRY,
]

# Distortion ranges (lo, hi) per attack type when attacking
_BASE_DISTORTION: Dict[AttackType, Tuple[float, float]] = {
    AttackType.BYZANTINE:         (0.05, 9.0),   # large inflation
    AttackType.BURST:             (0.02, 10.0),  # extreme burst inflation
    AttackType.REPUTATION_POISON: (0.40, 2.50),  # mimicry-style after warmup
    AttackType.TRUST_MIMICRY:     (0.40, 2.50),  # mild — stealth is the goal
    AttackType.COMPOUND:          (0.05, 9.0),   # overridden per sub-type
}


# =============================================================================
# PREDICTION SCENARIOS  (from NSW_Only, unchanged)
# =============================================================================

PREDICTION_SCENARIOS = (
    "mild_noise",
    "ideal",
    "severe_over",
    "severe_under",
    "asymmetric_outlier",
    "adversarial",
    "random_permutation",
    "half_over_half_under",
)

PREDICTION_NOISE_RANGE = 0.30


def generate_V_tilde(v_true: np.ndarray,
                     scenario: str,
                     rng: np.random.RandomState) -> np.ndarray:
    """
    Generate predicted total valuations Ṽ_i for a named prediction scenario.

    Parameters
    ----------
    v_true   : (N, T) true valuation matrix
    scenario : one of PREDICTION_SCENARIOS
    rng      : seeded random state for reproducibility

    Returns
    -------
    V_tilde : (N,) array of predicted total valuations, all > 0
    """
    V = v_true.sum(axis=1).astype(float)
    N = len(V)
    V_tilde = np.zeros(N)

    if scenario == "ideal":
        V_tilde = V.copy()
    elif scenario == "mild_noise":
        V_tilde = V * rng.uniform(0.7, 1.3, N)
    elif scenario == "severe_over":
        V_tilde = V * 10.0
    elif scenario == "severe_under":
        V_tilde = V * 0.1
    elif scenario == "asymmetric_outlier":
        V_tilde = V.copy()
        outlier_idx = int(rng.randint(N))
        V_tilde[outlier_idx] = V[outlier_idx] * 1000.0
    elif scenario == "adversarial":
        order_V = np.argsort(V)
        V_tilde[order_V] = np.sort(V)[::-1]
    elif scenario == "random_permutation":
        V_tilde = V[rng.permutation(N)]
    elif scenario == "half_over_half_under":
        V_tilde = V.copy()
        half = N // 2
        V_tilde[:half] *= 2.0
        V_tilde[half:] *= 0.5
    else:
        V_tilde = V * (1 + rng.uniform(-PREDICTION_NOISE_RANGE,
                                        PREDICTION_NOISE_RANGE, N))

    return np.maximum(V_tilde, 1e-10)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    All simulation parameters in one dataclass.
    Use dataclasses.replace(base, field=value) for sweep variants —
    this preserves every other field automatically.
    """
    n_legitimate:   int   = 10
    n_malicious:    int   = 4
    n_rounds:       int   = 200
    n_trials:       int   = 10

    attack_type:      AttackType = AttackType.BYZANTINE
    # [R7] Higher attack rates to produce visible misclassification dynamics
    base_attack_prob: float = 0.75  # raised from 0.55

    # --- Trust observation parameters ---
    # E_L = trust_legitimate: expected trust observation from a legitimate neighbour
    # E_M = trust_malicious_attack: expected trust observation during an active attack
    # Gap = E_L - E_M = 0.11; β gap grows as (E_L-E_M)*p_attack*t (linear in t)
    # xi_t = xi_0*sqrt((1+eps)(t+1)*ln(t+2)) grows sublinearly → eventual separation
    trust_legitimate:          float = 0.63
    trust_malicious_attack:    float = 0.52
    trust_malicious_no_attack: float = 0.63
    trust_std:                 float = 0.14

    # Algorithm 1 detection threshold parameters
    xi_0:    float = 0.25
    epsilon: float = 0.10

    # --- [PART A] Allocation alpha ---
    # alpha = fraction of budget set aside equally among trusted agents
    # 0.0 → pure greedy water-filling (maximises NSW, lowest fairness floor)
    # 0.5 → half/half (Banerjee et al. recommendation)
    # 1.0 → pure equal split (maximum fairness, lower NSW)
    allocation_alpha: float = 0.50

    # Synchronisation for COMPOUND attack [R4]
    coalition_sync_every: int   = 10   # every K rounds all agents attack together
    coalition_amplify:    float = 9.0

    # Reputation Poison parameters [R3]
    poison_warmup_rounds: int   = 50   # Phase 1 length (all appear legitimate)
    mimicry_margin:       float = 0.02 # safety margin below xi_t for mimicry

    # [R5] Full information — connectivity is always 1.0
    connectivity: float = 1.0

    # Valuation model
    valuation_model: str = "lognormal"  # "lognormal" | "pareto" | "dirichlet"

    # Moradi graph parameters [M1–M4]
    moradi_sigma:            float = 0.60
    sparsest_epsilon:        float = 0.30
    moradi_merge_threshold:  int   = 2
    moradi_bootstrap_rounds: int   = 15
    moradi_decision_interval: int  = 20
    # Algorithm 1 remains the allocation gate. Moradi runs every 20 rounds as
    # a shadow detector until its quality has been evaluated independently.
    online_detector: str = "algorithm1"  # "moradi" | "algorithm1" | "hybrid"

    # NSW_Only prediction scenario
    prediction_scenario: str = "mild_noise"

    # Label for plots and reports
    scenario_label: str = "baseline"


# =============================================================================
# TRUST OBSERVATION GENERATOR
# =============================================================================

class TrustObservationGenerator:
    """
    Generates stochastic side-channel trust observations α_ij(t) ∈ [0,1].

    α_ij(t) ~ Clip(Normal(μ, σ), 0, 1) where:
      μ = E_L  if agent j is legitimate
      μ = E_M  if agent j is malicious and attacking this round
      μ = E_L  if agent j is malicious but NOT attacking (blends in)

    These observations are the raw material for β accumulation in Algorithm 1.
    The key property: E[α_legit] = E_L, E[α_attacking] = E_M < E_L.
    """

    def __init__(self, config: ExperimentConfig, seed: int = None):
        self.cfg = config
        self.rng = np.random.RandomState(seed)

    def observe(self, observed_legit: bool, observed_attacking: bool,
                t: int = 0) -> float:
        if observed_legit:
            mu = self.cfg.trust_legitimate
        elif observed_attacking:
            mu = self.cfg.trust_malicious_attack
        else:
            # Malicious but not attacking this round — indistinguishable
            mu = self.cfg.trust_malicious_no_attack
        return float(np.clip(self.rng.normal(mu, self.cfg.trust_std), 0.0, 1.0))


# =============================================================================
# [R4] MALICIOUS AGENT — compound sub-type support
# =============================================================================

class MaliciousAgent:
    """
    Malicious agent implementing the four focused attack strategies.

    For COMPOUND attacks [R4], the agent is assigned a sub_type at init
    drawn from _COMPOUND_SUBTYPES. The sub-type controls its individual
    attack probability and distortion range, while synchronisation is
    handled externally by the CoalitionCoordinator.

    For REPUTATION_POISON [R3], the agent moves through two phases:
      Phase 1 (t < warmup): nearly always honest to build reputation
      Phase 2 (t >= warmup): switches to full Trust Mimicry evasion,
        using global β gap information [R5] to stay below detection.
    """

    def __init__(self, agent_id: int, attack_type: AttackType,
                 base_prob: float, config: ExperimentConfig,
                 seed: int = None,
                 sub_type: Optional[AttackType] = None):
        self.id          = agent_id
        self.attack_type = attack_type
        self.base_prob   = base_prob
        self.cfg         = config
        self.rng         = np.random.RandomState(seed)
        # Sub-type for COMPOUND — assigned at init [R4]
        self.sub_type    = sub_type  # None unless COMPOUND
        self.attack_history:     List[bool]  = []
        self.distortion_history: List[float] = []

    def decide_attack(self,
                      t: int,
                      is_detected: bool,
                      sync_signal: bool = False,
                      current_xi_t: float = 0.0,
                      global_beta_gap: float = 0.0,
                      ) -> Tuple[bool, float]:
        """
        Returns (attacking: bool, phi: float).

        Parameters
        ----------
        t               : current round
        is_detected     : whether this agent is currently flagged by Algorithm 1
        sync_signal     : True on coalition synchronisation rounds [R4]
        current_xi_t    : current detection threshold value (for mimicry)
        global_beta_gap : max_beta - beta[self.id] averaged over ALL legitimate
                          detectors simultaneously [R5 full information]
        """
        # ── BYZANTINE: constant high attack rate ──────────────────────────
        if self.attack_type == AttackType.BYZANTINE:
            prob   = self.base_prob          # 0.75 → clearly detectable
            lo, hi = _BASE_DISTORTION[AttackType.BYZANTINE]

        # ── BURST: alternating intense bursts and quiet periods ───────────
        elif self.attack_type == AttackType.BURST:
            # 3-round burst every 10 rounds: extremely aggressive when active,
            # nearly silent otherwise to partially recover β in quiet periods
            in_burst = (t % 10) < 3
            prob     = 0.95 if in_burst else 0.08
            lo, hi   = (0.02, 10.0) if in_burst else (0.88, 1.12)

        # ── REPUTATION POISON: warm-up then hidden mimicry [R3] ──────────
        elif self.attack_type == AttackType.REPUTATION_POISON:
            if t < self.cfg.poison_warmup_rounds:
                # Phase 1: be completely honest to build a high β reputation.
                # Attack probability near zero; distortion near 1 (no distortion).
                prob   = 0.01
                lo, hi = 0.98, 1.02
            else:
                # Phase 2: switch to Trust Mimicry evasion [R3].
                # Uses the same threshold-aware logic as TRUST_MIMICRY,
                # but with a higher attack rate because the warm-up β credit
                # means the gap to xi_t is large enough to absorb more attacks.
                safe = global_beta_gap < (current_xi_t - self.cfg.mimicry_margin)
                prob   = 0.80 if safe else 0.03   # [R7] higher rate Phase 2
                lo, hi = _BASE_DISTORTION[AttackType.REPUTATION_POISON]

        # ── TRUST MIMICRY: continuously monitors detection margin ─────────
        elif self.attack_type == AttackType.TRUST_MIMICRY:
            # Attacks only when the global β gap (across ALL observers [R5])
            # is safely below the current detection threshold minus a margin.
            # This prevents any single observer from crossing the threshold.
            safe = global_beta_gap < (current_xi_t - self.cfg.mimicry_margin)
            prob   = 0.75 if safe else 0.03   # [R7] higher active rate
            lo, hi = _BASE_DISTORTION[AttackType.TRUST_MIMICRY]

        # ── COMPOUND: use per-agent sub-type, synchronise on signal [R4] ─
        elif self.attack_type == AttackType.COMPOUND:
            effective = self.sub_type or AttackType.BYZANTINE

            if sync_signal:
                # On coalition rounds ALL agents attack at maximum rate
                # regardless of sub-type, creating a coordinated spike
                prob   = 0.97
                lo, hi = _BASE_DISTORTION[effective]
            elif effective == AttackType.BYZANTINE:
                prob   = self.base_prob
                lo, hi = _BASE_DISTORTION[AttackType.BYZANTINE]
            elif effective == AttackType.BURST:
                in_burst = (t % 10) < 3
                prob     = 0.95 if in_burst else 0.08
                lo, hi   = (0.02, 10.0) if in_burst else (0.88, 1.12)
            elif effective == AttackType.TRUST_MIMICRY:
                safe   = global_beta_gap < (current_xi_t - self.cfg.mimicry_margin)
                prob   = 0.75 if safe else 0.03
                lo, hi = _BASE_DISTORTION[AttackType.TRUST_MIMICRY]
            else:
                prob   = self.base_prob
                lo, hi = (0.05, 9.0)

        else:
            prob   = self.base_prob
            lo, hi = (0.05, 9.0)

        attacking = bool(self.rng.rand() < prob)
        phi       = float(self.rng.uniform(lo, hi)) if attacking else 1.0
        self.attack_history.append(attacking)
        self.distortion_history.append(phi)
        return attacking, phi

    def empirical_attack_rate(self) -> float:
        if not self.attack_history:
            return 0.0
        return float(np.mean(self.attack_history))


# =============================================================================
# [R4] COALITION COORDINATOR — synchronised attack signal
# =============================================================================

class CoalitionCoordinator:
    """
    Issues a synchronisation signal every K rounds.

    When the signal fires, all malicious agents attack simultaneously,
    each using their own sub-type's attack rate and distortion range [R4].
    For COMPOUND attacks it also applies a targeted coalition move: malicious
    reports are amplified together while one rotating legitimate victim is
    suppressed. This preserves the coordinated victim attack from the earlier
    COORDINATED model without bringing back the whole old attack taxonomy.
    """

    def __init__(self, malicious_ids: List[int], n_legitimate: int,
                 sync_every: int, seed: int = 0):
        self.mal_ids = malicious_ids
        self.n_legit = n_legitimate
        self.K       = sync_every
        self.rng     = np.random.RandomState(seed + 77777)
        self._target = 0

    def get_signal(self, t: int) -> bool:
        """Returns True on synchronisation rounds (t divisible by K)."""
        return (t % self.K) == 0

    def apply(self, reports: np.ndarray, true_vals_round: np.ndarray,
              amplify: float) -> Tuple[np.ndarray, int]:
        """
        Apply a synchronized coalition report attack.

        All malicious agents inflate their reports together, while a rotating
        legitimate victim's report is suppressed near zero. Returns the updated
        report vector and the victim id for diagnostics.
        """
        victim = self._target % self.n_legit
        self._target += 1

        attacked_reports = reports.copy()
        for m in self.mal_ids:
            attacked_reports[m] = (
                true_vals_round[m] * amplify * self.rng.uniform(0.8, 1.2))
        attacked_reports[victim] = true_vals_round[victim] * self.rng.uniform(0.01, 0.08)
        return attacked_reports, victim


# =============================================================================
# ════════════════════════════════════════════════════════════════════════════
# PART B — DETECTION MECHANISM  (your contribution)
# ════════════════════════════════════════════════════════════════════════════
# =============================================================================

class TrustDetector:
    """
    PART B — Trusted Neighbourhood Learning (Algorithm 1).

    Implements the detection mechanism from Akgun et al. (arXiv:2504.07189).

    State
    -----
    beta[j] = β_ij(t) = Σ_{k=0}^{t} α_ij(k)
      Cumulative sum of trust observations from agent j as seen by this agent i.
      Legitimate neighbours accumulate higher β than attacking malicious ones
      because E[α_legit] = E_L > E_M = E[α_attacking].

    Detection rule (Algorithm 1 line 3–4)
    --------------------------------------
    Let j̄ = argmax_j β_ij(t)  (most trusted neighbour).
    Agent j is classified as trusted iff:
        β_ij̄(t) − β_ij(t) ≤ ξ_t

    Threshold
    ---------
    ξ_t = ξ_0 · √((1+ε)(t+1) · ln(t+2))

    This grows sublinearly (as √(t log t)), which is slower than the linear
    accumulation of β gaps. This guarantees:
      (a) False positive probability decays near-exponentially (Lemma 4)
      (b) False negative probability also decays near-exponentially (Lemma 5)
      (c) Almost sure correct classification after finite random time (Lemma 7)

    [R5] Full information: beta_gap() returns the gap using the maximum β
    across all j, giving malicious agents perfect knowledge of their margin.
    """

    def __init__(self, agent_id: int, all_ids: List[int],
                 xi_0: float = 0.25, epsilon: float = 0.10):
        self.id   = agent_id
        self.xi_0 = xi_0
        self.eps  = epsilon
        # Initialise all β values to zero; never includes self
        self.beta: Dict[int, float] = {j: 0.0 for j in all_ids if j != agent_id}

    def threshold(self, t: int) -> float:
        """
        Compute ξ_t = ξ_0 · √((1+ε)(t+1) · ln(t+2)).

        The √(t log t) growth ensures the threshold stays competitive with
        the √t noise fluctuations of β while falling behind the linear
        signal accumulation of persistent attackers.
        """
        if t <= 0:
            return self.xi_0 * math.sqrt((1 + self.eps) * math.log(2))
        return self.xi_0 * math.sqrt((1 + self.eps) * (t + 1) * math.log(t + 2))

    def update(self, j: int, alpha: float) -> None:
        """Add one trust observation to β_ij(t)."""
        if j in self.beta:
            self.beta[j] += alpha

    def classify(self, t: int) -> Tuple[Set[int], Set[int]]:
        """
        Apply Algorithm 1 to classify all neighbours as trusted or detected.

        Returns
        -------
        trusted  : set of agent IDs classified as legitimate this round
        detected : set of agent IDs classified as malicious this round
        """
        if not self.beta:
            return {self.id}, set()
        j_bar    = max(self.beta, key=self.beta.get)
        max_beta = self.beta[j_bar]
        xi_t     = self.threshold(t)
        trusted, detected = set(), set()
        for j, b in self.beta.items():
            if max_beta - b <= xi_t:
                trusted.add(j)
            else:
                detected.add(j)
        trusted.add(self.id)   # always trust yourself
        return trusted, detected

    def beta_gap(self, j: int) -> float:
        """
        Return max_β − β[j] for agent j as observed by this detector.
        Used by Trust Mimicry and Reputation Poison agents [R5].
        A gap of zero means j is indistinguishable from the most trusted agent.
        A large gap means j is clearly below the trusted cluster.
        """
        if not self.beta:
            return 0.0
        max_beta = max(self.beta.values())
        return max_beta - self.beta.get(j, 0.0)


# =============================================================================
# [M1] BLENDED EDGE WEIGHT MATRIX  (Moradi Eq.1, PART B)
# =============================================================================

def build_blended_matrix(beta_matrix: np.ndarray,
                          hist_reports: List[np.ndarray],
                          n_agents: int,
                          sigma: float) -> np.ndarray:
    """
    PART B — Construct the blended Moradi affinity matrix W.

    W_ij = σ · T_ij + (1−σ) · S_ij

    where:
      T_ij = normalised trust: beta_ij / max_beta (row-wise), then symmetrised
      S_ij = Pearson correlation similarity of agents' reported valuation
             histories, mapped from [−1,1] → [0,1]

    The two channels are complementary:
      T_ij catches attacks that generate low trust observations (BYZANTINE, BURST)
      S_ij catches attacks that distort reports in anomalous patterns (TRUST_MIMICRY,
           REPUTATION_POISON Phase 2) even when their trust signal is suppressed.
    """
    N = n_agents
    T = beta_matrix.copy()
    row_max = T.max(axis=1, keepdims=True)
    row_max = np.where(row_max < 1e-12, 1.0, row_max)
    T = T / row_max
    T = (T + T.T) / 2.0
    np.fill_diagonal(T, 0.0)

    S = np.zeros((N, N))
    report_count = hist_reports.shape[1] if isinstance(hist_reports, np.ndarray) else len(hist_reports)
    if report_count >= 2:
        V = hist_reports if isinstance(hist_reports, np.ndarray) else np.stack(hist_reports, axis=1)
        V_centred = V - V.mean(axis=1, keepdims=True)
        norms     = np.linalg.norm(V_centred, axis=1, keepdims=True)
        norms     = np.where(norms < 1e-12, 1.0, norms)
        V_norm    = V_centred / norms
        corr      = V_norm @ V_norm.T
        S         = (corr + 1.0) / 2.0
        np.fill_diagonal(S, 0.0)

    W = sigma * T + (1.0 - sigma) * S
    W = np.clip(W, 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    return W


# =============================================================================
# [M2] SPARSEST SUBGRAPH DETECTION  (Moradi Algorithm 1, PART B)
# =============================================================================

def sparsest_subgraph_detection(W: np.ndarray,
                                  n_agents: int,
                                  epsilon: float = 0.30) -> np.ndarray:
    """
    PART B — Count-free sparse-group detector (Moradi 2015, Sec. 3.2).

    Malicious agents are expected to have lower weighted affinity to the graph.
    Their number is deliberately NOT supplied.  Instead, the detector sorts
    normalised weighted degree and uses the strongest low-degree separation as
    a data-driven boundary.  It assumes only that attackers are a strict
    minority, which is the threat-model assumption rather than knowledge of
    their exact count.

    ``epsilon`` is the minimum selected gap as a fraction of the complete
    weighted-degree range.  If the graph has no sufficiently clear separation,
    the detector returns no malicious labels instead of forcing a known-sized
    answer.
    """
    A = np.asarray(W, dtype=float)
    if A.shape != (n_agents, n_agents):
        raise ValueError(
            f"W must have shape ({n_agents}, {n_agents}), got {A.shape}"
        )

    N      = n_agents
    labels = np.zeros(N, dtype=int)
    if N < 3 or A.sum() < 1e-10:
        return labels

    A = np.clip((A + A.T) / 2.0, 0.0, None)
    np.fill_diagonal(A, 0.0)
    degree = A.sum(axis=1) / max(N - 1, 1)
    order = np.argsort(degree, kind="stable")
    sorted_degree = degree[order]

    # Search only strict-minority group sizes; no exact attacker count is used.
    max_sparse_size = max(1, (N - 1) // 2)
    gaps = np.diff(sorted_degree)[:max_sparse_size]
    if gaps.size == 0:
        return labels

    split = int(np.argmax(gaps)) + 1
    selected_gap = float(gaps[split - 1])
    degree_range = float(sorted_degree[-1] - sorted_degree[0])
    if degree_range <= 1e-12 or selected_gap < epsilon * degree_range:
        return labels

    labels[order[:split]] = 1
    return labels


# =============================================================================
# [M3] CLUSTER MERGING SAFETY NET  (PART B)
# =============================================================================

def merge_small_clusters(trusted_ids: List[int],
                          beta_matrix: np.ndarray,
                          all_agent_ids: List[int],
                          merge_threshold: int) -> List[int]:
    """
    PART B — Moradi merging step.

    If the trusted set falls below merge_threshold agents (which can happen
    due to false positives), expand it by absorbing the nearest available
    agents ranked by symmetric β similarity.

    Intentionally label-agnostic: a malicious agent with a high β score
    (e.g. REPUTATION_POISON Phase 1) may be re-admitted. This is the honest
    tradeoff — the safety net prevents starvation at the cost of occasionally
    re-admitting a sophisticated attacker.
    """
    if len(trusted_ids) >= merge_threshold:
        return trusted_ids

    trusted_set = set(trusted_ids)
    candidates  = [j for j in all_agent_ids if j not in trusted_set]
    if not candidates:
        return trusted_ids

    scores = {
        c: np.mean([beta_matrix[t, c] + beta_matrix[c, t]
                    for t in trusted_ids]) if trusted_ids else 0.0
        for c in candidates
    }
    merged = list(trusted_ids)
    for c in sorted(candidates, key=lambda x: scores[x], reverse=True):
        if len(merged) >= merge_threshold:
            break
        merged.append(c)
    return merged


# =============================================================================
# [M4] IMPLICIT TRUST BOOTSTRAP  (PART B)
# =============================================================================

def implicit_trust_score(agent_i: int,
                          agent_j: int,
                          hist_reports: List[np.ndarray],
                          bootstrap_rounds: int) -> float:
    """
    PART B — Moradi Eq.(5) adapted for valuation report overlap.

    implicit_trust(i,j) = |rounds where v_i > median(v_i) AND v_j > median(v_j)|
                          / |rounds where v_i > median(v_i)|

    This supplements β for agents with insufficient observation history (cold-start).
    Two agents that consistently need resources together are likely in the same
    legitimate cluster. Attackers that distort reports create unusual patterns
    that diverge from the legitimate cluster.

    Returns 0.5 (neutral) when fewer than 3 rounds of history are available.
    """
    T_obs = min(len(hist_reports), bootstrap_rounds)
    if T_obs < 3:
        return 0.5

    vi       = np.array([hist_reports[t][agent_i] for t in range(T_obs)])
    vj       = np.array([hist_reports[t][agent_j] for t in range(T_obs)])
    med_i    = np.median(vi)
    active_i = vi > med_i
    n_active = active_i.sum()
    if n_active == 0:
        return 0.5

    active_both = active_i & (vj > np.median(vj))
    return float(active_both.sum() / n_active)


# =============================================================================
# [E] SPECTRAL CLUSTERING DETECTOR  (PART B, benchmark)
# =============================================================================

def spectral_detection(beta_matrix: np.ndarray,
                        n_agents: int) -> np.ndarray:
    """
    PART B — Spectral clustering on the blended affinity matrix W.

    Treats W as a weighted graph and finds the two-community partition
    that maximises intra-cluster affinity (honest cluster) and minimises
    inter-cluster affinity (honest vs malicious separation).

    The exact number of malicious agents is not supplied.  Because scikit-learn
    cluster labels are arbitrary, the cluster with lower mean weighted degree
    is labelled suspicious.  Limitation: this still assumes exactly two
    clusters and can fail when attack patterns blur the graph boundary.
    """
    A = (beta_matrix + beta_matrix.T) / 2.0
    A = np.clip(A, 0, None)
    np.fill_diagonal(A, 0)
    if A.sum() < 1e-10 or SpectralClustering is None:
        return np.zeros(n_agents, dtype=int)
    try:
        sc     = SpectralClustering(n_clusters=2, affinity='precomputed',
                                    n_init=10, random_state=42)
        labels = sc.fit_predict(A)
    except Exception:
        return np.zeros(n_agents, dtype=int)
    degree = A.sum(axis=1) / max(n_agents - 1, 1)
    cluster_means = [
        float(degree[labels == cluster].mean())
        if np.any(labels == cluster) else float("inf")
        for cluster in (0, 1)
    ]
    suspicious_cluster = int(np.argmin(cluster_means))
    return (labels == suspicious_cluster).astype(int)


# =============================================================================
# ════════════════════════════════════════════════════════════════════════════
# PART C — ALGORITHM DESIGN  (third teammate's contribution)
# ════════════════════════════════════════════════════════════════════════════
# =============================================================================

def compute_nsw(utilities: np.ndarray) -> float:
    """
    PART C — Nash Social Welfare: geometric mean of utilities.

    NSW(u) = (∏_i u_i)^{1/N} = exp(mean(log(u)))

    Returns 0.0 if any agent has zero or negative utility, because the
    geometric mean is zero whenever any factor is zero. This is the correct
    behaviour — NSW penalises starvation of any single agent infinitely.
    """
    if np.any(utilities <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(utilities))))


def offline_optimal_nsw(v_true: np.ndarray,
                         legit_mask: np.ndarray,
                         ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    PART C — Offline NSW optimum over legitimate agents only.

    Solves: max Σ_i log(u_i)  s.t. Σ_i x[i,t] = 1 ∀t, x ≥ 0
    via SLSQP. This is the clairvoyant benchmark — it has full knowledge
    of all future valuations, which the online algorithm does not.

    Returns
    -------
    nsw_opt       : float — optimal NSW value
    utilities_opt : (N_legit,) — per-agent optimal utilities
    X_opt         : (N_legit, T) — optimal allocation matrix
    """
    idx   = np.where(legit_mask)[0]
    v_opt = v_true[idx, :]
    N, T  = v_opt.shape

    def objective(x_flat):
        X = x_flat.reshape(N, T)
        utils = (v_opt * X).sum(axis=1)
        if np.any(utils <= 0):
            return 1e6
        return -np.sum(np.log(utils))

    constraints = [{'type': 'eq',
                    'fun': lambda x_flat, t=t: np.sum(x_flat.reshape(N, T)[:, t]) - 1.0}
                   for t in range(T)]
    x0     = np.ones(N * T) / N
    bounds = [(0.0, 1.0)] * (N * T)
    result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 1000})
    X_opt         = result.x.reshape(N, T) if result.success else np.ones((N, T)) / N
    utilities_opt = (v_opt * X_opt).sum(axis=1)
    return compute_nsw(utilities_opt), utilities_opt, X_opt


# =============================================================================
# ════════════════════════════════════════════════════════════════════════════
# PART A — ALPHA COMPUTATION  (first teammate's contribution)
# ════════════════════════════════════════════════════════════════════════════
# =============================================================================

def compute_allocation_min_price_sec74(predicted_util: np.ndarray,
                                        values_round: np.ndarray,
                                        budget: float) -> np.ndarray:
    """
    PART A — Section 7.4 water-filling / KKT allocation rule.

    Solves the per-round greedy allocation by equalising marginal ratios:
        v_i / u_i(z) for all i with positive value

    The KKT conditions for the greedy NSW maximisation give:
        z_i* = (1/λ - u_i/v_i) for agents with positive value,
    where λ is chosen so Σ z_i = budget.

    Implemented via the O(N log N) peeling algorithm from Banerjee et al.
    (Section 7.4): sort by v_i/u_i descending, then iteratively equalise.
    """
    N = predicted_util.size
    z = np.zeros(N)
    u = np.maximum(predicted_util.astype(float), 1e-10)
    v = np.maximum(values_round.astype(float), 0.0)

    positive = v > 1e-12
    if not np.any(positive):
        z[:] = budget / N
        return z

    order    = np.argsort(-(v / u))
    u_sorted = u[order].copy()
    v_sorted = v[order].copy()
    n_pos    = int(np.sum(positive))
    z_sorted = np.zeros(n_pos)
    B        = float(budget)

    if n_pos == 1:
        z_sorted[0] = B
    else:
        for i in range(n_pos - 1):
            r_target = v_sorted[i + 1] / u_sorted[i + 1]
            if r_target <= 0:
                continue
            delta_i      = max(0.0, (1.0 / r_target) - (u_sorted[i] / v_sorted[i]))
            total_needed = (i + 1) * delta_i
            if total_needed <= 0:
                continue
            if total_needed <= B:
                z_sorted[:i + 1] += delta_i
                u_sorted[:i + 1] += v_sorted[:i + 1] * delta_i
                B -= total_needed
            else:
                z_sorted[:i + 1] += B / (i + 1)
                B = 0.0
                break
        if B > 0:
            z_sorted[:] += B / n_pos

    z[order[:n_pos]] = z_sorted
    total = z.sum()
    if total > 0 and abs(total - budget) > 1e-10:
        z *= budget / total
    return z


def trusted_online_allocation(
        reports: np.ndarray,
        trusted_ids: List[int],
        V_tilde: np.ndarray,
        u_greedy: np.ndarray,
        budget: float = 1.0,
        alpha: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PART A — NSW_Only online set-aside greedy allocation over trusted agents.

    Implements the Banerjee et al. algorithm restricted to the trusted set
    selected by the configured online detector (PART B) each round.

    Allocation split
    ----------------
    Set-aside (equal guarantee):
        x_i = α·B / N_trusted  for each i in trusted_ids
        Guarantees every trusted agent a minimum floor allocation regardless
        of their valuation. This is the fairness component.

    Greedy (water-filling):
        y_i via Section 7.4 KKT rule on budget (1−α)·B
        Allocates remaining budget to maximise predicted NSW.
        Uses predicted utility baseline: ũ_i = u_greedy_i + Ṽ_i·α/N_trusted
        This tracks how much greedy utility the agent has accumulated so far
        plus its predicted future set-aside share.

    Total: z_i = x_i + y_i, normalised to sum exactly to B.

    Parameters
    ----------
    reports     : (N,) reported valuations (distorted for malicious agents)
    trusted_ids : agents passing the trust filter this round
    V_tilde     : (N,) predicted total valuations (fixed at trial start)
    u_greedy    : (N,) cumulative greedy-component utility (maintained by sim)
    budget      : total resource budget per round (always 1.0)
    alpha       : set-aside fraction

    Returns
    -------
    alloc           : (N,) total allocation
    set_aside_alloc : (N,) equal-share component
    greedy_alloc    : (N,) water-filling component
    """
    N_total   = len(reports)
    alloc     = np.zeros(N_total)
    set_aside = np.zeros(N_total)
    greedy    = np.zeros(N_total)

    if len(trusted_ids) == 0:
        alloc[:] = budget / N_total
        return alloc, set_aside, greedy

    N                = len(trusted_ids)
    set_aside_budget = alpha * budget
    greedy_budget    = (1.0 - alpha) * budget

    for i in trusted_ids:
        set_aside[i] = set_aside_budget / N

    pred_util_local = np.array([
        max(u_greedy[i] + V_tilde[i] * alpha / N, 1e-10)
        for i in trusted_ids
    ])
    v_local = np.array([max(reports[i], 0.0) for i in trusted_ids])

    greedy_local = compute_allocation_min_price_sec74(
        predicted_util=pred_util_local,
        values_round=v_local,
        budget=greedy_budget,
    )
    total_g = greedy_local.sum()
    if total_g > 1e-12 and abs(total_g - greedy_budget) > 1e-10:
        greedy_local *= greedy_budget / total_g

    for idx, i in enumerate(trusted_ids):
        greedy[i] = greedy_local[idx]
        alloc[i]  = set_aside[i] + greedy[i]

    total = alloc.sum()
    if total > 1e-12 and abs(total - budget) > 1e-10:
        alloc *= budget / total

    return alloc, set_aside, greedy


# =============================================================================
# VALUATION GENERATOR
# =============================================================================

def _generate_valuations(rng: np.random.RandomState,
                          N: int, T: int, model: str) -> np.ndarray:
    """
    Generate (N, T) true valuation matrix, column-normalised so each
    period's values sum to 1 (budget = 1 convention).
    """
    if model == "lognormal":
        raw = rng.lognormal(mean=0.0, sigma=0.80, size=(N, T))
    elif model == "pareto":
        raw = (rng.pareto(a=2.0, size=(N, T)) + 1.0)
    else:
        raw = rng.dirichlet(np.ones(N), size=T).T
        return raw
    col_sums = raw.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
    return raw / col_sums


# =============================================================================
# PART C — MAIN SIMULATION
# =============================================================================

class NSWTrustworthySim:
    """
    PART C — Full NSW trustworthy consensus simulation.

    Integrates all three parts each round:
      1. PART B updates Algorithm 1 continuously
      2. Count-free Moradi produces a shadow decision every 20 rounds
      3. Algorithm 1 produces trusted_ids for allocation by default
      4. PART A allocates resources to trusted_ids via water-filling
      5. PART C updates utilities, computes NSW, records statistics

    Key design decisions [R5]:
      - Full connectivity (all agents observe all others)
      - Malicious agents receive global β gap for precise evasion
      - COMPOUND agents have heterogeneous sub-types assigned at init [R4]
    """

    def __init__(self, config: ExperimentConfig, seed: int = 42):
        self.cfg       = config
        self.N         = config.n_legitimate + config.n_malicious
        self.T         = config.n_rounds
        self.legit_ids = list(range(config.n_legitimate))
        self.mal_ids   = list(range(config.n_legitimate, self.N))

        self.trust_gen = TrustObservationGenerator(config, seed=seed)
        rng_main       = np.random.RandomState(seed)
        rng_subtype    = np.random.RandomState(seed + 5555)

        # True valuations: (N, T) matrix, column-normalised
        self.true_vals = _generate_valuations(
            rng_main, self.N, self.T, config.valuation_model)

        # V_tilde: predicted totals, fixed at trial start — cannot be poisoned
        self.V_tilde = generate_V_tilde(
            self.true_vals, config.prediction_scenario,
            np.random.RandomState(seed + 1000))

        # [R5] Full connectivity — no mask needed, every agent observes all
        # (connectivity field kept for config compatibility but always 1.0)

        self.rngs = [np.random.RandomState(seed + 7 * i + 3) for i in range(self.N)]

        # Assign sub-types for COMPOUND attack [R4]
        def _make_agent(m: int) -> MaliciousAgent:
            sub = None
            if config.attack_type == AttackType.COMPOUND:
                idx = rng_subtype.randint(0, len(_COMPOUND_SUBTYPES))
                sub = _COMPOUND_SUBTYPES[idx]
            return MaliciousAgent(
                m, config.attack_type, config.base_attack_prob,
                config=config, seed=seed + m * 17 + 5, sub_type=sub)

        self.mal_agents: Dict[int, MaliciousAgent] = {
            m: _make_agent(m) for m in self.mal_ids}

        # Log sub-types for analysis
        self.mal_subtypes: Dict[int, Optional[AttackType]] = {
            m: self.mal_agents[m].sub_type for m in self.mal_ids}

        self.coalition = CoalitionCoordinator(
            self.mal_ids, config.n_legitimate,
            config.coalition_sync_every, seed=seed)

        # PART B: one TrustDetector per legitimate agent
        self.detectors: Dict[int, TrustDetector] = {
            i: TrustDetector(i, list(range(self.N)), config.xi_0, config.epsilon)
            for i in self.legit_ids}

        self.observation_counts = np.zeros((self.N, self.N), dtype=int)

        # PART C: utility accumulators
        self.utilities = np.zeros(self.N)
        self.u_greedy  = np.zeros(self.N)  # greedy component only (for PART A)

        self.hist_reports: List[np.ndarray] = []
        self.t        = 0
        self.history: List[Dict] = []
        self._nsw_opt: Optional[float] = None

        # Blended graph matrices (built at end of trial for M1–M2)
        self.beta_matrix    = np.zeros((self.N, self.N))
        self.blended_matrix = np.zeros((self.N, self.N))
        self.moradi_detected: Set[int] = set()
        self.moradi_decision_rounds: List[int] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _nsw_optimal(self) -> float:
        """Compute and cache the offline NSW optimum (legitimate agents only)."""
        if self._nsw_opt is None:
            mask = np.array([i in self.legit_ids for i in range(self.N)])
            nsw_opt, _, _ = offline_optimal_nsw(self.true_vals, mask)
            self._nsw_opt = nsw_opt
        return self._nsw_opt

    def _majority_vote_detect(self) -> Set[int]:
        """
        PART B — Majority vote over all legitimate detectors.

        Agent j is flagged as malicious iff more than half of all legitimate
        agents have classified j as detected by their individual Algorithm 1.
        This aggregation step makes the detector robust to isolated false
        positives from individual agents with noisy β observations.
        """
        votes: Dict[int, int] = {j: 0 for j in range(self.N)}
        for i in self.legit_ids:
            _, det = self.detectors[i].classify(self.t)
            for j in det:
                votes[j] += 1
        thresh = len(self.legit_ids) / 2.0
        return {j for j, v in votes.items() if v > thresh}

    def _global_beta_gap(self, m: int) -> float:
        """
        [R5] Full information: return the global β gap for malicious agent m,
        computed as the worst observer's max_beta_im - beta_im gap over ALL
        legitimate detectors.

        This gives the attacker perfect knowledge of the tightest detection
        margin. Using the maximum gap is more conservative than the previous
        average: one observer near the threshold is enough to create majority
        pressure once similar evidence accumulates elsewhere.
        With full connectivity every legitimate agent has observed m, so there
        are no missing observations to hide behind.
        """
        gaps = [self.detectors[i].beta_gap(m) for i in self.legit_ids]
        return float(np.max(gaps)) if gaps else 0.0

    def _current_xi_t(self) -> float:
        """Return the current threshold ξ_t from the first detector (all are identical)."""
        return self.detectors[self.legit_ids[0]].threshold(self.t)

    def _update_online_moradi(self, current_reports: np.ndarray) -> bool:
        """Recompute the count-free Moradi decision at a fixed checkpoint.

        At round 20, 40, 60, ... the graph is built only from trust and report
        observations available through that round.  The inferred suspicious
        set replaces the previous checkpoint's set and remains active until
        the next decision, allowing both detection and recovery from a false
        positive.  Returns True exactly when a new decision was made.
        """
        interval = max(1, int(self.cfg.moradi_decision_interval))
        if (self.t + 1) % interval != 0:
            return False

        reports_so_far = self.hist_reports + [current_reports.copy()]
        W_t = build_blended_matrix(
            beta_matrix=self.beta_matrix,
            hist_reports=reports_so_far,
            n_agents=self.N,
            sigma=self.cfg.moradi_sigma,
        )
        labels = sparsest_subgraph_detection(
            W=W_t,
            n_agents=self.N,
            epsilon=self.cfg.sparsest_epsilon,
        )
        self.moradi_detected = {
            int(agent_id) for agent_id in np.flatnonzero(labels)
        }
        self.moradi_decision_rounds.append(self.t + 1)
        return True

    def _select_online_detected(self, alg1_detected: Set[int]) -> Set[int]:
        """Choose the detector that controls the current online allocation."""
        mode = self.cfg.online_detector.lower()
        if mode == "moradi":
            return set(self.moradi_detected)
        if mode == "algorithm1":
            return set(alg1_detected)
        if mode == "hybrid":
            return set(alg1_detected) | set(self.moradi_detected)
        raise ValueError(
            "online_detector must be 'moradi', 'algorithm1', or 'hybrid'"
        )

    def _snapshot_beta_matrix(self) -> None:
        """Build the beta and blended matrices for post-hoc M1/M2 analysis."""
        for i in self.legit_ids:
            for j, b in self.detectors[i].beta.items():
                self.beta_matrix[i, j] = b
        self.blended_matrix = build_blended_matrix(
            beta_matrix=self.beta_matrix,
            hist_reports=self.hist_reports,
            n_agents=self.N,
            sigma=self.cfg.moradi_sigma,
        )

    # ── one simulation round ──────────────────────────────────────────────────

    def step(self) -> Dict:
        t    = self.t
        xi_t = self._current_xi_t()

        # (A) The most recent online decision is visible to adaptive attackers.
        alg1_detected_pre = self._majority_vote_detect() if t > 0 else set()
        detected_pre = self._select_online_detected(alg1_detected_pre)

        # (B) Generate reports — legitimate agents report truthfully;
        #     malicious agents apply distortion factor φ if attacking.
        #     [R5] Malicious agents know their global β gap exactly.
        reports = self.true_vals[:, t].copy()
        attacks: Dict[int, bool] = {}
        sync_signal = self.coalition.get_signal(t)   # [R4]

        for m in self.mal_ids:
            gap        = self._global_beta_gap(m)    # [R5]
            attacking, phi = self.mal_agents[m].decide_attack(
                t=t,
                is_detected=(m in detected_pre),
                sync_signal=sync_signal,
                current_xi_t=xi_t,
                global_beta_gap=gap,
            )
            attacks[m] = attacking
            if attacking:
                reports[m] = self.true_vals[m, t] * phi

        coalition_victim = None
        if self.cfg.attack_type == AttackType.COMPOUND and sync_signal:
            reports, coalition_victim = self.coalition.apply(
                reports=reports,
                true_vals_round=self.true_vals[:, t],
                amplify=self.cfg.coalition_amplify,
            )
            for m in self.mal_ids:
                attacks[m] = True

        # (C) Side-channel trust observations — [R5] full connectivity means
        #     every legitimate agent observes every other agent every round.
        for i in self.legit_ids:
            for j in range(self.N):
                if j == i:
                    continue
                alpha_obs = self.trust_gen.observe(
                    observed_legit=(j in self.legit_ids),
                    observed_attacking=attacks.get(j, False),
                    t=t,
                )
                self.detectors[i].update(j, alpha_obs)
                self.observation_counts[i, j] += 1

        # (D) Refresh the graph after this round's trust observations.
        alg1_detected = self._majority_vote_detect()
        self.beta_matrix.fill(0.0)
        for i in self.legit_ids:
            for j, b in self.detectors[i].beta.items():
                self.beta_matrix[i, j] = b

        # Moradi makes a count-free online decision every configured interval.
        moradi_decision = self._update_online_moradi(reports)
        detected = self._select_online_detected(alg1_detected)
        trusted  = [j for j in range(self.N) if j not in detected]

        # [M4] Implicit trust bootstrap — supplement cold-start agents (PART B)
        if t >= self.cfg.moradi_bootstrap_rounds and len(self.hist_reports) >= 3:
            demote = set()
            for j in list(trusted):
                obs = int(self.observation_counts[self.legit_ids, j].sum())
                if obs < self.cfg.moradi_bootstrap_rounds:
                    refs = [k for k in trusted if k != j]
                    if refs:
                        avg_imp = float(np.mean([
                            implicit_trust_score(j, k, self.hist_reports,
                                                 self.cfg.moradi_bootstrap_rounds)
                            for k in refs]))
                        if avg_imp < 0.35:
                            demote.add(j)
            trusted = [j for j in trusted if j not in demote]

        # [M3] Cluster merging safety net (PART B)
        trusted = merge_small_clusters(
            trusted_ids=trusted,
            beta_matrix=self.beta_matrix,
            all_agent_ids=list(range(self.N)),
            merge_threshold=self.cfg.moradi_merge_threshold,
        )

        # (E) Allocation — PART A: trusted_online_allocation over trusted set
        alloc, set_aside_alloc, greedy_alloc = trusted_online_allocation(
            reports=reports,
            trusted_ids=trusted,
            V_tilde=self.V_tilde,
            u_greedy=self.u_greedy,
            budget=1.0,
            alpha=self.cfg.allocation_alpha,
        )

        # (F) Update cumulative utilities with TRUE valuations (PART C)
        self.utilities += self.true_vals[:, t] * alloc
        self.u_greedy  += self.true_vals[:, t] * greedy_alloc  # PART A state

        # (G) Append reports to history (used by M1 similarity and M4 bootstrap)
        self.hist_reports.append(reports.copy())

        # (H) Compute statistics for this round
        tp  = len(detected & set(self.mal_ids))
        fp  = len(detected & set(self.legit_ids))
        fn  = len(set(self.mal_ids) - detected)
        alg1_tp = len(alg1_detected & set(self.mal_ids))
        alg1_fp = len(alg1_detected & set(self.legit_ids))
        moradi_tp = len(self.moradi_detected & set(self.mal_ids))
        moradi_fp = len(self.moradi_detected & set(self.legit_ids))

        nsw_legit    = compute_nsw(self.utilities[self.legit_ids])
        nsw_opt      = self._nsw_optimal()
        nsw_ratio    = nsw_legit / nsw_opt if nsw_opt > 0 else 0.0
        legit_utils  = self.utilities[self.legit_ids]
        min_util     = float(legit_utils.min()) if len(legit_utils) else 0.0
        fairness_gap = float(legit_utils.max() - legit_utils.min()) \
                       if len(legit_utils) > 1 else 0.0

        # β trajectory snapshots for per-agent analysis [R6]
        beta_legit = float(np.mean([
            np.mean(list(self.detectors[i].beta.values()))
            for i in self.legit_ids]))
        mal_gap_values = [
            self.detectors[i].beta_gap(m)
            for i in self.legit_ids
            for m in self.mal_ids
        ]
        legit_gap_values = [
            self.detectors[i].beta_gap(j)
            for i in self.legit_ids
            for j in self.legit_ids
            if j != i
        ]
        beta_mal_gap_avg = float(np.mean(mal_gap_values)) if mal_gap_values else 0.0
        beta_legit_gap_avg = float(np.mean(legit_gap_values)) if legit_gap_values else 0.0

        row = dict(
            t=t, tp=tp, fp=fp, fn=fn,
            detection_rate = tp / max(len(self.mal_ids), 1),
            fp_rate        = fp / max(len(self.legit_ids), 1),
            fn_rate        = fn / max(len(self.mal_ids), 1),
            n_attacking    = sum(attacks.values()),
            detected       = detected,
            alg1_detected  = set(alg1_detected),
            moradi_detected = set(self.moradi_detected),
            alg1_detection_rate = alg1_tp / max(len(self.mal_ids), 1),
            alg1_fp_rate = alg1_fp / max(len(self.legit_ids), 1),
            moradi_detection_rate = moradi_tp / max(len(self.mal_ids), 1),
            moradi_fp_rate = moradi_fp / max(len(self.legit_ids), 1),
            moradi_decision = moradi_decision,
            trusted        = list(trusted),
            alloc          = alloc.copy(),
            set_aside_alloc = set_aside_alloc.copy(),
            greedy_alloc    = greedy_alloc.copy(),
            utilities      = self.utilities.copy(),
            beta_matrix    = self.beta_matrix.copy(),
            nsw_legit      = nsw_legit,
            nsw_opt        = nsw_opt,
            nsw_ratio      = nsw_ratio,
            min_util       = min_util,
            fairness_gap   = fairness_gap,
            beta_legit_avg = beta_legit,
            beta_mal_gap_avg = beta_mal_gap_avg,
            beta_legit_gap_avg = beta_legit_gap_avg,
            xi_t           = xi_t,
            sync_signal    = sync_signal,
            coalition_victim = coalition_victim,
        )
        self.history.append(row)
        self.t += 1
        return row

    def run(self, verbose: bool = False) -> List[Dict]:
        for t in range(self.T):
            self.step()
            if verbose and (t % 50 == 0 or t == self.T - 1):
                r = self.history[-1]
                print(f"  t={t+1:3d}/{self.T}  "
                      f"DR={r['detection_rate']:.0%}  "
                      f"FPR={r['fp_rate']:.0%}  "
                      f"NSW={r['nsw_legit']:.4f}  "
                      f"ratio={r['nsw_ratio']:.3f}  "
                      f"α={self.cfg.allocation_alpha:.2f}  "
                      f"atk={r['n_attacking']}/{self.cfg.n_malicious}")
        self._snapshot_beta_matrix()
        return self.history


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def convergence_round(history: List[Dict], threshold: float = 0.95) -> int:
    """First one-based round reaching the detection threshold; T if never."""
    for row in history:
        if row['detection_rate'] >= threshold:
            return row['t'] + 1
    return len(history)


def run_trials(config: ExperimentConfig, verbose: bool = False) -> List[Dict]:
    """
    Run n_trials independent simulations with different seeds.
    Returns list of trial result dicts, each containing full history and
    summary statistics.
    """
    results = []
    for trial in range(config.n_trials):
        if verbose:
            print(f"  Trial {trial+1}/{config.n_trials}")
        sim  = NSWTrustworthySim(config, seed=42 + trial * 13)
        hist = sim.run(verbose=False)

        # Detectors receive only the observed graph.  Ground-truth composition
        # is used below for evaluation after predictions have been produced.
        spec_labels = spectral_detection(
            sim.blended_matrix,
            sim.N)

        sparse_labels = sparsest_subgraph_detection(
            W=sim.blended_matrix,
            n_agents=sim.N,
            epsilon=config.sparsest_epsilon,
        )

        true_labels = np.array(
            [0] * config.n_legitimate + [1] * config.n_malicious
        )
        spec_f1 = binary_f1_score(true_labels, spec_labels)
        sparse_f1 = binary_f1_score(true_labels, sparse_labels)

        online_dr = hist[-1]['detection_rate']
        online_labels = np.array([
            1 if i in hist[-1]['detected'] else 0
            for i in range(sim.N)
        ])
        online_f1 = binary_f1_score(true_labels, online_labels)

        alg1_dr = hist[-1]['alg1_detection_rate']
        alg1_labels = np.array([
            1 if i in hist[-1]['alg1_detected'] else 0
            for i in range(sim.N)
        ])
        alg1_f1 = binary_f1_score(true_labels, alg1_labels)

        moradi_online_labels = np.array([
            1 if i in hist[-1]['moradi_detected'] else 0
            for i in range(sim.N)
        ])
        moradi_online_f1 = binary_f1_score(
            true_labels, moradi_online_labels
        )

        # Attack rates per malicious agent
        atk_rates = {m: sim.mal_agents[m].empirical_attack_rate()
                     for m in sim.mal_ids}

        results.append(dict(
            history            = hist,
            final_nsw_legit    = hist[-1]['nsw_legit'],
            final_nsw_ratio    = hist[-1]['nsw_ratio'],
            final_dr           = online_dr,
            final_fpr          = hist[-1]['fp_rate'],
            final_min_util     = hist[-1]['min_util'],
            final_fairness_gap = hist[-1]['fairness_gap'],
            conv_round         = convergence_round(hist),
            spectral_f1        = spec_f1,
            sparse_f1          = sparse_f1,
            alg1_dr            = alg1_dr,
            alg1_f1            = alg1_f1,
            online_f1          = online_f1,
            moradi_online_f1   = moradi_online_f1,
            empirical_atk_rates = atk_rates,
            sim                = sim,
        ))
    return results


def compare_focused_types(base: ExperimentConfig) -> Dict:
    """
    [R2][R6] Run the 4 focused attack types with 25 trials each.
    Returns dict keyed by AttackType with aggregate statistics.
    """
    focused = [
        AttackType.BYZANTINE,
        AttackType.BURST,
        AttackType.REPUTATION_POISON,
        AttackType.TRUST_MIMICRY,
    ]
    out = {}
    for at in focused:
        cfg    = replace(base, attack_type=at)
        trials = run_trials(cfg, verbose=False)
        out[at] = dict(
            avg_nsw       = float(np.mean([r['final_nsw_legit']   for r in trials])),
            std_nsw       = float(np.std( [r['final_nsw_legit']   for r in trials])),
            avg_ratio     = float(np.mean([r['final_nsw_ratio']   for r in trials])),
            std_ratio     = float(np.std( [r['final_nsw_ratio']   for r in trials])),
            avg_dr        = float(np.mean([r['final_dr']          for r in trials])),
            std_dr        = float(np.std( [r['final_dr']          for r in trials])),
            avg_conv      = float(np.mean([r['conv_round']        for r in trials])),
            avg_min_util  = float(np.mean([r['final_min_util']    for r in trials])),
            avg_spec_f1   = float(np.mean([r['spectral_f1']       for r in trials])),
            avg_sparse_f1 = float(np.mean([r['sparse_f1']         for r in trials])),
            avg_alg1_f1   = float(np.mean([r['alg1_f1']           for r in trials])),
            avg_online_f1 = float(np.mean([r['online_f1']         for r in trials])),
            avg_moradi_online_f1 = float(np.mean([
                r['moradi_online_f1'] for r in trials
            ])),
            avg_fpr       = float(np.mean([r['final_fpr']         for r in trials])),
            trials        = trials,
        )
        print(f"  {at.value:18s}  NSW={out[at]['avg_nsw']:.4f}  "
              f"ratio={out[at]['avg_ratio']:.3f}  DR={out[at]['avg_dr']:.0%}  "
              f"onlineF1={out[at]['avg_online_f1']:.3f}  "
              f"alg1F1={out[at]['avg_alg1_f1']:.3f}  "
              f"sparseF1={out[at]['avg_sparse_f1']:.3f}  "
              f"conv={out[at]['avg_conv']:.1f}")
    return out


# =============================================================================
# PLOTS
# =============================================================================

ATTACK_COLOURS = {
    AttackType.BYZANTINE:         '#E53935',   # red
    AttackType.BURST:             '#FB8C00',   # orange
    AttackType.REPUTATION_POISON: '#8E24AA',   # purple
    AttackType.TRUST_MIMICRY:     '#1E88E5',   # blue
    AttackType.COMPOUND:          '#43A047',   # green
}

ATTACK_LABELS = {
    AttackType.BYZANTINE:         'Byzantine',
    AttackType.BURST:             'Burst',
    AttackType.REPUTATION_POISON: 'Reputation Poison',
    AttackType.TRUST_MIMICRY:     'Trust Mimicry',
    AttackType.COMPOUND:          'Compound',
}


def _save(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")


# ----------------------------------------------------------------------------
# Plot 1: Main results over rounds (all 4 attack types on same axes) [R1]
# ----------------------------------------------------------------------------

def plot_main_results_all_types(
        results: Dict,
        config: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot1_main_results.png')):
    """
    [R1][R6] Four-panel line plot showing all 4 focused attack types
    simultaneously, with 95% CI bands. Replaces the single-attack-type
    bar chart from the previous version.
    """
    n_r    = config.n_rounds
    rounds = np.arange(n_r)
    if config.online_detector.lower() == "algorithm1":
        detection_title = (
            "Detection Rate (Algorithm 1 allocation gate; "
            f"Moradi shadow every {config.moradi_decision_interval} rounds)"
        )
    else:
        detection_title = f"Detection Rate (Online {config.online_detector.title()} Gate)"
    metrics = [
        ('detection_rate', 'Detection Rate', detection_title),
        ('fp_rate',        'False Positive Rate',      'FPR — Legitimate Agents Flagged'),
        ('nsw_legit',      'NSW (legitimate)',          'Nash Social Welfare — Legitimate Agents'),
        ('nsw_ratio',      'NSW Ratio (achieved/opt)', 'NSW Ratio vs Offline Optimum'),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f'All Four Attack Types — Main Metrics over {n_r} Rounds\n'
        f'α={config.allocation_alpha:.2f}  σ={config.moradi_sigma:.2f}  '
        f'n_legit={config.n_legitimate}  n_mal={config.n_malicious}  '
        f'trials={config.n_trials}  (bands = 95% CI)',
        fontsize=10, fontweight='bold')

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for at, res in results.items():
            trials = res['trials']
            n_t    = len(trials)
            data   = np.zeros((n_t, n_r))
            for k, tr in enumerate(trials):
                for row in tr['history']:
                    data[k, row['t']] = row[key]
            mu  = data.mean(0)
            err = ci95(data)
            col = ATTACK_COLOURS[at]
            ax.plot(rounds, mu, color=col, lw=2, label=ATTACK_LABELS[at])
            lower = mu - err
            upper = mu + err
            if key in ('detection_rate', 'fp_rate'):
                lower = np.clip(lower, 0, 1)
                upper = np.clip(upper, 0, 1)
            ax.fill_between(rounds, lower, upper, alpha=0.18, color=col)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        if key in ('detection_rate', 'fp_rate'):
            ax.set_ylim(0, 1.05)

    axes[3].axhline(1.0, ls=':', color='gray', alpha=0.6, label='Optimal = 1')
    axes[3].set_xlabel('Round', fontsize=11)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[3].legend(fontsize=9)
    _save(path)


# ----------------------------------------------------------------------------
# Plot 2: Per-agent utility trajectories
# ----------------------------------------------------------------------------

def plot_utility_trajectories(trial: Dict, config: ExperimentConfig,
                               path: str = str(OUTPUT_DIR / 'plot2_utility_traj.png')):
    hist   = trial['history']
    N_l    = config.n_legitimate
    N_m    = config.n_malicious
    utils  = np.array([row['utilities'] for row in hist])
    rounds = np.arange(len(hist))

    at_col = ATTACK_COLOURS.get(config.attack_type, '#333333')
    fig, ax = plt.subplots(figsize=(13, 5))
    for i in range(N_l):
        ax.plot(rounds, utils[:, i], color='#1E88E5', lw=1.0, alpha=0.55)
    for m in range(N_l, N_l + N_m):
        ax.plot(rounds, utils[:, m], color=at_col, lw=1.4, alpha=0.75, ls='--')
    ax.plot([], [], color='#1E88E5', label=f'Legitimate ({N_l})', lw=2)
    ax.plot([], [], color=at_col, label=f'Malicious ({N_m}) — {ATTACK_LABELS[config.attack_type]}',
            lw=2, ls='--')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Cumulative Utility', fontsize=11)
    ax.set_title(
        f'Per-Agent Cumulative Utility  |  {ATTACK_LABELS[config.attack_type]}  |  '
        f'α={config.allocation_alpha:.2f}  |  {config.scenario_label}',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    _save(path)


# ----------------------------------------------------------------------------
# Plot 3: Allocation heatmap
# ----------------------------------------------------------------------------

def plot_allocation_heatmap(trial: Dict, config: ExperimentConfig,
                             path: str = str(OUTPUT_DIR / 'plot3_alloc_heatmap.png')):
    hist  = trial['history']
    N     = config.n_legitimate + config.n_malicious
    alloc = np.array([row['alloc'] for row in hist]).T

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(alloc, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.axhline(config.n_legitimate - 0.5, color='cyan', lw=2,
               label='Legitimate / Malicious boundary')
    plt.colorbar(im, ax=ax, label='Allocation share z_{i,t}')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Agent index', fontsize=11)
    ax.set_title(
        f'Allocation Heatmap  |  {ATTACK_LABELS[config.attack_type]}  |  '
        f'α={config.allocation_alpha:.2f}  |  {config.scenario_label}',
        fontsize=11, fontweight='bold')
    lbls = [f'L{i}' for i in range(config.n_legitimate)] + \
           [f'M{i}' for i in range(config.n_malicious)]
    ax.set_yticks(range(N))
    ax.set_yticklabels(lbls, fontsize=7)
    ax.legend(fontsize=9, loc='upper right')
    _save(path)


# ----------------------------------------------------------------------------
# Plot 4: Misclassification rates — all 4 types [R1][R6]
# ----------------------------------------------------------------------------

def plot_misclassification_all_types(
        results: Dict,
        config: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot4_misclass.png')):
    """
    [R1][R6][R7] Line plot of FPR and FNR over time for all 4 attack types.
    Higher base attack rate [R7] makes both curves more dynamic.
    95% CI bands from 25 trials [R6].
    """
    n_r    = config.n_rounds
    rounds = np.arange(n_r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Misclassification Rates — All Four Attack Types  |  '
        f'{n_r} rounds  |  {config.n_trials} trials  (bands = 95% CI)',
        fontsize=11, fontweight='bold')

    for at, res in results.items():
        trials = res['trials']
        n_t    = len(trials)
        col    = ATTACK_COLOURS[at]
        lbl    = ATTACK_LABELS[at]
        for ax, key, ylabel in zip(axes,
                                    ['fp_rate', 'fn_rate'],
                                    ['False Positive Rate', 'False Negative Rate']):
            data = np.zeros((n_t, n_r))
            for k, tr in enumerate(trials):
                for row in tr['history']:
                    data[k, row['t']] = row[key]
            mu  = data.mean(0)
            err = ci95(data)
            ax.plot(rounds, mu, color=col, lw=2, label=lbl)
            ax.fill_between(rounds, np.clip(mu - err, 0, 1),
                            np.clip(mu + err, 0, 1), alpha=0.18, color=col)

    titles = ['False Positive Rate\n(legitimate agents incorrectly flagged)',
              'False Negative Rate\n(malicious agents not yet detected)']
    for ax, title, ylabel in zip(axes, titles, ['FPR', 'FNR']):
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    _save(path)


# ----------------------------------------------------------------------------
# Plot 5: Attack-type summary — line plot [R1]
# ----------------------------------------------------------------------------

def plot_attack_summary_lines(
        results: Dict,
        config: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot5_attack_comparison.png')):
    """
    [R1] Summary line plot: each metric as a line across the 4 attack types.
    X-axis = attack type (categorical), but rendered as connected markers
    so trends between types are visible.
    """
    types  = list(results.keys())
    labels = [ATTACK_LABELS[at] for at in types]
    x      = np.arange(len(types))

    metrics = [
        ('avg_nsw',   'std_nsw',   'Avg NSW (legitimate)',         '#1E88E5'),
        ('avg_ratio', 'std_ratio', 'Avg NSW Ratio (achieved/opt)', '#8E24AA'),
        ('avg_dr',    'std_dr',    'Avg Detection Rate',           '#43A047'),
        ('avg_moradi_online_f1', None, 'Avg Shadow Online Moradi F1', '#E53935'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f'Attack Type Summary — Four Focused Types  |  '
        f'{config.n_trials} trials  |  {config.n_rounds} rounds',
        fontsize=12, fontweight='bold')

    for ax, (mu_key, std_key, title, col) in zip(axes.flat, metrics):
        vals = [results[at][mu_key] for at in types]
        errs = [results[at][std_key] for at in types] if std_key else None
        ax.plot(x, vals, 'o-', color=col, lw=2.5, ms=9, zorder=3)
        if errs:
            ax.errorbar(x, vals, yerr=errs, fmt='none', color=col,
                        capsize=5, lw=1.5, zorder=2)
        # Colour each marker by attack type
        for xi, at in zip(x, types):
            ax.scatter([xi], [results[at][mu_key]], color=ATTACK_COLOURS[at],
                       s=80, zorder=4, edgecolors='white', linewidths=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
        # Annotate values
        label_offset = 0.01 * max(max(vals), 1e-9)
        for xi, v in zip(x, vals):
            ax.text(xi, v + label_offset, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)
    _save(path)


# ----------------------------------------------------------------------------
# Plot 6: β trajectory — trust accumulation per agent type over time [R6]
# ----------------------------------------------------------------------------

def plot_beta_trajectories(
        results: Dict,
        config: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot6_beta_trajectories.png')):
    """
    [R6] Deep analysis plot: shows the actual Algorithm 1 decision quantity,
    max_β − β_j, for malicious and legitimate agents.

    Algorithm 1 compares this β gap to ξ_t. Plotting raw average β would be
    visually misleading because detection is triggered by relative evidence,
    not by the absolute β level.
    """
    n_r    = config.n_rounds
    rounds = np.arange(n_r)

    # Compute xi_t curve (same for all trials, use a dummy detector)
    dummy = TrustDetector(0, [0, 1], config.xi_0, config.epsilon)
    xi_curve = np.array([dummy.threshold(t) for t in rounds])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f'β Accumulation and Detection Threshold ξ_t — Per Attack Type\n'
        f'Average β gap as observed by legitimate agents  |  {config.n_trials} trials  '
        f'(bands = 95% CI)',
        fontsize=10, fontweight='bold')

    for ax, (at, res) in zip(axes.flat, results.items()):
        trials = res['trials']
        n_t    = len(trials)
        col    = ATTACK_COLOURS[at]

        mal_gap_data = np.zeros((n_t, n_r))
        legit_gap_data = np.zeros((n_t, n_r))
        for k, tr in enumerate(trials):
            for row in tr['history']:
                mal_gap_data[k, row['t']] = row['beta_mal_gap_avg']
                legit_gap_data[k, row['t']] = row['beta_legit_gap_avg']

        mal_mu  = mal_gap_data.mean(0)
        mal_err = ci95(mal_gap_data)
        legit_mu = legit_gap_data.mean(0)
        legit_err = ci95(legit_gap_data)
        ax.plot(rounds, mal_mu, color=col, lw=2,
                label=f'Malicious gap — {ATTACK_LABELS[at]}')
        ax.fill_between(rounds, np.clip(mal_mu - mal_err, 0, None),
                        mal_mu + mal_err, alpha=0.20, color=col)
        ax.plot(rounds, legit_mu, color='#1E88E5', lw=1.5, ls=':',
                label='Legitimate gap')
        ax.fill_between(rounds, np.clip(legit_mu - legit_err, 0, None),
                        legit_mu + legit_err, alpha=0.12, color='#1E88E5')

        # Overlay ξ_t threshold
        ax.plot(rounds, xi_curve, color='black', lw=1.5, ls='--', alpha=0.6,
                label='ξ_t (detection threshold)')

        # Mark warm-up end for REPUTATION_POISON
        if at == AttackType.REPUTATION_POISON:
            ax.axvline(config.poison_warmup_rounds, color='red', ls=':',
                       lw=1.5, alpha=0.8, label=f'Warmup end (t={config.poison_warmup_rounds})')

        # Mark burst cadence; COMPOUND is not part of this four-type plot.
        if at == AttackType.BURST:
            for t_sync in range(0, n_r, config.coalition_sync_every):
                if t_sync == 0:
                    ax.axvline(t_sync, color='orange', ls=':', lw=0.8,
                               alpha=0.4, label='Coalition sync rounds')
                else:
                    ax.axvline(t_sync, color='orange', ls=':', lw=0.8, alpha=0.4)

        ax.set_title(ATTACK_LABELS[at], fontsize=11, fontweight='bold')
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('β gap: max_β − β_j', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    _save(path)


# ----------------------------------------------------------------------------
# Plot 7: NSW ratio sensitivity (adversary count) — line plot [R1]
# ----------------------------------------------------------------------------

def plot_nsw_ratio_sensitivity(
        base: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot7_nsw_ratio.png')):
    """
    [R1][R6] NSW ratio vs number of malicious agents for all 4 attack types.
    Line plot with 95% CI for each attack type — shows how each attack strategy
    scales differently with adversary count.
    """
    mal_counts = [1, 2, 3, 4, 5, 6]
    focused    = [AttackType.BYZANTINE, AttackType.BURST,
                  AttackType.REPUTATION_POISON, AttackType.TRUST_MIMICRY]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(
        f'NSW Ratio vs Adversary Count — Four Attack Types\n'
        f'{base.n_trials} trials, {base.n_rounds} rounds  (bars = 95% CI)',
        fontsize=11, fontweight='bold')

    for at in focused:
        ratios = []
        errs   = []
        for nm in mal_counts:
            cfg    = replace(base, n_malicious=nm, attack_type=at)
            trials = run_trials(cfg, verbose=False)
            vals   = np.array([r['final_nsw_ratio'] for r in trials])
            ratios.append(float(vals.mean()))
            errs.append(float(1.96 * vals.std() / math.sqrt(len(vals))))
        col = ATTACK_COLOURS[at]
        ax.plot(mal_counts, ratios, 'o-', color=col, lw=2, ms=7,
                label=ATTACK_LABELS[at])
        ax.errorbar(mal_counts, ratios, yerr=errs, fmt='none',
                    color=col, capsize=5, lw=1.5)

    ax.axhline(1.0, ls=':', color='gray', alpha=0.6, label='Optimal NSW ratio')
    ax.set_xlabel('Number of Malicious Agents', fontsize=11)
    ax.set_ylabel('NSW Ratio (achieved / offline-optimal)', fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(path)


# ----------------------------------------------------------------------------
# Plot 8: Alpha sweep — line plot [R1]
# ----------------------------------------------------------------------------

def plot_alpha_sweep(base: ExperimentConfig,
                     path: str = str(OUTPUT_DIR / 'plot8_alpha_sweep.png')):
    """
    [R1] Alpha sweep (efficiency-fairness frontier) as a line plot.
    Shows all 4 attack types simultaneously so the interaction between
    alpha and attack type is visible.
    """
    alphas  = np.linspace(0.0, 1.0, 11)
    focused = [AttackType.BYZANTINE, AttackType.BURST,
               AttackType.REPUTATION_POISON, AttackType.TRUST_MIMICRY]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'Alpha Sweep: Efficiency vs Fairness Frontier — {base.n_rounds} rounds',
        fontsize=12, fontweight='bold')

    metric_keys = ['final_nsw_ratio', 'final_min_util', 'final_fairness_gap']
    ylabels     = ['NSW Ratio (achieved/opt)', 'Min-Agent Utility', 'Fairness Gap (max−min)']
    titles      = ['NSW Efficiency vs α', 'Fairness Floor vs α', 'Utility Spread vs α']

    for at in focused:
        col    = ATTACK_COLOURS[at]
        lbl    = ATTACK_LABELS[at]
        data   = {k: [] for k in metric_keys}
        for alpha in alphas:
            cfg    = replace(base, allocation_alpha=float(alpha), attack_type=at)
            trials = run_trials(cfg, verbose=False)
            for k in metric_keys:
                data[k].append(float(np.mean([r[k] for r in trials])))
        for ax, k, ylabel, title in zip(axes, metric_keys, ylabels, titles):
            ax.plot(alphas, data[k], 'o-', color=col, lw=2, ms=5, label=lbl)

    for ax, ylabel, title in zip(axes, ylabels, titles):
        ax.axvline(0.5, ls='--', color='gray', alpha=0.6, label='α=0.5 (paper default)')
        ax.set_xlabel('α  (0 = pure greedy, 1 = pure equal)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    _save(path)


# ----------------------------------------------------------------------------
# Plot 9: Detector benchmark [R6]
# ----------------------------------------------------------------------------

def plot_three_way_detector(
        results: Dict,
        path: str = str(OUTPUT_DIR / 'plot9_three_way_detector.png')):
    """
    Four-way F1 comparison: Algorithm 1, spectral clustering, the final
    count-free Moradi graph, and Moradi's active online checkpoint decision.
    """
    types    = list(results.keys())
    labels   = [ATTACK_LABELS[at] for at in types]
    alg1_v   = [results[at]['avg_alg1_f1']   for at in types]
    spec_v   = [results[at]['avg_spec_f1']   for at in types]
    sparse_v = [results[at]['avg_sparse_f1'] for at in types]
    online_v = [results[at]['avg_moradi_online_f1'] for at in types]
    trial_count = min(len(results[at]['trials']) for at in types) if types else 0

    x = np.arange(len(types))
    w = 0.20

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*w, alg1_v, width=w, color='#1E88E5', alpha=0.85,
           label='Algorithm 1 (PART B)', edgecolor='white')
    ax.bar(x - 0.5*w, spec_v, width=w, color='#FB8C00', alpha=0.85,
           label='Spectral Clustering', edgecolor='white')
    ax.bar(x + 0.5*w, sparse_v, width=w, color='#43A047', alpha=0.85,
           label='Final Moradi graph', edgecolor='white')
    ax.bar(x + 1.5*w, online_v, width=w, color='#8E24AA', alpha=0.85,
           label='Online Moradi checkpoints', edgecolor='white')

    for vals, offset, col in [
        (alg1_v, -1.5*w, '#0D47A1'),
        (spec_v, -0.5*w, '#E65100'),
        (sparse_v, 0.5*w, '#1B5E20'),
        (online_v, 1.5*w, '#4A148C'),
    ]:
        for xi, v in zip(x, vals):
            ax.text(xi + offset, v + 0.01, f'{v:.2f}', ha='center',
                    va='bottom', fontsize=9, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_ylim(0, 1.20)
    ax.set_title(
        'Detection Benchmark: Algorithm 1 vs Spectral vs Final and Online Moradi\n'
        '(blended trust+similarity graph W  |  four focused attack types  |  '
        f'{trial_count} trials)',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    _save(path)


# ----------------------------------------------------------------------------
# Plot 10: Sigma sweep — line plot [R1]
# ----------------------------------------------------------------------------

def plot_sigma_sweep(
        base: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot10_sigma_sweep.png')):
    """
    [R1] Sigma sweep: blended graph weight σ vs detector F1 and NSW ratio.
    Shows all 4 attack types to reveal which σ value is most robust.
    """
    sigmas  = np.linspace(0.0, 1.0, 11)
    focused = [AttackType.BYZANTINE, AttackType.BURST,
               AttackType.REPUTATION_POISON, AttackType.TRUST_MIMICRY]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'Sigma Sweep: Trust-Similarity Blend σ in W_ij = σ·T_ij + (1−σ)·S_ij\n'
        f'{base.n_rounds} rounds  |  {base.n_trials} trials per attack type',
        fontsize=11, fontweight='bold')

    for at in focused:
        col       = ATTACK_COLOURS[at]
        lbl       = ATTACK_LABELS[at]
        spec_f1s  = []
        sparse_f1s = []
        nsw_ratios = []
        for sigma in sigmas:
            cfg    = replace(base, moradi_sigma=float(sigma), attack_type=at)
            trials = run_trials(cfg, verbose=False)
            spec_f1s.append( float(np.mean([r['spectral_f1']      for r in trials])))
            sparse_f1s.append(float(np.mean([r['sparse_f1']         for r in trials])))
            nsw_ratios.append(float(np.mean([r['final_nsw_ratio']   for r in trials])))
        for ax, vals in zip(axes, [spec_f1s, sparse_f1s, nsw_ratios]):
            ax.plot(sigmas, vals, 'o-', color=col, lw=2, ms=5, label=lbl)

    titles  = ['Spectral Clustering F1 vs σ',
               'Sparsest Subgraph F1 vs σ (Moradi)',
               'NSW Ratio vs σ']
    ylabels = ['Spectral F1', 'Sparsest-Subgraph F1', 'NSW Ratio']

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.axvline(0.6, ls='--', color='gray', alpha=0.6, label='σ=0.6 (default)')
        ax.set_xlabel('σ  (0=similarity only, 1=trust only)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    _save(path)


# ----------------------------------------------------------------------------
# Plot 11: Compound attack sub-type breakdown [R4]
# ----------------------------------------------------------------------------

def plot_compound_breakdown(
        base: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot11_compound_breakdown.png')):
    """
    [R4][R6] Deep analysis of COMPOUND attack: shows per-agent empirical attack
    rates grouped by their assigned sub-type (BYZANTINE / BURST / TRUST_MIMICRY).
    Also shows detection rate over time for COMPOUND vs individual types.
    """
    cfg    = replace(base, attack_type=AttackType.COMPOUND)
    trials = run_trials(cfg, verbose=False)
    n_r    = base.n_rounds
    rounds = np.arange(n_r)

    # Compare detection rate: COMPOUND vs individual types
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Compound Attack Analysis — Heterogeneous Sub-Types per Agent\n'
        f'{base.n_trials} trials  |  coalition sync every {base.coalition_sync_every} rounds',
        fontsize=11, fontweight='bold')

    # Left: detection rate COMPOUND vs baselines
    compare_types = [AttackType.COMPOUND, AttackType.BYZANTINE,
                     AttackType.BURST, AttackType.TRUST_MIMICRY]
    for at in compare_types:
        cfg_at = replace(base, attack_type=at)
        tr_at  = run_trials(cfg_at, verbose=False) if at != AttackType.COMPOUND else trials
        n_t    = len(tr_at)
        data   = np.zeros((n_t, n_r))
        for k, tr in enumerate(tr_at):
            for row in tr['history']:
                data[k, row['t']] = row['detection_rate']
        mu  = data.mean(0)
        err = ci95(data)
        col = ATTACK_COLOURS[at]
        axes[0].plot(rounds, mu, color=col, lw=2, label=ATTACK_LABELS[at])
        axes[0].fill_between(rounds, mu - err, mu + err, alpha=0.18, color=col)

    axes[0].set_xlabel('Round', fontsize=10)
    axes[0].set_ylabel('Detection Rate', fontsize=10)
    axes[0].set_title('Detection Rate: Compound vs Individual Types', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Mark coalition sync rounds
    for t_sync in range(0, n_r, base.coalition_sync_every):
        axes[0].axvline(t_sync, color='gray', ls=':', lw=0.6, alpha=0.3)

    # Right: empirical attack rates per sub-type across trials
    subtype_rates: Dict[AttackType, List[float]] = {
        st: [] for st in _COMPOUND_SUBTYPES}
    for tr in trials:
        sim = tr['sim']
        for m, agent in sim.mal_agents.items():
            st = agent.sub_type or AttackType.BYZANTINE
            subtype_rates[st].append(agent.empirical_attack_rate())

    st_labels = [ATTACK_LABELS[st] for st in _COMPOUND_SUBTYPES]
    st_vals   = [np.mean(subtype_rates[st]) if subtype_rates[st] else 0.0
                 for st in _COMPOUND_SUBTYPES]
    st_errs   = [1.96 * np.std(subtype_rates[st]) / math.sqrt(max(len(subtype_rates[st]), 1))
                 for st in _COMPOUND_SUBTYPES]
    st_cols   = [ATTACK_COLOURS[st] for st in _COMPOUND_SUBTYPES]

    axes[1].bar(st_labels, st_vals, color=st_cols, alpha=0.82,
                edgecolor='white', linewidth=1.2)
    axes[1].errorbar(st_labels, st_vals, yerr=st_errs,
                     fmt='none', color='black', capsize=6, lw=1.5)
    for xi, (v, e) in enumerate(zip(st_vals, st_errs)):
        axes[1].text(xi, v + e + 0.01, f'{v:.2f}', ha='center',
                     va='bottom', fontsize=10)
    axes[1].set_ylabel('Empirical Attack Rate', fontsize=10)
    axes[1].set_title('Attack Rate by Sub-Type (Compound Agents)', fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', alpha=0.3)
    _save(path)


# ----------------------------------------------------------------------------
# Plot 12: Reputation Poison phase analysis [R3][R6]
# ----------------------------------------------------------------------------

def plot_reputation_poison_phases(
        base: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot12_reputation_poison_phases.png')):
    """
    [R3][R6] Deep analysis of the redesigned Reputation Poison attack.

    Shows three panels:
      Left:   β accumulation for REPUTATION_POISON vs BYZANTINE — the warmup
              phase lets the poisoner build high β that must be overcome
      Centre: Detection rate with vertical markers at warmup end and at
              the round where detection first exceeds 50%
      Right:  NSW over time — the characteristic cliff at warmup end followed
              by gradual recovery as detection catches up

    This plot directly validates the redesign in [R3].
    """
    n_r    = base.n_rounds
    rounds = np.arange(n_r)
    warmup = base.poison_warmup_rounds

    # Run both REPUTATION_POISON and BYZANTINE for comparison
    cfg_rp  = replace(base, attack_type=AttackType.REPUTATION_POISON)
    cfg_byz = replace(base, attack_type=AttackType.BYZANTINE)
    trials_rp  = run_trials(cfg_rp,  verbose=False)
    trials_byz = run_trials(cfg_byz, verbose=False)
    n_t = len(trials_rp)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'Reputation Poison — Phase Analysis  |  '
        f'Warmup = {warmup} rounds  |  Phase 2 = Trust Mimicry evasion\n'
        f'{base.n_trials} trials  |  {n_r} rounds  (bands = 95% CI)',
        fontsize=10, fontweight='bold')

    # Left: β trajectory
    for trials, at, col, lbl in [
        (trials_rp,  AttackType.REPUTATION_POISON, '#8E24AA', 'Reputation Poison'),
        (trials_byz, AttackType.BYZANTINE,          '#E53935', 'Byzantine (baseline)'),
    ]:
        beta_data = np.zeros((n_t, n_r))
        for k, tr in enumerate(trials):
            for row in tr['history']:
                beta_data[k, row['t']] = row['beta_legit_avg']
        mu  = beta_data.mean(0)
        err = ci95(beta_data)
        axes[0].plot(rounds, mu, color=col, lw=2, label=lbl)
        axes[0].fill_between(rounds, mu - err, mu + err, alpha=0.18, color=col)

    axes[0].axvline(warmup, color='red', ls='--', lw=1.5,
                    label=f'Warmup end (t={warmup})')
    axes[0].set_xlabel('Round', fontsize=10)
    axes[0].set_ylabel('Avg β (legitimate observers)', fontsize=10)
    axes[0].set_title('β Trajectory: Warmup → Evasion', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Centre: detection rate
    for trials, col, lbl in [
        (trials_rp,  '#8E24AA', 'Reputation Poison'),
        (trials_byz, '#E53935', 'Byzantine'),
    ]:
        data = np.zeros((n_t, n_r))
        for k, tr in enumerate(trials):
            for row in tr['history']:
                data[k, row['t']] = row['detection_rate']
        mu  = data.mean(0)
        err = ci95(data)
        axes[1].plot(rounds, mu, color=col, lw=2, label=lbl)
        axes[1].fill_between(rounds, mu - err, mu + err, alpha=0.18, color=col)

    axes[1].axvline(warmup, color='red', ls='--', lw=1.5)
    axes[1].axhline(0.50, color='gray', ls=':', lw=1.2, alpha=0.7,
                    label='50% detection')
    axes[1].set_xlabel('Round', fontsize=10)
    axes[1].set_ylabel('Detection Rate', fontsize=10)
    axes[1].set_title('Detection Rate Over Time', fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Right: NSW
    for trials, col, lbl in [
        (trials_rp,  '#8E24AA', 'Reputation Poison'),
        (trials_byz, '#E53935', 'Byzantine'),
    ]:
        data = np.zeros((n_t, n_r))
        for k, tr in enumerate(trials):
            for row in tr['history']:
                data[k, row['t']] = row['nsw_legit']
        mu  = data.mean(0)
        err = ci95(data)
        axes[2].plot(rounds, mu, color=col, lw=2, label=lbl)
        axes[2].fill_between(rounds, mu - err, mu + err, alpha=0.18, color=col)

    axes[2].axvline(warmup, color='red', ls='--', lw=1.5,
                    label=f'Warmup end (t={warmup})')
    axes[2].set_xlabel('Round', fontsize=10)
    axes[2].set_ylabel('NSW (legitimate agents)', fontsize=10)
    axes[2].set_title('NSW: Cliff at Betrayal → Recovery', fontsize=10, fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    _save(path)


# ----------------------------------------------------------------------------
# Plot 13: Utility trajectories with Moradi blended graph overlay
#          Shows per-agent cumulative utility alongside the blended W_ij
#          trust+similarity edge weight for each agent, so you can see
#          whether the online Moradi graph identifies low-affinity agents at
#          its 20-round decision checkpoints.
# ----------------------------------------------------------------------------

def plot_utility_with_moradi(
        trial: Dict,
        config: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot13_utility_moradi.png')):
    """
    Two-panel plot per attack type.

    Top panel — Cumulative utility trajectories (as in Plot 2) split by
      legitimate (blue) vs malicious (attack-type colour, dashed).

    Bottom panel — Moradi blended W_ij score for each malicious agent
      as seen from the average legitimate observer.  W_ij = σ·T_ij + (1−σ)·S_ij
      is computed cumulatively at every round using the β matrix and the
      report history accumulated so far.  A falling W_ij for a malicious
      agent means the Moradi graph is beginning to isolate it.

    Separate vertical markers show the first Algorithm 1 flag and the first
    shadow Moradi flag. Moradi decisions occur only at the configured interval
    and do not affect allocation in the default configuration.
    """
    hist   = trial['history']
    sim    = trial['sim']
    T      = len(hist)
    N_l    = config.n_legitimate
    N_m    = config.n_malicious
    N      = N_l + N_m
    rounds = np.arange(T)

    at_col = ATTACK_COLOURS.get(config.attack_type, '#333333')
    at_lbl = ATTACK_LABELS.get(config.attack_type, config.attack_type.value)

    # ── Utility array ──────────────────────────────────────────────────────
    utils = np.array([row['utilities'] for row in hist])  # (T, N)

    # ── Moradi W scores per round ──────────────────────────────────────────
    # Rebuild W cumulatively at every round so the plotted Moradi trajectory
    # matches the documented trust+similarity graph rather than an interpolation.
    moradi_scores = np.zeros((T, N_m))
    report_matrix = np.stack(sim.hist_reports, axis=1) if sim.hist_reports else np.zeros((N, 0))
    for t_idx, row in enumerate(hist):
        W_t = build_blended_matrix(
            beta_matrix=row['beta_matrix'],
            hist_reports=report_matrix[:, :t_idx + 1],
            n_agents=N,
            sigma=config.moradi_sigma,
        )
        for k, m_idx in enumerate(range(N_l, N)):
            moradi_scores[t_idx, k] = float(np.mean(W_t[:N_l, m_idx]))

    # First malicious flag from each detector (zero-based history index).
    alg1_onset = T
    moradi_onset = T
    for row in hist:
        malicious_ids = set(range(N_l, N))
        if alg1_onset == T and row['alg1_detected'] & malicious_ids:
            alg1_onset = row['t']
        if moradi_onset == T and row['moradi_detected'] & malicious_ids:
            moradi_onset = row['t']

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f'Utility Trajectories with Moradi Blended Graph Check  |  '
        f'{at_lbl}  |  α={config.allocation_alpha:.2f}  '
        f'σ={config.moradi_sigma:.2f}  |  {config.scenario_label}',
        fontsize=10, fontweight='bold')

    # Top: utility
    ax = axes[0]
    for i in range(N_l):
        ax.plot(rounds, utils[:, i], color='#1E88E5', lw=1.0, alpha=0.50)
    for k, m in enumerate(range(N_l, N)):
        ax.plot(rounds, utils[:, m], color=at_col, lw=1.5, alpha=0.80,
                ls='--', label=f'M{k} (malicious)' if k == 0 else '')
    ax.plot([], [], color='#1E88E5', lw=2,
            label=f'Legitimate agents ({N_l})')
    ax.plot([], [], color=at_col, lw=2, ls='--',
            label=f'Malicious agents ({N_m}) — {at_lbl}')
    if alg1_onset < T:
        ax.axvline(alg1_onset, color='black', ls=':', lw=1.5,
                   alpha=0.7, label=f'Algorithm 1 onset (round {alg1_onset + 1})')
    if moradi_onset < T:
        ax.axvline(moradi_onset, color='#8E24AA', ls='-.', lw=1.5,
                   alpha=0.8, label=f'Shadow Moradi onset (round {moradi_onset + 1})')
    ax.set_ylabel('Cumulative Utility', fontsize=10)
    ax.set_title('Per-Agent Cumulative Utility', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Bottom: Moradi W score
    ax2 = axes[1]
    mal_colours = ['#E53935', '#FB8C00', '#8E24AA', '#1E88E5']
    for k in range(N_m):
        col = mal_colours[k % len(mal_colours)]
        ax2.plot(rounds, moradi_scores[:, k], color=col, lw=2,
                 label=f'M{k} Moradi W score')
    ax2.axhline(0.5, color='gray', ls='--', lw=1.2, alpha=0.7,
                label='Neutral W = 0.5')
    if alg1_onset < T:
        ax2.axvline(alg1_onset, color='black', ls=':', lw=1.5,
                    alpha=0.7, label=f'Algorithm 1 onset (round {alg1_onset + 1})')
    if moradi_onset < T:
        ax2.axvline(moradi_onset, color='#8E24AA', ls='-.', lw=1.5,
                    alpha=0.8, label=f'Shadow Moradi onset (round {moradi_onset + 1})')
    ax2.set_xlabel('Round', fontsize=10)
    ax2.set_ylabel('Moradi W_ij  (avg over legit observers)', fontsize=10)
    ax2.set_title(
        'Moradi Blended Graph Score per Malicious Agent  '
        f'(online decision every {config.moradi_decision_interval} rounds)',
        fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)

    _save(path)


# ----------------------------------------------------------------------------
# Plot 14: Attack rate graph
#          Shows the empirical round-by-round attack rate for each malicious
#          agent type — how often they actually attacked vs how often they
#          could have.  Uses a rolling window average so burst and mimicry
#          patterns are visible rather than noisy per-round binary values.
# ----------------------------------------------------------------------------

def plot_attack_rates(
        results: Dict,
        config: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot14_attack_rates.png')):
    """
    Three-panel attack rate analysis across all four attack types.

    Left panel — Rolling mean attack rate per attack type (window=15 rounds),
      averaged across all 25 trials.  Shows the dynamic pattern of when and
      how often each type attacks.  Byzantine = flat line; Burst = periodic
      spikes; Reputation Poison = near-zero then step up; Trust Mimicry =
      variable, self-regulated curve.

    Centre panel — Cumulative attack fraction over time (total attacks so far
      / total rounds so far).  Converges to the empirical attack rate.
      Useful for showing that Trust Mimicry and Reputation Poison Phase 2
      achieve nearly the same long-run rate as Byzantine but with a different
      distribution.

    Right panel — Final empirical attack rate summary (bar + 95% CI) per
      attack type, separated into Phase 1 (rounds 0–49) and Phase 2
      (rounds 50–299) for all types.  Highlights that Reputation Poison
      Phase 1 is near-zero while Phase 2 matches Trust Mimicry.
    """
    n_r    = config.n_rounds
    rounds = np.arange(n_r)
    window = 15   # rolling mean window for left panel
    focused = [AttackType.BYZANTINE, AttackType.BURST,
               AttackType.REPUTATION_POISON, AttackType.TRUST_MIMICRY]

    warmup = config.poison_warmup_rounds

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Attack Rate Analysis — Four Focused Types  |  '
        f'{n_r} rounds  |  {config.n_trials} trials  '
        f'(rolling window = {window} rounds)',
        fontsize=11, fontweight='bold')

    # ── Precompute round-level attack rate per trial per type ──────────────
    # n_attacking / n_malicious each round
    rate_data = {}
    for at in focused:
        trials = results[at]['trials']
        n_t    = len(trials)
        data   = np.zeros((n_t, n_r))
        for k, tr in enumerate(trials):
            for row in tr['history']:
                t   = row['t']
                data[k, t] = row['n_attacking'] / max(config.n_malicious, 1)
        rate_data[at] = data

    # ── Left: rolling mean attack rate ────────────────────────────────────
    ax = axes[0]
    for at in focused:
        data = rate_data[at]
        data_roll = np.array([rolling_mean_edge(row, window) for row in data])
        mu_roll = data_roll.mean(0)
        err  = ci95(data_roll)
        col  = ATTACK_COLOURS[at]
        lbl  = ATTACK_LABELS[at]
        ax.plot(rounds, mu_roll, color=col, lw=2, label=lbl)
        ax.fill_between(rounds,
                        np.clip(mu_roll - err, 0, 1),
                        np.clip(mu_roll + err, 0, 1),
                        alpha=0.15, color=col)

    ax.axvline(warmup, color='gray', ls=':', lw=1.5, alpha=0.7,
               label=f'Poison warmup end (t={warmup})')
    # Mark coalition sync rounds for Burst
    for t_sync in range(0, n_r, config.coalition_sync_every):
        ax.axvline(t_sync, color='#FB8C00', ls=':', lw=0.5, alpha=0.18)

    ax.set_xlabel('Round', fontsize=10)
    ax.set_ylabel('Attack Rate (rolling mean)', fontsize=10)
    ax.set_title(f'Round-by-Round Attack Rate\n(rolling window = {window} rounds)',
                 fontsize=10, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Centre: cumulative attack fraction ────────────────────────────────
    ax2 = axes[1]
    for at in focused:
        data   = rate_data[at]
        mu     = data.mean(0)
        cumsum = np.cumsum(mu) / (rounds + 1)
        err    = ci95(np.cumsum(data, axis=1) /
                      (rounds + 1)[np.newaxis, :])
        col    = ATTACK_COLOURS[at]
        ax2.plot(rounds, cumsum, color=col, lw=2, label=ATTACK_LABELS[at])
        ax2.fill_between(rounds,
                         np.clip(cumsum - err, 0, 1),
                         np.clip(cumsum + err, 0, 1),
                         alpha=0.15, color=col)

    ax2.axvline(warmup, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax2.axhline(config.base_attack_prob, color='black', ls='--', lw=1.2,
                alpha=0.5, label=f'Base prob = {config.base_attack_prob:.2f}')
    ax2.set_xlabel('Round', fontsize=10)
    ax2.set_ylabel('Cumulative Attack Fraction', fontsize=10)
    ax2.set_title('Cumulative Attack Rate Over Time',
                  fontsize=10, fontweight='bold')
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Right: Phase 1 vs Phase 2 empirical rates ─────────────────────────
    ax3 = axes[2]
    x = np.arange(len(focused))
    w = 0.35

    phase1_means, phase1_errs = [], []
    phase2_means, phase2_errs = [], []

    for at in focused:
        data = rate_data[at]            # (n_trials, n_rounds)
        p1   = data[:, :warmup].mean(axis=1)         # per-trial phase 1 mean
        p2   = data[:, warmup:].mean(axis=1)         # per-trial phase 2 mean
        phase1_means.append(float(p1.mean()))
        phase1_errs.append(float(1.96 * p1.std() / math.sqrt(max(len(p1), 1))))
        phase2_means.append(float(p2.mean()))
        phase2_errs.append(float(1.96 * p2.std() / math.sqrt(max(len(p2), 1))))

    bars1 = ax3.bar(x - w/2, phase1_means, width=w, color='#90CAF9',
                    alpha=0.88, edgecolor='white', linewidth=1.2,
                    label=f'Phase 1  (t < {warmup})')
    bars2 = ax3.bar(x + w/2, phase2_means, width=w, color='#EF9A9A',
                    alpha=0.88, edgecolor='white', linewidth=1.2,
                    label=f'Phase 2  (t ≥ {warmup})')

    ax3.errorbar(x - w/2, phase1_means, yerr=phase1_errs,
                 fmt='none', color='#1565C0', capsize=5, lw=1.5)
    ax3.errorbar(x + w/2, phase2_means, yerr=phase2_errs,
                 fmt='none', color='#B71C1C', capsize=5, lw=1.5)

    # Colour-coded x-tick labels
    ax3.set_xticks(x)
    ax3.set_xticklabels([ATTACK_LABELS[at] for at in focused], fontsize=10)
    for tick, at in zip(ax3.get_xticklabels(), focused):
        tick.set_color(ATTACK_COLOURS[at])

    # Annotate bar values
    for xi, (v1, v2) in enumerate(zip(phase1_means, phase2_means)):
        ax3.text(xi - w/2, v1 + 0.02, f'{v1:.2f}', ha='center',
                 va='bottom', fontsize=9, color='#1565C0')
        ax3.text(xi + w/2, v2 + 0.02, f'{v2:.2f}', ha='center',
                 va='bottom', fontsize=9, color='#B71C1C')

    ax3.set_ylabel('Mean Attack Rate', fontsize=10)
    ax3.set_title('Phase 1 vs Phase 2 Attack Rate\n(95% CI error bars)',
                  fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 1.10)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    _save(path)


# ----------------------------------------------------------------------------
# Plot 16: Shadow online Moradi checkpoints
# ----------------------------------------------------------------------------

def plot_online_moradi_checkpoints(
        results: Dict,
        config: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot16_online_moradi.png')):
    """Compare continuous Algorithm 1 with non-controlling Moradi decisions.

    Moradi is recomputed only at rounds 20, 40, 60, ... and its most recent
    decision is held between checkpoints.  These curves are shadow metrics:
    Algorithm 1 remains the allocation gate in the default experiment.
    """
    focused = [AttackType.BYZANTINE, AttackType.BURST,
               AttackType.REPUTATION_POISON, AttackType.TRUST_MIMICRY]
    rounds = np.arange(1, config.n_rounds + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)

    for ax, attack_type in zip(axes.flat, focused):
        trials = results[attack_type]['trials']
        alg1_dr = np.array([
            [row['alg1_detection_rate'] for row in trial['history']]
            for trial in trials
        ])
        moradi_dr = np.array([
            [row['moradi_detection_rate'] for row in trial['history']]
            for trial in trials
        ])
        moradi_fpr = np.array([
            [row['moradi_fp_rate'] for row in trial['history']]
            for trial in trials
        ])

        alg1_mean = alg1_dr.mean(axis=0)
        moradi_mean = moradi_dr.mean(axis=0)
        fpr_mean = moradi_fpr.mean(axis=0)
        ax.plot(rounds, alg1_mean, color='#1E88E5', lw=1.8,
                label='Algorithm 1 DR')
        ax.plot(rounds, moradi_mean, color='#8E24AA', lw=2.2,
                drawstyle='steps-post', label='Shadow Moradi DR')
        ax.plot(rounds, fpr_mean, color='#E53935', lw=1.6, ls=':',
                drawstyle='steps-post', label='Shadow Moradi FPR')
        ax.fill_between(
            rounds,
            np.clip(moradi_mean - ci95(moradi_dr), 0, 1),
            np.clip(moradi_mean + ci95(moradi_dr), 0, 1),
            step='post', color='#8E24AA', alpha=0.14,
        )
        ax.set_title(ATTACK_LABELS[attack_type], fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    for ax in axes[:, 0]:
        ax.set_ylabel('Rate', fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel('Round', fontsize=10)
    axes[0, 0].legend(fontsize=9, loc='lower right')
    fig.suptitle(
        'Shadow Online Moradi Evaluation — Decisions Every '
        f'{config.moradi_decision_interval} Rounds\n'
        'Moradi predictions are measured but do not control allocation',
        fontsize=12, fontweight='bold',
    )
    _save(path)


# ----------------------------------------------------------------------------
# Plot 15: Prediction scenario comparison (Banerjee NSW-with-predictions)
# ----------------------------------------------------------------------------

def plot_prediction_scenario_comparison(
        base: ExperimentConfig,
        path: str = str(OUTPUT_DIR / 'plot15_prediction_scenarios.png')):
    """
    Banerjee-side stress test: keep the focused attack model fixed and vary
    prediction quality. This is intentionally retained even though the main
    professor revision focuses on attacks, because prediction quality is part
    of the NSW-with-predictions mechanism being integrated.
    """
    n_trials = min(base.n_trials, 8)
    labels = [s.replace('_', '\n') for s in PREDICTION_SCENARIOS]
    x = np.arange(len(PREDICTION_SCENARIOS))

    metrics = {
        'NSW Ratio': [],
        'Min Utility': [],
        'Detection Rate': [],
    }
    errs = {k: [] for k in metrics}

    for scenario in PREDICTION_SCENARIOS:
        cfg = replace(base, n_trials=n_trials, prediction_scenario=scenario)
        trials = run_trials(cfg, verbose=False)
        values = {
            'NSW Ratio': np.array([r['final_nsw_ratio'] for r in trials]),
            'Min Utility': np.array([r['final_min_util'] for r in trials]),
            'Detection Rate': np.array([r['final_dr'] for r in trials]),
        }
        for key, vals in values.items():
            metrics[key].append(float(vals.mean()))
            errs[key].append(float(1.96 * vals.std() / math.sqrt(max(len(vals), 1))))
        print(f"  pred={scenario:22s}  "
              f"NSW_ratio={metrics['NSW Ratio'][-1]:.3f}  "
              f"min_util={metrics['Min Utility'][-1]:.4f}  "
              f"DR={metrics['Detection Rate'][-1]:.0%}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Prediction Scenario Stress Test — {ATTACK_LABELS[base.attack_type]}\n'
        f'{n_trials} trials per scenario  |  α={base.allocation_alpha:.2f}  '
        f'σ={base.moradi_sigma:.2f}',
        fontsize=11, fontweight='bold')

    colours = ['#8E24AA', '#43A047', '#1E88E5']
    for ax, (key, vals), col in zip(axes, metrics.items(), colours):
        ax.plot(x, vals, 'o-', color=col, lw=2.2, ms=6)
        ax.errorbar(x, vals, yerr=errs[key], fmt='none',
                    color=col, capsize=5, lw=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(key, fontsize=10)
        ax.set_title(key, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    _save(path)


# =============================================================================
# MAIN
# =============================================================================

def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the trust-aware NSW simulator locally. The default reproduces "
            "the full experiment suite; use --quick for a short local check."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated plots (default: %(default)s).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Rounds per trial (default: 200, or 20 with --quick).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Trials per experiment cell (default: 10, or 1 with --quick).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a small local check (20 rounds, 1 trial by default) and skip "
            "the expensive parameter sweeps."
        ),
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run the four focused attacks and derived plots, but skip repeated sweeps.",
    )
    args = parser.parse_args(argv)
    args.rounds = args.rounds if args.rounds is not None else (20 if args.quick else 200)
    args.trials = args.trials if args.trials is not None else (1 if args.quick else 10)
    if args.rounds <= 0:
        parser.error("--rounds must be greater than zero")
    if args.trials <= 0:
        parser.error("--trials must be greater than zero")
    return args


def main(argv: Optional[List[str]] = None):
    args = _parse_cli_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_expensive_sweeps = not (args.quick or args.core_only)

    print("=" * 72)
    print("NSW TRUSTWORTHY CONSENSUS — Professor Revision")
    print("  [R1] Line plots   [R2] 4 focused attacks  [R3] Poison redesign")
    print("  [R4] Compound     [R5] Full information   [R6] Deep analysis")
    print("  [R7] Higher rates [R8] 3-part structure   [R9] Expanded docs")
    print(f"  Local output: {output_dir}")
    print(f"  Run size: {args.rounds} rounds x {args.trials} trial(s)")
    if not run_expensive_sweeps:
        print("  Mode: core-only (expensive parameter sweeps skipped)")
    print("=" * 72)

    # Base configuration — full information, higher attack rates, 25 trials
    base_cfg = ExperimentConfig(
        n_legitimate      = 10,
        n_malicious       = 4,
        n_rounds          = args.rounds,
        n_trials          = args.trials,
        attack_type       = AttackType.BYZANTINE,
        base_attack_prob  = 0.75,
        trust_legitimate          = 0.63,
        trust_malicious_attack    = 0.52,
        trust_malicious_no_attack = 0.63,
        trust_std                 = 0.14,
        xi_0              = 0.25,
        allocation_alpha  = 0.50,
        moradi_sigma      = 0.60,
        coalition_sync_every = 10,
        # Preserve the 50-round research default while keeping short local
        # runs meaningful (both poison phases must fit inside the horizon).
        poison_warmup_rounds = min(50, max(1, args.rounds // 4)),
        connectivity      = 1.0,
        prediction_scenario = "mild_noise",
        scenario_label    = "professor revision",
    )
    print(
        f"  Online gate: {base_cfg.online_detector} "
        f"(decision every {base_cfg.moradi_decision_interval} rounds)"
    )

    # ── Phase 1: Run all four focused attack types ──────────────────────────
    print(
        f"\n[Phase 1] Running all four focused attack types "
        f"({args.trials} trial(s) each) ..."
    )
    results = compare_focused_types(base_cfg)

    # ── Plots 1–5: main diagnostics ─────────────────────────────────────────
    print("\n[Plots 1-5] Main diagnostic plots ...")
    plot_main_results_all_types(
        results, base_cfg, path=str(output_dir / "plot1_main_results.png")
    )

    # Single-attack utility trajectory and heatmap (Byzantine as illustrative)
    byz_trials = results[AttackType.BYZANTINE]['trials']
    plot_utility_trajectories(
        byz_trials[0],
        replace(base_cfg, attack_type=AttackType.BYZANTINE),
        path=str(output_dir / "plot2_utility_traj.png"),
    )
    plot_allocation_heatmap(
        byz_trials[0],
        replace(base_cfg, attack_type=AttackType.BYZANTINE),
        path=str(output_dir / "plot3_alloc_heatmap.png"),
    )
    plot_misclassification_all_types(
        results, base_cfg, path=str(output_dir / "plot4_misclass.png")
    )
    plot_attack_summary_lines(
        results, base_cfg, path=str(output_dir / "plot5_attack_comparison.png")
    )

    # ── Plot 6: β trajectory analysis ──────────────────────────────────────
    print("\n[Plot 6] β accumulation and detection threshold ...")
    plot_beta_trajectories(
        results, base_cfg, path=str(output_dir / "plot6_beta_trajectories.png")
    )

    if run_expensive_sweeps:
        # ── Plot 7: NSW ratio sensitivity ──────────────────────────────────
        print("\n[Plot 7] NSW ratio sensitivity to adversary count ...")
        plot_nsw_ratio_sensitivity(
            base_cfg, path=str(output_dir / "plot7_nsw_ratio.png")
        )

        # ── Plot 8: Alpha sweep ─────────────────────────────────────────────
        print("\n[Plot 8] Alpha sweep ...")
        plot_alpha_sweep(
            base_cfg, path=str(output_dir / "plot8_alpha_sweep.png")
        )

    # ── Plot 9: Detector benchmark ─────────────────────────────────────────
    print("\n[Plot 9] Offline and online detector benchmark ...")
    plot_three_way_detector(
        results, path=str(output_dir / "plot9_three_way_detector.png")
    )

    if run_expensive_sweeps:
        # ── Plot 10: Sigma sweep ────────────────────────────────────────────
        print("\n[Plot 10] Sigma sweep ...")
        plot_sigma_sweep(
            base_cfg, path=str(output_dir / "plot10_sigma_sweep.png")
        )

        # ── Plot 11: Compound attack breakdown [R4] ─────────────────────────
        print("\n[Plot 11] Compound attack sub-type breakdown ...")
        plot_compound_breakdown(
            base_cfg, path=str(output_dir / "plot11_compound_breakdown.png")
        )

        # ── Plot 12: Reputation Poison phase analysis [R3] ──────────────────
        print("\n[Plot 12] Reputation Poison phase analysis ...")
        plot_reputation_poison_phases(
            base_cfg, path=str(output_dir / "plot12_reputation_poison_phases.png")
        )

    # ── Plot 13: Utility trajectories with Moradi blended graph check ────────
    print("\n[Plot 13] Utility trajectories with Moradi blended graph ...")
    for at in [AttackType.BYZANTINE, AttackType.BURST,
               AttackType.REPUTATION_POISON, AttackType.TRUST_MIMICRY]:
        trials = results[at]['trials']
        plot_utility_with_moradi(
            trials[0],
            replace(base_cfg, attack_type=at),
            path=str(output_dir / f'plot13_utility_moradi_{at.value}.png'))

    # ── Plot 14: Attack rate analysis ─────────────────────────────────────────
    print("\n[Plot 14] Attack rate analysis ...")
    plot_attack_rates(
        results, base_cfg, path=str(output_dir / "plot14_attack_rates.png")
    )

    # ── Plot 16: Online Moradi shadow evaluation ───────────────────────────
    print("\n[Plot 16] Shadow online Moradi decisions ...")
    plot_online_moradi_checkpoints(
        results, base_cfg, path=str(output_dir / "plot16_online_moradi.png")
    )

    if run_expensive_sweeps:
        # ── Plot 15: Prediction scenario stress test ─────────────────────────
        print("\n[Plot 15] Prediction scenario comparison ...")
        plot_prediction_scenario_comparison(
            base_cfg, path=str(output_dir / "plot15_prediction_scenarios.png")
        )

    print("\n" + "=" * 72)
    print("All scenarios complete. Output files:")
    core_outputs = [
        "plot1_main_results.png        — all 4 types, 4 metrics, 95% CI bands",
        "plot2_utility_traj.png        — per-agent cumulative utility (Byzantine)",
        "plot3_alloc_heatmap.png       — allocation heatmap (Byzantine)",
        "plot4_misclass.png            — FPR and FNR for all 4 types",
        "plot5_attack_comparison.png   — summary line plot with 95% CI",
        "plot6_beta_trajectories.png   — β accumulation + ξ_t per attack type",
        "plot9_three_way_detector.png  — Alg1, Spectral, final + online Moradi F1",
        "plot13_utility_moradi_*.png   — Utility + Moradi W score per type",
        "plot14_attack_rates.png       — Attack rate rolling mean + phases",
        "plot16_online_moradi.png      — shadow Moradi DR/FPR every 20 rounds",
    ]
    sweep_outputs = [
        "plot7_nsw_ratio.png           — NSW ratio vs adversary count (4 types)",
        "plot8_alpha_sweep.png         — α sweep for all 4 types",
        "plot10_sigma_sweep.png        — σ sweep for all 4 types",
        "plot11_compound_breakdown.png — Compound sub-type analysis [R4]",
        "plot12_reputation_poison_phases.png — Poison phase analysis [R3]",
        "plot15_prediction_scenarios.png — Prediction quality stress test",
    ]
    outputs = core_outputs + (sweep_outputs if run_expensive_sweeps else [])
    for i, name in enumerate(outputs, 1):
        print(f"  {i:2d}. {output_dir / name.split()[0]}")
    print("=" * 72)


if __name__ == "__main__":
    main()
