"""
Approach 1: Adaptive trust-aware alpha.

This is the report's recommended next step.  It keeps the trusted set-aside
greedy allocator from NSW_Moradi_Final.py, but replaces the fixed alpha with
alpha_t.  The controller increases alpha_t when trust votes or prediction
signals look unstable, making the mechanism more conservative under attack.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from approach_common import VariantStepMixin
from NSW_Moradi_Final import NSWTrustworthySim, trusted_online_allocation


@dataclass
class AdaptiveAlphaParams:
    alpha_min: float = 0.20
    alpha_max: float = 0.85
    trust_weight: float = 0.65
    prediction_weight: float = 0.35
    smoothing: float = 0.80
    mismatch_scale: float = 2.0


class AdaptiveAlphaSim(VariantStepMixin, NSWTrustworthySim):
    approach_name = "adaptive_alpha"

    def __init__(self, *args, adaptive_params: AdaptiveAlphaParams = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_params = adaptive_params or AdaptiveAlphaParams()
        self.alpha_t = float(self.cfg.allocation_alpha)

    def _trust_instability(self) -> float:
        """Fractional detector disagreement using only observable votes."""
        if not self.legit_ids:
            return 0.0
        votes = np.zeros(self.N)
        for i in self.legit_ids:
            _, detected = self.detectors[i].classify(self.t)
            for j in detected:
                votes[j] += 1.0
        p = votes / max(len(self.legit_ids), 1)
        # p*(1-p) peaks at 0.25 when detectors split 50/50.
        disagreement = 4.0 * p * (1.0 - p)
        return float(np.clip(np.mean(disagreement), 0.0, 1.0))

    def _prediction_instability(self, reports: np.ndarray, trusted: List[int]) -> float:
        """Robust mismatch between current reports and per-round predictions."""
        if not trusted:
            return 1.0
        predicted_round = self.V_tilde / max(self.T, 1)
        ratios = np.abs(reports[trusted] - predicted_round[trusted]) / (
            predicted_round[trusted] + 1e-10)
        mismatch = float(np.median(ratios))
        return float(np.clip(mismatch / self.adaptive_params.mismatch_scale, 0.0, 1.0))

    def _choose_alpha(self, reports: np.ndarray, trusted: List[int]) -> Tuple[float, Dict]:
        p = self.adaptive_params
        trust_risk = self._trust_instability()
        prediction_risk = self._prediction_instability(reports, trusted)
        risk = np.clip(
            p.trust_weight * trust_risk + p.prediction_weight * prediction_risk,
            0.0, 1.0,
        )
        raw_alpha = p.alpha_min + (p.alpha_max - p.alpha_min) * risk
        self.alpha_t = p.smoothing * self.alpha_t + (1.0 - p.smoothing) * raw_alpha
        self.alpha_t = float(np.clip(self.alpha_t, p.alpha_min, p.alpha_max))
        return self.alpha_t, {
            "alpha_t": self.alpha_t,
            "trust_risk": trust_risk,
            "prediction_risk": prediction_risk,
            "controller_risk": float(risk),
        }

    def allocate_current_round(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        alpha_t, meta = self._choose_alpha(reports, trusted)
        alloc, set_aside, greedy = trusted_online_allocation(
            reports=reports,
            trusted_ids=trusted,
            V_tilde=self.V_tilde,
            u_greedy=self.u_greedy,
            budget=1.0,
            alpha=alpha_t,
        )
        return alloc, set_aside, greedy, meta


def run_adaptive_alpha_trial(config, seed: int = 42):
    sim = AdaptiveAlphaSim(config, seed=seed)
    return sim.run(verbose=False), sim

