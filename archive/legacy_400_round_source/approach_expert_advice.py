"""
Approach 5: Expert-advice controller over several allocator candidates.

This meta-controller keeps multiple allocation candidates and combines them
with multiplicative weights.  It is useful when no single allocator is best in
both peaceful and adversarial regimes.  Rewards are based on reported value
inside the trusted set, so this remains an online-style controller rather than
peeking at true utilities for its weight update.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from approach_common import VariantStepMixin, normalize_budget
from NSW_Moradi_Final import (
    NSWTrustworthySim,
    trusted_online_allocation,
)


@dataclass
class ExpertAdviceParams:
    eta: float = 0.20
    min_weight: float = 0.05
    defensive_alpha: float = 0.80
    aggressive_alpha: float = 0.20
    gm_rho: float = -0.50


class ExpertAdviceSim(VariantStepMixin, NSWTrustworthySim):
    approach_name = "expert_advice"

    def __init__(self, *args, expert_params: ExpertAdviceParams = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_params = expert_params or ExpertAdviceParams()
        self.expert_names = ["defensive_set_aside", "aggressive_set_aside", "gm_greedy"]
        self.expert_weights = np.ones(len(self.expert_names)) / len(self.expert_names)

    def _gm_candidate(self, reports: np.ndarray, trusted: List[int]) -> np.ndarray:
        p = self.expert_params
        alloc = np.zeros(self.N)
        if not trusted:
            alloc[:] = 1.0 / self.N
            return alloc
        values = np.maximum(reports[trusted], 0.0)
        baseline = np.maximum(self.utilities[trusted], 1e-6)
        marginal = values * np.power(baseline, p.gm_rho - 1.0)
        if marginal.sum() <= 1e-12:
            local = np.ones(len(trusted)) / len(trusted)
        else:
            local = marginal / marginal.sum()
        for idx, i in enumerate(trusted):
            alloc[i] = local[idx]
        return normalize_budget(alloc)

    def _candidate_allocations(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> List[np.ndarray]:
        p = self.expert_params
        defensive, _, _ = trusted_online_allocation(
            reports, trusted, self.V_tilde, self.u_greedy, alpha=p.defensive_alpha)
        aggressive, _, _ = trusted_online_allocation(
            reports, trusted, self.V_tilde, self.u_greedy, alpha=p.aggressive_alpha)
        gm = self._gm_candidate(reports, trusted)
        return [defensive, aggressive, gm]

    def _update_weights(self, candidates: List[np.ndarray], reports: np.ndarray) -> None:
        rewards = np.array([
            float(np.dot(np.maximum(reports, 0.0), alloc))
            for alloc in candidates
        ])
        if rewards.max() > rewards.min():
            rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
        else:
            rewards = np.zeros_like(rewards)
        self.expert_weights *= np.exp(self.expert_params.eta * rewards)
        self.expert_weights = np.maximum(self.expert_weights, self.expert_params.min_weight)
        self.expert_weights /= self.expert_weights.sum()

    def allocate_current_round(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        candidates = self._candidate_allocations(reports, trusted)
        self._update_weights(candidates, reports)
        alloc = np.zeros(self.N)
        for w, candidate in zip(self.expert_weights, candidates):
            alloc += w * candidate
        alloc = normalize_budget(alloc)
        meta = {
            "expert_defensive_weight": float(self.expert_weights[0]),
            "expert_aggressive_weight": float(self.expert_weights[1]),
            "expert_gm_weight": float(self.expert_weights[2]),
        }
        return alloc, np.zeros(self.N), alloc.copy(), meta


def run_expert_advice_trial(config, seed: int = 42):
    sim = ExpertAdviceSim(config, seed=seed)
    return sim.run(verbose=False), sim
