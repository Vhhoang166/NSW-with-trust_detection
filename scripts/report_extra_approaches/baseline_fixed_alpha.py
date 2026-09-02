from typing import Dict, List, Tuple

import numpy as np

from NSW_Moradi_Final import NSWTrustworthySim, trusted_online_allocation

from .common import TrustModeStepMixin


class BaselineFixedAlphaSim(TrustModeStepMixin, NSWTrustworthySim):
    approach_name = "baseline_fixed_alpha"

    def allocate_current_round(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        alloc, set_aside, greedy = trusted_online_allocation(
            reports=reports,
            trusted_ids=trusted,
            V_tilde=self.V_tilde,
            u_greedy=self.u_greedy,
            budget=1.0,
            alpha=self.cfg.allocation_alpha,
        )
        return alloc, set_aside, greedy, {"fixed_alpha": float(self.cfg.allocation_alpha)}


class PureGreedyAlpha0Sim(BaselineFixedAlphaSim):
    approach_name = "pure_greedy_alpha0"


class PureEqualAlpha1Sim(BaselineFixedAlphaSim):
    approach_name = "pure_equal_alpha1"
