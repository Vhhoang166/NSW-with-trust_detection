"""
Shared simulation glue for allocator variants.

The main simulator in NSW_Moradi_Final.py is intentionally left intact.  Each
new approach subclasses NSWTrustworthySim and mixes in VariantStepMixin, which
keeps the same attack, trust-update, Moradi M3/M4, utility, and metric logic,
but delegates the allocation decision to allocate_current_round(...).
"""

from typing import Dict, List, Tuple

import numpy as np

from NSW_Moradi_Final import (
    AttackType,
    compute_nsw,
    implicit_trust_score,
    merge_small_clusters,
)


class VariantStepMixin:
    """Override only the allocation stage of NSWTrustworthySim.step()."""

    approach_name = "variant"

    def allocate_current_round(
        self,
        reports: np.ndarray,
        trusted: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Return (alloc, set_aside_alloc, greedy_alloc, metadata).

        Subclasses implement this.  The two component arrays are kept so the
        existing plotting/evaluation code can still inspect allocations.  For
        approaches without a set-aside/greedy split, return zeros for whichever
        component does not apply.
        """
        raise NotImplementedError

    def step(self) -> Dict:
        t = self.t
        xi_t = self._current_xi_t()

        detected_pre = self._majority_vote_detect() if t > 0 else set()

        reports = self.true_vals[:, t].copy()
        attacks: Dict[int, bool] = {}
        sync_signal = self.coalition.get_signal(t)

        for m in self.mal_ids:
            gap = self._global_beta_gap(m)
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

        detected = self._majority_vote_detect()
        trusted = [j for j in range(self.N) if j not in detected]

        self.beta_matrix.fill(0.0)
        for i in self.legit_ids:
            for j, b in self.detectors[i].beta.items():
                self.beta_matrix[i, j] = b

        if t >= self.cfg.moradi_bootstrap_rounds and len(self.hist_reports) >= 3:
            demote = set()
            for j in list(trusted):
                obs = int(self.observation_counts[self.legit_ids, j].sum())
                if obs < self.cfg.moradi_bootstrap_rounds:
                    refs = [k for k in trusted if k != j]
                    if refs:
                        avg_imp = float(np.mean([
                            implicit_trust_score(
                                j, k, self.hist_reports,
                                self.cfg.moradi_bootstrap_rounds)
                            for k in refs
                        ]))
                        if avg_imp < 0.35:
                            demote.add(j)
            trusted = [j for j in trusted if j not in demote]

        trusted = merge_small_clusters(
            trusted_ids=trusted,
            beta_matrix=self.beta_matrix,
            all_agent_ids=list(range(self.N)),
            merge_threshold=self.cfg.moradi_merge_threshold,
        )

        alloc, set_aside_alloc, greedy_alloc, extra = self.allocate_current_round(
            reports=reports,
            trusted=trusted,
        )

        self.utilities += self.true_vals[:, t] * alloc
        self.u_greedy += self.true_vals[:, t] * greedy_alloc
        self.hist_reports.append(reports.copy())

        tp = len(detected & set(self.mal_ids))
        fp = len(detected & set(self.legit_ids))
        fn = len(set(self.mal_ids) - detected)

        nsw_legit = compute_nsw(self.utilities[self.legit_ids])
        nsw_opt = self._nsw_optimal()
        nsw_ratio = nsw_legit / nsw_opt if nsw_opt > 0 else 0.0
        legit_utils = self.utilities[self.legit_ids]
        min_util = float(legit_utils.min()) if len(legit_utils) else 0.0
        fairness_gap = (
            float(legit_utils.max() - legit_utils.min())
            if len(legit_utils) > 1 else 0.0
        )

        beta_legit = float(np.mean([
            np.mean(list(self.detectors[i].beta.values()))
            for i in self.legit_ids
        ]))
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
            approach=self.approach_name,
            detection_rate=tp / max(len(self.mal_ids), 1),
            fp_rate=fp / max(len(self.legit_ids), 1),
            fn_rate=fn / max(len(self.mal_ids), 1),
            n_attacking=sum(attacks.values()),
            detected=detected,
            trusted=list(trusted),
            alloc=alloc.copy(),
            set_aside_alloc=set_aside_alloc.copy(),
            greedy_alloc=greedy_alloc.copy(),
            utilities=self.utilities.copy(),
            beta_matrix=self.beta_matrix.copy(),
            nsw_legit=nsw_legit,
            nsw_opt=nsw_opt,
            nsw_ratio=nsw_ratio,
            min_util=min_util,
            fairness_gap=fairness_gap,
            beta_legit_avg=beta_legit,
            beta_mal_gap_avg=beta_mal_gap_avg,
            beta_legit_gap_avg=beta_legit_gap_avg,
            xi_t=xi_t,
            sync_signal=sync_signal,
            coalition_victim=coalition_victim,
        )
        row.update(extra)
        self.history.append(row)
        self.t += 1
        return row


def normalize_budget(alloc: np.ndarray, budget: float = 1.0) -> np.ndarray:
    """Project a nonnegative allocation vector onto the round budget."""
    alloc = np.maximum(np.asarray(alloc, dtype=float), 0.0)
    total = float(alloc.sum())
    if total <= 1e-12:
        alloc[:] = budget / len(alloc)
        return alloc
    return alloc * (budget / total)

