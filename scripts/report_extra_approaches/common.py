from typing import Dict, List, Set, Tuple

import numpy as np

from NSW_Moradi_Final import (
    AttackType,
    NSWTrustworthySim,
    build_blended_matrix,
    compute_nsw,
    implicit_trust_score,
    merge_small_clusters,
    sparsest_subgraph_detection,
)


def rate(count: int, total: int) -> float:
    return float(count / max(total, 1))


def normalize_budget(alloc: np.ndarray, budget: float = 1.0) -> np.ndarray:
    z = np.maximum(np.asarray(alloc, dtype=float), 0.0)
    total = float(z.sum())
    if total <= 1e-12:
        z[:] = budget / max(len(z), 1)
        return z
    return z * (budget / total)


class TrustModeStepMixin:
    """Reuse each allocator, but swap the trust gate used before allocation."""

    trust_mode = "algorithm1"
    approach_name = "trust_mode_variant"

    def __init__(self, *args, trust_mode: str = "algorithm1", **kwargs):
        self.trust_mode = trust_mode
        super().__init__(*args, **kwargs)

    def _algorithm1_trusted(self, detected: Set[int]) -> List[int]:
        trusted = [j for j in range(self.N) if j not in detected]

        if self.t >= self.cfg.moradi_bootstrap_rounds and len(self.hist_reports) >= 3:
            demote = set()
            for j in list(trusted):
                obs = int(self.observation_counts[self.legit_ids, j].sum())
                if obs < self.cfg.moradi_bootstrap_rounds:
                    refs = [k for k in trusted if k != j]
                    if refs:
                        avg_imp = float(np.mean([
                            implicit_trust_score(
                                j, k, self.hist_reports,
                                self.cfg.moradi_bootstrap_rounds,
                            )
                            for k in refs
                        ]))
                        if avg_imp < 0.35:
                            demote.add(j)
            trusted = [j for j in trusted if j not in demote]

        return merge_small_clusters(
            trusted_ids=trusted,
            beta_matrix=self.beta_matrix,
            all_agent_ids=list(range(self.N)),
            merge_threshold=self.cfg.moradi_merge_threshold,
        )

    def _sparse_online_detect(self, algorithm1_detected: Set[int]) -> Tuple[Set[int], str]:
        if len(self.hist_reports) < 3:
            return set(algorithm1_detected), "algorithm1_cold_start"

        W = build_blended_matrix(
            beta_matrix=self.beta_matrix,
            hist_reports=self.hist_reports,
            n_agents=self.N,
            sigma=self.cfg.moradi_sigma,
        )
        if W.sum() < 1e-10:
            return set(algorithm1_detected), "algorithm1_empty_graph"

        labels = sparsest_subgraph_detection(
            W=W,
            n_agents=self.N,
            n_malicious=self.cfg.n_malicious,
            epsilon=self.cfg.sparsest_epsilon,
        )
        return {i for i, label in enumerate(labels) if int(label) == 1}, "sparse_online"

    def _choose_trust_gate(self, algorithm1_detected: Set[int]) -> Tuple[Set[int], List[int], str]:
        if self.trust_mode == "no_trust":
            return set(), list(range(self.N)), "no_trust"
        if self.trust_mode == "oracle":
            detected = set(self.mal_ids)
            return detected, list(self.legit_ids), "oracle"
        if self.trust_mode == "sparse_online":
            detected, source = self._sparse_online_detect(algorithm1_detected)
            trusted = [j for j in range(self.N) if j not in detected]
            return detected, trusted, source

        detected = set(algorithm1_detected)
        return detected, self._algorithm1_trusted(detected), "algorithm1"

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

        algorithm1_detected = self._majority_vote_detect()

        self.beta_matrix.fill(0.0)
        for i in self.legit_ids:
            for j, beta in self.detectors[i].beta.items():
                self.beta_matrix[i, j] = beta

        detected, trusted, gate_source = self._choose_trust_gate(algorithm1_detected)

        alloc, set_aside_alloc, greedy_alloc, extra = self.allocate_current_round(
            reports=reports,
            trusted=trusted,
        )

        self.utilities += self.true_vals[:, t] * alloc
        self.u_greedy += self.true_vals[:, t] * greedy_alloc
        self.hist_reports.append(reports.copy())

        mal_set = set(self.mal_ids)
        legit_set = set(self.legit_ids)
        tp = len(detected & mal_set)
        fp = len(detected & legit_set)
        fn = len(mal_set - detected)
        alg_tp = len(algorithm1_detected & mal_set)
        alg_fp = len(algorithm1_detected & legit_set)
        alg_fn = len(mal_set - algorithm1_detected)

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

        row = dict(
            t=t,
            tp=tp,
            fp=fp,
            fn=fn,
            approach=self.approach_name,
            trust_mode=self.trust_mode,
            trust_gate_source=gate_source,
            detection_rate=rate(tp, len(self.mal_ids)),
            fp_rate=rate(fp, len(self.legit_ids)),
            fn_rate=rate(fn, len(self.mal_ids)),
            algorithm1_detection_rate=rate(alg_tp, len(self.mal_ids)),
            algorithm1_fp_rate=rate(alg_fp, len(self.legit_ids)),
            algorithm1_fn_rate=rate(alg_fn, len(self.mal_ids)),
            n_attacking=sum(attacks.values()),
            detected=detected,
            algorithm1_detected=algorithm1_detected,
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
            beta_mal_gap_avg=float(np.mean(mal_gap_values)) if mal_gap_values else 0.0,
            beta_legit_gap_avg=float(np.mean(legit_gap_values)) if legit_gap_values else 0.0,
            xi_t=xi_t,
            sync_signal=sync_signal,
            coalition_victim=coalition_victim,
        )
        row.update(extra)
        self.history.append(row)
        self.t += 1
        return row
