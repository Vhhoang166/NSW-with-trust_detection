#!/usr/bin/env python3
"""Frank-Wolfe runner for the Moradi trust-aware NSW experiment.

This module deliberately imports ``NSW_Moradi_Final`` and replaces only its
offline NSW benchmark.  Attack generation, trust detection, online allocation,
metrics, and plots therefore remain identical to the original implementation.

Defaults that differ from the original runner:
  * 10 trials per experiment cell, matching ``NSW_Moradi_Final.py``;
  * 2,000 rounds per trial unless ``--rounds`` is supplied;
  * output is written to ``outputs_frank_wolfe``;
  * the offline benchmark uses Frank-Wolfe with an exact line search.
  * online metrics are exported from an explicit round-0 baseline and then
    every round against the paper's complete-horizon offline benchmark.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import NSW_Moradi_Final as moradi


FW_MAX_ITERATIONS = 1000
FW_TOLERANCE = 1e-6
ONLINE_CHECKPOINT_INTERVAL = 1

# Repeated attack types use the same seed and true valuations.  Cache the
# corresponding offline result so it is solved only once per unique dataset.
_OFFLINE_CACHE: Dict[
    Tuple[Tuple[int, int], bytes],
    Tuple[float, np.ndarray, np.ndarray],
] = {}


def frank_wolfe_offline_nsw(
    v_true: np.ndarray,
    legit_mask: np.ndarray,
    max_iterations: int = FW_MAX_ITERATIONS,
    tolerance: float = FW_TOLERANCE,
    *,
    report: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
    """Approximately maximize offline log-NSW with Frank-Wolfe.

    The linear minimization oracle assigns each round to the legitimate agent
    with the largest current marginal value ``v[i,t] / utility[i]``.  An exact
    one-dimensional line search then chooses the feasible step size.

    The returned Frank-Wolfe gap is an upper bound on the remaining error in
    the sum-log-utility objective.  A result is still a feasible lower bound on
    the true offline optimum if the iteration limit is reached first.
    """
    started = time.perf_counter()
    idx = np.where(np.asarray(legit_mask, dtype=bool))[0]
    values = np.asarray(v_true[idx, :], dtype=float)
    n_agents, n_rounds = values.shape
    if n_agents == 0 or n_rounds == 0:
        raise ValueError("Frank-Wolfe requires at least one agent and one round")

    allocation = np.full((n_agents, n_rounds), 1.0 / n_agents, dtype=float)
    utilities = np.maximum((values * allocation).sum(axis=1), 1e-15)
    rounds = np.arange(n_rounds)
    final_gap = float("inf")
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        marginal_scores = values / utilities[:, None]
        winners = np.argmax(marginal_scores, axis=0)

        oracle_allocation = np.zeros_like(allocation)
        oracle_allocation[winners, rounds] = 1.0
        oracle_utilities = np.bincount(
            winners,
            weights=values[winners, rounds],
            minlength=n_agents,
        ).astype(float)

        # <gradient f(x), s-x>; for concave maximization this bounds f* - f(x).
        final_gap = float(marginal_scores[winners, rounds].sum() - n_agents)
        iterations = iteration
        if final_gap <= tolerance:
            break

        utility_direction = oracle_utilities - utilities

        def directional_derivative(step: float) -> float:
            candidate = np.maximum(
                utilities + step * utility_direction,
                1e-15,
            )
            return float(np.sum(utility_direction / candidate))

        if directional_derivative(0.0) <= 0.0:
            break

        upper = 1.0 - 1e-12
        if directional_derivative(upper) >= 0.0:
            step_size = upper
        else:
            lower = 0.0
            # Exact-enough bisection for the concave one-dimensional objective.
            for _ in range(40):
                midpoint = (lower + upper) / 2.0
                if directional_derivative(midpoint) > 0.0:
                    lower = midpoint
                else:
                    upper = midpoint
            step_size = (lower + upper) / 2.0

        allocation += step_size * (oracle_allocation - allocation)
        utilities = np.maximum(
            utilities + step_size * utility_direction,
            1e-15,
        )

    elapsed = time.perf_counter() - started
    nsw = moradi.compute_nsw(utilities)
    diagnostics = {
        "iterations": float(iterations),
        "final_gap": final_gap,
        "elapsed_seconds": elapsed,
        "converged": float(final_gap <= tolerance),
    }
    if report:
        status = "converged" if final_gap <= tolerance else "iteration limit"
        print(
            "  Frank-Wolfe offline benchmark: "
            f"NSW={nsw:.8f}, iterations={iterations}, "
            f"gap={final_gap:.3e}, time={elapsed:.3f}s ({status})"
        )
    return nsw, utilities, allocation, diagnostics


def cached_frank_wolfe_offline_nsw(
    v_true: np.ndarray,
    legit_mask: np.ndarray,
    *,
    report: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Drop-in cached replacement for ``moradi.offline_optimal_nsw``."""
    idx = np.where(np.asarray(legit_mask, dtype=bool))[0]
    legitimate_values = np.ascontiguousarray(v_true[idx, :], dtype=float)
    key = (legitimate_values.shape, legitimate_values.tobytes())
    if key not in _OFFLINE_CACHE:
        nsw, utilities, allocation, _ = frank_wolfe_offline_nsw(
            v_true,
            legit_mask,
            report=report,
        )
        _OFFLINE_CACHE[key] = (nsw, utilities, allocation)
    return _OFFLINE_CACHE[key]


def _checkpoint_rounds(n_rounds: int, interval: int) -> List[int]:
    """Return round zero plus checkpoints, always including the final round."""
    checkpoints = [0] + list(range(interval, n_rounds + 1, interval))
    if not checkpoints or checkpoints[-1] != n_rounds:
        checkpoints.append(n_rounds)
    return checkpoints


def write_online_checkpoint_outputs(
    results: Dict,
    config: moradi.ExperimentConfig,
    output_dir: Path,
    interval: int = ONLINE_CHECKPOINT_INTERVAL,
) -> Tuple[Path, Path]:
    """Export online progress against the full-information offline benchmark.

    The online metrics include an explicit all-zero row at round 0. The offline
    solver sees the complete valuation sequence before allocation, exactly as
    in the paper, and its result stays fixed from round 0 onward.
    Therefore, the intermediate ``nsw_ratio`` is progress toward the final
    offline optimum. Only the final checkpoint compares equal horizons.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"online_checkpoints_{interval}.csv"
    plot_path = output_dir / f"plot17_online_checkpoints_{interval}.png"

    metric_keys = [
        "nsw_legit",
        "detection_rate",
        "fp_rate",
        "fn_rate",
        "alg1_detection_rate",
        "alg1_fp_rate",
        "moradi_detection_rate",
        "moradi_fp_rate",
        "min_util",
        "fairness_gap",
        "beta_mal_gap_avg",
        "beta_legit_gap_avg",
        "n_attacking",
    ]
    fieldnames = ["attack_type", "checkpoint_round", "trials"]
    for key in metric_keys:
        fieldnames.extend((f"{key}_mean", f"{key}_ci95"))
    fieldnames.extend((
        "offline_nsw_full_horizon_mean",
        "offline_nsw_full_horizon_ci95",
        "nsw_ratio_mean",
        "nsw_ratio_ci95",
    ))
    fieldnames.extend(("attacking_fraction_mean", "attacking_fraction_ci95"))

    checkpoint_rows = []
    for attack_type, result in results.items():
        trials = result["trials"]
        n_rounds = min(len(trial["history"]) for trial in trials)
        for checkpoint in _checkpoint_rounds(n_rounds, interval):
            history_rows = (
                [trial["history"][checkpoint - 1] for trial in trials]
                if checkpoint > 0 else None
            )
            output_row = {
                "attack_type": attack_type.value,
                "checkpoint_round": checkpoint,
                "trials": len(trials),
            }
            for key in metric_keys:
                values = (
                    np.asarray([row[key] for row in history_rows], dtype=float)
                    if history_rows is not None
                    else np.zeros(len(trials), dtype=float)
                )
                output_row[f"{key}_mean"] = float(values.mean())
                output_row[f"{key}_ci95"] = float(
                    1.96 * values.std() / np.sqrt(max(len(values), 1))
                )

            online_nsw = (
                np.asarray([row["nsw_legit"] for row in history_rows], dtype=float)
                if history_rows is not None
                else np.zeros(len(trials), dtype=float)
            )
            full_offline = np.asarray(
                [trial["history"][-1]["nsw_opt"] for trial in trials],
                dtype=float,
            )
            nsw_ratio = np.divide(
                online_nsw,
                full_offline,
                out=np.zeros_like(online_nsw),
                where=full_offline > 0,
            )
            derived_metrics = {
                "offline_nsw_full_horizon": full_offline,
                "nsw_ratio": nsw_ratio,
            }
            for key, values in derived_metrics.items():
                output_row[f"{key}_mean"] = float(values.mean())
                output_row[f"{key}_ci95"] = float(
                    1.96 * values.std() / np.sqrt(max(len(values), 1))
                )

            attacking_fraction = (
                np.asarray(
                    [row["n_attacking"] / max(config.n_malicious, 1)
                     for row in history_rows],
                    dtype=float,
                )
                if history_rows is not None
                else np.zeros(len(trials), dtype=float)
            )
            output_row["attacking_fraction_mean"] = float(attacking_fraction.mean())
            output_row["attacking_fraction_ci95"] = float(
                1.96 * attacking_fraction.std()
                / np.sqrt(max(len(attacking_fraction), 1))
            )
            checkpoint_rows.append(output_row)

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    plot_specs = [
        ("nsw_legit", "Cumulative online NSW", None),
        ("offline_nsw_full_horizon", "Full-information offline NSW", None),
        ("nsw_ratio", "Online / full-horizon offline NSW", None),
        ("detection_rate", "Online detection rate", (0.0, 1.05)),
        ("fp_rate", "Online false-positive rate", (0.0, 1.05)),
        ("moradi_detection_rate", "Shadow Moradi detection rate", (0.0, 1.05)),
        ("min_util", "Minimum legitimate utility", None),
        ("fairness_gap", "Legitimate utility gap", None),
        ("attacking_fraction", "Active malicious fraction", (0.0, 1.05)),
    ]
    fig, axes = moradi.plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    axes = axes.ravel()
    for attack_type, result in results.items():
        attack_rows = [
            row for row in checkpoint_rows
            if row["attack_type"] == attack_type.value
        ]
        rounds = np.asarray([row["checkpoint_round"] for row in attack_rows])
        color = moradi.ATTACK_COLOURS.get(attack_type, "#333333")
        label = moradi.ATTACK_LABELS.get(attack_type, attack_type.value)
        for axis, (key, title, ylim) in zip(axes, plot_specs):
            means = np.asarray([row[f"{key}_mean"] for row in attack_rows])
            errors = np.asarray([row[f"{key}_ci95"] for row in attack_rows])
            axis.plot(rounds, means, lw=1.6, color=color, label=label)
            axis.fill_between(rounds, means - errors, means + errors,
                              color=color, alpha=0.12)
            axis.set_title(title, fontsize=10, fontweight="bold")
            axis.grid(True, alpha=0.25)
            if ylim is not None:
                axis.set_ylim(*ylim)
    for axis in axes:
        axis.set_xlabel("Completed rounds")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Online progress every {interval} rounds | offline sees all "
        f"{config.n_rounds} rounds before deciding",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    moradi.plt.close(fig)

    print(f"  Checkpoint data: {csv_path}")
    print(f"  Checkpoint plot: {plot_path}")
    return csv_path, plot_path


def install_online_checkpoint_reporting(output_dir: Path) -> None:
    """Add checkpoint reporting around the shared experiment runner."""
    current = moradi.compare_focused_types
    original = getattr(current, "_online_checkpoint_original", current)

    def compare_with_checkpoints(base: moradi.ExperimentConfig) -> Dict:
        results = original(base)
        write_online_checkpoint_outputs(results, base, output_dir)
        return results

    compare_with_checkpoints._online_checkpoint_original = original
    moradi.compare_focused_types = compare_with_checkpoints


def plot_utility_with_moradi_all_agents(
    trial: Dict,
    config: moradi.ExperimentConfig,
    path: str,
) -> None:
    """Plot utilities and detector-aligned Moradi scores for every agent."""
    history = trial["history"]
    sim = trial["sim"]
    n_rounds = len(history)
    n_legitimate = config.n_legitimate
    n_malicious = config.n_malicious
    n_agents = n_legitimate + n_malicious
    rounds = np.arange(n_rounds + 1)

    attack_colour = moradi.ATTACK_COLOURS.get(config.attack_type, "#333333")
    attack_label = moradi.ATTACK_LABELS.get(
        config.attack_type, config.attack_type.value
    )

    # Explicit round-0 baseline followed by the cumulative utilities.
    utilities = np.vstack((
        np.zeros((1, n_agents), dtype=float),
        np.asarray([row["utilities"] for row in history], dtype=float),
    ))

    # Use the same symmetrized normalized weighted degree used by the
    # count-free sparse-group detector, now shown for all agents rather than
    # only the known-malicious evaluation subset.
    scores = np.zeros((n_rounds + 1, n_agents), dtype=float)
    report_matrix = (
        np.stack(sim.hist_reports, axis=1)
        if sim.hist_reports else np.zeros((n_agents, 0), dtype=float)
    )
    for round_index, row in enumerate(history, start=1):
        graph = moradi.build_blended_matrix(
            beta_matrix=row["beta_matrix"],
            hist_reports=report_matrix[:, :round_index],
            n_agents=n_agents,
            sigma=config.moradi_sigma,
        )
        graph = np.clip((graph + graph.T) / 2.0, 0.0, None)
        np.fill_diagonal(graph, 0.0)
        scores[round_index, :] = graph.sum(axis=1) / max(n_agents - 1, 1)

    malicious_ids = set(range(n_legitimate, n_agents))
    algorithm1_onset = next(
        (row["t"] + 1 for row in history
         if row["alg1_detected"] & malicious_ids),
        None,
    )
    moradi_onset = next(
        (row["t"] + 1 for row in history
         if row["moradi_detected"] & malicious_ids),
        None,
    )

    figure, axes = moradi.plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    figure.suptitle(
        "Utility Trajectories and Moradi Scores for All Agents  |  "
        f"{attack_label}  |  alpha={config.allocation_alpha:.2f}  "
        f"sigma={config.moradi_sigma:.2f}",
        fontsize=11,
        fontweight="bold",
    )

    utility_axis = axes[0]
    for agent in range(n_legitimate):
        utility_axis.plot(
            rounds, utilities[:, agent], color="#1E88E5", lw=1.0, alpha=0.5,
        )
    for malicious_index, agent in enumerate(range(n_legitimate, n_agents)):
        utility_axis.plot(
            rounds,
            utilities[:, agent],
            color=attack_colour,
            lw=1.5,
            alpha=0.8,
            ls="--",
            label=f"M{malicious_index}" if malicious_index == 0 else None,
        )
    utility_axis.plot([], [], color="#1E88E5", lw=2,
                      label=f"Legitimate agents ({n_legitimate})")
    utility_axis.plot([], [], color=attack_colour, lw=2, ls="--",
                      label=f"Malicious agents ({n_malicious})")
    utility_axis.set_ylabel("Cumulative utility")
    utility_axis.set_title("Per-agent cumulative utility", fontweight="bold")
    utility_axis.grid(True, alpha=0.3)
    utility_axis.legend(fontsize=8, loc="upper left")

    score_axis = axes[1]
    legitimate_colours = moradi.plt.cm.Blues(np.linspace(0.35, 0.9, n_legitimate))
    malicious_colours = ["#E53935", "#FB8C00", "#8E24AA", "#00897B"]
    for agent in range(n_legitimate):
        score_axis.plot(
            rounds,
            scores[:, agent],
            color=legitimate_colours[agent],
            lw=1.1,
            alpha=0.85,
            label=f"L{agent}",
        )
    for malicious_index, agent in enumerate(range(n_legitimate, n_agents)):
        score_axis.plot(
            rounds,
            scores[:, agent],
            color=malicious_colours[malicious_index % len(malicious_colours)],
            lw=1.8,
            ls="--",
            label=f"M{malicious_index}",
        )
    if algorithm1_onset is not None:
        score_axis.axvline(
            algorithm1_onset, color="black", ls=":", lw=1.3, alpha=0.7,
            label=f"Algorithm 1 onset ({algorithm1_onset})",
        )
    if moradi_onset is not None:
        score_axis.axvline(
            moradi_onset, color="#8E24AA", ls="-.", lw=1.3, alpha=0.7,
            label=f"Moradi onset ({moradi_onset})",
        )
    score_axis.set_xlabel("Completed rounds")
    score_axis.set_ylabel("Normalized Moradi weighted degree")
    score_axis.set_title(
        "Moradi graph score for all agents  |  lower means more suspicious",
        fontweight="bold",
    )
    score_axis.set_ylim(bottom=0.0)
    score_axis.grid(True, alpha=0.3)
    score_axis.legend(fontsize=7, ncol=4, loc="best")

    moradi._save(path)


def organize_output_files(output_dir: Path, *, control: bool = False) -> None:
    """Keep one ordered copy of each useful result and remove repetitions."""
    output_dir = output_dir.expanduser().resolve()
    if control:
        ordered = {
            "plot17_online_checkpoints_1.png": "01_primary_online_progress.png",
            "online_checkpoints_1.csv": "02_primary_online_data.csv",
            "plot5_attack_comparison.png": "03_final_attack_summary.png",
            "plot3_alloc_heatmap.png": "04_allocation_heatmap.png",
            "plot13_utility_moradi_byzantine.png": "05_byzantine_agent_utilities.png",
            "plot7_nsw_ratio.png": "06_adversary_count_sensitivity.png",
            "plot8_alpha_sweep.png": "07_alpha_tradeoff.png",
        }
        redundant = {
            "plot1_main_results.png",
            "plot2_utility_traj.png",
            "plot4_misclass.png",
            "plot6_beta_trajectories.png",
            "plot9_three_way_detector.png",
            "plot10_sigma_sweep.png",
            "plot11_compound_breakdown.png",
            "plot12_reputation_poison_phases.png",
            "plot13_utility_moradi_burst.png",
            "plot13_utility_moradi_reputation_poison.png",
            "plot13_utility_moradi_trust_mimicry.png",
            "plot14_attack_rates.png",
            "plot15_prediction_scenarios.png",
            "plot16_online_moradi.png",
        }
    else:
        ordered = {
            "plot17_online_checkpoints_1.png": "01_primary_online_progress.png",
            "online_checkpoints_1.csv": "02_primary_online_data.csv",
            "plot5_attack_comparison.png": "03_final_attack_summary.png",
            "plot3_alloc_heatmap.png": "04_allocation_heatmap.png",
            "plot13_utility_moradi_byzantine.png": "05_byzantine_agent_utilities.png",
            "plot13_utility_moradi_reputation_poison.png": "06_reputation_poison_agent_utilities.png",
            "plot9_three_way_detector.png": "07_detector_comparison.png",
            "plot6_beta_trajectories.png": "08_beta_detection_evidence.png",
            "plot14_attack_rates.png": "09_attack_behaviour.png",
            "plot7_nsw_ratio.png": "10_adversary_count_sensitivity.png",
            "plot8_alpha_sweep.png": "11_alpha_tradeoff.png",
            "plot10_sigma_sweep.png": "12_sigma_detector_sensitivity.png",
            "plot12_reputation_poison_phases.png": "13_reputation_poison_phases.png",
            "plot15_prediction_scenarios.png": "14_prediction_robustness.png",
            "plot11_compound_breakdown.png": "15_compound_attack_appendix.png",
        }
        redundant = {
            "plot1_main_results.png",
            "plot2_utility_traj.png",
            "plot4_misclass.png",
            "plot13_utility_moradi_burst.png",
            "plot13_utility_moradi_trust_mimicry.png",
            "plot16_online_moradi.png",
        }

    for name in redundant:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for source_name, destination_name in ordered.items():
        source = output_dir / source_name
        destination = output_dir / destination_name
        if source.exists():
            source.replace(destination)

    print("\nOrdered retained outputs (most important first):")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path}")


def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse Moradi options with a 10-trial default."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the trust-aware NSW simulator with a cached Frank-Wolfe "
            "offline benchmark."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_frank_wolfe",
        help="Directory for generated plots (default: %(default)s).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Rounds per trial (default: 2000, or 20 with --quick).",
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
        help="Run a 20-round local check and skip expensive parameter sweeps.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run the four focused attacks and plots, but skip parameter sweeps.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    args.rounds = args.rounds if args.rounds is not None else (20 if args.quick else 2000)
    args.trials = args.trials if args.trials is not None else (1 if args.quick else 10)
    if args.rounds <= 0:
        parser.error("--rounds must be greater than zero")
    if args.trials <= 0:
        parser.error("--trials must be greater than zero")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_cli_args(argv)
    moradi.offline_optimal_nsw = cached_frank_wolfe_offline_nsw
    moradi._parse_cli_args = _parse_cli_args
    moradi.plot_utility_with_moradi = plot_utility_with_moradi_all_agents
    install_online_checkpoint_reporting(args.output_dir)
    print(
        "Using cached Frank-Wolfe for the offline NSW benchmark "
        f"(max_iterations={FW_MAX_ITERATIONS}, tolerance={FW_TOLERANCE:g})."
    )
    moradi.main(argv)
    organize_output_files(args.output_dir)


if __name__ == "__main__":
    main()
