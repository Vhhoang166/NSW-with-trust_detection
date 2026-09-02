#!/usr/bin/env python3
"""
Generate the extra experiment outputs needed for the allocation-focused report.

This script is meant to be run on a stronger CPU machine.  It does not replace
the existing 400-round benchmark; it adds the evidence that the report now asks
for:

  * no-trust vs current Algorithm 1 vs oracle-trust vs online sparse trust
  * malicious resource share
  * allocation heatmaps for stealth attacks
  * adaptive-alpha trajectories
  * expert-advice weight trajectories
  * compound-attack summaries
  * paired lift estimates against the fixed-alpha baseline

Every trial is cached as a compressed NPZ file.  If a run is interrupted, launch
the same command again and it will reuse completed trials.
"""

import argparse
import csv
import gzip
import json
import math
import os
import shutil
import sys
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent if (SCRIPT_PATH.parent / "NSW_Moradi_Final.py").exists() else SCRIPT_PATH.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

plt = None
_MATPLOTLIB_STUBS: List[str] = []


def install_matplotlib_stub_for_moradi_import() -> None:
    """Avoid paying Moradi Final's plotting import cost for simulator-only use."""
    if "matplotlib" in sys.modules:
        return
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.__runner_stub__ = True
    matplotlib_stub.use = lambda *args, **kwargs: None

    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    pyplot_stub.__runner_stub__ = True

    ticker_stub = types.ModuleType("matplotlib.ticker")
    ticker_stub.__runner_stub__ = True

    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub
    sys.modules["matplotlib.ticker"] = ticker_stub
    _MATPLOTLIB_STUBS.extend(["matplotlib.pyplot", "matplotlib.ticker", "matplotlib"])


install_matplotlib_stub_for_moradi_import()

import NSW_Moradi_Final as base
from NSW_Moradi_Final import (
    AttackType,
    ExperimentConfig,
)

ATTACKS = {
    "byzantine": AttackType.BYZANTINE,
    "burst": AttackType.BURST,
    "reputation_poison": AttackType.REPUTATION_POISON,
    "trust_mimicry": AttackType.TRUST_MIMICRY,
    "compound": AttackType.COMPOUND,
}

ATTACK_LABELS = {
    "byzantine": "Byzantine",
    "burst": "Burst",
    "reputation_poison": "Reputation Poison",
    "trust_mimicry": "Trust Mimicry",
    "compound": "Compound",
}

TRUST_MODES = {
    "no_trust": "No trust gate",
    "algorithm1": "Current Algorithm 1 gate",
    "oracle": "Oracle trust gate",
    "sparse_online": "Online sparse gate",
}

APPROACH_LABELS = {
    "baseline_fixed_alpha": "Fixed-alpha set-aside",
    "adaptive_alpha": "Adaptive alpha",
    "pace_trusted": "PACE-inspired",
    "generalized_mean_greedy": "Generalized mean",
    "sample_resolving": "Sample resolving",
    "expert_advice": "Expert advice",
    "robust_aggregation": "Robust aggregation",
    "pure_greedy_alpha0": "Pure greedy alpha=0",
    "pure_equal_alpha1": "Pure equal alpha=1",
}

BASE_APPROACHES = [
    "baseline_fixed_alpha",
    "adaptive_alpha",
    "pace_trusted",
    "generalized_mean_greedy",
    "sample_resolving",
    "expert_advice",
    "robust_aggregation",
]

EXTRA_APPROACHES = ["pure_greedy_alpha0", "pure_equal_alpha1"]

CORE_TRAJECTORY_KEYS = [
    "nsw_ratio",
    "nsw_legit",
    "min_util",
    "fairness_gap",
    "detection_rate",
    "fp_rate",
    "fn_rate",
    "algorithm1_detection_rate",
    "algorithm1_fp_rate",
    "algorithm1_fn_rate",
    "n_attacking",
    "trusted_count",
    "detected_count",
    "malicious_trusted_count",
    "malicious_detected_count",
    "malicious_resource_share",
    "legit_resource_share",
    "beta_mal_gap_avg",
    "beta_legit_gap_avg",
    "xi_t",
]

EXTRA_TRAJECTORY_KEYS = [
    "alpha_t",
    "trust_risk",
    "prediction_risk",
    "controller_risk",
    "expert_defensive_weight",
    "expert_aggressive_weight",
    "expert_gm_weight",
    "pace_avg_price",
    "gm_rho",
    "sample_window_used",
    "robust_report_delta",
]

ALL_TRAJECTORY_KEYS = CORE_TRAJECTORY_KEYS + EXTRA_TRAJECTORY_KEYS

_ORIGINAL_INIT = base.NSWTrustworthySim.__init__
_FW_ITERS = 160
_OFFLINE_CACHE: Dict[Tuple, float] = {}


def ensure_matplotlib():
    global plt
    if plt is None:
        for name in _MATPLOTLIB_STUBS:
            module = sys.modules.get(name)
            if getattr(module, "__runner_stub__", False):
                del sys.modules[name]
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        plt = pyplot
    return plt


def _fast_offline_nsw_from_vals(
    v_true: np.ndarray,
    legit_ids: Sequence[int],
    iters: int,
) -> float:
    """Frank-Wolfe approximation for max sum_i log(u_i)."""
    values = np.asarray(v_true[list(legit_ids), :], dtype=float)
    n_agents, n_rounds = values.shape
    allocation = np.ones((n_agents, n_rounds), dtype=float) / n_agents
    utilities = np.maximum((values * allocation).sum(axis=1), 1e-12)

    for k in range(int(iters)):
        gradient = values / utilities[:, None]
        winners = np.argmax(gradient, axis=0)
        direction = np.zeros_like(allocation)
        direction[winners, np.arange(n_rounds)] = 1.0
        gamma = 2.0 / (k + 2.0)
        allocation = (1.0 - gamma) * allocation + gamma * direction
        utilities = np.maximum((values * allocation).sum(axis=1), 1e-12)

    return float(np.exp(np.mean(np.log(utilities))))


def _patched_init(self, config: ExperimentConfig, seed: int = 42):
    _ORIGINAL_INIT(self, config, seed=seed)
    self._seed_for_cache = int(seed)


def _patched_nsw_optimal(self) -> float:
    if self._nsw_opt is not None:
        return self._nsw_opt
    key = (
        getattr(self, "_seed_for_cache", None),
        self.cfg.n_legitimate,
        self.cfg.n_malicious,
        self.cfg.n_rounds,
        self.cfg.valuation_model,
        _FW_ITERS,
    )
    if key not in _OFFLINE_CACHE:
        _OFFLINE_CACHE[key] = _fast_offline_nsw_from_vals(
            self.true_vals, self.legit_ids, iters=_FW_ITERS
        )
    self._nsw_opt = _OFFLINE_CACHE[key]
    return self._nsw_opt


base.NSWTrustworthySim.__init__ = _patched_init
base.NSWTrustworthySim._nsw_optimal = _patched_nsw_optimal


def _as_float(value, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return default
    except Exception:
        return default


from report_extra_approaches import APPROACH_CLASSES


def make_config(
    n_rounds: int,
    n_trials: int,
    attack_type: AttackType,
    approach: str,
) -> ExperimentConfig:
    cfg = ExperimentConfig(
        n_legitimate=10,
        n_malicious=4,
        n_rounds=int(n_rounds),
        n_trials=int(n_trials),
        attack_type=attack_type,
        base_attack_prob=0.75,
        trust_legitimate=0.63,
        trust_malicious_attack=0.52,
        trust_malicious_no_attack=0.63,
        trust_std=0.14,
        xi_0=0.25,
        allocation_alpha=0.50,
        moradi_sigma=0.60,
        coalition_sync_every=10,
        poison_warmup_rounds=50,
        connectivity=1.0,
        prediction_scenario="mild_noise",
        scenario_label="report extra experiments",
    )
    if approach == "pure_greedy_alpha0":
        cfg = replace(cfg, allocation_alpha=0.0)
    elif approach == "pure_equal_alpha1":
        cfg = replace(cfg, allocation_alpha=1.0)
    return cfg


def cache_path(cache_dir: Path, approach: str, attack: str, trust_mode: str, trial: int) -> Path:
    name = f"{approach}__{attack}__{trust_mode}__trial_{trial:03d}.npz"
    return cache_dir / name


def load_cached(path: Path) -> Dict:
    with np.load(path, allow_pickle=False) as data:
        summary = json.loads(str(data["summary_json"].item()))
        metric_keys = json.loads(str(data["metric_keys_json"].item()))
        metrics = {key: data[f"metric_{key}"].astype(float) for key in metric_keys}
        alloc = data["alloc"].astype(float)
        utilities = data["utilities"].astype(float)
    return {"summary": summary, "metrics": metrics, "alloc": alloc, "utilities": utilities}


def save_cached(path: Path, result: Dict) -> None:
    arrays = {
        "summary_json": np.array(json.dumps(result["summary"], sort_keys=True)),
        "metric_keys_json": np.array(json.dumps(list(result["metrics"].keys()))),
        "alloc": np.asarray(result["alloc"], dtype=float),
        "utilities": np.asarray(result["utilities"], dtype=float),
    }
    for key, values in result["metrics"].items():
        arrays[f"metric_{key}"] = np.asarray(values, dtype=float)
    tmp = path.with_suffix(".tmp.npz")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def run_trial_uncached(task: Dict) -> Dict:
    global _FW_ITERS
    _FW_ITERS = int(task["fw_iters"])

    approach = task["approach"]
    attack = task["attack"]
    trust_mode = task["trust_mode"]
    trial = int(task["trial"])
    n_rounds = int(task["n_rounds"])
    n_trials = int(task["n_trials"])
    seed = int(task["seed_base"] + trial * task["seed_stride"])

    cfg = make_config(
        n_rounds=n_rounds,
        n_trials=n_trials,
        attack_type=ATTACKS[attack],
        approach=approach,
    )
    sim_class = APPROACH_CLASSES[approach]
    sim = sim_class(cfg, seed=seed, trust_mode=trust_mode)
    history = sim.run(verbose=False)

    metrics = {key: [] for key in ALL_TRAJECTORY_KEYS}
    alloc_rows = []
    utility_rows = []
    for row in history:
        alloc = np.asarray(row["alloc"], dtype=float)
        utilities = np.asarray(row["utilities"], dtype=float)
        trusted = set(row.get("trusted", []))
        detected = set(row.get("detected", []))
        mal_set = set(sim.mal_ids)
        legit_set = set(sim.legit_ids)

        values = {
            "trusted_count": len(trusted),
            "detected_count": len(detected),
            "malicious_trusted_count": len(trusted & mal_set),
            "malicious_detected_count": len(detected & mal_set),
            "malicious_resource_share": float(alloc[sim.mal_ids].sum() / max(alloc.sum(), 1e-12)),
            "legit_resource_share": float(alloc[sim.legit_ids].sum() / max(alloc.sum(), 1e-12)),
        }
        for key in ALL_TRAJECTORY_KEYS:
            if key in values:
                metrics[key].append(values[key])
            else:
                metrics[key].append(_as_float(row.get(key, np.nan)))
        alloc_rows.append(alloc)
        utility_rows.append(utilities)

    for key in ALL_TRAJECTORY_KEYS:
        metrics[key] = np.asarray(metrics[key], dtype=float)

    final = history[-1]
    avg_mal_share = float(np.nanmean(metrics["malicious_resource_share"]))
    summary = {
        "approach": approach,
        "attack_type": attack,
        "trust_mode": trust_mode,
        "trial": trial,
        "seed": seed,
        "n_rounds": n_rounds,
        "final_nsw_ratio": float(final["nsw_ratio"]),
        "final_nsw_legit": float(final["nsw_legit"]),
        "offline_nsw": float(final["nsw_opt"]),
        "final_min_util": float(final["min_util"]),
        "final_fairness_gap": float(final["fairness_gap"]),
        "final_detection_rate": float(final["detection_rate"]),
        "final_fp_rate": float(final["fp_rate"]),
        "final_fn_rate": float(final["fn_rate"]),
        "final_algorithm1_detection_rate": float(final["algorithm1_detection_rate"]),
        "final_algorithm1_fp_rate": float(final["algorithm1_fp_rate"]),
        "avg_malicious_resource_share": avg_mal_share,
        "final_malicious_resource_share": float(metrics["malicious_resource_share"][-1]),
        "avg_malicious_trusted_count": float(np.nanmean(metrics["malicious_trusted_count"])),
        "final_malicious_trusted_count": float(metrics["malicious_trusted_count"][-1]),
        "avg_alpha_t": float(np.nanmean(metrics["alpha_t"])),
        "avg_expert_defensive_weight": float(np.nanmean(metrics["expert_defensive_weight"])),
        "avg_expert_aggressive_weight": float(np.nanmean(metrics["expert_aggressive_weight"])),
        "avg_expert_gm_weight": float(np.nanmean(metrics["expert_gm_weight"])),
    }
    return {
        "summary": summary,
        "metrics": metrics,
        "alloc": np.stack(alloc_rows, axis=0),
        "utilities": np.stack(utility_rows, axis=0),
    }


def run_or_load_trial(task: Dict) -> Dict:
    path = cache_path(
        Path(task["cache_dir"]),
        task["approach"],
        task["attack"],
        task["trust_mode"],
        int(task["trial"]),
    )
    if path.exists():
        result = load_cached(path)
        result["cache_hit"] = True
        return result
    result = run_trial_uncached(task)
    save_cached(path, result)
    result["cache_hit"] = False
    return result


def ci95(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.96 * np.std(x, ddof=0) / math.sqrt(x.size))


class CellAccumulator:
    def __init__(self, n_rounds: int):
        self.n = 0
        self.sums = {key: np.zeros(n_rounds, dtype=float) for key in ALL_TRAJECTORY_KEYS}
        self.sq_sums = {key: np.zeros(n_rounds, dtype=float) for key in ALL_TRAJECTORY_KEYS}
        self.counts = {key: np.zeros(n_rounds, dtype=float) for key in ALL_TRAJECTORY_KEYS}
        self.final_ratios: List[float] = []
        self.final_min_utils: List[float] = []
        self.avg_mal_shares: List[float] = []

    def add(self, result: Dict) -> None:
        self.n += 1
        for key, series in result["metrics"].items():
            values = np.asarray(series, dtype=float)
            mask = np.isfinite(values)
            self.sums[key][mask] += values[mask]
            self.sq_sums[key][mask] += values[mask] ** 2
            self.counts[key][mask] += 1.0
        summary = result["summary"]
        self.final_ratios.append(float(summary["final_nsw_ratio"]))
        self.final_min_utils.append(float(summary["final_min_util"]))
        self.avg_mal_shares.append(float(summary["avg_malicious_resource_share"]))

    def mean(self, key: str) -> np.ndarray:
        counts = np.maximum(self.counts[key], 1.0)
        out = self.sums[key] / counts
        out[self.counts[key] == 0] = np.nan
        return out

    def sem95(self, key: str) -> np.ndarray:
        counts = np.maximum(self.counts[key], 1.0)
        mean = self.mean(key)
        var = np.maximum(self.sq_sums[key] / counts - mean**2, 0.0)
        err = 1.96 * np.sqrt(var) / np.sqrt(counts)
        err[self.counts[key] == 0] = np.nan
        return err


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def trajectory_fieldnames(n_agents: int) -> List[str]:
    fields = [
        "approach",
        "attack_type",
        "trust_mode",
        "trial",
        "seed",
        "t",
    ]
    fields.extend(ALL_TRAJECTORY_KEYS)
    fields.extend([f"alloc_{i}" for i in range(n_agents)])
    fields.extend([f"utility_{i}" for i in range(n_agents)])
    return fields


def result_to_trajectory_rows(result: Dict) -> Iterable[Dict]:
    summary = result["summary"]
    metrics = result["metrics"]
    alloc = result["alloc"]
    utilities = result["utilities"]
    T = alloc.shape[0]
    N = alloc.shape[1]
    for t in range(T):
        row = {
            "approach": summary["approach"],
            "attack_type": summary["attack_type"],
            "trust_mode": summary["trust_mode"],
            "trial": summary["trial"],
            "seed": summary["seed"],
            "t": t,
        }
        for key in ALL_TRAJECTORY_KEYS:
            row[key] = _as_float(metrics[key][t])
        for i in range(N):
            row[f"alloc_{i}"] = float(alloc[t, i])
            row[f"utility_{i}"] = float(utilities[t, i])
        yield row


def build_cell_summary(accumulators: Dict[Tuple[str, str, str], CellAccumulator]) -> List[Dict]:
    rows = []
    for (approach, attack, trust_mode), acc in sorted(accumulators.items()):
        ratios = np.asarray(acc.final_ratios, dtype=float)
        min_utils = np.asarray(acc.final_min_utils, dtype=float)
        mal_shares = np.asarray(acc.avg_mal_shares, dtype=float)
        rows.append(
            {
                "approach": approach,
                "attack_type": attack,
                "trust_mode": trust_mode,
                "n_trials_completed": acc.n,
                "avg_final_nsw_ratio": float(np.nanmean(ratios)),
                "ci95_final_nsw_ratio": ci95(ratios),
                "std_final_nsw_ratio": float(np.nanstd(ratios)),
                "avg_final_min_util": float(np.nanmean(min_utils)),
                "ci95_final_min_util": ci95(min_utils),
                "avg_malicious_resource_share": float(np.nanmean(mal_shares)),
                "ci95_malicious_resource_share": ci95(mal_shares),
                "final_detection_rate": float(np.nanmean(acc.mean("detection_rate")[-1])),
                "final_algorithm1_detection_rate": float(
                    np.nanmean(acc.mean("algorithm1_detection_rate")[-1])
                ),
                "final_malicious_trusted_count": float(
                    np.nanmean(acc.mean("malicious_trusted_count")[-1])
                ),
            }
        )
    return rows


def build_paired_lift_rows(trial_summaries: Sequence[Dict]) -> List[Dict]:
    by_key = {}
    for row in trial_summaries:
        key = (
            row["attack_type"],
            row["trust_mode"],
            int(row["trial"]),
            row["approach"],
        )
        by_key[key] = float(row["final_nsw_ratio"])

    groups = sorted(
        {
            (row["approach"], row["attack_type"], row["trust_mode"])
            for row in trial_summaries
            if row["approach"] != "baseline_fixed_alpha"
        }
    )
    rows = []
    for approach, attack, trust_mode in groups:
        diffs = []
        rels = []
        for row in trial_summaries:
            if (
                row["approach"] == approach
                and row["attack_type"] == attack
                and row["trust_mode"] == trust_mode
            ):
                trial = int(row["trial"])
                baseline = by_key.get((attack, trust_mode, trial, "baseline_fixed_alpha"))
                if baseline is None or baseline == 0:
                    continue
                current = float(row["final_nsw_ratio"])
                diffs.append(current - baseline)
                rels.append((current - baseline) / baseline)
        if diffs:
            rows.append(
                {
                    "approach": approach,
                    "attack_type": attack,
                    "trust_mode": trust_mode,
                    "n_pairs": len(diffs),
                    "mean_ratio_lift": float(np.mean(diffs)),
                    "ci95_ratio_lift": ci95(np.asarray(diffs)),
                    "mean_percent_lift": float(100.0 * np.mean(rels)),
                    "ci95_percent_lift": float(100.0 * ci95(np.asarray(rels))),
                }
            )
    return rows


def savefig(path: Path) -> None:
    ensure_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def plot_ratio_heatmaps(out_dir: Path, summary_rows: Sequence[Dict], approaches: List[str], attacks: List[str]) -> None:
    for trust_mode in TRUST_MODES:
        rows = [r for r in summary_rows if r["trust_mode"] == trust_mode]
        if not rows:
            continue
        matrix = np.full((len(approaches), len(attacks)), np.nan)
        for row in rows:
            if row["approach"] in approaches and row["attack_type"] in attacks:
                i = approaches.index(row["approach"])
                j = attacks.index(row["attack_type"])
                matrix[i, j] = float(row["avg_final_nsw_ratio"])
        fig, ax = plt.subplots(figsize=(12, 0.55 * len(approaches) + 2.5))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=ax, label="Mean final NSW ratio")
        ax.set_title(f"Allocation approaches under {TRUST_MODES[trust_mode]}")
        ax.set_yticks(np.arange(len(approaches)))
        ax.set_yticklabels([APPROACH_LABELS.get(a, a) for a in approaches])
        ax.set_xticks(np.arange(len(attacks)))
        ax.set_xticklabels([ATTACK_LABELS[a] for a in attacks], rotation=20, ha="right")
        threshold = np.nanmean(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    color = "white" if matrix[i, j] < threshold else "black"
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
        savefig(out_dir / f"ratio_heatmap_{trust_mode}.png")


def plot_trust_ablation_focus(out_dir: Path, summary_rows: Sequence[Dict], attacks: List[str]) -> None:
    focus = [a for a in ["baseline_fixed_alpha", "adaptive_alpha", "expert_advice"]]
    fig, axes = plt.subplots(len(focus), 1, figsize=(13, 9), sharex=True)
    x = np.arange(len(attacks))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(TRUST_MODES))
    for ax, approach in zip(axes, focus):
        for offset, trust_mode in zip(offsets, TRUST_MODES):
            vals = []
            errs = []
            for attack in attacks:
                match = [
                    r
                    for r in summary_rows
                    if r["approach"] == approach
                    and r["attack_type"] == attack
                    and r["trust_mode"] == trust_mode
                ]
                vals.append(float(match[0]["avg_final_nsw_ratio"]) if match else np.nan)
                errs.append(float(match[0]["ci95_final_nsw_ratio"]) if match else np.nan)
            ax.bar(x + offset, vals, width=width, yerr=errs, capsize=2, label=TRUST_MODES[trust_mode])
        ax.set_ylabel(APPROACH_LABELS[approach])
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([ATTACK_LABELS[a] for a in attacks], rotation=20, ha="right")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("No-trust vs current trust vs oracle-trust ablation", fontweight="bold")
    savefig(out_dir / "trust_ablation_focus_ratio.png")


def plot_malicious_share_stealth(out_dir: Path, summary_rows: Sequence[Dict]) -> None:
    attacks = ["reputation_poison", "trust_mimicry"]
    approaches = [a for a in BASE_APPROACHES if a in {r["approach"] for r in summary_rows}]
    for attack in attacks:
        fig, ax = plt.subplots(figsize=(13, 5.5))
        x = np.arange(len(approaches))
        width = 0.18
        offsets = np.linspace(-1.5 * width, 1.5 * width, len(TRUST_MODES))
        for offset, trust_mode in zip(offsets, TRUST_MODES):
            vals = []
            errs = []
            for approach in approaches:
                match = [
                    r
                    for r in summary_rows
                    if r["approach"] == approach
                    and r["attack_type"] == attack
                    and r["trust_mode"] == trust_mode
                ]
                vals.append(float(match[0]["avg_malicious_resource_share"]) if match else np.nan)
                errs.append(float(match[0]["ci95_malicious_resource_share"]) if match else np.nan)
            ax.bar(x + offset, vals, width=width, yerr=errs, capsize=2, label=TRUST_MODES[trust_mode])
        ax.set_title(f"Malicious resource share under {ATTACK_LABELS[attack]}")
        ax.set_ylabel("Mean allocation share sent to malicious agents")
        ax.set_xticks(x)
        ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in approaches], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        savefig(out_dir / f"malicious_resource_share_{attack}.png")


def plot_adaptive_trajectories(
    out_dir: Path,
    accumulators: Dict[Tuple[str, str, str], CellAccumulator],
    attacks: List[str],
    trust_modes: List[str],
) -> None:
    for attack in attacks:
        for trust_mode in trust_modes:
            key = ("adaptive_alpha", attack, trust_mode)
            acc = accumulators.get(key)
            if acc is None or not np.isfinite(acc.mean("alpha_t")).any():
                continue
            rounds = np.arange(len(acc.mean("alpha_t")))
            fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
            series = [
                ("alpha_t", "alpha_t"),
                ("trust_risk", "trust instability"),
                ("prediction_risk", "prediction/report mismatch"),
                ("nsw_ratio", "NSW ratio"),
            ]
            for ax, (metric, label) in zip(axes, series):
                mu = acc.mean(metric)
                err = acc.sem95(metric)
                ax.plot(rounds, mu, lw=2)
                ax.fill_between(rounds, mu - err, mu + err, alpha=0.18)
                ax.set_ylabel(label)
                ax.grid(alpha=0.25)
            axes[-1].set_xlabel("Round")
            fig.suptitle(
                f"Adaptive alpha trajectory: {ATTACK_LABELS[attack]}, {TRUST_MODES[trust_mode]}",
                fontweight="bold",
            )
            savefig(out_dir / f"adaptive_alpha_trajectory_{attack}_{trust_mode}.png")


def plot_expert_weight_trajectories(
    out_dir: Path,
    accumulators: Dict[Tuple[str, str, str], CellAccumulator],
    attacks: List[str],
    trust_modes: List[str],
) -> None:
    metrics = [
        ("expert_defensive_weight", "Defensive set-aside"),
        ("expert_aggressive_weight", "Aggressive set-aside"),
        ("expert_gm_weight", "Generalized mean"),
    ]
    for attack in attacks:
        for trust_mode in trust_modes:
            key = ("expert_advice", attack, trust_mode)
            acc = accumulators.get(key)
            if acc is None or not np.isfinite(acc.mean("expert_defensive_weight")).any():
                continue
            rounds = np.arange(len(acc.mean("expert_defensive_weight")))
            fig, ax = plt.subplots(figsize=(13, 5.5))
            for metric, label in metrics:
                mu = acc.mean(metric)
                err = acc.sem95(metric)
                ax.plot(rounds, mu, lw=2, label=label)
                ax.fill_between(rounds, mu - err, mu + err, alpha=0.12)
            ax.set_title(
                f"Expert-advice weight trajectory: {ATTACK_LABELS[attack]}, {TRUST_MODES[trust_mode]}",
                fontweight="bold",
            )
            ax.set_xlabel("Round")
            ax.set_ylabel("Expert weight")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend()
            savefig(out_dir / f"expert_weight_trajectory_{attack}_{trust_mode}.png")


def plot_allocation_heatmaps(
    out_dir: Path,
    samples: Dict[Tuple[str, str, str], np.ndarray],
    n_legitimate: int,
) -> None:
    wanted_attacks = {"reputation_poison", "trust_mimicry", "compound"}
    wanted_approaches = {"baseline_fixed_alpha", "adaptive_alpha", "expert_advice"}
    for (approach, attack, trust_mode), alloc in samples.items():
        if attack not in wanted_attacks or approach not in wanted_approaches:
            continue
        fig, ax = plt.subplots(figsize=(13, 5.5))
        im = ax.imshow(alloc.T, aspect="auto", cmap="magma", vmin=0, vmax=max(0.25, float(np.nanmax(alloc))))
        plt.colorbar(im, ax=ax, label="Allocation share")
        ax.axhline(n_legitimate - 0.5, color="cyan", lw=1.5, ls="--")
        ax.set_title(
            f"Allocation heatmap: {APPROACH_LABELS.get(approach, approach)}, "
            f"{ATTACK_LABELS[attack]}, {TRUST_MODES[trust_mode]}, trial 0",
            fontweight="bold",
        )
        ax.set_xlabel("Round")
        ax.set_ylabel("Agent index")
        savefig(out_dir / f"allocation_heatmap_{approach}_{attack}_{trust_mode}_trial0.png")


def plot_paired_lifts(out_dir: Path, lift_rows: Sequence[Dict]) -> None:
    rows = [r for r in lift_rows if r["trust_mode"] == "algorithm1"]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (r["attack_type"], r["mean_percent_lift"]), reverse=False)
    labels = [f"{APPROACH_LABELS.get(r['approach'], r['approach'])}\n{ATTACK_LABELS[r['attack_type']]}" for r in rows]
    vals = np.asarray([float(r["mean_percent_lift"]) for r in rows])
    errs = np.asarray([float(r["ci95_percent_lift"]) for r in rows])
    fig, ax = plt.subplots(figsize=(max(12, 0.42 * len(rows)), 6))
    x = np.arange(len(rows))
    colors = ["#2a9d8f" if v >= 0 else "#c1121f" for v in vals]
    ax.bar(x, vals, yerr=errs, capsize=2, color=colors, alpha=0.9)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Percent lift vs fixed-alpha baseline")
    ax.set_title("Paired lift under current Algorithm 1 trust gate", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    savefig(out_dir / "paired_lift_vs_baseline_algorithm1.png")


def parse_csv_arg(value: str, allowed: Sequence[str]) -> List[str]:
    if value.strip().lower() == "all":
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    bad = [item for item in selected if item not in allowed]
    if bad:
        raise SystemExit(f"Unknown value(s): {bad}. Allowed: {', '.join(allowed)}")
    return selected


def build_tasks(args) -> List[Dict]:
    attacks = parse_csv_arg(args.attacks, list(ATTACKS.keys()))
    trust_modes = parse_csv_arg(args.trust_modes, list(TRUST_MODES.keys()))
    all_approach_choices = BASE_APPROACHES + EXTRA_APPROACHES
    if args.approaches.strip().lower() == "all":
        approaches = BASE_APPROACHES + (EXTRA_APPROACHES if args.include_extremes else [])
    else:
        approaches = parse_csv_arg(args.approaches, all_approach_choices)

    tasks = []
    for approach in approaches:
        for attack in attacks:
            for trust_mode in trust_modes:
                for trial in range(args.n_trials):
                    tasks.append(
                        {
                            "approach": approach,
                            "attack": attack,
                            "trust_mode": trust_mode,
                            "trial": trial,
                            "n_rounds": args.n_rounds,
                            "n_trials": args.n_trials,
                            "seed_base": args.seed_base,
                            "seed_stride": args.seed_stride,
                            "fw_iters": args.fw_iters,
                            "cache_dir": str(Path(args.out_dir) / "trial_cache"),
                        }
                    )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="output_report_extra", help="Output directory.")
    parser.add_argument("--n-rounds", type=int, default=400)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--fw-iters", type=int, default=160, help="Frank-Wolfe iterations for offline benchmark.")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--seed-stride", type=int, default=13)
    parser.add_argument("--attacks", default="all", help="Comma-separated attacks or all.")
    parser.add_argument("--trust-modes", default="all", help="Comma-separated trust modes or all.")
    parser.add_argument("--approaches", default="all", help="Comma-separated approaches or all.")
    parser.add_argument("--include-extremes", action="store_true", help="Also run alpha=0 and alpha=1 baseline extremes.")
    parser.add_argument("--skip-trajectory-csv", action="store_true", help="Only write summaries and figures.")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "trial_cache"
    plot_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args)
    if not tasks:
        raise SystemExit("No tasks selected.")

    approaches = sorted({task["approach"] for task in tasks}, key=lambda a: list(APPROACH_LABELS).index(a))
    attacks = sorted({task["attack"] for task in tasks}, key=lambda a: list(ATTACKS).index(a))
    trust_modes = sorted({task["trust_mode"] for task in tasks}, key=lambda t: list(TRUST_MODES).index(t))

    n_agents = make_config(args.n_rounds, args.n_trials, ATTACKS[attacks[0]], approaches[0]).n_legitimate
    n_agents += make_config(args.n_rounds, args.n_trials, ATTACKS[attacks[0]], approaches[0]).n_malicious

    trial_summary_rows: List[Dict] = []
    accumulators: Dict[Tuple[str, str, str], CellAccumulator] = {}
    samples: Dict[Tuple[str, str, str], np.ndarray] = {}
    errors = []

    trajectory_path = out_dir / "extra_trajectory_rows.csv.gz"
    trajectory_handle = None
    trajectory_writer = None
    if not args.skip_trajectory_csv:
        trajectory_handle = gzip.open(trajectory_path, "wt", newline="")
        trajectory_writer = csv.DictWriter(
            trajectory_handle, fieldnames=trajectory_fieldnames(n_agents)
        )
        trajectory_writer.writeheader()

    def consume(result: Dict) -> None:
        summary = result["summary"]
        trial_summary_rows.append(summary)
        cell = (summary["approach"], summary["attack_type"], summary["trust_mode"])
        if cell not in accumulators:
            accumulators[cell] = CellAccumulator(args.n_rounds)
        accumulators[cell].add(result)
        if int(summary["trial"]) == 0:
            samples[cell] = result["alloc"]
        if trajectory_writer is not None:
            for row in result_to_trajectory_rows(result):
                trajectory_writer.writerow(row)

    print(f"Selected tasks: {len(tasks)} trials")
    print(f"Output directory: {out_dir}")
    print(f"Workers: {args.jobs}")

    if args.jobs <= 1:
        for idx, task in enumerate(tasks, start=1):
            try:
                result = run_or_load_trial(task)
                consume(result)
                source = "cache" if result.get("cache_hit") else "run"
                print(
                    f"[{idx}/{len(tasks)}] {source} "
                    f"{task['approach']} {task['attack']} {task['trust_mode']} trial={task['trial']}"
                )
            except Exception as exc:
                errors.append({"task": task, "error": repr(exc), "traceback": traceback.format_exc()})
                print(f"ERROR {task}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_task = {pool.submit(run_or_load_trial, task): task for task in tasks}
            for idx, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    result = future.result()
                    consume(result)
                    source = "cache" if result.get("cache_hit") else "run"
                    print(
                        f"[{idx}/{len(tasks)}] {source} "
                        f"{task['approach']} {task['attack']} {task['trust_mode']} trial={task['trial']}"
                    )
                except Exception as exc:
                    errors.append({"task": task, "error": repr(exc), "traceback": traceback.format_exc()})
                    print(f"ERROR {task}: {exc}")

    if trajectory_handle is not None:
        trajectory_handle.close()

    if not trial_summary_rows:
        raise SystemExit("No successful trials. Check the errors JSON.")

    trial_fields = list(trial_summary_rows[0].keys())
    write_csv(out_dir / "extra_trial_summary.csv", trial_summary_rows, trial_fields)

    cell_summary_rows = build_cell_summary(accumulators)
    cell_fields = list(cell_summary_rows[0].keys())
    write_csv(out_dir / "extra_cell_summary.csv", cell_summary_rows, cell_fields)

    lift_rows = build_paired_lift_rows(trial_summary_rows)
    if lift_rows:
        write_csv(out_dir / "paired_lift_vs_baseline.csv", lift_rows, list(lift_rows[0].keys()))

    manifest = {
        "n_rounds": args.n_rounds,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len(trial_summary_rows),
        "approaches": approaches,
        "attacks": attacks,
        "trust_modes": trust_modes,
        "fw_iters": args.fw_iters,
        "offline_benchmark": "Frank-Wolfe approximation to the same offline NSW objective",
        "trajectory_csv": None if args.skip_trajectory_csv else str(trajectory_path),
        "errors": errors,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    source_dir = out_dir / "source_files"
    source_dir.mkdir(exist_ok=True)
    for src in [
        ROOT / "NSW_Moradi_Final.py",
        SCRIPT_PATH,
    ]:
        if src.exists():
            dest = source_dir / src.name
            shutil.copy2(src, dest)
    approach_source_dir = source_dir / "report_extra_approaches"
    approach_source_dir.mkdir(exist_ok=True)
    approach_module_dir = SCRIPT_PATH.parent / "report_extra_approaches"
    if approach_module_dir.exists():
        for src in approach_module_dir.glob("*.py"):
            shutil.copy2(src, approach_source_dir / src.name)

    if not args.skip_plots:
        ensure_matplotlib()
        plot_ratio_heatmaps(plot_dir, cell_summary_rows, approaches, attacks)
        plot_trust_ablation_focus(plot_dir, cell_summary_rows, attacks)
        plot_malicious_share_stealth(plot_dir, cell_summary_rows)
        plot_adaptive_trajectories(plot_dir, accumulators, attacks, trust_modes)
        plot_expert_weight_trajectories(plot_dir, accumulators, attacks, trust_modes)
        plot_allocation_heatmaps(plot_dir, samples, n_legitimate=10)
        plot_paired_lifts(plot_dir, lift_rows)

    zip_path = shutil.make_archive(str(out_dir), "zip", out_dir)
    print("Done.")
    print(f"Trial summary: {out_dir / 'extra_trial_summary.csv'}")
    print(f"Cell summary: {out_dir / 'extra_cell_summary.csv'}")
    if not args.skip_trajectory_csv:
        print(f"Trajectory CSV: {trajectory_path}")
    print(f"Figures: {plot_dir}")
    print(f"Zip archive: {zip_path}")
    if errors:
        print(f"Completed with {len(errors)} errors. See run_manifest.json.")


if __name__ == "__main__":
    main()
