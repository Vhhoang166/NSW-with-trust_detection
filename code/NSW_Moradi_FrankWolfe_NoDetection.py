#!/usr/bin/env python3
"""Malicious-agent baseline with the online detection gate disabled.

This runner is an apples-to-apples control for ``NSW_Moradi_FrankWolfe.py``:

* it uses the same valuations, malicious agents, attacks, seeds, allocator,
  Frank-Wolfe offline benchmark, 2,000-round default, and 10-trial default;
* neither Algorithm 1 nor online Moradi removes an agent from the trusted set,
  so malicious agents remain allocation-eligible for the entire experiment;
* trust observations and the Moradi matrices are still recorded as shadow
  diagnostics.  They do not affect admission to the allocator.

Keeping the attack model and seeds unchanged isolates the welfare effect of
the detection/exclusion gate.  Results are written to a separate directory so
they cannot overwrite the detection-enabled outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set

import NSW_Moradi_Final as moradi
import NSW_Moradi_FrankWolfe as frank_wolfe


class NSWNoDetectionSim(moradi.NSWTrustworthySim):
    """Run the original simulation while never excluding an agent online."""

    def _select_online_detected(self, alg1_detected: Set[int]) -> Set[int]:
        # Algorithm 1 and Moradi continue to run as shadow diagnostics, but
        # neither prediction is allowed to change the allocation-eligible set.
        return set()


def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Moradi malicious-agent baseline with Frank-Wolfe and "
            "the online detection/exclusion gate disabled."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "outputs_frank_wolfe_no_detection"
        ),
        help="Directory for generated control plots (default: %(default)s).",
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
        help="Use 20 rounds and 1 trial, and skip expensive parameter sweeps.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run the four focused attacks and core plots only.",
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
    # Install the fast benchmark and the no-detection simulation class into the
    # original experiment module.  All other experiment code remains shared.
    moradi.offline_optimal_nsw = frank_wolfe.cached_frank_wolfe_offline_nsw
    moradi.NSWTrustworthySim = NSWNoDetectionSim
    moradi._parse_cli_args = _parse_cli_args
    moradi.plot_utility_with_moradi = frank_wolfe.plot_utility_with_moradi_all_agents
    frank_wolfe.install_online_checkpoint_reporting(args.output_dir)

    print(
        "CONTROL RUN: malicious agents enabled; online detection/exclusion "
        "disabled."
    )
    print(
        "Trust and Moradi values are recorded only as shadow diagnostics; "
        "all agents remain allocation-eligible."
    )
    print(
        "Using cached Frank-Wolfe for the offline NSW benchmark "
        f"(max_iterations={frank_wolfe.FW_MAX_ITERATIONS}, "
        f"tolerance={frank_wolfe.FW_TOLERANCE:g})."
    )
    moradi.main(argv)
    frank_wolfe.organize_output_files(args.output_dir, control=True)


if __name__ == "__main__":
    main()
