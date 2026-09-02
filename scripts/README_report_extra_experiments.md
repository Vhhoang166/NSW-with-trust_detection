# Extra Report Experiments

Use `scripts/run_report_extra_experiments.py` on a stronger CPU machine to
generate the missing report evidence without editing the main simulator files.
The runner is split into one small file per allocator variant under
`scripts/report_extra_approaches/`.

## File Layout

- `run_report_extra_experiments.py`: orchestration, caching, CSV export, plots.
- `report_extra_approaches/common.py`: shared trust-gate and trust-ablation logic.
- `report_extra_approaches/baseline_fixed_alpha.py`: fixed-alpha set-aside plus alpha extremes.
- `report_extra_approaches/adaptive_alpha.py`: adaptive trust-aware alpha.
- `report_extra_approaches/pace_trusted.py`: PACE-inspired trusted allocator.
- `report_extra_approaches/generalized_mean.py`: generalized-mean greedy allocator.
- `report_extra_approaches/sample_resolving.py`: sample-resolving approximation.
- `report_extra_approaches/expert_advice.py`: expert-advice meta-controller.
- `report_extra_approaches/robust_aggregation.py`: robust report aggregation.

## Install

From the repository root:

```bash
pip install -r requirements_report_extra.txt
```

Google Colab normally already has most of these packages, but running the
install command is still fine.

If you upload files manually to another website, upload:

- `NSW_Moradi_Final.py`
- `scripts/run_report_extra_experiments.py`
- `scripts/report_extra_approaches/`
- `requirements_report_extra.txt`

The runner also works if you place `run_report_extra_experiments.py` in the
same folder as `NSW_Moradi_Final.py`; put `report_extra_approaches/` next to
it too, then call it as `python run_report_extra_experiments.py ...`.

## Full report run

This runs all seven report approaches, all five attack settings, and all four
trust settings for 25 trials and 400 rounds:

```bash
python scripts/run_report_extra_experiments.py \
  --out-dir output_report_extra \
  --n-rounds 400 \
  --n-trials 25 \
  --jobs 8
```

Use `--jobs` equal to the number of CPU cores the website gives you.  GPU does
not help much because this simulator is CPU-heavy Python and NumPy code, not a
neural-network workload.

## Smaller but most important run

If the full run is too slow, run the core evidence first:

```bash
python scripts/run_report_extra_experiments.py \
  --out-dir output_report_extra_core \
  --n-rounds 400 \
  --n-trials 25 \
  --jobs 8 \
  --approaches baseline_fixed_alpha,adaptive_alpha,expert_advice \
  --attacks reputation_poison,trust_mimicry,compound \
  --trust-modes no_trust,algorithm1,oracle,sparse_online
```

This is the strongest evidence for the report because it tests the main
baseline, the clean proposed method, and the best empirical controller under
the attacks where trust is most likely to fail.

## Outputs

The script creates:

- `extra_trial_summary.csv`: one row per completed trial.
- `extra_cell_summary.csv`: mean and 95% CI by approach, attack, and trust mode.
- `extra_trajectory_rows.csv.gz`: round-by-round metrics, allocations, and utilities.
- `paired_lift_vs_baseline.csv`: paired lift against fixed-alpha set-aside.
- `figures/ratio_heatmap_*.png`: approach ratio heatmaps under each trust mode.
- `figures/trust_ablation_focus_ratio.png`: no-trust/current/oracle/sparse comparison.
- `figures/malicious_resource_share_*.png`: resource share captured by malicious agents.
- `figures/adaptive_alpha_trajectory_*.png`: alpha, trust risk, prediction risk, NSW.
- `figures/expert_weight_trajectory_*.png`: defensive/aggressive/generalized weights.
- `figures/allocation_heatmap_*.png`: stealth and compound allocation heatmaps.
- `run_manifest.json`: exact run settings plus any errors.
- `output_report_extra.zip`: archive of the whole output folder.

Every trial is cached in `trial_cache/`.  If the run stops halfway, rerun the
same command and it will continue from the cached completed trials.

## Optional extremes

To also include pure greedy and pure equal split reference rules:

```bash
python scripts/run_report_extra_experiments.py \
  --out-dir output_report_extra_with_extremes \
  --n-rounds 400 \
  --n-trials 25 \
  --jobs 8 \
  --include-extremes
```
