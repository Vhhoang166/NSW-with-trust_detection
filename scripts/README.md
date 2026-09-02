# Scripts

This folder contains the maintained experiment and report-building code.

## Current Runners

- `run_report_extra_experiments.py` - full benchmark runner for allocation variants, trust modes, attacks, resource leakage, confidence summaries, and report-ready CSV outputs.
- `build_trust_aware_report.py` - report generator that reads completed CSV outputs and writes the HTML and DOCX manuscript.

## Current Allocation Variants

The maintained modular variants live in `report_extra_approaches/`:

- `baseline_fixed_alpha.py`
- `adaptive_alpha.py`
- `pace_trusted.py`
- `generalized_mean.py`
- `sample_resolving.py`
- `expert_advice.py`
- `robust_aggregation.py`
- `common.py`

The older root-level `approach_*.py` files were archived because they duplicated the 400-round source snapshot rather than the current report benchmark modules.
