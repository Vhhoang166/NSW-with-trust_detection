# NSW With Trust Detection

This repository studies online Nash social welfare allocation under strategic reporting attacks and trust filtering.

The current project focus is the allocation-side comparison built around `NSW_Moradi_Final.py`: fixed-alpha set-aside allocation, adaptive-alpha allocation, paper-inspired allocation variants, expert-advice control, and trust-mode ablations.

## Main Files

- `NSW_Moradi_Final.py` - main Moradi-final simulator and baseline trust-aware allocation code.
- `NSW_Only.py` - NSW-only/reference simulation code.
- `scripts/run_report_extra_experiments.py` - runner for the full trust-mode and allocation-variant benchmark.
- `scripts/report_extra_approaches/` - maintained modular implementations for the report benchmark variants.
- `scripts/build_trust_aware_report.py` - builds the final HTML and DOCX manuscript from completed CSV outputs.

## Reports

- `docs/trust_aware_online_nsw_manuscript.html` - current HTML manuscript.
- `docs/trust_aware_online_nsw_report.docx` - current DOCX report.
- `docs/combined_full_4500_summary/` - full 4,500-trial summary tables and the five primary report figures.

## Output Folders

- `output_400_rounds/` - earlier 400-round all-approach benchmark outputs, plots, trial CSVs, and source snapshots.
- `outputs/Output_Moradi/` - legacy Moradi-focused run logs and outputs.
- `archive/` - duplicate or older materials kept for provenance but removed from the root project view.

## Typical Workflow

Run extra experiments on a stronger CPU machine. If you use the local project environment, call:

```bash
.venv-report-extra/bin/python scripts/run_report_extra_experiments.py --help
```

Rebuild the report from existing outputs:

```bash
.venv-report-extra/bin/python scripts/build_trust_aware_report.py
```

If you use a fresh environment instead, install the dependencies first:

```bash
python -m pip install -r requirements_report_extra.txt
```

The report builder expects the main full-benchmark CSVs in:

```text
docs/combined_full_4500_summary/full_4500_cell_summary.csv
docs/combined_full_4500_summary/full_4500_trial_summary.csv
```

## Notes

- Root-level `approach_*.py` files from the older 400-round runner were duplicate source snapshots. They are archived under `archive/legacy_400_round_source/`.
- Duplicate root-level 400-round summary CSVs are archived under `archive/root_duplicate_outputs/`; the active copies remain in `output_400_rounds/`.
- Old rendered report draft folders are archived under `docs/archive/rendered_report_drafts/`.
