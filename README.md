# NSW With Trust Detection

This repository studies online Nash social welfare allocation under strategic reporting attacks and trust filtering.

The current project focus is the trust-aware NSW simulation built around
`NSW_Moradi_Final.py`, with a faster Frank-Wolfe offline benchmark for
long-horizon experiments.

## Main Files

- `NSW_Moradi_Final.py` - main Moradi-final simulator and baseline trust-aware allocation code.
- `NSW_Moradi_FrankWolfe.py` - fast Moradi runner using a cached Frank-Wolfe offline benchmark.
- `NSW_Only.py` - NSW-only/reference simulation code.
- `compare_offline_solvers_500.py` - reproducible 500-round SLSQP versus Frank-Wolfe comparison.

## Reports

- `docs/trust_aware_online_nsw_manuscript.html` - current HTML manuscript.
- `docs/trust_aware_online_nsw_report.docx` - current DOCX report.
- `docs/combined_full_4500_summary/` - full 4,500-trial summary tables and the five primary report figures.

Generated output folders are intentionally excluded from the maintained project
tree. New experiment runs should write to a fresh output directory.

## Typical Workflow

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the faster Moradi experiment with one trial and only the core plots:

```bash
python3 NSW_Moradi_FrankWolfe.py --rounds 1000 --core-only
```

Reproduce the 500-round solver comparison:

```bash
python3 compare_offline_solvers_500.py
```

## Solver Validation

The measured 500-round solver comparison is documented in
`docs/offline_solver_comparison_500_rounds.md`.
