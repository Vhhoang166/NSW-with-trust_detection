# Nash Social Welfare (NSW) Maximization with Trust-Based Malicious Detection

Online resource allocation with Nash Social Welfare objective and trust-based detection of malicious agents. Implements and extends ideas from **"Online Nash Social Welfare Maximization with Predictions"** with malicious agents and detection.

## Overview

- **Version 1**: Proportional allocation (greedy half allocated by reported value / predicted utility).
- **Version 2**: Water-filling allocation (iterative equalization of marginal utilities, with trust weighting).
- **Version 0**: Baseline prototype (reference only).

Experiments compare detection effectiveness (Option 1: vs offline ideal; Option 2: vs online clean) and plot utility trajectories over time.

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

```bash
pip install numpy scipy matplotlib
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

## How to Run

**Sensitivity analysis (parameter sweep over xi0, generates plots):**
```bash
python "nsw with trust version 1.py"
python "nsw with trust version 2.py"
```

**Utility trajectories (time-series of cumulative utility, uses optimal xi0):**
```bash
python utility_trajectories_version1.py   # xi0 = 0.29
python utility_trajectories_version2.py  # xi0 = 0.27
```

## Outputs

- `option1_detection_effectiveness_version1.png`, `option2_detection_effectiveness_version1.png`, `detection_rate_reference_version1.png`
- Same set for version 2
- `utility_trajectories_over_time_version1.png`, `utility_trajectories_over_time_version2.png`

## Parameters

- **Optimal xi0**: Version 1 = 0.29, Version 2 = 0.27 (Option 2, non-crashed region).
- Main globals: `N_trust`, `N_mal`, `T`, `c_mal`, `gamma`, `alpha_pairwise`; see script headers.

## Documentation

See `COMPARISON_V0_V1_V2.md` for a detailed comparison of Version 0, 1, and 2 (allocation, methodology, and usage).
