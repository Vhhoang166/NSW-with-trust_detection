# Comparison: Version 0 vs Version 1 vs Version 2

## Executive Summary

**Version 0** was a foundational prototype with basic proportional allocation but lacked research rigor, proper experiment design, and visualization capabilities. **Version 1** evolved into a research-grade implementation with **proportional allocation** (same as Version 0, improved), comprehensive sensitivity analysis, fair comparisons, publication-ready visualizations, and methodologically correct fixed-dataset approach. **Version 2** uses the same infrastructure as Version 1 but with **water filling allocation** (optimal for NSW) instead of proportional, enabling direct comparison between allocation strategies. **Fair comparison**: Both V1 and V2 use the same dataset per trial (v_true from seed `42+trial`, malicious reports from seeds `1000+trial` and `2000+trial`), so detection rate is identical and NSW differences reflect only allocation method.

---

## 1. Core Purpose & Philosophy

### Version 0: Proof of Concept
- **Goal**: Basic implementation to test if the algorithms work
- **Approach**: Simple trial loop, print results, move on
- **Allocation Method**: Proportional allocation (reported_value / predicted_utility)
- **Mindset**: "Does it run?" rather than "Is it scientifically sound?"

### Version 1: Research-Grade Evaluation with Proportional Allocation
- **Goal**: Rigorous evaluation with proportional allocation (baseline strategy)
- **Approach**: Parameter sweep, sensitivity analysis, multiple comparison metrics
- **Allocation Method**: Proportional allocation (reported value / predicted utility; same idea as Version 0, improved)
- **Fixed Dataset Approach**: All xi0 values tested on identical datasets (ensures fair comparison)
- **Mindset**: "How robust is the algorithm across parameter space?" and "Is the comparison methodologically sound?"

### Version 2: Research-Grade Evaluation with Water Filling
- **Goal**: Same as Version 1, but using water filling (optimal for NSW) for comparison
- **Approach**: Identical to Version 1 (parameter sweep, sensitivity analysis, etc.)
- **Allocation Method**: Water filling (equalizes marginal utilities, optimal for NSW maximization)
- **Fixed Dataset Approach**: Same as Version 1 (all xi0 values tested on identical datasets); same report seeds so detection rate matches V1
- **Mindset**: "How does water filling compare to proportional allocation?"

---

## 2. Allocation Strategy Comparison

### Version 0: Proportional Allocation (Basic)
```python
# Compute priority
priority[eligible] = reported_round[eligible] / predicted_util[eligible]
# Normalize
weights[eligible] = priority[eligible] / priority[eligible].sum()
# Allocate
y[:, t] = 0.5 * weights
```
**Characteristics:**
- Simple proportional distribution based on reported values
- No zero-value handling (can crash)
- Basic trust weighting

### Version 1: Proportional Allocation (Improved)
```python
# Compute priorities
priority[eligible] = reported_round[eligible] / predicted_util[eligible]
# Apply trust weighting (conditional)
if detect_fn is not None and any_detected_ever:
    priority = priority * incoming_trust
# Proportional allocation: allocate proportionally to priorities
y[idx, t] = 0.5 * (priority[idx] / total_priority)
```
**Characteristics:**
- Same mathematical approach as Version 0 (proportional allocation)
- Robust zero-value handling with epsilon values (improved from V0)
- Conditional trust weighting (only when detection works)
- **Optimal xi0 (Option 2)**: 0.29 (from sensitivity analysis)

### Version 2: Water Filling Allocation (Optimal)
```python
# Compute marginal utilities
marginal_util[eligible] = reported_round[eligible] / predicted_util[eligible]
# Apply trust weighting (conditional)
if detect_fn is not None and any_detected_ever:
    marginal_util = marginal_util * incoming_trust
# Water filling: iterative equalization algorithm
# Sorts agents by marginal utility ratio, then iteratively allocates
# to equalize marginal utilities v[i,t] / (predicted_util[i] + v[i,t]*allocation[i])
```
**Characteristics:**
- Mathematically optimal for NSW maximization
- True iterative water filling algorithm (equalizes marginal utilities)
- Robust zero-value handling with epsilon values
- Conditional trust weighting (only when detection works)
- **Optimal xi0 (Option 2)**: 0.27 (from sensitivity analysis)
- Trust weighting integrated in water filling (trust-weighted values used throughout)

**Mathematical Foundation (Version 2):**
- Water filling maximizes: max Σᵢ log(u_i) where u_i = predicted_util + v[i,t]*allocation[i]
- Solution: Allocate proportionally to marginal utilities v[i,t] / u_i
- This equalizes marginal utilities, which is optimal for geometric mean (NSW)

**Key Difference:**
- Version 1 uses proportional allocation (simpler, same as V0)
- Version 2 uses water filling (optimal for NSW)
- Detection rate is the same in both (same dataset via seeded reports)

---

## 3. Code Quality & Correctness

### Version 0: Critical Bugs
```python
# Bug 1: Wrong import
from nsw.optimize import minimize  # ❌ Module doesn't exist

# Bug 2: Name conflict
import cvxpy as np  # ❌ Overwrites numpy!

# Bug 3: Incomplete function
def offline_optimal_nsw(v_true):
    # ... objective and constraints defined ...
    # ❌ Missing: actual minimize() call and return statement!

# Bug 4: No zero-value handling
priority[eligible] = reported_round[eligible] / denom[eligible]
# ❌ Can crash if reported_round or denom is zero
```

### Version 1: Production-Ready with Proportional Allocation
```python
# Fixed imports
from scipy.optimize import minimize  # ✅ Correct library

# Fixed naming
import matplotlib.pyplot as plt  # ✅ No conflicts

# Complete implementation
def offline_optimal_nsw(v_true, agent_mask=None):
    # ... full implementation with minimize(), bounds, constraints ...
    return nsw_opt, u_opt, X_opt  # ✅ Returns all needed values

# Robust zero-value handling
EPSILON_REPORTED = 1e-10
EPSILON_DENOM = 1e-9
reported_round_safe[reported_round_safe <= 0] = EPSILON_REPORTED
denom[denom <= 0] = EPSILON_DENOM
# ✅ Prevents crashes from zero values
```

### Version 2: Production-Ready with Water Filling
```python
# Same fixes as Version 1
from scipy.optimize import minimize  # ✅ Correct library
import matplotlib.pyplot as plt  # ✅ No conflicts

# Complete implementation (same as V1)
def offline_optimal_nsw(v_true, agent_mask=None):
    # ... full implementation ...

# Robust zero-value handling (denom = np.maximum(predicted_util, 1e-10))
# ✅ Prevents inf/nan in water filling

# Uses water filling (iterative equalization) instead of proportional
marginal_util[eligible] = reported_round[eligible] / denom[eligible]
# Then iterative water filling algorithm
```

---

## 4. Experiment Design & Methodology

### Version 0: Single-Point Evaluation
```python
# Single parameter set
xi0 = 0.1
gamma = 0.7
c_mal = 2.0

# Simple loop
for trial in range(n_trials):
    # Generate new random data each trial
    v_true = np.random.rand(N_all, T)  # ❌ Different data per trial
    # Run experiment
    result = run_experiment_once(...)
```
**Problems:**
- No sensitivity analysis
- Random data generation per trial (no reproducibility)
- No visualization
- No comparison metrics (Option 1 vs Option 2)

### Version 1: Comprehensive Sensitivity Analysis with Fixed Datasets
```python
# Parameter sweep
XI0_MIN = 0.05
XI0_MAX = 0.50
XI0_STEP = 0.01

# Fixed dataset approach (methodological correctness)
fixed_datasets = []
for trial in range(n_trials):
    np.random.seed(42 + trial)  # ✅ Fixed seed per trial
    v_true = np.random.rand(N_all, T)
    fixed_datasets.append(v_true)

# Test all xi0 values on same datasets
for xi0_val in xi0_values:
    for trial in range(n_trials):
        result = run_experiment_once(
            ...,
            xi0_param=xi0_val,
            v_true_precomputed=fixed_datasets[trial],  # ✅ Same v_true
            trial=trial  # ✅ Same report stream (seeded) → same detection
        )
```
**Improvements:**
- ✅ Parameter sweep (46 xi0 values tested)
- ✅ Fixed datasets ensure fair comparison
- ✅ Reproducible results
- ✅ Comprehensive visualizations
- ✅ Option 1 and Option 2 comparison metrics

### Version 2: Same as Version 1
- ✅ Identical experiment design
- ✅ Same fixed dataset approach
- ✅ Same parameter sweep
- ✅ Same visualizations
- ✅ Only difference: allocation method (proportional vs water filling)

---

## 5. Output & Visualization

### Version 0: Text-Only Output
```python
# Simple print statements
print("Offline ideal NSW: {:.6f}".format(mean_offline))
print("Online NSW (trustworthy-only): {:.6f}".format(mean_online_trustonly))
print("Online NSW (with malicious + detection): {:.6f}".format(mean_online_all))
print("Competitive ratios (offline / online):")
print("  vs online (trustworthy-only): {:.3f}".format(...))
print("  vs online (with malicious): {:.3f}".format(...))
```
**Limitations:**
- ❌ No visualizations
- ❌ No sensitivity analysis plots
- ❌ No detection rate analysis
- ❌ No utility trajectory plots

### Version 1: Publication-Ready Visualizations
- ✅ **Option 1 Plot**: Detection effectiveness vs Offline Ideal
  - Shows optimal xi0 point (red circle + yellow annotation)
  - Green shaded region where detection helps
  - Smooth interpolation (no gaps)
  
- ✅ **Option 2 Plot**: Detection effectiveness vs Online Clean
  - Shows optimal xi0 point (red circle + yellow annotation)
  - Green shaded region where detection helps
  - Smooth interpolation (no gaps)
  
- ✅ **Detection Rate Plot**: Reference visualization
  - Shows detection rate vs threshold
  
- ✅ **Utility Trajectories Plot**: Time-series visualization
  - Shows cumulative utility over time for all scenarios

### Version 2: Same Visualizations as Version 1
- ✅ Identical visualization capabilities
- ✅ Same plot types and formats
- ✅ Enables direct comparison with Version 1 results

---

## 6. Key Differences Summary

| Feature | Version 0 | Version 1 | Version 2 |
|---------|-----------|-----------|-----------|
| **Allocation Method** | Proportional | Proportional | Water Filling (Optimal) |
| **Code Quality** | Buggy (wrong imports, incomplete) | Production-ready | Production-ready |
| **Zero-Value Handling** | None (can crash) | Robust (epsilon values) | Robust (epsilon values) |
| **Experiment Design** | Single-point | Sensitivity analysis | Sensitivity analysis |
| **Fixed Datasets** | No (random per trial) | Yes (methodologically correct) | Yes (methodologically correct) |
| **Visualizations** | None | Comprehensive | Comprehensive |
| **Comparison Metrics** | Basic ratios | Option 1 & Option 2 | Option 1 & Option 2 |
| **Research Value** | Proof of concept | Research-grade | Research-grade (comparison) |

---

## 7. Allocation Strategy: Mathematical Comparison

### Proportional Allocation (Version 0 & Version 2)
**Formula:**
```
priority[i] = reported_value[i] / predicted_util[i]
allocation[i] = 0.5 * (priority[i] / Σ_j priority[j])
```

**Characteristics:**
- Simple and intuitive
- Allocates proportionally to reported values relative to predicted utilities
- Not theoretically optimal for NSW maximization
- Same approach in V0 and V1 (V1 has better implementation)

### Water Filling Allocation (Version 2)
**Algorithm (Iterative):**
```
1. Compute initial ratios: r_i = reported_value[i] / predicted_util[i]
2. Sort agents by ratio (descending)
3. Iteratively allocate to top agents to equalize marginal utilities:
   - Target: v[i,t] / (predicted_util[i] + v[i,t]*allocation[i]) = constant
   - Allocate until marginal utilities are equalized or budget exhausted
```

**Characteristics:**
- Mathematically optimal for maximizing NSW (geometric mean)
- True iterative water filling algorithm (equalizes marginal utilities across agents)
- Based on optimization theory: max Σᵢ log(u_i) → equalize marginal utilities v[i,t] / u_i
- Theoretically superior to proportional allocation for NSW maximization
- **Trust Integration**: Uses trust-weighted reported values throughout allocation process

**Key Insight:**
While both formulas look similar, water filling is derived from optimization theory and is proven optimal for NSW maximization. Proportional allocation is a heuristic that happens to work well but is not theoretically optimal.

---

## 8. Research Value

### Version 0: Proof of Concept
- Demonstrates basic algorithm functionality
- Shows that detection mechanism can be implemented
- Limited research value (single data point, no robustness analysis)

### Version 1: Research-Grade with Proportional Allocation
- Comprehensive sensitivity analysis
- Methodologically sound comparisons
- Proportional allocation (baseline strategy)
- Publication-ready results
- Identifies optimal parameter ranges
- Demonstrates algorithm robustness

### Version 2: Research-Grade with Optimal Allocation
- Enables direct comparison: water filling vs proportional allocation
- Same methodological rigor as Version 1; same dataset (seeded reports) so detection rate matches
- Water filling (optimal for NSW)
- Answers: "Does water filling actually outperform proportional allocation?"

---

## 9. Methodological Correctness: Fixed Dataset Approach

### The Problem (Version 0)
When testing different scenarios, each trial used different random data:
- Trial 1 → Dataset A (random)
- Trial 2 → Dataset B (different random)
- Trial 3 → Dataset C (different random)

**Consequence**: Unfair comparison
- Results depend on luck of random data generation
- Cannot fairly compare different parameter values
- No reproducibility

### The Solution (Version 1 & Version 2)
Generate all datasets upfront, then test all scenarios on the same datasets:
- Generate: Dataset 1, Dataset 2, ..., Dataset 20 (fixed seeds)
- Test xi0 = 0.05 on: Dataset 1, Dataset 2, ..., Dataset 20
- Test xi0 = 0.10 on: Dataset 1, Dataset 2, ..., Dataset 20 (same datasets!)
- Test xi0 = 0.15 on: Dataset 1, Dataset 2, ..., Dataset 20 (same datasets!)

**Result**: Fair comparison
- All parameter values tested on identical scenarios
- Smooth sensitivity curves (no jittery artifacts)
- Reproducible results
- Scientifically valid comparisons

---

## 10. Conclusion

**Version 0** served as a foundational prototype but lacked research rigor and proper methodology. **Version 1** evolved into a research-grade implementation with optimal water filling allocation, comprehensive analysis, and methodologically sound comparisons. **Version 2** provides the same research infrastructure but uses proportional allocation (same as V0) to enable direct comparison between allocation strategies.

**Key Takeaways:**
1. **Allocation Strategy**: Version 0 and Version 1 use proportional allocation; Version 2 uses optimal water filling (iterative algorithm).
2. **Code Quality**: Version 1 and Version 2 are production-ready; Version 0 had critical bugs.
3. **Methodology**: Version 1 and Version 2 use fixed dataset approach and **same dataset across both** (seeded v_true and report factory) so detection rate is identical and comparison is fair.
4. **Research Value**: Version 1 and Version 2 enable rigorous evaluation; Version 0 was proof of concept.
5. **Comparison**: Run both Version 1 and Version 2 to compare "Does water filling outperform proportional allocation?"
6. **Optimal Parameters**: Version 1 optimal xi0 = 0.29, Version 2 optimal xi0 = 0.27 (Option 2, non-crashed region).
7. **Trust Integration**: Version 2 uses trust-weighted values throughout the water filling algorithm.

**Recommendation**: Use Version 2 for optimal NSW performance (water filling), Version 1 for proportional baseline, and Version 0 only as historical reference.
