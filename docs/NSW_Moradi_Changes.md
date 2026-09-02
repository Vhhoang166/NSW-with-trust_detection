# NSW_Moradi.py: What Changed From the Original Moradi Version

The goal of the current version is:

- keep the Moradi-inspired ideas,
- fix logic problems from the first Moradi implementation,
- make the code runnable locally,
- make the results easier to trust and explain.

## What Stayed

The current Moradi version still keeps the main Moradi ideas:

- A blended trust/similarity graph.
- A sparsest-subgraph detector.
- A cluster-merging safety net.
- An implicit trust bootstrap idea.
- A three-way detector comparison.
- A sigma sweep for the Moradi trust/similarity blend.

## Main Changes

### 1. The file now runs locally

The original Moradi file used hardcoded output paths like:

```text
/mnt/user-data/outputs/...
```

The current file saves plots into this repository's local `outputs/` folder.
It also creates that folder automatically.

Why this matters:

- Other people can clone the branch and run the file without changing paths.
- The plots are saved in a predictable project-local location.

### 2. Matplotlib setup is safer

The current file sets a writable matplotlib cache directory before importing
matplotlib.

Why this matters:

- It avoids local runtime warnings or failures when the default user cache is
  not writable.

### 3. `sklearn` is optional

The original Moradi file required `sklearn`.

The current file still uses spectral clustering when `sklearn` is installed,
but it does not crash if `sklearn` is missing.

Why this matters:

- The main simulation can still run even without spectral clustering support.

### 4. NSW is computed correctly

The original Moradi file ignored agents with zero utility when computing NSW.

The current file returns NSW as `0.0` if any included agent has zero or negative
utility.

Why this matters:

- Nash Social Welfare is a geometric mean.
- If one included agent gets zero utility, the product is zero.
- Ignoring zero-utility agents makes welfare look better than it really is.

### 5. The offline benchmark is stronger

The original Moradi file used a greedy offline benchmark.

The current file uses an optimizer-based offline benchmark.

Why this matters:

- The NSW ratio is now compared against a better offline optimum.
- The result is more meaningful than comparing against a heuristic.

### 6. Predictions are no longer built from reports during the run

The original Moradi file updated prediction-like values using reports over
time.

The current file generates predicted total valuations once at the beginning of
the simulation.

Why this matters:

- Reports can be manipulated by malicious agents.
- Predictions should behave like external side information, not like a value
that attackers can directly poison during the run.

### 7. Allocation now uses the corrected water-filling rule

The original Moradi file used a softmax-style greedy allocation.

The current file uses the Paper 1 Section 7.4 water-filling rule.

The allocation is now split into:

- `set_aside_alloc`
- `greedy_alloc`
- total `alloc`

Why this matters:

- The allocation logic is closer to the NSW-with-predictions algorithm.
- It is easier to check that the budget is conserved.
- It makes the alpha sweep easier to explain.

### 8. The alpha sweep is now logically consistent

The original Moradi file allowed changing `allocation_alpha`, but the predicted
utility formula still assumed the half/half case.

The current file makes the predicted utility baseline depend on
`allocation_alpha`.

Why this matters:

- If `alpha = 0.0`, all budget is greedy.
- If `alpha = 0.5`, half is set-aside and half is greedy.
- If `alpha = 1.0`, all budget is equal share.
- The alpha sweep now measures the thing it claims to measure.

### 9. The online algorithm no longer uses hidden true valuations

The original Moradi file used true valuation history in some online logic.

The current file uses reported valuation history where the online algorithm
needs history.

Why this matters:

- The algorithm should not peek at hidden true values.
- True valuations are still used to evaluate actual utility, which is fine.
- But allocation and Moradi similarity should use observable information.

### 10. Coordinated attacks are simulated more consistently

The original Moradi file could call `decide_attack(...)` twice for a malicious
agent in the same round during coordinated attacks.

The current file calls it once per malicious agent per round.

Why this matters:

- Attack history and report distortion stay synchronized.
- The simulation is easier to reason about.

### 11. Moradi similarity now uses reported valuations

The original Moradi graph similarity used true valuation history.

The current Moradi graph similarity uses reported valuation history.

Why this matters:

- Report history is observable.
- It also helps expose attackers whose reports create unusual patterns.

### 12. Spectral clustering now uses the Moradi blended graph

The current file evaluates spectral clustering on the blended Moradi graph,
not only the raw beta trust matrix.

Why this matters:

- Spectral and sparsest-subgraph detectors are now compared on the same graph.
- The comparison is more fair.

### 13. Detector F1 scores are now computed correctly

The original Moradi file depended on `sklearn.metrics.f1_score`.

The current file includes a small local `binary_f1_score(...)` function.

Why this matters:

- Detector F1 still works if `sklearn.metrics` is unavailable.
- Algorithm 1 F1, spectral F1, and sparsest-subgraph F1 are compared fairly.

### 14. Plot 9 is now a true three-way F1 comparison

The original Moradi plot mixed different kinds of scores.

The current Plot 9 compares:

- Algorithm 1 F1,
- spectral clustering F1,
- sparsest-subgraph F1.

Why this matters:

- This is an apples-to-apples detector comparison.

### 15. Moradi bootstrap no longer leaks ground truth labels

The original Moradi bootstrap logic used known legitimate IDs in places where
the online algorithm should not know who is legitimate.

The current bootstrap logic is label-agnostic.

It also tracks actual observation counts instead of using positive beta as a
proxy for observation count.

Why this matters:

- The algorithm no longer gets hidden information about who is legitimate.
- Cold-start logic is based on actual observations.

### 16. Moradi cluster merging no longer leaks ground truth labels

The original Moradi merge step used known legitimate IDs as merge candidates.

The current merge step can consider all agents.

Why this matters:

- This removes ground-truth leakage.
- It keeps the Moradi safety-net idea, but with the real tradeoff:
  a malicious agent can be merged back if it looks trust-similar enough.

### 17. Experiment helpers preserve the full configuration

The original Moradi helper experiments rebuilt `ExperimentConfig` manually.
That could accidentally drop fields like:

- `connectivity`,
- `valuation_model`,
- `xi_0`,
- `prediction_noise_range`,
- `moradi_sigma`,
- `sparsest_epsilon`.

The current file uses `dataclasses.replace(...)`.

Why this matters:

- Sweeps and comparisons now keep the base configuration unless they
  intentionally change one field.

### 18. Some comments were corrected

The original Moradi file described the sparsest-subgraph detector as fully
count-free.

The current comments are more precise:

- the detector avoids a two-cluster partition assumption,
- but the benchmark still uses the configured malicious count to trim the final
  predicted malicious set when needed.

Why this matters:

- The code comments now match what the code actually does.

## Important Tradeoffs That Still Remain

Some behavior is intentional and was not removed.

### Trust filtering can hurt fairness

If a legitimate agent is falsely detected as malicious, it can lose allocation.
This is a security/fairness tradeoff.

The Moradi merge step may help reduce starvation, but it cannot guarantee that
false positives never hurt legitimate agents.

### Moradi merging can also help attackers

Because the merge step is now label-agnostic, it can also merge back a
malicious agent if that agent appears sufficiently trust-similar.

This is more honest than using ground-truth labels, but it is a real tradeoff.

### The sparsest-subgraph detector still uses the malicious count for reporting

The Moradi sparsest-subgraph step does not require spectral two-cluster
partitioning.

However, for binary F1 reporting, the implementation still trims large suspect
sets using the configured number of malicious agents.

## Verification

The current Moradi file was checked with:

```bash
env PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile NSW_Moradi.py
```

Additional smoke tests were also run for:

- `run_trials(...)`,
- allocation budget sums,
- set-aside and greedy budget split,
- Moradi blended matrix creation,
- bootstrap observation-count tracking,
- all attack types on a small configuration,
- Algorithm 1 F1 in a known perfect-detection case.

## Short Version

The current `NSW_Moradi.py` is the original Moradi idea cleaned up and made
consistent:

- no hidden true-value leakage in online logic,
- no ground-truth legitimate-ID leakage in Moradi bootstrap/merge,
- corrected NSW and offline benchmark,
- corrected allocation,
- corrected detector metrics,
- local runnable outputs,
- Moradi graph, sparsest detector, bootstrap, merge, and sigma sweep preserved.

