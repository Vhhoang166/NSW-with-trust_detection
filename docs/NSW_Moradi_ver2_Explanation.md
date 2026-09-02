# NSW_Moradi_ver2.py Explained

This note explains [`NSW_Moradi_ver2.py`](/Users/admin/Prof%20Sarper%20Aydin/NSW-with-trust_detection/NSW_Moradi_ver2.py) in plain language for someone who has read:

1. the Banerjee et al. NSW-with-predictions paper, and
2. the Moradi-style trust / clustering paper.

The goal of `NSW_Moradi_ver2.py` is:

- keep the trust-detection and graph ideas from the Moradi-based code,
- keep the attacker models and simulation machinery,
- replace the older allocation core with the `NSW_Only` style prediction-and-allocation engine,
- then test everything under many attack types and many plots.

This file is long, so the right way to read it is block by block.

## 1. Big Picture

At a high level, each round does this:

1. Generate true valuations for this round.
2. Let malicious agents decide whether to attack and how strongly to distort reports.
3. Let legitimate agents observe noisy trust signals about others.
4. Run Algorithm 1 style trust detection.
5. Optionally apply Moradi add-ons:
   - M4 implicit trust bootstrap
   - M3 merge-small-clusters safety net
6. Allocate the round's budget only over the currently trusted set using an `NSW_Only` style set-aside + greedy rule.
7. Update cumulative utilities and save statistics.

So the file is really:

- `NSW_Only` for allocation,
- `NSW_Moradi` for trust and attacks,
- extra plots and scenario sweeps around them.

## 2. File Header and Imports

Code block: top docstring and imports near the top of the file.

What it does:

- The docstring tells you this is a merged file.
- It says which ideas came from `NSW_Only.py`.
- It says which ideas came from `NSW_Moradi.py`.

Important imports:

- `numpy` is used for almost all math and arrays.
- `matplotlib` is used for the plots.
- `dataclass` stores experiment parameters cleanly.
- `Enum` defines attack types.
- `Path` builds output paths.
- `minimize` from SciPy solves the offline optimal NSW benchmark.
- `SpectralClustering` from sklearn is optional. If sklearn is missing, spectral detection just falls back safely.

Important environment line:

- `os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")`

This avoids matplotlib cache problems on systems where the normal cache folder is not writable.

## 3. Output Directory

Code block: `OUTPUT_DIR`

Purpose:

- All plots are written into an `outputs/` folder next to the Python file.

So when `main()` runs, it saves figures there instead of cluttering the project root.

## 4. Utility Helper: `binary_f1_score`

Code block: `binary_f1_score(...)`

Purpose:

- Compute F1 score without needing sklearn metrics.

Why it exists:

- The code compares three detectors:
  - Algorithm 1 trust detector
  - spectral clustering
  - sparsest-subgraph detector
- F1 is the main quality score for those detectors.

How it works:

- `tp` = true positives
- `fp` = false positives
- `fn` = false negatives
- F1 = `2*tp / (2*tp + fp + fn)`

If the denominator is zero, it returns `0.0`.

## 5. Attack Types

Code block: `class AttackType(Enum)`

This defines the 10 malicious behaviors the simulation can test:

- `CONSTANT_HIGH`: attacks often and strongly
- `CONSTANT_LOW`: attacks less strongly
- `INTERMITTENT`: alternates between stronger and weaker periods
- `STRATEGIC_BURST`: attacks in bursts
- `ADAPTIVE`: attacks less when already detected
- `RANDOM`: random behavior each round
- `COORDINATED`: group attack with synchronized bursts
- `TRUST_MIMICRY`: tries to stay just under the trust threshold
- `REPUTATION_POISON`: acts normal first, then betrays later
- `DRIFTING`: gets gradually harder to distinguish from legitimate agents

This enum is important because many later blocks switch behavior based on attack type.

## 6. Prediction Scenarios from `NSW_Only`

Code block:

- `PREDICTION_SCENARIOS`
- `PREDICTION_NOISE_RANGE`
- `generate_V_tilde(...)`

What `V_tilde` means:

- In the Banerjee paper, `V_tilde_i` is the predicted total value of agent `i` across all rounds.
- This file generates that prediction once at the start of the simulation.

The scenarios:

- `ideal`: perfect predictions
- `mild_noise`: realistic noisy predictions
- `severe_over`: everyone is overestimated by 10x
- `severe_under`: everyone is underestimated by 10x
- `asymmetric_outlier`: one agent gets a massive overestimate
- `adversarial`: high-value agents get low predictions and vice versa
- `random_permutation`: predictions are shuffled across agents
- `half_over_half_under`: half overestimated, half underestimated

What the function does:

1. Compute true total value `V = v_true.sum(axis=1)`.
2. Build a predicted version depending on the scenario.
3. Clip values below `1e-10` so logs and ratios do not break later.

Why this matters:

- The greedy NSW allocator depends heavily on predicted totals.
- So this block controls how "good" or "bad" the prediction model is.

## 7. Experiment Configuration

Code block: `@dataclass class ExperimentConfig`

This is the central parameter container.

Main groups of parameters:

- population:
  - `n_legitimate`
  - `n_malicious`
  - `n_rounds`
  - `n_trials`
- attack behavior:
  - `attack_type`
  - `base_attack_prob`
- trust signal model:
  - `trust_legitimate`
  - `trust_malicious_attack`
  - `trust_malicious_no_attack`
  - `trust_std`
- Algorithm 1 detector:
  - `xi_0`
  - `epsilon`
- allocation:
  - `allocation_alpha`
- coordinated attack:
  - `coalition_sync_every`
  - `coalition_amplify`
- new attack tuning:
  - `poison_warmup_rounds`
  - `mimicry_margin`
  - `drift_rate`
- valuation environment:
  - `valuation_model`
  - `connectivity`
- Moradi add-ons:
  - `moradi_sigma`
  - `sparsest_epsilon`
  - `moradi_merge_threshold`
  - `moradi_bootstrap_rounds`
- NSW_Only prediction scenario:
  - `prediction_scenario`
- labeling:
  - `scenario_label`

How to think about it:

- Banerjee-side parameters control allocation.
- Moradi-side parameters control trust graph construction and clustering behavior.
- Simulation-side parameters control how hard the world is.

## 8. Scenario Factory

Code block: `class ScenarioFactory`

Purpose:

- Build named presets so experiments feel like interpretable application settings instead of random numbers.

The presets:

- `iot_network()`
- `spectrum_sharing()`
- `compute_cluster()`
- `adversarial_overlap()`

Each preset chooses:

- number of legit and malicious agents
- trust gap difficulty
- valuation model
- connectivity
- attack type
- allocation alpha

This is very useful for plots because it lets the code say:

- "run the IoT scenario"
- instead of "run this exact long parameter list"

## 9. Trust Observation Generator

Code block: `class TrustObservationGenerator`

What it models:

- Legitimate agents do not see the hidden truth directly.
- They see noisy trust observations `alpha_ij(t)` in `[0,1]`.

How `observe(...)` works:

- If the observed agent is legitimate:
  - mean trust is `trust_legitimate`
- If the observed agent is malicious and attacking:
  - mean trust is `trust_malicious_attack`
- If the observed agent is malicious but not attacking:
  - mean trust is `trust_malicious_no_attack`
- If the attack type is `DRIFTING`:
  - malicious attackers slowly move upward toward legitimate trust values over time

Then:

- Gaussian noise with standard deviation `trust_std` is added
- result is clipped into `[0,1]`

Interpretation:

- This is the side-channel observation process feeding Algorithm 1.
- Lower malicious trust means the detector can eventually separate attackers from legitimate agents.

## 10. Attack Distortion Table

Code block: `_BASE_DISTORTION`

Purpose:

- For each attack type, store a default interval `(low, high)` for distortion factor `phi`.

Meaning of `phi`:

- If attacker attacks, reported value becomes:
  - `reported_value = true_value * phi`

So:

- very small `phi` means strong underreporting
- very large `phi` means strong overreporting
- values near `1.0` mean subtle attack or near-honest reporting

## 11. Malicious Agent

Code block: `class MaliciousAgent`

This is one of the most important blocks.

Role:

- It decides whether a malicious agent attacks in round `t`.
- If it attacks, it also picks how much to distort the report.

Internal state:

- `attack_history` stores attack decisions over time
- `distortion_history` stores distortion factors
- `_poison_active` tracks whether a poison-style agent has entered betrayal mode

Main method: `decide_attack(...)`

Inputs:

- current round `t`
- whether the agent is already detected
- optional coalition signal
- current detector threshold `xi_t`
- current beta gap relative to the detector leader

Output:

- `attacking`: `True` or `False`
- `phi`: distortion multiplier

How each attack type behaves:

- `CONSTANT_HIGH`
  - attacks with base probability
- `CONSTANT_LOW`
  - attacks less often and more mildly
- `INTERMITTENT`
  - uses a sinusoidal then decaying phase pattern
- `STRATEGIC_BURST`
  - strong bursts in certain modular windows
- `ADAPTIVE`
  - attacks less if already detected
- `RANDOM`
  - random probability every round
- `COORDINATED`
  - follows group burst signal
- `TRUST_MIMICRY`
  - attacks only when its beta gap is still below the trust threshold margin
- `REPUTATION_POISON`
  - stays almost honest during warm-up, then becomes aggressive
- `DRIFTING`
  - moderate attacks while becoming harder to detect through the trust channel

Final lines of the method:

- sample attack decision from Bernoulli probability
- if attacking, sample `phi` uniformly inside the chosen distortion interval
- store the history
- return decision and distortion

The helper `empirical_attack_rate()` just computes the average attack frequency over time.

## 12. Coalition Coordinator

Code block: `class CoalitionCoordinator`

Purpose:

- Handle synchronized group attacks.

Key idea:

- Every `K` rounds, the malicious group launches a burst.
- During burst:
  - malicious agents amplify their own reports
  - one legitimate target gets suppressed

Important methods:

- `get_signal(t)`
  - says whether the current round is a burst
  - picks which legitimate agent is the current target
- `apply(...)`
  - modifies the report vector in place during a coordinated burst

This models more dangerous collaborative behavior than independent attackers.

## 13. Trust Detector: Algorithm 1

Code block: `class TrustDetector`

This is the Banerjee/consensus-style trust detector inside the trustworthy mechanism.

State:

- `beta[j]` is cumulative trust evidence about agent `j`

Main methods:

- `threshold(t)`
  - computes `xi_t`
  - grows sublinearly over time
- `update(j, alpha)`
  - adds a new trust observation to `beta[j]`
- `classify(t)`
  - finds the best-scoring neighbor `j_bar`
  - calls agent `j` trusted if `max_beta - beta[j] <= xi_t`
  - otherwise marks `j` as detected
- `beta_gap(j)`
  - how far `j` is below the current leader

Plain-language meaning:

- Every legitimate observer keeps a scoreboard over other agents.
- The agents close to the current leader stay trusted.
- The agents too far below are considered suspicious.

## 14. NSW Objective and Offline Optimum

Code block:

- `compute_nsw(...)`
- `offline_optimal_nsw(...)`

`compute_nsw(...)`:

- takes a vector of utilities
- returns geometric mean
- if any utility is zero or negative, NSW is `0.0`

`offline_optimal_nsw(...)`:

- computes the benchmark with full future information
- only over legitimate agents
- solves:
  - maximize sum of log utilities
  - subject to each round's allocation summing to 1
  - allocations between 0 and 1

Important reason this block exists:

- The online mechanism is judged against this offline optimum.
- The reported `nsw_ratio` is:
  - achieved legitimate NSW / offline legitimate NSW

Why log objective:

- maximizing geometric mean is equivalent to maximizing sum of logs

Why fallback exists:

- if the optimizer fails, the code falls back to uniform allocation

## 15. Water-Filling Greedy Allocation

Code block: `compute_allocation_min_price_sec74(...)`

This is the Banerjee Section 7.4 style greedy rule.

Inputs:

- `predicted_util`: current predicted utility base for each candidate agent
- `values_round`: current round values
- `budget`: how much resource the greedy stage can distribute

Key idea:

- give more budget to agents with high ratio `v_i / u_i`
- that means:
  - high current value
  - but still under-served compared with others

How the function works:

1. Start with current predicted utilities `u`.
2. Sort agents by descending `v/u`.
3. Equalize marginal ratios gradually.
4. Spend budget until the water-filling condition is met.
5. Renormalize to exact budget if needed.

This is the mathematical heart of the greedy half.

## 16. Trusted Online Allocation

Code block: `trusted_online_allocation(...)`

This is the main merge point between the two papers.

Inputs:

- `reports`: current round reported values
- `trusted_ids`: who is currently allowed to receive trusted allocation
- `V_tilde`: predicted total values
- `u_greedy`: cumulative utility from previous greedy allocations
- `budget`
- `alpha`

What it returns:

- `alloc`: total allocation this round
- `set_aside_alloc`: equal-share part
- `greedy_alloc`: water-filling part

How it works:

1. If `trusted_ids` is empty, it falls back to uniform allocation over all agents.
2. Otherwise split budget into:
   - set-aside budget = `alpha * budget`
   - greedy budget = `(1-alpha) * budget`
3. Give equal set-aside share to trusted agents only.
4. Build greedy baseline for each trusted agent:
   - `u_greedy[i] + V_tilde[i] * alpha / N_trusted`
5. Run water-filling greedy allocation over trusted agents only.
6. Combine set-aside and greedy parts.
7. Renormalize total allocation if needed.

Interpretation:

- This is `NSW_Only` allocation after the trust filter has removed some agents.

## 17. Spectral Detection

Code block: `spectral_detection(...)`

Purpose:

- Use spectral clustering as a benchmark detector.

Input:

- `beta_matrix` or blended matrix transformed into adjacency form

How it works:

1. Symmetrize the matrix.
2. Clip negatives.
3. Zero the diagonal.
4. Run 2-cluster spectral clustering.
5. Flip labels if needed so the cluster size closer to `n_malicious` is treated as the malicious cluster.

Why it exists:

- Not part of the core mechanism.
- It is used only as a benchmark detector to compare against Algorithm 1 and Moradi's sparsest-subgraph idea.

## 18. Moradi M1: Blended Matrix

Code block: `build_blended_matrix(...)`

Purpose:

- Build a graph weight matrix combining trust and report similarity.

Formula:

- `W = sigma * T + (1-sigma) * S`

Where:

- `T` = normalized trust matrix from cumulative beta values
- `S` = similarity matrix from report-history correlation

How it works:

1. Copy beta matrix into `T`.
2. Normalize each row by its row max.
3. Symmetrize and zero the diagonal.
4. If there is enough history:
   - stack report vectors over time
   - center each agent's report sequence
   - normalize
   - compute cosine-style correlation
   - map correlation from `[-1,1]` into `[0,1]`
5. Blend trust and similarity with `sigma`.

Interpretation:

- `sigma=1` means pure trust graph
- `sigma=0` means pure report-similarity graph

## 19. Moradi M2: Sparsest-Subgraph Detection

Code block: `sparsest_subgraph_detection(...)`

Purpose:

- Find the most weakly connected group in the blended graph.
- That weakly connected group is interpreted as the malicious / outlier set.

How it works:

1. Start with all nodes in `S`.
2. Compute current density `rho_S`.
3. Keep track of the best lowest-density subset found so far.
4. Remove high-degree candidates iteratively.
5. Continue until the subset becomes small.
6. If the best subset is larger than `n_malicious`, trim to the weakest-connected nodes.
7. Return binary labels.

Why it matters:

- This is the main Moradi detector benchmark in the file.

## 20. Moradi M3: Merge Small Clusters

Code block: `merge_small_clusters(...)`

Purpose:

- Prevent the trusted set from becoming too tiny.

How it works:

1. If the trusted set already has at least `merge_threshold` agents, do nothing.
2. Otherwise consider all outside agents as candidates.
3. Score each candidate by similarity to the trusted set using beta values.
4. Add the strongest candidates back until the trusted set reaches the threshold.

Interpretation:

- This is a safety net.
- It can help restore isolated legitimate agents.
- But because it is label-agnostic, it can also re-admit an attacker.

## 21. Moradi M4: Implicit Trust Bootstrap

Code block: `implicit_trust_score(...)`

Purpose:

- Estimate trust when there are too few direct beta observations.

How it works:

1. Look at limited report history.
2. Mark rounds where agent `i` was "active" relative to its own median.
3. Check how often agent `j` is also active in those rounds.
4. Return that overlap fraction.

If history is too short, return neutral `0.5`.

Interpretation:

- This is not direct trust.
- It is a backup signal based on behavior overlap.

## 22. Valuation Generator

Code block: `_generate_valuations(...)`

Purpose:

- Create the hidden true valuation matrix `(N, T)`.

Supported models:

- `lognormal`
- `pareto`
- `dirichlet`

Why normalize columns:

- Each round represents one divisible resource with total budget `1`.
- So each column is normalized to sum to 1.

This means:

- the values are comparable across rounds
- allocation budget is consistent

## 23. Main Simulation Class

Code block: `class NSWTrustworthySim`

This is the engine that actually runs the simulation.

### 23.1 Constructor: `__init__`

This prepares everything.

Main fields:

- `self.N`, `self.T`: total agents and rounds
- `self.legit_ids`, `self.mal_ids`: index split
- `self.true_vals`: hidden true valuation matrix
- `self.V_tilde`: prediction vector created once at the start
- `self.observe_mask`: who can observe whom
- `self.mal_agents`: dictionary of malicious agents
- `self.coalition`: group attack controller
- `self.detectors`: one trust detector per legitimate agent
- `self.observation_counts`: counts of side-channel trust observations
- `self.utilities`: cumulative realized utilities
- `self.u_greedy`: cumulative utility from greedy allocations only
- `self.hist_reports`: history of reported valuations
- `self.history`: full round-by-round saved output
- `self.beta_matrix`, `self.blended_matrix`: final matrices for detector benchmarks

Key design choice:

- `V_tilde` is generated once from `true_vals` at startup, not re-estimated every round.

### 23.2 Helper: `_nsw_optimal`

Purpose:

- Cache the offline benchmark so it is solved only once.

### 23.3 Helper: `_majority_vote_detect`

Purpose:

- Each legitimate observer has its own detector.
- This method combines them by majority vote.

So an agent is globally detected if more than half of legitimate observers classify it as detected.

### 23.4 Helper: `_get_beta_gap`

Purpose:

- Compute average beta gap for a malicious agent across legitimate observers.

Why:

- `TRUST_MIMICRY` attackers use this to decide whether it is safe to attack.

### 23.5 Helper: `_current_xi_t`

Purpose:

- Get the current detector threshold from one representative legitimate detector.

### 23.6 Helper: `_snapshot_beta_matrix`

Purpose:

- At the end of a trial, save the final beta matrix.
- Then build the final blended matrix for spectral and sparsest-subgraph evaluation.

## 24. One Round of Simulation: `step()`

This is the most important function in the whole file.

### Step A: Pre-round detection

- If `t > 0`, compute who is already detected.
- Compute current threshold `xi_t`.

Why:

- Some attacker strategies depend on whether they were already detected.

### Step B: Generate reports

- Start from truthful reports: `reports = true_vals[:, t]`.
- For each malicious agent:
  - call `decide_attack(...)`
  - if attacking, distort that report

Special coordinated case:

- coordinated attacks use `CoalitionCoordinator`
- bursts can amplify malicious reports and suppress one legitimate target

### Step C: Side-channel trust observations

- Each legitimate agent observes other agents if `observe_mask[i, j]` is true.
- For each such pair, generate `alpha_obs` from `TrustObservationGenerator`.
- Update that observer's detector.
- Increment observation count.

Meaning:

- this is how trust evidence accumulates over time

### Step D: Post-observation detection and trusted set

- Run majority-vote detection again using the latest observations.
- Build the current trusted set as everyone not detected.

Then refresh `beta_matrix`.

Then Moradi M4:

- If history is long enough, use implicit trust to demote thinly observed suspicious agents.

Then Moradi M3:

- If trusted set is too small, merge back similar agents.

### Step E: Trusted allocation

- Call `trusted_online_allocation(...)`
- Give the full round budget according to:
  - trusted set
  - prediction vector
  - current cumulative greedy utility
  - chosen `alpha`

### Step F: Update realized utilities

- Add `true_value * allocation` to cumulative utilities.

Why true value, not report:

- welfare should be measured using real utility, not fake reports

Then:

- update `u_greedy` using only greedy allocation part

That mirrors the `NSW_Only` logic.

### Step G: Save history

- Save the reported valuation vector for this round.

This is later used for:

- blended graph similarity
- implicit trust bootstrap

### Step H: Compute statistics

The function computes:

- `tp`, `fp`, `fn`
- detection rate
- false positive rate
- false negative rate
- number of attackers active this round
- current allocation vectors
- cumulative utilities
- current legitimate NSW
- offline benchmark
- NSW ratio
- minimum legitimate utility
- fairness gap

Then it stores everything in one dictionary `row`.

Finally:

- append `row` to history
- increment time `t`
- return the row

## 25. Full Trial: `run()`

Code block: `run(...)`

Purpose:

- Repeat `step()` for all rounds.

Also:

- print periodic progress if `verbose=True`
- after final round, call `_snapshot_beta_matrix()`
- return the full round history

## 26. Evaluation Helpers

Code block:

- `convergence_round(...)`
- `run_trials(...)`
- `compare_all_types(...)`

### `convergence_round(...)`

- returns the first round where detection rate reaches a threshold, default `0.95`

### `run_trials(...)`

This is the experiment loop over random seeds.

For each trial:

1. create a new simulator
2. run it
3. build ground-truth malicious labels
4. run spectral detector on the final blended matrix
5. run sparsest-subgraph detector on the final blended matrix
6. build Algorithm 1 final labels from final detected set
7. compute F1 scores
8. collect final welfare and fairness statistics

This function returns one dictionary per trial.

### `compare_all_types(...)`

- repeats `run_trials(...)` for every attack type
- averages the key metrics
- prints a compact table

This is used later for attack comparison plots.

## 27. Plot Saving Helper

Code block: `_save(...)`

Purpose:

- make output directory if needed
- apply `tight_layout`
- save plot
- close figure
- print saved path

## 28. Plot Blocks

The plotting section does not change the mechanism. It only visualizes results.

### Plot 1: `plot_main_results`

Shows average over trials of:

- detection rate
- false positive rate
- legitimate NSW
- NSW ratio

This is the most standard summary plot.

### Plot 2: `plot_utility_trajectories`

Shows cumulative utility per agent over time:

- blue solid lines for legitimate agents
- red dashed lines for malicious agents

Useful for seeing who benefits over time.

### Plot 3: `plot_allocation_heatmap`

Shows round-by-round allocation matrix as a heatmap.

Useful for seeing whether malicious agents keep receiving budget.

### Plot 4: `plot_misclassification`

Shows:

- false positive rate
- false negative rate

over time.

Useful for judging detection quality dynamically.

### Plot 5: `plot_attack_comparison`

Compares average metrics across all 10 attack types.

### Plot 6: `plot_coordinated_vs_independent`

Focuses on:

- constant high
- strategic burst
- coordinated
- reputation poison

to compare harder attack coordination patterns.

### Plot 7: `plot_nsw_ratio_sensitivity`

Sweeps number of malicious agents and tracks final NSW ratio.

### Plot 8: `plot_alpha_sweep`

Sweeps allocation parameter `alpha` from 0 to 1 and records:

- NSW ratio
- minimum legitimate utility
- fairness gap

This shows the efficiency-fairness tradeoff.

### Plot 9: `plot_three_way_detector`

Compares detector F1 across attack types for:

- Algorithm 1
- spectral clustering
- Moradi sparsest-subgraph

### Plot 10: `plot_scenario_comparison`

Runs the named domain presets and compares:

- NSW ratio
- detection rate
- convergence round
- minimum utility

### Plot 11: `plot_sigma_sweep`

Sweeps Moradi blend parameter `sigma` and tracks:

- spectral F1
- sparsest-subgraph F1
- NSW ratio

It focuses on `TRUST_MIMICRY` because that attack is especially interesting for trust-vs-similarity tradeoffs.

### Plot 12: `plot_prediction_scenario_comparison`

This is the new merged-file plot.

It sweeps the `NSW_Only` prediction scenarios and compares:

- NSW ratio
- minimum legitimate utility
- detection rate

Its purpose is:

- not to change detection,
- but to show how prediction quality interacts with the trust-aware allocation system.

## 29. Main Program

Code block: `main()`

This is the script entry point.

What it does:

1. Create the output directory.
2. Print a banner showing which components are included.
3. Build `base_cfg`.
4. Run Scenario 1 baseline.
5. Print summary metrics.
6. Save Plots 1 to 4.
7. Run all later scenarios and save Plots 5 to 12.
8. Print the list of output files.

The scenarios in `main()` are:

1. baseline run
2. compare all attack types
3. coordinated vs independent attacks
4. NSW ratio sensitivity to number of malicious agents
5. alpha sweep
6. domain preset comparison
7. three-way detector benchmark
8. sigma sweep
9. prediction scenario comparison

## 30. Last Line: Script Execution

Code block:

```python
if __name__ == "__main__":
    main()
```

Meaning:

- If the file is run directly as a script, execute `main()`.
- If it is imported from somewhere else, do not automatically run the experiments.

## 31. How to Read This File as a Beginner

If you are new, read in this order:

1. `ExperimentConfig`
2. `AttackType`
3. `TrustDetector`
4. `generate_V_tilde`
5. `compute_nsw`
6. `offline_optimal_nsw`
7. `compute_allocation_min_price_sec74`
8. `trusted_online_allocation`
9. `NSWTrustworthySim.__init__`
10. `NSWTrustworthySim.step`
11. `run_trials`
12. the plot functions only after the simulation makes sense

That order matches the mental story:

- what world are we in?
- what attacks exist?
- how is trust learned?
- how are predictions built?
- how is budget allocated?
- how does one full round work?
- how do we evaluate the final result?

## 32. One-Sentence Summary of Each Major Component

- `generate_V_tilde`: creates predicted future importance of each agent.
- `TrustDetector`: learns who looks trustworthy from noisy observations.
- `MaliciousAgent`: controls how attackers behave.
- `trusted_online_allocation`: gives this round's resource using trust-filtered NSW logic.
- `NSWTrustworthySim.step`: connects attacks, trust, allocation, and welfare in one round.
- `run_trials`: repeats the whole process over many seeds and collects metrics.
- plot functions: turn those metrics into figures for analysis.

## 33. Final Mental Model

You can think of `NSW_Moradi_ver2.py` as a three-layer system:

Layer 1: world generation

- true values
- predictions
- malicious behavior
- noisy trust observations

Layer 2: decision system

- Algorithm 1 trust detector
- Moradi M1 to M4 add-ons
- trusted NSW allocation

Layer 3: evaluation

- welfare
- fairness
- detection quality
- plots

That is the whole file in one picture.
