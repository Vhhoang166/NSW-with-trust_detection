# Offline NSW Solver Comparison at 500 Rounds

## Experiment setup

The experiment compared SLSQP and Frank–Wolfe on exactly the same offline
Nash social welfare optimization problem and deterministic valuation dataset.

- Legitimate agents: 10
- Malicious agents: 4
- Rounds: 500
- Random seed: 42
- Script: `compare_offline_solvers_500.py`
- Computer: Apple M1 MacBook Air

## Results

| Metric | SLSQP | Frank–Wolfe |
|---|---:|---:|
| Offline NSW | 9.6803455894 | 9.6805788244 |
| Runtime | 45,077.845 s | 0.102 s |
| Iterations | 164 | 632 |
| Solver status | Successful | Converged |
| Final optimality gap | Not reported | 9.249e-07 |
| Maximum budget error | 6.217e-15 | 8.882e-16 |

The SLSQP runtime was approximately **12 hours, 31 minutes, and 18 seconds**.
Frank–Wolfe completed in approximately **0.10 seconds**, producing a measured
speedup of **441,645×**.

The absolute difference between the reported NSW values was
`2.3323499133e-04`, corresponding to a relative difference of only
`0.002409%`. Frank–Wolfe produced the slightly higher NSW value. This does not
mean that it exceeded the mathematical optimum; it indicates that the two
numerical methods stopped at different numerical tolerances. SLSQP reported a
successful termination, while Frank–Wolfe satisfied the configured convergence
tolerance with a duality gap below `1e-6`.

Both methods respected the per-round allocation budget to floating-point
precision. Their maximum budget errors were below `1e-14`, so the comparison
does not reveal a feasibility problem in either solution.

## Conclusion

For this 500-round instance, Frank–Wolfe reproduced the SLSQP offline NSW value
to within `0.002409%` while reducing runtime from more than twelve hours to a
fraction of a second. SLSQP is therefore unsuitable for the planned
1,000–2,000-round experiments. Frank–Wolfe should be used as the offline NSW
benchmark for those experiments, with its final duality gap reported alongside
the result.

This is one deterministic comparison, so it establishes suitability for this
project configuration rather than a universal performance guarantee for every
possible dataset.
