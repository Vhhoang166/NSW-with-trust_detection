#!/usr/bin/env python3
import os, time, importlib.util
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')

spec=importlib.util.spec_from_file_location('runner','/mnt/data/run_nsw_cpu_fast.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
nsw=runner.nsw

base25=nsw.ExperimentConfig(
    n_legitimate=10,n_malicious=4,n_rounds=200,n_trials=25,
    attack_type=nsw.AttackType.BYZANTINE,base_attack_prob=0.75,
    trust_legitimate=0.63,trust_malicious_attack=0.52,trust_malicious_no_attack=0.63,trust_std=0.14,
    xi_0=0.25,allocation_alpha=0.50,moradi_sigma=0.60,coalition_sync_every=10,poison_warmup_rounds=50,
    connectivity=1.0,prediction_scenario='mild_noise',scenario_label='professor revision')
base5=nsw.replace(base25, n_trials=5)

print('[Recompute] 25-trial focused results for plots 9, 13, 14', flush=True)
st=time.time(); results25=nsw.compare_focused_types(base25); print('elapsed', time.time()-st, flush=True)

print('[Plot 9] Three-way detector benchmark', flush=True)
st=time.time(); nsw.plot_three_way_detector(results25); print('elapsed', time.time()-st, flush=True)

print('[Plot 13] Utility trajectories with Moradi blended graph', flush=True)
st=time.time()
for at in [nsw.AttackType.BYZANTINE, nsw.AttackType.BURST, nsw.AttackType.REPUTATION_POISON, nsw.AttackType.TRUST_MIMICRY]:
    nsw.plot_utility_with_moradi(results25[at]['trials'][0], nsw.replace(base25, attack_type=at), path=str(nsw.OUTPUT_DIR / f'plot13_utility_moradi_{at.value}.png'))
print('elapsed', time.time()-st, flush=True)

print('[Plot 14] Attack rate analysis', flush=True)
st=time.time(); nsw.plot_attack_rates(results25, base25); print('elapsed', time.time()-st, flush=True)

# Longer sweeps: reduced to 5 trials each so each plot completes in this runtime.
for label, func in [
    ('Plot 7', nsw.plot_nsw_ratio_sensitivity),
    ('Plot 8', nsw.plot_alpha_sweep),
    ('Plot 10', nsw.plot_sigma_sweep),
    ('Plot 11', nsw.plot_compound_breakdown),
    ('Plot 12', nsw.plot_reputation_poison_phases),
    ('Plot 15', nsw.plot_prediction_scenario_comparison),
]:
    print(f'[{label}] start with n_trials=5', flush=True)
    st=time.time(); func(base5); print(f'[{label}] elapsed {time.time()-st}', flush=True)
