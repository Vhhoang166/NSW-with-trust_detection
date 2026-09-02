#!/usr/bin/env python3
import os, sys, time, importlib.util
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
spec=importlib.util.spec_from_file_location('runner','/mnt/data/run_nsw_cpu_fast.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
nsw=runner.nsw
base=nsw.ExperimentConfig(n_legitimate=10,n_malicious=4,n_rounds=200,n_trials=5,attack_type=nsw.AttackType.BYZANTINE,base_attack_prob=0.75,trust_legitimate=0.63,trust_malicious_attack=0.52,trust_malicious_no_attack=0.63,trust_std=0.14,xi_0=0.25,allocation_alpha=0.50,moradi_sigma=0.60,coalition_sync_every=10,poison_warmup_rounds=50,connectivity=1.0,prediction_scenario='mild_noise',scenario_label='professor revision')
name=sys.argv[1]
funcs={
 'plot7': nsw.plot_nsw_ratio_sensitivity,
 'plot8': nsw.plot_alpha_sweep,
 'plot10': nsw.plot_sigma_sweep,
 'plot11': nsw.plot_compound_breakdown,
 'plot12': nsw.plot_reputation_poison_phases,
 'plot15': nsw.plot_prediction_scenario_comparison,
}
st=time.time(); print(f'[{name}] start', flush=True); funcs[name](base); print(f'[{name}] elapsed {time.time()-st}', flush=True)
