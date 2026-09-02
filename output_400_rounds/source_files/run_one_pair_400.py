#!/usr/bin/env python3
import os, sys, pickle, csv, json, time
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
ROOT=Path('/mnt/data')
sys.path.insert(0,str(ROOT))
import run_all_approaches_400 as r

def main():
    approach=sys.argv[1]; attack=sys.argv[2]
    n_rounds=int(os.environ.get('NSW_ROUNDS','400'))
    n_trials=int(os.environ.get('NSW_TRIALS','25'))
    out_dir=ROOT/f'nsw_pair_results_{n_rounds}'
    out_dir.mkdir(exist_ok=True)
    start=time.time(); rows=[]
    for trial in range(n_trials):
        res=r.run_one((approach, attack, trial, n_rounds, n_trials))
        rows.append(res)
    path=out_dir/f'{approach}__{attack}.pkl'
    with open(path,'wb') as f: pickle.dump(rows,f,protocol=pickle.HIGHEST_PROTOCOL)
    # quick aggregate json
    agg=r.aggregate_attack(rows, attack)
    agg['approach']=approach; agg['n_rounds']=n_rounds; agg['n_trials_completed']=len(rows)
    with open(out_dir/f'{approach}__{attack}.json','w') as f: json.dump(agg,f,indent=2)
    print(f'DONE {approach} {attack} trials={len(rows)} ratio={agg["avg_ratio"]:.3f} NSW={agg["avg_nsw"]:.4f} elapsed={time.time()-start:.1f}s path={path}')
if __name__=='__main__': main()
