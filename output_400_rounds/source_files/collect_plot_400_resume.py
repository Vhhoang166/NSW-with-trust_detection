#!/usr/bin/env python3
import os, sys, pickle, json, csv, shutil
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
ROOT=Path('/mnt/data'); sys.path.insert(0,str(ROOT))
import run_all_approaches_400 as r

def write_csv(path, rows, fieldnames):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def load_results(approach,n_rounds):
    pair_dir=ROOT/f'nsw_pair_results_{n_rounds}'
    results={at.value: [] for at in r.ATTACKS}
    agg_rows=[]; trial_rows=[]
    for at in r.ATTACKS:
        p=pair_dir/f'{approach}__{at.value}.pkl'
        with open(p,'rb') as f: rows=pickle.load(f)
        results[at.value]=rows
        agg=r.aggregate_attack(rows, at.value)
        agg['approach']=approach; agg['n_rounds']=n_rounds; agg['n_trials_completed']=len(rows)
        agg_rows.append(agg)
        trial_rows.extend([tr['summary'] for tr in rows])
    return results,agg_rows,trial_rows

def main():
    n_rounds=int(os.environ.get('NSW_ROUNDS','400')); n_trials=int(os.environ.get('NSW_TRIALS','25'))
    out_root=ROOT/f'nsw_all_approaches_{n_rounds}rounds_outputs'; out_root.mkdir(parents=True,exist_ok=True)
    (out_root/'source_files').mkdir(exist_ok=True)
    for fname in ['NSW_Moradi_Final.py','approach_common.py','approach_adaptive_alpha.py','approach_pace_trusted.py','approach_generalized_mean.py','approach_sample_resolving.py','approach_expert_advice.py','approach_robust_aggregation.py','run_all_approaches_400.py','run_one_pair_400.py','collect_plot_400.py','collect_plot_400_resume.py']:
        src=ROOT/fname
        if src.exists(): shutil.copy2(src,out_root/'source_files'/fname)
    cfg0=r.make_config(n_rounds,n_trials); errors=[]
    for approach in r.APPROACHES:
        app_dir=out_root/approach; app_dir.mkdir(exist_ok=True)
        # if already plotted enough, still regenerate summaries later; skip plotting if plot9 exists and no need.
        results,agg_rows,trial_rows=load_results(approach,n_rounds)
        agg_fields=['approach','attack_type','n_rounds','n_trials_completed','avg_nsw','std_nsw','avg_ratio','std_ratio','avg_dr','std_dr','avg_fpr','avg_min_util','avg_fairness_gap','avg_conv','avg_alg1_f1','avg_spec_f1','avg_sparse_f1','avg_offline_nsw']
        write_csv(app_dir/f'{approach}_summary.csv',agg_rows,agg_fields)
        write_csv(app_dir/f'{approach}_trial_level.csv',trial_rows,list(trial_rows[0].keys()))
        if not (app_dir/'plot9_trusted_detected_counts.png').exists() or (approach in ['sample_resolving','expert_advice','robust_aggregation']):
            try:
                r.plot_main(app_dir,approach,results,n_rounds,n_trials)
                r.plot_utility_and_alloc(app_dir,approach,results,cfg0)
                r.plot_misclass(app_dir,approach,results,n_rounds,n_trials)
                r.plot_summary(app_dir,approach,agg_rows)
                r.plot_beta(app_dir,approach,results,n_rounds,n_trials)
                r.plot_detector_f1(app_dir,approach,agg_rows)
                r.plot_attack_rates(app_dir,approach,results,n_rounds,n_trials,cfg0.n_malicious)
                r.plot_operational(app_dir,approach,results,n_rounds)
                r.plot_extra_keys(app_dir,approach,results,n_rounds)
            except Exception as e:
                errors.append({'approach':approach,'attack_type':'plot','error':repr(e)})
        with open(app_dir/f'{approach}_aggregate.json','w') as f: json.dump(agg_rows,f,indent=2)
        print(f'DONE APPROACH {approach} png={len(list(app_dir.glob("*.png")))}')
    # combined summary
    all_agg=[]; all_trials=[]
    for approach in r.APPROACHES:
        _,agg_rows,trial_rows=load_results(approach,n_rounds)
        all_agg.extend(agg_rows); all_trials.extend(trial_rows)
    agg_fields=['approach','attack_type','n_rounds','n_trials_completed','avg_nsw','std_nsw','avg_ratio','std_ratio','avg_dr','std_dr','avg_fpr','avg_min_util','avg_fairness_gap','avg_conv','avg_alg1_f1','avg_spec_f1','avg_sparse_f1','avg_offline_nsw']
    write_csv(ROOT/f'all_approaches_{n_rounds}rounds_summary.csv',all_agg,agg_fields)
    write_csv(out_root/f'all_approaches_{n_rounds}rounds_summary.csv',all_agg,agg_fields)
    write_csv(out_root/f'all_approaches_{n_rounds}rounds_trial_level.csv',all_trials,list(all_trials[0].keys()))
    # comparison with prior 200-round summary
    comparison=[]; prev={}; previous_path=ROOT/'all_approaches_summary.csv'
    if previous_path.exists():
        with open(previous_path,newline='') as f:
            for row in csv.DictReader(f):
                a=row.get('approach') or row.get('Approach')
                attack=row.get('attack_type') or row.get('Attack type') or row.get('attack')
                ratio=row.get('avg_ratio') or row.get('Avg ratio') or row.get('ratio')
                if a and attack and ratio:
                    try: prev[(a,attack)]=float(ratio)
                    except: pass
    for row in all_agg:
        old=prev.get((row['approach'],row['attack_type']),float('nan')); new=row['avg_ratio']
        comparison.append({'approach':row['approach'],'attack_type':row['attack_type'],'ratio_200_rounds':old,'ratio_400_rounds':new,'delta_400_minus_200':new-old if old==old else float('nan')})
    write_csv(ROOT/'all_approaches_200_vs_400_ratio_comparison.csv',comparison,['approach','attack_type','ratio_200_rounds','ratio_400_rounds','delta_400_minus_200'])
    write_csv(out_root/'all_approaches_200_vs_400_ratio_comparison.csv',comparison,['approach','attack_type','ratio_200_rounds','ratio_400_rounds','delta_400_minus_200'])
    # heatmap
    try:
        import numpy as np, matplotlib.pyplot as plt
        approaches=list(r.APPROACHES.keys()); attacks=[at.value for at in r.ATTACKS]
        mat=np.full((len(approaches),len(attacks)),np.nan)
        for row in all_agg: mat[approaches.index(row['approach']),attacks.index(row['attack_type'])]=row['avg_ratio']
        fig,ax=plt.subplots(figsize=(12,6)); im=ax.imshow(mat,aspect='auto',cmap='viridis')
        plt.colorbar(im,ax=ax,label='Avg NSW ratio vs offline')
        ax.set_yticks(np.arange(len(approaches))); ax.set_yticklabels(approaches)
        ax.set_xticks(np.arange(len(attacks))); ax.set_xticklabels([r.ATTACK_LABELS[r.AttackType(a)] for a in attacks],rotation=20,ha='right')
        mean=np.nanmean(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]): ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if mat[i,j] < mean else 'black')
        ax.set_title(f'All approaches: {n_rounds}-round average NSW ratio')
        r.savefig(out_root/'all_approaches_ratio_heatmap.png')
    except Exception as e: errors.append({'approach':'combined','attack_type':'plot','error':repr(e)})
    notes={'n_rounds':n_rounds,'n_trials_requested':n_trials,'approaches':list(r.APPROACHES.keys()),'attacks':[a.value for a in r.ATTACKS],'cpu_only':True,'execution_mode':'chunked one approach/attack at a time to avoid container timeout','offline_benchmark':'Frank-Wolfe approximation for same offline concave NSW objective; avoids SLSQP fallback-to-equal-allocation','errors':errors,'png_count':len(list(out_root.rglob('*.png')))}
    with open(ROOT/'NSW_ALL_APPROACHES_400_RUN_NOTES.txt','w') as f: json.dump(notes,f,indent=2)
    with open(out_root/'NSW_ALL_APPROACHES_400_RUN_NOTES.txt','w') as f: json.dump(notes,f,indent=2)
    zip_path=shutil.make_archive(str(ROOT/f'nsw_all_approaches_{n_rounds}rounds_outputs'),'zip',out_root)
    print('DONE RESUME COLLECT')
    print(f'png_count={notes["png_count"]}')
    print(f'errors={len(errors)}')
    print(f'zip={zip_path}')
if __name__=='__main__': main()
