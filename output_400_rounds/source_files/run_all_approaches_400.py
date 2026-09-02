#!/usr/bin/env python3
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

import sys, math, time, json, csv, shutil, traceback
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path('/mnt/data')
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import NSW_Moradi_Final as base
from NSW_Moradi_Final import AttackType, ExperimentConfig

# approach classes
from approach_adaptive_alpha import AdaptiveAlphaSim
from approach_pace_trusted import PaceTrustedSim
from approach_generalized_mean import GeneralizedMeanGreedySim
from approach_sample_resolving import SampleResolvingSim
from approach_expert_advice import ExpertAdviceSim
from approach_robust_aggregation import RobustAggregationSim

# ---------------- fast offline NSW benchmark ----------------
# Same objective as offline_optimal_nsw: max sum_i log(u_i), u_i=sum_t v_it x_it.
# Frank-Wolfe is much faster and avoids SLSQP fallback-to-equal-allocation.
_ORIG_INIT = base.NSWTrustworthySim.__init__
_OFFLINE_CACHE = {}

def _patched_init(self, config, seed=42):
    _ORIG_INIT(self, config, seed=seed)
    self._seed_for_cache = int(seed)

base.NSWTrustworthySim.__init__ = _patched_init

def _fast_offline_nsw_from_vals(v_true, legit_ids, iters=160):
    v = np.asarray(v_true[legit_ids, :], dtype=float)
    N, T = v.shape
    X = np.ones((N, T), dtype=float) / N
    u = np.maximum((v * X).sum(axis=1), 1e-12)
    # FW with diminishing steps. Starting equal split gives safe positive utilities.
    for k in range(iters):
        grad = v / u[:, None]
        winners = np.argmax(grad, axis=0)
        S = np.zeros_like(X)
        S[winners, np.arange(T)] = 1.0
        gamma = 2.0 / (k + 2.0)
        X = (1.0 - gamma) * X + gamma * S
        u = np.maximum((v * X).sum(axis=1), 1e-12)
    return float(np.exp(np.mean(np.log(u))))

def _patched_nsw_optimal(self):
    if self._nsw_opt is not None:
        return self._nsw_opt
    key = (getattr(self, '_seed_for_cache', None), self.cfg.n_legitimate, self.N, self.T, self.cfg.valuation_model)
    if key not in _OFFLINE_CACHE:
        _OFFLINE_CACHE[key] = _fast_offline_nsw_from_vals(self.true_vals, self.legit_ids)
    self._nsw_opt = _OFFLINE_CACHE[key]
    return self._nsw_opt

base.NSWTrustworthySim._nsw_optimal = _patched_nsw_optimal

# Make subclasses inherit patched base methods too (they already do via class MRO)

APPROACHES = {
    'baseline_fixed_alpha': base.NSWTrustworthySim,
    'adaptive_alpha': AdaptiveAlphaSim,
    'pace_trusted': PaceTrustedSim,
    'generalized_mean_greedy': GeneralizedMeanGreedySim,
    'sample_resolving': SampleResolvingSim,
    'expert_advice': ExpertAdviceSim,
    'robust_aggregation': RobustAggregationSim,
}
ATTACKS = [AttackType.BYZANTINE, AttackType.BURST, AttackType.REPUTATION_POISON, AttackType.TRUST_MIMICRY]
ATTACK_LABELS = {
    AttackType.BYZANTINE: 'Byzantine',
    AttackType.BURST: 'Burst',
    AttackType.REPUTATION_POISON: 'Reputation Poison',
    AttackType.TRUST_MIMICRY: 'Trust Mimicry',
}
ATTACK_COLORS = {
    AttackType.BYZANTINE: '#E53935',
    AttackType.BURST: '#FB8C00',
    AttackType.REPUTATION_POISON: '#8E24AA',
    AttackType.TRUST_MIMICRY: '#1E88E5',
}

METRIC_KEYS = ['detection_rate','fp_rate','fn_rate','nsw_legit','nsw_ratio','min_util','fairness_gap','beta_mal_gap_avg','beta_legit_gap_avg','xi_t','n_attacking']
EXTRA_KEYS = ['alpha_t','trust_risk','prediction_risk','controller_risk','pace_avg_price','pace_min_price','pace_max_price','gm_rho','gm_avg_baseline','gm_min_baseline','sample_window_used','sample_current_weight','sample_rho','expert_defensive_weight','expert_aggressive_weight','expert_gm_weight','robust_alpha','robust_report_delta']

def ci95(x, axis=0):
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    return 1.96 * np.nanstd(x, axis=axis) / math.sqrt(max(n, 1))

def rolling_mean_edge(values, window):
    x = np.asarray(values, dtype=float)
    if window <= 1 or x.size == 0:
        return x.copy()
    window = min(int(window), x.size)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(x, (left, right), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')

def binary_f1_score(true_labels, pred_labels):
    y_true = np.asarray(true_labels, dtype=int)
    y_pred = np.asarray(pred_labels, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = 2*tp + fp + fn
    return float(2*tp/denom) if denom > 0 else 0.0

def convergence_round(history, threshold=0.95):
    for row in history:
        if row['detection_rate'] >= threshold:
            return int(row['t'])
    return len(history)

def make_config(n_rounds=400, n_trials=25, attack_type=AttackType.BYZANTINE):
    return ExperimentConfig(
        n_legitimate=10,
        n_malicious=4,
        n_rounds=n_rounds,
        n_trials=n_trials,
        attack_type=attack_type,
        base_attack_prob=0.75,
        trust_legitimate=0.63,
        trust_malicious_attack=0.52,
        trust_malicious_no_attack=0.63,
        trust_std=0.14,
        xi_0=0.25,
        allocation_alpha=0.50,
        moradi_sigma=0.60,
        coalition_sync_every=10,
        poison_warmup_rounds=50,
        connectivity=1.0,
        prediction_scenario='mild_noise',
        scenario_label='all approaches 400 rounds',
    )

def _sanitize_extra(v):
    try:
        if v is None: return np.nan
        if isinstance(v, (int,float,np.integer,np.floating)): return float(v)
        return np.nan
    except Exception:
        return np.nan

def run_one(args):
    approach_name, attack_value, trial_idx, n_rounds, n_trials = args
    attack_type = AttackType(attack_value)
    cfg = make_config(n_rounds=n_rounds, n_trials=n_trials, attack_type=attack_type)
    seed = 42 + trial_idx * 13
    SimClass = APPROACHES[approach_name]
    sim = SimClass(cfg, seed=seed)
    hist = sim.run(verbose=False)
    T = len(hist)
    arrays = {k: np.array([row.get(k, np.nan) for row in hist], dtype=float) for k in METRIC_KEYS}
    extras = {}
    for k in EXTRA_KEYS:
        vals = [_sanitize_extra(row.get(k, np.nan)) for row in hist]
        if not np.all(np.isnan(vals)):
            extras[k] = np.array(vals, dtype=float)
    trusted_count = np.array([len(row.get('trusted', [])) for row in hist], dtype=float)
    detected_count = np.array([len(row.get('detected', [])) for row in hist], dtype=float)
    arrays['trusted_count'] = trusted_count
    arrays['detected_count'] = detected_count

    true_labels = np.array([0]*cfg.n_legitimate + [1]*cfg.n_malicious)
    spec_labels = base.spectral_detection(sim.blended_matrix, cfg.n_legitimate+cfg.n_malicious, cfg.n_malicious)
    sparse_labels = base.sparsest_subgraph_detection(sim.blended_matrix, cfg.n_legitimate+cfg.n_malicious, cfg.n_malicious, epsilon=cfg.sparsest_epsilon)
    alg1_labels = np.array([1 if i in hist[-1]['detected'] else 0 for i in range(cfg.n_legitimate+cfg.n_malicious)])
    summary = {
        'approach': approach_name,
        'attack_type': attack_type.value,
        'trial': trial_idx,
        'seed': seed,
        'final_nsw_legit': float(hist[-1]['nsw_legit']),
        'final_nsw_ratio': float(hist[-1]['nsw_ratio']),
        'final_dr': float(hist[-1]['detection_rate']),
        'final_fpr': float(hist[-1]['fp_rate']),
        'final_min_util': float(hist[-1]['min_util']),
        'final_fairness_gap': float(hist[-1]['fairness_gap']),
        'conv_round': convergence_round(hist),
        'spectral_f1': binary_f1_score(true_labels, spec_labels),
        'sparse_f1': binary_f1_score(true_labels, sparse_labels),
        'alg1_f1': binary_f1_score(true_labels, alg1_labels),
        'offline_nsw': float(hist[-1]['nsw_opt']),
    }
    sample = None
    if trial_idx == 0:
        sample = {
            'utilities': np.stack([row['utilities'] for row in hist], axis=0),
            'alloc': np.stack([row['alloc'] for row in hist], axis=0),
        }
    return {'summary': summary, 'arrays': arrays, 'extras': extras, 'sample': sample}

# ---------------- plots from compact results ----------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def stack_metric(trials, key):
    return np.stack([tr['arrays'][key] for tr in trials], axis=0)

def plot_main(approach_dir, approach_name, results, n_rounds, n_trials):
    rounds = np.arange(n_rounds)
    metrics = [('detection_rate','Detection Rate'),('fp_rate','False Positive Rate'),('nsw_legit','NSW legitimate'),('nsw_ratio','NSW ratio vs offline')]
    fig, axes = plt.subplots(4,1,figsize=(14,16),sharex=True)
    fig.suptitle(f'{approach_name}: Main Metrics, 400 Rounds, {n_trials} Trials', fontsize=12, fontweight='bold')
    for ax,(key,label) in zip(axes,metrics):
        for at in ATTACKS:
            trials = results[at.value]
            data = stack_metric(trials,key)
            mu=data.mean(axis=0); err=ci95(data, axis=0)
            col=ATTACK_COLORS[at]
            ax.plot(rounds,mu,color=col,lw=2,label=ATTACK_LABELS[at])
            lo,hi=mu-err,mu+err
            if key in ('detection_rate','fp_rate','nsw_ratio'):
                lo=np.clip(lo,0,None)
            if key in ('detection_rate','fp_rate'):
                hi=np.clip(hi,0,1); ax.set_ylim(0,1.05)
            ax.fill_between(rounds,lo,hi,color=col,alpha=.17)
        ax.set_ylabel(label); ax.grid(True,alpha=.3); ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel('Round')
    savefig(approach_dir/'plot1_main_metrics.png')

def plot_utility_and_alloc(approach_dir, approach_name, results, cfg):
    byz = results[AttackType.BYZANTINE.value][0]['sample']
    if byz is None: return
    utils=byz['utilities']; alloc=byz['alloc']; T,N=utils.shape; rounds=np.arange(T)
    fig,ax=plt.subplots(figsize=(13,5))
    for i in range(cfg.n_legitimate): ax.plot(rounds,utils[:,i],color='#1E88E5',lw=1,alpha=.55)
    for m in range(cfg.n_legitimate,N): ax.plot(rounds,utils[:,m],color='#E53935',lw=1.3,alpha=.7,ls='--')
    ax.plot([],[],color='#1E88E5',lw=2,label='Legitimate')
    ax.plot([],[],color='#E53935',lw=2,ls='--',label='Malicious')
    ax.set_title(f'{approach_name}: Per-Agent Cumulative Utility, Byzantine sample')
    ax.set_xlabel('Round'); ax.set_ylabel('Cumulative utility'); ax.grid(True,alpha=.3); ax.legend()
    savefig(approach_dir/'plot2_utility_trajectories_byzantine.png')
    fig,ax=plt.subplots(figsize=(14,5))
    im=ax.imshow(alloc.T,aspect='auto',interpolation='nearest',cmap='YlOrRd')
    ax.axhline(cfg.n_legitimate-.5,color='cyan',lw=2,label='Legitimate/Malicious boundary')
    plt.colorbar(im,ax=ax,label='Allocation share')
    ax.set_title(f'{approach_name}: Allocation heatmap, Byzantine sample')
    ax.set_xlabel('Round'); ax.set_ylabel('Agent id'); ax.legend()
    savefig(approach_dir/'plot3_allocation_heatmap_byzantine.png')

def plot_misclass(approach_dir, approach_name, results, n_rounds, n_trials):
    rounds=np.arange(n_rounds); fig,axes=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f'{approach_name}: Misclassification Rates, {n_rounds} rounds, {n_trials} trials',fontweight='bold')
    for ax,key,title in [(axes[0],'fp_rate','False Positive Rate'),(axes[1],'fn_rate','False Negative Rate')]:
        for at in ATTACKS:
            data=stack_metric(results[at.value],key); mu=data.mean(0); err=ci95(data,0); col=ATTACK_COLORS[at]
            ax.plot(rounds,mu,color=col,lw=2,label=ATTACK_LABELS[at])
            ax.fill_between(rounds,np.clip(mu-err,0,1),np.clip(mu+err,0,1),color=col,alpha=.17)
        ax.set_title(title); ax.set_xlabel('Round'); ax.set_ylim(0,1.05); ax.grid(True,alpha=.3); ax.legend(fontsize=8)
    savefig(approach_dir/'plot4_misclassification_rates.png')

def plot_summary(approach_dir, approach_name, agg_rows):
    labels=[ATTACK_LABELS[AttackType(r['attack_type'])] for r in agg_rows]
    x=np.arange(len(labels))
    metrics=[('avg_nsw','Avg NSW'),('avg_ratio','Avg NSW ratio'),('avg_dr','Avg detection rate'),('avg_alg1_f1','Avg Algorithm 1 F1')]
    fig,axes=plt.subplots(2,2,figsize=(14,9)); fig.suptitle(f'{approach_name}: Attack Summary',fontweight='bold')
    for ax,(key,title) in zip(axes.flat,metrics):
        vals=[r[key] for r in agg_rows]
        ax.plot(x,vals,'o-',lw=2.5)
        ax.set_xticks(x); ax.set_xticklabels(labels,rotation=10); ax.set_title(title); ax.grid(True,alpha=.3,axis='y'); ax.set_ylim(bottom=0)
        for xi,v in zip(x,vals): ax.text(xi,v+.01*max(max(vals),1e-9),f'{v:.3f}',ha='center',fontsize=9)
    savefig(approach_dir/'plot5_attack_summary.png')

def plot_beta(approach_dir, approach_name, results, n_rounds, n_trials):
    rounds=np.arange(n_rounds); fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
    fig.suptitle(f'{approach_name}: Beta Gap Trajectories and Threshold',fontweight='bold')
    for ax,at in zip(axes.flat,ATTACKS):
        mal=stack_metric(results[at.value],'beta_mal_gap_avg'); leg=stack_metric(results[at.value],'beta_legit_gap_avg'); xi=stack_metric(results[at.value],'xi_t')
        m_mu=mal.mean(0); m_err=ci95(mal,0); l_mu=leg.mean(0); xi_mu=xi.mean(0); col=ATTACK_COLORS[at]
        ax.plot(rounds,m_mu,color=col,lw=2,label='Malicious gap')
        ax.fill_between(rounds,np.clip(m_mu-m_err,0,None),m_mu+m_err,color=col,alpha=.18)
        ax.plot(rounds,l_mu,color='#1E88E5',lw=1.5,ls=':',label='Legitimate gap')
        ax.plot(rounds,xi_mu,color='black',lw=1.5,ls='--',label='Threshold')
        if at==AttackType.REPUTATION_POISON: ax.axvline(50,color='red',ls=':',alpha=.8)
        ax.set_title(ATTACK_LABELS[at]); ax.grid(True,alpha=.3); ax.set_ylim(bottom=0); ax.legend(fontsize=8)
    savefig(approach_dir/'plot6_beta_gap_trajectories.png')

def plot_detector_f1(approach_dir, approach_name, agg_rows):
    labels=[ATTACK_LABELS[AttackType(r['attack_type'])] for r in agg_rows]; x=np.arange(len(labels)); w=.25
    alg=[r['avg_alg1_f1'] for r in agg_rows]; spec=[r['avg_spec_f1'] for r in agg_rows]; sparse=[r['avg_sparse_f1'] for r in agg_rows]
    fig,ax=plt.subplots(figsize=(12,6))
    ax.bar(x-w,alg,width=w,label='Algorithm 1')
    ax.bar(x,spec,width=w,label='Spectral')
    ax.bar(x+w,sparse,width=w,label='Sparsest Subgraph')
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=10); ax.set_ylim(0,1.2); ax.set_ylabel('F1'); ax.set_title(f'{approach_name}: Detector F1 Comparison'); ax.legend(); ax.grid(True,alpha=.3,axis='y')
    savefig(approach_dir/'plot7_detector_f1_comparison.png')

def plot_attack_rates(approach_dir, approach_name, results, n_rounds, n_trials, n_mal=4):
    rounds=np.arange(n_rounds); fig,axes=plt.subplots(1,2,figsize=(14,5)); fig.suptitle(f'{approach_name}: Attack Rate Analysis',fontweight='bold')
    for at in ATTACKS:
        data=stack_metric(results[at.value],'n_attacking')/max(n_mal,1); roll=np.array([rolling_mean_edge(row,15) for row in data]); mu=roll.mean(0); err=ci95(roll,0); col=ATTACK_COLORS[at]
        axes[0].plot(rounds,mu,color=col,lw=2,label=ATTACK_LABELS[at]); axes[0].fill_between(rounds,np.clip(mu-err,0,1),np.clip(mu+err,0,1),color=col,alpha=.15)
        cum=np.cumsum(data,axis=1)/np.arange(1,n_rounds+1)[None,:]; c_mu=cum.mean(0); axes[1].plot(rounds,c_mu,color=col,lw=2,label=ATTACK_LABELS[at])
    for ax,title in zip(axes,['Rolling attack rate','Cumulative attack fraction']):
        ax.set_title(title); ax.set_xlabel('Round'); ax.set_ylim(0,1.05); ax.grid(True,alpha=.3); ax.legend(fontsize=8)
    savefig(approach_dir/'plot8_attack_rates.png')

def plot_operational(approach_dir, approach_name, results, n_rounds):
    rounds=np.arange(n_rounds); fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True); fig.suptitle(f'{approach_name}: Trusted and Detected Set Sizes',fontweight='bold')
    for at in ATTACKS:
        tr=stack_metric(results[at.value],'trusted_count'); det=stack_metric(results[at.value],'detected_count'); col=ATTACK_COLORS[at]
        axes[0].plot(rounds,tr.mean(0),color=col,lw=2,label=ATTACK_LABELS[at])
        axes[1].plot(rounds,det.mean(0),color=col,lw=2,label=ATTACK_LABELS[at])
    axes[0].set_ylabel('Trusted agents'); axes[1].set_ylabel('Detected agents'); axes[1].set_xlabel('Round')
    for ax in axes: ax.grid(True,alpha=.3); ax.legend(fontsize=8)
    savefig(approach_dir/'plot9_trusted_detected_counts.png')

def plot_extra_keys(approach_dir, approach_name, results, n_rounds):
    # Pick approach-specific keys that exist.
    keys=[]
    for trlist in results.values():
        for tr in trlist:
            for k in tr.get('extras',{}).keys():
                if k not in keys: keys.append(k)
    if not keys: return
    # Limit to 4 meaningful panels.
    preferred=[k for k in ['alpha_t','controller_risk','trust_risk','prediction_risk','pace_avg_price','gm_avg_baseline','sample_window_used','expert_defensive_weight','expert_aggressive_weight','expert_gm_weight','robust_report_delta'] if k in keys]
    if not preferred: preferred=keys[:4]
    preferred=preferred[:4]
    fig,axes=plt.subplots(len(preferred),1,figsize=(13,3.2*len(preferred)),sharex=True)
    if len(preferred)==1: axes=[axes]
    fig.suptitle(f'{approach_name}: Approach-Specific Signals',fontweight='bold')
    for ax,k in zip(axes,preferred):
        for at in ATTACKS:
            vals=[]
            for tr in results[at.value]:
                if k in tr['extras']: vals.append(tr['extras'][k])
            if vals:
                data=np.stack(vals,axis=0); mu=np.nanmean(data,axis=0); col=ATTACK_COLORS[at]
                ax.plot(np.arange(len(mu)),mu,color=col,lw=2,label=ATTACK_LABELS[at])
        ax.set_ylabel(k); ax.grid(True,alpha=.3); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Round')
    savefig(approach_dir/'plot10_approach_specific_signals.png')

def aggregate_attack(trials, attack_value):
    sums=[tr['summary'] for tr in trials]
    return {
        'attack_type': attack_value,
        'avg_nsw': float(np.mean([s['final_nsw_legit'] for s in sums])),
        'std_nsw': float(np.std([s['final_nsw_legit'] for s in sums])),
        'avg_ratio': float(np.mean([s['final_nsw_ratio'] for s in sums])),
        'std_ratio': float(np.std([s['final_nsw_ratio'] for s in sums])),
        'avg_dr': float(np.mean([s['final_dr'] for s in sums])),
        'std_dr': float(np.std([s['final_dr'] for s in sums])),
        'avg_fpr': float(np.mean([s['final_fpr'] for s in sums])),
        'avg_min_util': float(np.mean([s['final_min_util'] for s in sums])),
        'avg_fairness_gap': float(np.mean([s['final_fairness_gap'] for s in sums])),
        'avg_conv': float(np.mean([s['conv_round'] for s in sums])),
        'avg_alg1_f1': float(np.mean([s['alg1_f1'] for s in sums])),
        'avg_spec_f1': float(np.mean([s['spectral_f1'] for s in sums])),
        'avg_sparse_f1': float(np.mean([s['sparse_f1'] for s in sums])),
        'avg_offline_nsw': float(np.mean([s['offline_nsw'] for s in sums])),
    }

def write_csv(path, rows, fieldnames):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def main():
    n_rounds = int(os.environ.get('NSW_ROUNDS','400'))
    n_trials = int(os.environ.get('NSW_TRIALS','25'))
    workers = int(os.environ.get('NSW_WORKERS','8'))
    out_root = ROOT / f'nsw_all_approaches_{n_rounds}rounds_outputs'
    if out_root.exists(): shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root/'source_files').mkdir(exist_ok=True)
    for fname in ['NSW_Moradi_Final.py','approach_common.py','approach_adaptive_alpha.py','approach_pace_trusted.py','approach_generalized_mean.py','approach_sample_resolving.py','approach_expert_advice.py','approach_robust_aggregation.py', Path(__file__).name]:
        src=ROOT/fname
        if src.exists(): shutil.copy2(src,out_root/'source_files'/fname)
    print(f'Running {len(APPROACHES)} approaches x {len(ATTACKS)} attacks x {n_trials} trials x {n_rounds} rounds')
    print(f'CPU-only, workers={workers}')
    start=time.time()
    all_agg=[]; all_trials=[]; errors=[]
    cfg0=make_config(n_rounds,n_trials)
    for approach_name in APPROACHES:
        app_start=time.time()
        app_dir=out_root/approach_name
        app_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n[{approach_name}] starting')
        results={at.value: [] for at in ATTACKS}
        tasks=[]
        for at in ATTACKS:
            for trial in range(n_trials):
                tasks.append((approach_name, at.value, trial, n_rounds, n_trials))
        # Run with conservative parallelism to use CPU but avoid RAM reset.
        done=0
        if workers <= 1:
            for task in tasks:
                try:
                    res = run_one(task)
                    results[task[1]].append(res)
                except Exception as e:
                    err={'approach':task[0],'attack_type':task[1],'trial':task[2],'error':repr(e),'traceback':traceback.format_exc()}
                    errors.append(err)
                    print('ERROR',err['approach'],err['attack_type'],err['trial'],err['error'])
                done += 1
                if done % max(1, n_trials) == 0:
                    print(f'  {approach_name}: {done}/{len(tasks)} trials finished')
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs={ex.submit(run_one,t): t for t in tasks}
                for fut in as_completed(futs):
                    task=futs[fut]
                    try:
                        res=fut.result()
                        results[task[1]].append(res)
                    except Exception as e:
                        err={'approach':task[0],'attack_type':task[1],'trial':task[2],'error':repr(e),'traceback':traceback.format_exc()}
                        errors.append(err)
                        print('ERROR',err['approach'],err['attack_type'],err['trial'],err['error'])
                    done+=1
                    if done % max(1, n_trials) == 0:
                        print(f'  {approach_name}: {done}/{len(tasks)} trials finished')
        # Sort trials by trial id
        for at in ATTACKS:
            results[at.value].sort(key=lambda r: r['summary']['trial'])
        # aggregate
        agg_rows=[]
        for at in ATTACKS:
            if results[at.value]:
                agg=aggregate_attack(results[at.value], at.value)
                agg['approach']=approach_name
                agg['n_rounds']=n_rounds
                agg['n_trials_completed']=len(results[at.value])
                agg_rows.append(agg)
                all_agg.append(agg.copy())
                for tr in results[at.value]: all_trials.append(tr['summary'])
                print(f"  {at.value:18s} ratio={agg['avg_ratio']:.3f} NSW={agg['avg_nsw']:.4f} DR={agg['avg_dr']:.0%} F1={agg['avg_alg1_f1']:.3f}")
        # csv per approach
        agg_fields=['approach','attack_type','n_rounds','n_trials_completed','avg_nsw','std_nsw','avg_ratio','std_ratio','avg_dr','std_dr','avg_fpr','avg_min_util','avg_fairness_gap','avg_conv','avg_alg1_f1','avg_spec_f1','avg_sparse_f1','avg_offline_nsw']
        write_csv(app_dir/f'{approach_name}_summary.csv', agg_rows, agg_fields)
        trial_fields=list(all_trials[-1].keys()) if all_trials else []
        write_csv(app_dir/f'{approach_name}_trial_level.csv', [s for s in all_trials if s['approach']==approach_name], trial_fields)
        # plots
        try:
            plot_main(app_dir, approach_name, results, n_rounds, n_trials)
            plot_utility_and_alloc(app_dir, approach_name, results, cfg0)
            plot_misclass(app_dir, approach_name, results, n_rounds, n_trials)
            plot_summary(app_dir, approach_name, agg_rows)
            plot_beta(app_dir, approach_name, results, n_rounds, n_trials)
            plot_detector_f1(app_dir, approach_name, agg_rows)
            plot_attack_rates(app_dir, approach_name, results, n_rounds, n_trials, cfg0.n_malicious)
            plot_operational(app_dir, approach_name, results, n_rounds)
            plot_extra_keys(app_dir, approach_name, results, n_rounds)
        except Exception as e:
            errors.append({'approach':approach_name,'attack_type':'plot','trial':-1,'error':repr(e),'traceback':traceback.format_exc()})
            print('PLOT ERROR', approach_name, repr(e))
        # JSON compact summary
        with open(app_dir/f'{approach_name}_aggregate.json','w') as f: json.dump(agg_rows,f,indent=2)
        print(f'[{approach_name}] finished in {(time.time()-app_start)/60:.2f} min; png={len(list(app_dir.glob("*.png")))}')
    # combined csv
    agg_fields=['approach','attack_type','n_rounds','n_trials_completed','avg_nsw','std_nsw','avg_ratio','std_ratio','avg_dr','std_dr','avg_fpr','avg_min_util','avg_fairness_gap','avg_conv','avg_alg1_f1','avg_spec_f1','avg_sparse_f1','avg_offline_nsw']
    write_csv(ROOT/f'all_approaches_400rounds_summary.csv', all_agg, agg_fields)
    write_csv(out_root/'all_approaches_400rounds_summary.csv', all_agg, agg_fields)
    if all_trials:
        write_csv(out_root/'all_approaches_400rounds_trial_level.csv', all_trials, list(all_trials[0].keys()))
    # Compare with previous 200-round summary when available.
    previous_path = ROOT/'all_approaches_summary.csv'
    comparison_rows=[]
    if previous_path.exists():
        prev={}
        with open(previous_path,newline='') as f:
            for row in csv.DictReader(f):
                # support either field names from prior runner
                a=row.get('approach') or row.get('Approach')
                attack=row.get('attack_type') or row.get('Attack type') or row.get('attack')
                ratio=row.get('avg_ratio') or row.get('Avg ratio') or row.get('ratio')
                if a and attack and ratio:
                    try: prev[(a,attack)] = float(ratio)
                    except: pass
        for row in all_agg:
            key=(row['approach'],row['attack_type'])
            old=prev.get(key, np.nan)
            new=row['avg_ratio']
            comparison_rows.append({'approach':row['approach'],'attack_type':row['attack_type'],'ratio_200_rounds':old,'ratio_400_rounds':new,'delta_400_minus_200':(new-old if not np.isnan(old) else np.nan)})
        write_csv(ROOT/'all_approaches_200_vs_400_ratio_comparison.csv', comparison_rows, ['approach','attack_type','ratio_200_rounds','ratio_400_rounds','delta_400_minus_200'])
        write_csv(out_root/'all_approaches_200_vs_400_ratio_comparison.csv', comparison_rows, ['approach','attack_type','ratio_200_rounds','ratio_400_rounds','delta_400_minus_200'])
    # combined plot: average ratio by approach and attack
    try:
        approaches=list(APPROACHES.keys()); attacks=[at.value for at in ATTACKS]
        mat=np.full((len(approaches),len(attacks)),np.nan)
        for r in all_agg:
            mat[approaches.index(r['approach']),attacks.index(r['attack_type'])]=r['avg_ratio']
        fig,ax=plt.subplots(figsize=(12,6))
        im=ax.imshow(mat,aspect='auto',cmap='viridis')
        plt.colorbar(im,ax=ax,label='Avg NSW ratio vs offline')
        ax.set_yticks(np.arange(len(approaches))); ax.set_yticklabels(approaches)
        ax.set_xticks(np.arange(len(attacks))); ax.set_xticklabels([ATTACK_LABELS[AttackType(a)] for a in attacks],rotation=20,ha='right')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',color='white' if mat[i,j]<np.nanmean(mat) else 'black',fontsize=9)
        ax.set_title(f'All approaches: 400-round average NSW ratio')
        savefig(out_root/'all_approaches_ratio_heatmap.png')
    except Exception as e:
        errors.append({'approach':'combined','attack_type':'plot','trial':-1,'error':repr(e),'traceback':traceback.format_exc()})
    # notes/errors
    notes = {
        'n_rounds': n_rounds,
        'n_trials_requested': n_trials,
        'approaches': list(APPROACHES.keys()),
        'attacks': [a.value for a in ATTACKS],
        'workers': workers,
        'cpu_only': True,
        'offline_benchmark': 'Frank-Wolfe approximation for same offline concave NSW objective; avoids SLSQP fallback-to-equal-allocation',
        'elapsed_seconds': time.time()-start,
        'errors': errors,
        'png_count': len(list(out_root.rglob('*.png'))),
    }
    with open(ROOT/'NSW_ALL_APPROACHES_400_RUN_NOTES.txt','w') as f: json.dump(notes,f,indent=2)
    with open(out_root/'NSW_ALL_APPROACHES_400_RUN_NOTES.txt','w') as f: json.dump(notes,f,indent=2)
    # zip
    zip_base=str(ROOT/f'nsw_all_approaches_{n_rounds}rounds_outputs')
    zip_path=shutil.make_archive(zip_base,'zip',out_root)
    print('\nDONE')
    print(f'elapsed_min={(time.time()-start)/60:.2f}')
    print(f'png_count={notes["png_count"]}')
    print(f'errors={len(errors)}')
    print(f'zip={zip_path}')
    print(f'summary={ROOT/"all_approaches_400rounds_summary.csv"}')
    if comparison_rows:
        print(f'comparison={ROOT/"all_approaches_200_vs_400_ratio_comparison.csv"}')

if __name__=='__main__':
    main()
