"""Pooled TC-STAG analysis. Usage: python analyze_tc.py <root>  where <root> contains run folders
named PEMS04_<variant>_seed<k>/ (te, tcstag) and baseline folders PEMS04_AG_seed<k>/ (each with test_preds.npz).
Search is recursive, so just unzip everything into one folder."""
import sys, glob, os, re, numpy as np, pandas as pd
root = sys.argv[1] if len(sys.argv) > 1 else '.'
START_DOW, B, rng = 0, 4000, np.random.default_rng(0)
def regime(t):
    tod = t % 288; dow = (t // 288 + START_DOW) % 7
    peak = ((tod >= 84) & (tod < 120)) | ((tod >= 192) & (tod < 228))
    return np.where(dow >= 5, 'weekend', np.where(peak, 'weekday_peak', 'weekday_offpeak'))
runs = {}
for p in glob.glob(os.path.join(root, '**', 'test_preds.npz'), recursive=True):
    d = os.path.basename(os.path.dirname(p)); m = re.match(r'PEMS04_(te|tcstag|AG)_seed(\d+)$', d)
    if m: runs[('baseline' if m[1] == 'AG' else m[1], int(m[2]))] = p
print('found runs:', sorted(runs))
E, T, ref = {}, None, None
for k, p in runs.items():
    z = np.load(p); pred, tgt, ts = z['prediction'], z['target'], z['timestamp'][:, 0].astype(int)
    if ref is None: ref = (tgt, ts)
    assert np.array_equal(ts, ref[1]) and np.allclose(tgt, ref[0]), 'test windows differ: %s' % (k,)
    E[k] = pred
tgt, ts = ref; day = ts // 288; reg = regime(ts); days = np.unique(day)
def win_mae(pred): return np.abs(pred - tgt).mean(axis=(1, 2))
def mae(pred, mask=None): w = win_mae(pred); return w.mean() if mask is None else w[mask].mean()
def ens(vs, seeds=None):
    ks = [k for k in E if k[0] == vs and (seeds is None or k[1] in seeds)]; return np.mean([E[k] for k in ks], 0), len(ks)
print('\n== per-seed MAE =='); print(pd.Series({k: mae(v) for k, v in sorted(E.items())}).round(3).to_string())
print('\n== regime MAE (mean over seeds of per-seed MAE) ==')
rows = {}
for v in ['baseline', 'te', 'tcstag']:
    ks = [k for k in E if k[0] == v]
    if ks: rows[v] = {r: np.mean([mae(E[k], reg == r) for k in ks]) for r in ['weekday_peak', 'weekday_offpeak', 'weekend']} | {'n_seeds': len(ks)}
df = pd.DataFrame(rows).T; print(df.round(3).to_string())
for rf in ['baseline', 'te']:
    if rf in df.index: print('\n%% change vs %s' % rf); print((100 * (df.drop(columns='n_seeds') / df.loc[rf, df.columns[:3]] - 1)).drop(rf).round(2).to_string())
print('\n== ensembles (mean of seed predictions) ==')
for v in ['baseline', 'te', 'tcstag']:
    p, n = ens(v)
    if n: print('%-9s %d seeds: MAE %.3f' % (v, n, mae(p)))
common = sorted(set(k[1] for k in E if k[0] == 'baseline') & set(k[1] for k in E if k[0] == 'te'))
if len(common) >= 2:
    for v in ['baseline', 'te', 'tcstag']:
        p, n = ens(v, common)
        if n: print('same seeds %s | %-9s MAE %.3f' % (common, v, mae(p)))
def boot(a, b, mask=None, label=''):
    wa, wb = win_mae(a), win_mae(b); m = np.ones(len(wa), bool) if mask is None else mask
    S = np.array([[(wa - wb)[m & (day == d)].sum(), (m & (day == d)).sum()] for d in days])
    S = S[S[:, 1] > 0]; n = len(S); idx = rng.integers(0, n, (B, n))
    diff = S[idx, 0].sum(1) / S[idx, 1].sum(1); pt = S[:, 0].sum() / S[:, 1].sum()
    print('%-34s diff %+.3f  95%% day-block CI [%+.3f, %+.3f]  (%d days)' % (label, pt, *np.percentile(diff, [2.5, 97.5]), n))
print('\n== paired day-block bootstrap (positive = first model worse) ==')
for s in sorted(set(k[1] for k in E if k[0] == 'tcstag')):
    if ('te', s) in E: boot(E[('tcstag', s)], E[('te', s)], label='tcstag - te, seed %d' % s)
pt, nt = ens('tcstag'); pe, ne = ens('te')
if nt and ne:
    boot(pt, pe, label='tcstag - te, seed-ensembles (%d,%d)' % (nt, ne))
    for r in ['weekday_peak', 'weekday_offpeak', 'weekend']: boot(pt, pe, reg == r, label='   %s' % r)
pb, nb = ens('baseline')
if nb and ne: boot(pe, pb, label='te - baseline, ensembles (%d,%d)' % (ne, nb))
print('\n== per-horizon MAE (seed-mean) ==')
for v in ['baseline', 'te', 'tcstag']:
    ks = [k for k in E if k[0] == v]
    if ks: print('%-9s' % v, [round(float(np.mean([np.abs(E[k][:, :, h] - tgt[:, :, h]).mean() for k in ks])), 3) for h in [0, 2, 5, 8, 11]])
print('\ntest days (index since Mon 1 Jan 2018):', days.min(), '-', days.max(), '| n windows', len(ts))
