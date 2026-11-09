"""Builds gate_weights_over_day.png (mean over seeds) and prints the numbers quoted in Section 'What the gate learned'.
Usage: python make_gate_figure.py tc      (tc = folder holding PEMS04_tcstag_seed*/gate_weights.npy)
Needs numpy and matplotlib:  pip install matplotlib"""
import sys, glob, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
root = sys.argv[1] if len(sys.argv) > 1 else '.'
files = sorted(glob.glob(os.path.join(root, '**', 'PEMS04_tcstag_seed*', 'gate_weights.npy'), recursive=True))
print('found:', files); assert files, 'no gate_weights.npy found'
gs = [np.load(f) for f in files]            # each (7 dow, 288 tod, 4 components); dow 0 = Monday
names = ['global', 'weekday peak', 'weekday off-peak', 'weekend']
for f, g in zip(files, gs):
    print(os.path.basename(os.path.dirname(f)))
    for c, n in enumerate(names):
        x = g[..., c]; print('  %-17s range over week %.3f | weekday mean %.3f | weekend mean %.3f | min %.3f max %.3f' % (n, x.max() - x.min(), x[:5].mean(), x[5:].mean(), x.min(), x.max()))
if len(gs) > 1:
    print('cross-seed correlation of the weight patterns (per component):')
    for c, n in enumerate(names):
        cs = [np.corrcoef(gs[i][..., c].ravel(), gs[j][..., c].ravel())[0, 1] for i in range(len(gs)) for j in range(i + 1, len(gs))]
        print('  %-17s %s' % (n, np.round(cs, 2)))
G = np.mean(gs, axis=0); hrs = np.arange(288) / 12
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
for a, (title, days) in zip(ax, [('Weekdays (Mon-Fri)', list(range(5))), ('Weekend (Sat-Sun)', [5, 6])]):
    sel = G[days].mean(0)
    for c in range(4): a.plot(hrs, sel[:, c], label=names[c])
    a.set_title(title, fontsize=9); a.set_xlabel('hour of day'); a.set_xlim(0, 24)
ax[0].set_ylabel('mixture weight'); ax[1].legend(fontsize=7); plt.tight_layout(); plt.savefig('gate_weights_over_day.png', dpi=200)
print('saved gate_weights_over_day.png')
