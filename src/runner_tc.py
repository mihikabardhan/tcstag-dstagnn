# TC-STAG runner: one training run = (dataset, variant, seed). Same protocol as runner.py (baseline), plus timestamps and regime graphs.
import os, sys, json, time, argparse, random, configparser
p = argparse.ArgumentParser()
p.add_argument('--repo', default='.'); p.add_argument('--dataset', default='PEMS04')
p.add_argument('--variant', default='tcstag'); p.add_argument('--seed', type=int, default=1); p.add_argument('--gpu', default='0')
p.add_argument('--max_epochs', type=int, default=60); p.add_argument('--patience', type=int, default=15)
p.add_argument('--out', required=True); p.add_argument('--deadline', type=float, default=0.0)
p.add_argument('--graph_dir', default=''); p.add_argument('--graph_k', default='', help="'' = 1% graphs, '_k5' = 5% graphs")
p.add_argument('--graph', default='AG', help='AG = learned STAG for the Chebyshev conv (the paper model); G = static road graph (DSTAGNN-G)'); p.add_argument('--start_dow', type=int, default=0); p.add_argument('--batch_size', type=int, default=0, help='0 = config value (32)'); p.add_argument('--debug_batches', type=int, default=0)
args = p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.chdir(args.repo); sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
if os.environ.get('TCSTAG_CPU_TEST'): torch.Tensor.cuda = lambda self, *a, **k: self     # CPU smoke tests only (repo hard-codes .cuda())
from model.tcstag import make_tc_model, VARIANTS
from lib.dataloader import load_weighted_adjacency_matrix, load_PA
from lib.utils1 import get_adjacency_matrix2
from lib.metrics import masked_mape_np

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
cfg = configparser.ConfigParser(); cfg.read('configurations/%s_dstagnn.conf' % args.dataset); D, T = cfg['Data'], cfg['Training']
ds, N = args.dataset, int(D['num_of_vertices']); bs = args.batch_size or int(T['batch_size']); lr = float(T['learning_rate'])
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); os.makedirs(args.out, exist_ok=True)

# ---- data (with timestamps) ----
f = np.load(D['graph_signal_matrix_filename'].replace('.npz', '') + '_r1_d0_w0_dstagnn.npz')
def loader(split, shuffle):
    x = torch.from_numpy(f[split + '_x'][:, :, 0:1, :]).float().to(DEVICE); y = torch.from_numpy(f[split + '_target']).float().to(DEVICE)
    t = torch.from_numpy(f[split + '_timestamp'][:, 0].astype(np.int64)).to(DEVICE)
    return DataLoader(TensorDataset(x, y, t), batch_size=bs, shuffle=shuffle), y
train_loader, _ = loader('train', True); val_loader, _ = loader('val', False); test_loader, test_target = loader('test', False)

# ---- graphs: component 0 = the baseline's own A_STRG; 1..3 = regime graphs (Jinay) ----
adj_mx = get_adjacency_matrix2(D['adj_filename'], N, id_filename=None)
adj_stag = load_weighted_adjacency_matrix(D['stag_filename'], N); adj_strg = load_PA(D['strg_filename'])
adj_merge = adj_mx if args.graph == 'G' else adj_stag     # exactly as the authors' train script (AG -> STAG)
print('Chebyshev graph for the base model:', 'road network (G)' if args.graph == 'G' else 'learned STAG (AG)', flush=True)
graphs = stags = None
if VARIANTS[args.variant]['graph'] != 'none':
    rd = lambda nm, kind='strg': (pd.read_csv(os.path.join(args.graph_dir, ds, '%s_001_%s_%s%s.csv' % (kind, ds, nm, args.graph_k)), header=None).values > 0).astype(np.float32)
    comp0 = adj_strg if args.graph_k == '' else rd('global')
    graphs = [comp0] + [rd(n) for n in ['weekday_peak', 'weekday_offpeak', 'weekend']]
    stag0 = adj_stag if args.graph_k == '' else rd('global', 'stag')            # component 0 of the Chebyshev graphs = the baseline's own STAG
    stags = [stag0] + [rd(n, 'stag') for n in ['weekday_peak', 'weekday_offpeak', 'weekend']]
    print('chebyshev graph neighbours per node (incl. self):', [float((s_ > 0).sum(1).mean()) for s_ in stags], flush=True)
    print('graph components, neighbours per node:', [float(g.sum(1).mean()) for g in graphs], flush=True)

net = make_tc_model(DEVICE, args.variant, graphs, stags, args.start_dow, int(T['in_channels']), int(T['nb_block']), int(T['in_channels']), int(T['K']),
                    int(T['nb_chev_filter']), int(T['nb_time_filter']), 1, adj_merge, adj_strg, adj_stag, int(D['num_for_predict']),
                    int(D['len_input']), N, int(T['d_model']), int(T['d_k']), int(T['d_k']), int(T['n_heads']))
n_params = int(sum(q.numel() for q in net.parameters())); print('variant', args.variant, '| params', n_params, '| device', DEVICE, flush=True)
crit = nn.SmoothL1Loss().to(DEVICE); opt = optim.Adam(net.parameters(), lr=lr)

def val_loss():
    net.eval(); tot = []
    with torch.no_grad():
        for i, (x, y, t) in enumerate(val_loader):
            if args.debug_batches and i >= args.debug_batches: break
            tot.append(crit(net(x, t), y).item())
    return float(np.mean(tot))

best_path = os.path.join(args.out, 'best.pt'); best_val, best_ep, bad, hist, stopped = float('inf'), -1, 0, [], 'max_epochs'; t_start = time.time()
for ep in range(args.max_epochs):
    t0 = time.time(); net.train(); tl = []
    for i, (x, y, t) in enumerate(train_loader):
        if args.debug_batches and i >= args.debug_batches: break
        opt.zero_grad(); loss = crit(net(x, t), y); loss.backward(); opt.step(); tl.append(loss.detach())
    tr = float(torch.stack(tl).mean()); vl = val_loss()
    hist.append(dict(epoch=ep, train_loss=tr, val_loss=vl, epoch_sec=time.time() - t0))
    improved = vl < best_val
    if improved: best_val, best_ep, bad = vl, ep, 0; torch.save(net.state_dict(), best_path)
    else: bad += 1
    print('ep %d train %.4f val %.4f %s  %.1fs' % (ep, tr, vl, '*' if improved else '', time.time() - t0), flush=True)
    if bad >= args.patience: stopped = 'early_stop'; break
    if args.deadline and time.time() > args.deadline: stopped = 'deadline'; break

net.load_state_dict(torch.load(best_path, map_location=DEVICE)); net.eval(); preds = []
with torch.no_grad():
    for i, (x, y, t) in enumerate(test_loader):
        if args.debug_batches and i >= args.debug_batches: break
        preds.append(net(x, t).cpu().numpy())
pred = np.concatenate(preds, 0); tgt = test_target.cpu().numpy()[:len(pred)]
def m(y, yh): return dict(MAE=float(np.mean(np.abs(y - yh))), RMSE=float(np.sqrt(np.mean((y - yh) ** 2))), MAPE=float(masked_mape_np(y, yh, 0)))
per_h = [m(tgt[:, :, i], pred[:, :, i]) for i in range(pred.shape[2])]; overall = m(tgt.reshape(-1), pred.reshape(-1))
np.savez_compressed(os.path.join(args.out, 'test_preds.npz'), prediction=pred.astype(np.float32), target=tgt.astype(np.float32), timestamp=f['test_timestamp'][:len(pred)])
if net.mixer is not None and net.mixer.mode != 'hard':      # learned gate weights for every (day-of-week, time-of-day): for the paper figure
    tt = torch.tensor([((d - args.start_dow) % 7) * 288 + s for d in range(7) for s in range(288)], device=DEVICE)
    with torch.no_grad(): gw = net.mixer.weights(tt).cpu().numpy().reshape(7, 288, -1)
    np.save(os.path.join(args.out, 'gate_weights.npy'), gw)
res = dict(dataset=ds, variant=args.variant, cheb_graph=args.graph, graph_k=args.graph_k, seed=args.seed, best_epoch=best_ep, best_val=best_val, epochs_run=len(hist),
           stopped_by=stopped, max_epochs=args.max_epochs, patience=args.patience, mean_epoch_sec=float(np.mean([h['epoch_sec'] for h in hist])),
           total_sec=time.time() - t_start, n_params=n_params, test_overall=overall, test_per_horizon=per_h, history=hist)
json.dump(res, open(os.path.join(args.out, 'results.json'), 'w'), indent=1); os.remove(best_path)
print('TEST overall:', overall, flush=True)
