# DSTAGNN runner: one training run = one (dataset, graph, seed). Used by every team notebook.
import os, sys, json, time, argparse, random, configparser, shutil

p = argparse.ArgumentParser()
p.add_argument('--repo', default='.')
p.add_argument('--dataset', default='PEMS04')
p.add_argument('--graph', default='AG', help='AG = learned STAG (paper), G = static road graph')
p.add_argument('--seed', type=int, default=1)
p.add_argument('--gpu', default='0')
p.add_argument('--max_epochs', type=int, default=60)
p.add_argument('--patience', type=int, default=15)
p.add_argument('--out', required=True)
p.add_argument('--deadline', type=float, default=0.0, help='unix time; stop training after this')
args = p.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.chdir(args.repo)
sys.path.insert(0, os.getcwd())
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from model.DSTAGNN_my import make_model
from lib.dataloader import load_weighted_adjacency_matrix, load_PA
from lib.utils1 import load_graphdata_channel1, get_adjacency_matrix2
from lib.metrics import masked_mape_np

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

cfg = configparser.ConfigParser()
cfg.read('configurations/%s_dstagnn.conf' % args.dataset)
D, T = cfg['Data'], cfg['Training']
N = int(D['num_of_vertices']); n_pred = int(D['num_for_predict']); len_in = int(D['len_input'])
bs = int(T['batch_size']); lr = float(T['learning_rate'])
hours, days, weeks = int(T['num_of_hours']), int(T['num_of_days']), int(T['num_of_weeks'])
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.out, exist_ok=True)

(_, train_loader, _, _, val_loader, _, _, test_loader, test_target, mean, std) = load_graphdata_channel1(
    D['graph_signal_matrix_filename'], hours, days, weeks, DEVICE, bs)
adj_mx = get_adjacency_matrix2(D['adj_filename'], N, id_filename=None)
adj_stag = load_weighted_adjacency_matrix(D['stag_filename'], N)
adj_strg = load_PA(D['strg_filename'])
adj_merge = adj_mx if args.graph == 'G' else adj_stag

net = make_model(DEVICE, int(T['in_channels']), int(T['nb_block']), int(T['in_channels']), int(T['K']),
                 int(T['nb_chev_filter']), int(T['nb_time_filter']), 1, adj_merge, adj_strg, adj_stag,
                 n_pred, len_in, N, int(T['d_model']), int(T['d_k']), int(T['d_k']), int(T['n_heads']))
n_params = int(sum(p_.numel() for p_ in net.parameters()))
print('params:', n_params, 'device:', DEVICE, flush=True)
crit = nn.SmoothL1Loss().to(DEVICE)   # Huber, threshold 1 (as in paper)
opt = optim.Adam(net.parameters(), lr=lr)

def val_loss():
    net.eval(); tot = []
    with torch.no_grad():
        for x, y in val_loader:
            tot.append(crit(net(x), y).item())
    return float(np.mean(tot))

best_path = os.path.join(args.out, 'best.pt')
best_val, best_ep, bad, hist, stopped = float('inf'), -1, 0, [], 'max_epochs'
t_start = time.time()
for ep in range(args.max_epochs):
    t0 = time.time(); net.train(); tl = []
    for x, y in train_loader:
        opt.zero_grad(); loss = crit(net(x), y); loss.backward(); opt.step(); tl.append(loss.detach())
    tr = float(torch.stack(tl).mean()); t_train = time.time() - t0
    vl = val_loss()
    hist.append(dict(epoch=ep, train_loss=tr, val_loss=vl, epoch_sec=time.time() - t0))
    improved = vl < best_val
    if improved:
        best_val, best_ep, bad = vl, ep, 0
        torch.save(net.state_dict(), best_path)
    else:
        bad += 1
    print('ep %d train %.4f val %.4f %s  %.1fs' % (ep, tr, vl, '*' if improved else '', time.time() - t0), flush=True)
    if bad >= args.patience:
        stopped = 'early_stop'; break
    if args.deadline and time.time() > args.deadline:
        stopped = 'deadline'; break

# ---- test at best epoch ----
net.load_state_dict(torch.load(best_path, map_location=DEVICE)); net.eval(); preds = []
with torch.no_grad():
    for x, y in test_loader:
        preds.append(net(x).cpu().numpy())
pred = np.concatenate(preds, 0); tgt = test_target.cpu().numpy()

def m(y, yh):
    return dict(MAE=float(np.mean(np.abs(y - yh))), RMSE=float(np.sqrt(np.mean((y - yh) ** 2))),
                MAPE=float(masked_mape_np(y, yh, 0)))

per_h = [m(tgt[:, :, i], pred[:, :, i]) for i in range(pred.shape[2])]
overall = m(tgt.reshape(-1), pred.reshape(-1))
prep = np.load(D['graph_signal_matrix_filename'].replace('.npz', '') + '_r%d_d%d_w%d_dstagnn.npz' % (hours, days, weeks))
np.savez_compressed(os.path.join(args.out, 'test_preds.npz'), prediction=pred.astype(np.float32),
                    target=tgt.astype(np.float32), timestamp=prep['test_timestamp'])
res = dict(dataset=args.dataset, graph=args.graph, seed=args.seed, best_epoch=best_ep, best_val=best_val,
           epochs_run=len(hist), stopped_by=stopped, max_epochs=args.max_epochs, patience=args.patience,
           mean_epoch_sec=float(np.mean([h['epoch_sec'] for h in hist])), total_sec=time.time() - t_start,
           n_params=n_params, test_overall=overall, test_per_horizon=per_h, history=hist)
json.dump(res, open(os.path.join(args.out, 'results.json'), 'w'), indent=1)
os.remove(best_path)   # keep outputs small; predictions + results are what we need
print('TEST overall:', overall, flush=True)
