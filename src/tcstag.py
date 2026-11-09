# TC-STAG: time-conditioned graph prior + time embeddings on top of the authors' DSTAGNN (model/DSTAGNN_my.py)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model.DSTAGNN_my import DSTAGNN_block, DSTAGNN_submodule, cheb_conv_withSAt
from lib.utils import scaled_Laplacian, cheb_polynomial

STEPS_PER_DAY = 288
# component order of the graph stack: 0 = global, 1 = weekday_peak, 2 = weekday_offpeak, 3 = weekend
def regime_index(t, start_dow):
    tod = t % STEPS_PER_DAY; dow = (t // STEPS_PER_DAY + start_dow) % 7
    peak = ((tod >= 84) & (tod < 120)) | ((tod >= 192) & (tod < 228))          # 7-10h, 16-19h
    return torch.where(dow >= 5, torch.full_like(t, 3), torch.where(peak, torch.full_like(t, 1), torch.full_like(t, 2)))

class TCChebConv(cheb_conv_withSAt):
    # authors' graph convolution, except T_k may be per-sample (B,N,N): the graph SUPPORT itself changes with time
    def forward(self, x, spatial_attention, adj_pa, T_mix=None):
        batch_size, N, _, num_t = x.shape; outputs = []
        for ts in range(num_t):
            gs = x[:, :, :, ts]; output = torch.zeros(batch_size, N, self.out_channels).to(self.DEVICE)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k] if T_mix is None else T_mix[:, k]
                att = F.softmax(spatial_attention[:, k, :, :] + adj_pa.mul(self.mask[k]), dim=1)
                rhs = T_k.mul(att).permute(0, 2, 1).matmul(gs)
                output = output + rhs.matmul(self.Theta[k])
            outputs.append(output.unsqueeze(-1))
        return self.relu(torch.cat(outputs, dim=-1))

class TimeEncoder(nn.Module):               # time-of-day + day-of-week embedding
    def __init__(self, dim, start_dow):
        super().__init__(); self.start_dow = start_dow
        self.tod = nn.Embedding(STEPS_PER_DAY, dim); self.dow = nn.Embedding(7, dim)
    def forward(self, t):
        return self.tod(t % STEPS_PER_DAY) + self.dow((t // STEPS_PER_DAY + self.start_dow) % 7)

class GraphMixer(nn.Module):
    # produces the attention-prior graph A_STRG(t). modes: hard | soft_static | soft_time
    def __init__(self, graphs, mode, start_dow, cheb=None, gate_dim=16):
        super().__init__()
        self.mode, self.start_dow = mode, start_dow
        self.register_buffer('graphs', graphs)                           # (C, N, N), C = 4 : attention priors A_STRG^(c)
        self.has_cheb = cheb is not None
        if self.has_cheb: self.register_buffer('cheb', cheb)             # (C, K, N, N): Chebyshev polynomials of each component's STAG
        C = graphs.shape[0]
        if mode == 'soft_static':
            self.logits = nn.Parameter(torch.zeros(C))
        elif mode == 'soft_time':
            self.e_tod = nn.Embedding(STEPS_PER_DAY, gate_dim); self.e_dow = nn.Embedding(7, gate_dim)
            self.mlp = nn.Sequential(nn.Linear(2 * gate_dim, 32), nn.ReLU(), nn.Linear(32, C))
    def post_init(self):                                                # called AFTER the repo's init loop (which overwrites biases)
        if self.mode == 'soft_static':
            nn.init.constant_(self.logits, 0.0); self.logits.data[0] = 2.0   # start close to the global graph (softmax ~ 0.71 / 0.10 each)
        elif self.mode == 'soft_time':
            nn.init.zeros_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias); self.mlp[2].bias.data[0] = 2.0
    def weights(self, t):                                               # (B, C)
        C = self.graphs.shape[0]
        if self.mode == 'hard':
            return F.one_hot(regime_index(t, self.start_dow), C).float()
        if self.mode == 'soft_static':
            return F.softmax(self.logits, 0).unsqueeze(0).expand(t.shape[0], -1)
        h = torch.cat([self.e_tod(t % STEPS_PER_DAY), self.e_dow((t // STEPS_PER_DAY + self.start_dow) % 7)], -1)
        return F.softmax(self.mlp(h), -1)
    def forward(self, t):                                               # -> prior (B,N,N), Chebyshev stack (B,K,N,N) or None
        w = self.weights(t)
        prior = torch.einsum('bc,cnm->bnm', w, self.graphs)
        T_mix = torch.einsum('bc,ckmn->bkmn', w, self.cheb) if self.has_cheb else None
        return prior, T_mix

class TCBlock(DSTAGNN_block):
    # identical to the authors' block, except: per-sample graph prior and an optional time embedding added in the spatial attention input
    def forward(self, x, res_att, adj_pa_t=None, te=None, T_mix=None):
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        TEmx = self.EmbedT(x, batch_size) if num_of_features == 1 else x.permute(0, 2, 3, 1)
        TATout, re_At = self.TAt(TEmx, TEmx, TEmx, None, res_att)
        x_TAt = self.pre_conv(TATout.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)
        SEmx_TAt = self.dropout(self.EmbedS(x_TAt, batch_size))
        if te is not None:
            SEmx_TAt = SEmx_TAt + te.unsqueeze(1)
        STAt = self.SAt(SEmx_TAt, SEmx_TAt, None)
        spatial_gcn = self.cheb_conv_SAt(x, STAt, self.adj_pa if adj_pa_t is None else adj_pa_t, T_mix)
        X = spatial_gcn.permute(0, 2, 1, 3)
        time_conv = self.fcmy(torch.cat([self.gtu3(X), self.gtu5(X), self.gtu7(X)], dim=-1))
        time_conv_output = self.relu(time_conv) if num_of_features == 1 else self.relu(X + time_conv)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3)) if num_of_features == 1 else x.permute(0, 2, 1, 3)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        return x_residual, re_At

class TCSubmodule(DSTAGNN_submodule):
    def __init__(self, *args, variant_cfg=None, graphs=None, cheb=None, start_dow=0, **kw):
        super().__init__(*args, **kw)
        d_model = args[14]                               # positional args of DSTAGNN_submodule
        for b in self.BlockList: b.__class__ = TCBlock; b.cheb_conv_SAt.__class__ = TCChebConv
        self.use_te, self.gmode = variant_cfg['te'], variant_cfg['graph']
        if self.use_te:
            self.time_enc = TimeEncoder(d_model, start_dow)
            self.te_head = nn.Linear(d_model, 128)
            self.fuse = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128))
        self.mixer = GraphMixer(graphs, self.gmode, start_dow, cheb if variant_cfg.get('cheb', True) else None) if self.gmode != 'none' else None
        self.to(self.DEVICE)
    def post_init(self):
        if self.use_te: nn.init.zeros_(self.fuse[2].weight); nn.init.zeros_(self.fuse[2].bias)   # start as the baseline
        if self.mixer is not None: self.mixer.post_init()
    def forward(self, x, t):
        te = self.time_enc(t) if self.use_te else None
        adj_t, T_mix = self.mixer(t) if self.mixer is not None else (None, None)
        need_concat, res_att = [], 0
        for block in self.BlockList:
            x, res_att = block(x, res_att, adj_t, te, T_mix); need_concat.append(x)
        final_x = torch.cat(need_concat, dim=-1)
        out1 = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)       # (B, N, 128)
        if self.use_te:
            h = torch.cat([out1, self.te_head(te).unsqueeze(1).expand(-1, out1.shape[1], -1)], -1)
            out1 = out1 + self.fuse(h)
        return self.final_fc(out1)

VARIANTS = {   # name -> config
    'base':        dict(te=False, graph='none'),
    'te':          dict(te=True,  graph='none'),         # time embeddings only
    'hard':        dict(te=False, graph='hard'),         # regime graph chosen by the clock, no learned gate, no time embeddings
    'soft_static': dict(te=False, graph='soft_static'),  # learned global mixture weights, not time-dependent
    'te_hard':     dict(te=True,  graph='hard'),
    'tcstag':      dict(te=True,  graph='soft_time'),    # FULL METHOD (prior AND Chebyshev support are time-conditioned)
    'tcstag_prior':dict(te=True,  graph='soft_time', cheb=False),  # ablation: only the attention prior is time-conditioned (mostly inert, see paper notes)
}

def make_tc_model(DEVICE, variant, graphs, stags, start_dow, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                  adj_mx, adj_pa, adj_TMD, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads):
    L_tilde = scaled_Laplacian(adj_mx)
    cheb = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    g = torch.tensor(np.stack(graphs), dtype=torch.float32).to(DEVICE) if graphs is not None else None
    ch = None
    if stags is not None:                                                # (C, K, N, N): one Chebyshev stack per component graph
        ch = torch.stack([torch.stack([torch.from_numpy(i).type(torch.FloatTensor) for i in cheb_polynomial(scaled_Laplacian(a), K)]) for a in stags]).to(DEVICE)
    model = TCSubmodule(DEVICE, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb, adj_pa, adj_TMD,
                        num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads,
                        variant_cfg=VARIANTS[variant], graphs=g, cheb=ch, start_dow=start_dow)
    for p in model.parameters():                                         # same init loop as the authors' make_model
        if p.dim() > 1: nn.init.xavier_uniform_(p)
        else: nn.init.uniform_(p)
    model.post_init()
    return model
