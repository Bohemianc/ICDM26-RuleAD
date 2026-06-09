import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from data import load_split

RDLogger.DisableLog("rdApp.*")

EPOCHS = 200
LR = 3e-3
WEIGHT_DECAY = 1e-6
HIDDEN = 128
LATENT = 128
N_LAYERS = 3
N_HEADS = 4
DROPOUT = 0.1
LR_DECAY_LAST = 40
ES_PATIENCE = 20
MAX_EDGES = 2_500_000
MISSING_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- helpers ---
def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(epoch, epochs, base_lr, decay_last):
    start = epochs - decay_last
    if epoch <= start:
        return base_lr
    t = (epoch - start) / max(decay_last, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def safe_auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def rankify(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(np.argsort(x))
    return order.astype(float) / max(len(x) - 1, 1)


# --- rule features ---
DESCRIPTOR_HANDLES = {
    "logp": lambda m: Descriptors.MolLogP(m),
    "mol_wt": lambda m: Descriptors.MolWt(m),
    "molwt": lambda m: Descriptors.MolWt(m),
    "mw": lambda m: Descriptors.MolWt(m),
    "tpsa": lambda m: Descriptors.TPSA(m),
    "hbd": lambda m: Lipinski.NumHDonors(m),
    "hba": lambda m: Lipinski.NumHAcceptors(m),
    "num_rotatable_bonds": lambda m: Lipinski.NumRotatableBonds(m),
    "rotatable_bonds": lambda m: Lipinski.NumRotatableBonds(m),
    "rot_bonds": lambda m: Lipinski.NumRotatableBonds(m),
    "num_aromatic_rings": lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
    "aromatic_rings": lambda m: rdMolDescriptors.CalcNumAromaticRings(m),
    "num_aliphatic_rings": lambda m: rdMolDescriptors.CalcNumAliphaticRings(m),
    "aliphatic_rings": lambda m: rdMolDescriptors.CalcNumAliphaticRings(m),
    "num_rings": lambda m: rdMolDescriptors.CalcNumRings(m),
    "rings": lambda m: rdMolDescriptors.CalcNumRings(m),
    "ring_count": lambda m: rdMolDescriptors.CalcNumRings(m),
    "num_heterocycles": lambda m: rdMolDescriptors.CalcNumHeterocycles(m),
    "heterocycles": lambda m: rdMolDescriptors.CalcNumHeterocycles(m),
    "num_amide_bonds": lambda m: rdMolDescriptors.CalcNumAmideBonds(m),
    "amide_bonds": lambda m: rdMolDescriptors.CalcNumAmideBonds(m),
    "fraction_csp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
    "fraction_sp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
    "csp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
    "num_spiro_atoms": lambda m: rdMolDescriptors.CalcNumSpiroAtoms(m),
    "num_bridgehead_atoms": lambda m: rdMolDescriptors.CalcNumBridgeheadAtoms(m),
    "heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
    "num_heavy_atoms": lambda m: m.GetNumHeavyAtoms(),
}


def _normalize(mol, name: str):
    name = (name or "heavy_atoms").lower()
    if name in ("heavy_atoms", "num_heavy_atoms"):
        return mol.GetNumHeavyAtoms()
    if name in ("atoms", "num_atoms"):
        return mol.GetNumAtoms()
    if name in ("num_rings", "rings", "ring_count"):
        return max(rdMolDescriptors.CalcNumRings(mol), 1)
    return mol.GetNumHeavyAtoms()


def _eval_numeric(rule: dict, mol):
    t = rule.get("type", "")
    try:
        if t == "smarts_count":
            pat = Chem.MolFromSmarts(rule["smarts"])
            if pat is None:
                return None
            return float(len(mol.GetSubstructMatches(pat)))
        if t == "smarts_ratio":
            pat = Chem.MolFromSmarts(rule["smarts"])
            if pat is None:
                return None
            n_match = len(mol.GetSubstructMatches(pat))
            denom = _normalize(mol, rule.get("normalizer", "heavy_atoms"))
            if denom <= 0:
                return 0.0
            return float(n_match) / float(denom)
        if t in ("descriptor", "desirability_window"):
            desc = (rule.get("descriptor") or "").lower()
            fn = DESCRIPTOR_HANDLES.get(desc)
            if fn is None:
                return None
            return float(fn(mol))
    except Exception:
        return None
    return None


def compute_rule_firings_and_values(smiles_list, rules_kept):
    N = len(smiles_list)
    R = len(rules_kept)
    phi = np.zeros((N, R), dtype=np.float32)
    X_raw = np.full((N, R), np.nan, dtype=np.float32)
    directions = [r["_filter"]["direction_resolved"] for r in rules_kept]
    for j, rule in enumerate(rules_kept):
        f = rule["_filter"]
        kind = f["threshold_kind"]
        thr = float(f["threshold"]) if f["threshold"] is not None else 0.0
        d = f["direction_resolved"]
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            v = _eval_numeric(rule, mol)
            if v is None:
                continue
            X_raw[i, j] = v
            if kind.startswith("count"):
                phi[i, j] = 1.0 if v >= thr else 0.0
            elif kind.startswith("desc>"):
                phi[i, j] = 1.0 if v > thr else 0.0
            elif kind.startswith("desc<"):
                phi[i, j] = 1.0 if v < thr else 0.0
            elif kind.startswith("ratio>"):
                phi[i, j] = 1.0 if v > thr else 0.0
            else:
                if d == "higher":
                    phi[i, j] = 1.0 if v > thr else 0.0
                elif d == "lower":
                    phi[i, j] = 1.0 if v < thr else 0.0
    return phi, X_raw, directions


# --- rule channel ---
class TailScorer:
    def __init__(self, topk_fraction=0.4, missing_threshold=0.5):
        self.topk_fraction = topk_fraction
        self.missing_threshold = missing_threshold

    def fit(self, X_train, directions):
        X = X_train.astype(float, copy=True)
        finite = np.isfinite(X)
        n_rows, n_cols = X.shape
        miss = 1.0 - finite.sum(axis=0) / max(n_rows, 1)
        too_m = miss > self.missing_threshold
        const = np.zeros(n_cols, dtype=bool)
        for j in range(n_cols):
            if too_m[j]:
                continue
            col = X[:, j][finite[:, j]]
            if col.size < 2 or float(col.min()) == float(col.max()):
                const[j] = True
        keep = ~(too_m | const)
        self.keep_idx = np.flatnonzero(keep)
        self.directions = [directions[i] for i in self.keep_idx.tolist()]
        Xs = X[:, self.keep_idx]
        self.fill = np.nanmedian(Xs, axis=0) if Xs.size else np.empty(0)
        if Xs.size:
            inds = np.where(~np.isfinite(Xs))
            Xs[inds] = np.take(self.fill, inds[1])
        self.sorted_train = np.sort(Xs, axis=0) if Xs.size else np.empty((0, 0))
        return self

    def per_rule_tail(self, X_test):
        if self.keep_idx.size == 0:
            return np.zeros((X_test.shape[0], 0))
        X = X_test[:, self.keep_idx].astype(float, copy=True)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(self.fill, inds[1])
        n = self.sorted_train.shape[0]
        floor = 1.0 / max(n, 1)
        tails = np.zeros_like(X, dtype=np.float64)
        for j, d in enumerate(self.directions):
            sc = self.sorted_train[:, j]
            v = X[:, j]
            if d == "higher":
                t = (n - np.searchsorted(sc, v, side="left")) / n
            elif d == "lower":
                t = np.searchsorted(sc, v, side="right") / n
            else:
                lh = np.searchsorted(sc, v, side="left") / n
                rh = 1.0 - np.searchsorted(sc, v, side="right") / n
                t = 2.0 * np.minimum(lh + floor, rh + floor)
            tails[:, j] = np.clip(t, floor, 1.0)
        return tails

    def score_from_tail(self, tails):
        if tails.shape[1] == 0:
            return np.zeros(tails.shape[0])
        per_rule = -np.log(np.maximum(tails, 1e-12))
        m = per_rule.shape[1]
        k = max(4, int(round(self.topk_fraction * m)))
        k = min(k, m)
        if k == m:
            return per_rule.mean(axis=1)
        part = np.partition(per_rule, kth=m - k, axis=1)
        return part[:, -k:].mean(axis=1)

    def score(self, X_test):
        return self.score_from_tail(self.per_rule_tail(X_test))


# --- graph channel ---
def filter_and_standardize(X_cf_tr: np.ndarray, X_cf_te: np.ndarray):
    """Drop too-missing/constant columns, median-impute, then z-score on train stats."""
    n_tr = X_cf_tr.shape[0]
    finite_tr = np.isfinite(X_cf_tr)
    miss_rate = 1.0 - finite_tr.sum(axis=0) / max(n_tr, 1)
    not_too_missing = miss_rate < MISSING_THRESHOLD
    not_constant = np.zeros(X_cf_tr.shape[1], dtype=bool)
    for j in range(X_cf_tr.shape[1]):
        if not_too_missing[j]:
            col = X_cf_tr[:, j][finite_tr[:, j]]
            if col.size >= 2 and float(col.min()) < float(col.max()):
                not_constant[j] = True
    keep = not_too_missing & not_constant
    if keep.sum() == 0:
        raise RuntimeError("filter_and_standardize: no valid columns after filtering")
    X_tr_kept = X_cf_tr[:, keep].astype(np.float32, copy=True)
    X_te_kept = X_cf_te[:, keep].astype(np.float32, copy=True)
    X_all = np.concatenate([X_tr_kept, X_te_kept], axis=0)

    finite_tr_kept = np.isfinite(X_tr_kept)
    n_kept = int(keep.sum())
    medians = np.zeros(n_kept, dtype=np.float32)
    for j in range(n_kept):
        col = X_tr_kept[:, j][finite_tr_kept[:, j]]
        medians[j] = float(np.median(col)) if col.size else 0.0
    inds = np.where(~np.isfinite(X_all))
    X_all[inds] = np.take(medians, inds[1])

    mu = X_all[:n_tr].mean(axis=0)
    sd = X_all[:n_tr].std(axis=0) + 1e-8
    X_std = ((X_all - mu[None, :]) / sd[None, :]).astype(np.float32)
    return X_std, n_kept, keep


class GraphAttentionLayer(nn.Module):
    """Sparse multi-head attention over an edge list (dst v, src u)."""

    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.Wq = nn.Linear(in_dim, out_dim)
        self.Wk = nn.Linear(in_dim, out_dim)
        self.Wv = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, h, edge_index):
        N = h.shape[0]
        H = self.n_heads
        D = self.head_dim
        q = self.Wq(h).view(N, H, D)
        k = self.Wk(h).view(N, H, D)
        v = self.Wv(h).view(N, H, D)

        dst = edge_index[0]
        src = edge_index[1]
        e = (q[dst] * k[src]).sum(dim=-1) / math.sqrt(D)
        e_max = torch.full((N, H), float("-inf"),
                            device=e.device, dtype=e.dtype)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(1).expand(-1, H), e,
                                      reduce="amax", include_self=True)
        e_shift = e - e_max[dst]
        exp_e = torch.exp(e_shift)
        denom = torch.zeros((N, H), device=e.device, dtype=e.dtype)
        denom = denom.scatter_add(0, dst.unsqueeze(1).expand(-1, H), exp_e)
        att = exp_e / (denom[dst] + 1e-12)
        att = self.dropout(att)
        msg = att.unsqueeze(-1) * v[src]
        out = torch.zeros((N, H, D), device=h.device, dtype=h.dtype)
        idx = dst.view(-1, 1, 1).expand(-1, H, D)
        out = out.scatter_add(0, idx, msg)
        out = out.reshape(N, H * D)
        out = self.out_proj(out)
        out = self.dropout(out)
        return F.relu(out + self.residual(h))


class GraphAttentionAutoencoder(nn.Module):
    def __init__(self, in_dim, hidden, latent, n_layers=3,
                 n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.attn_layers.append(GraphAttentionLayer(
                d, hidden, n_heads=n_heads, dropout=dropout))
            d = hidden
        self.enc_out = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, phi, edge_index):
        h = phi
        for layer in self.attn_layers:
            if self.training and h.requires_grad:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, edge_index, use_reentrant=False)
            else:
                h = layer(h, edge_index)
        return self.enc_out(h)

    def forward(self, phi, edge_index):
        z = self.encode(phi, edge_index)
        return self.dec(z), z


def build_intermolecule_edges(H: np.ndarray, max_edges: int = 20_000_000):
    """Edge list linking molecules that co-fire on a shared rule, with self-loops."""
    N = H.shape[0]
    H_bin = (np.abs(H) > 1e-6).astype(np.float32)
    BLOCK = 4096
    edge_pairs_dst = []
    edge_pairs_src = []
    total_edges = 0
    rng = np.random.default_rng(0)
    deg = np.zeros(N, dtype=np.int64)
    max_nbrs_per_node = max(1, int(max_edges // max(N, 1)))
    for i in range(0, N, BLOCK):
        j = min(i + BLOCK, N)
        block = H_bin[i:j]
        co = block @ H_bin.T
        co[:, :] = (co > 0.5).astype(np.float32)
        rows_local = np.arange(i, j)
        for k, r in enumerate(rows_local):
            nbrs = np.where(co[k] > 0.5)[0]
            if nbrs.size == 0:
                nbrs = np.asarray([r])
            elif r not in nbrs:
                nbrs = np.concatenate([nbrs, [r]])
            if nbrs.size > max_nbrs_per_node:
                sel = rng.choice(nbrs[nbrs != r], size=max_nbrs_per_node - 1,
                                  replace=False)
                nbrs = np.concatenate([sel, [r]])
            edge_pairs_dst.append(np.full(nbrs.shape, r, dtype=np.int64))
            edge_pairs_src.append(nbrs.astype(np.int64))
            deg[r] = nbrs.size
        total_edges = sum(e.size for e in edge_pairs_dst)
    dst = np.concatenate(edge_pairs_dst)
    src = np.concatenate(edge_pairs_src)
    edge_index = np.stack([dst, src], axis=0)
    stats = {
        "N": int(N),
        "E": int(edge_index.shape[1]),
        "avg_nbrs": float(deg.mean()),
        "max_nbrs": int(deg.max()),
        "max_nbrs_per_node_cap": int(max_nbrs_per_node),
    }
    return edge_index, stats


def build_inductive_test_edges(phi_cf_kept: np.ndarray, N_train: int,
                                max_edges: int):
    """Train-graph edges plus test->train and test self-loop edges (strict inductive)."""
    N = phi_cf_kept.shape[0]
    N_test = N - N_train
    if N_test < 0:
        raise ValueError(f"N_train ({N_train}) > N ({N})")

    train_edges, train_stats = build_intermolecule_edges(
        phi_cf_kept[:N_train], max_edges=max_edges,
    )

    H_bin = (np.abs(phi_cf_kept) > 1e-6).astype(np.float32)
    train_bin = H_bin[:N_train]
    test_bin = H_bin[N_train:]

    max_nbrs_per_node = max(1, int(max_edges // max(N, 1)))
    rng = np.random.default_rng(0)

    cross_dst_list, cross_src_list = [], []
    test_nbr_counts = np.zeros(N_test, dtype=np.int64)
    BLOCK = 2048
    if N_test > 0 and N_train > 0:
        for i in range(0, N_test, BLOCK):
            j = min(i + BLOCK, N_test)
            co = test_bin[i:j] @ train_bin.T
            for k in range(j - i):
                t_global = N_train + i + k
                nbrs = np.where(co[k] > 0.5)[0]
                if nbrs.size + 1 > max_nbrs_per_node:
                    keep = max_nbrs_per_node - 1
                    if keep > 0:
                        nbrs = rng.choice(nbrs, size=keep, replace=False)
                    else:
                        nbrs = np.empty(0, dtype=np.int64)
                test_nbr_counts[i + k] = nbrs.size
                if nbrs.size:
                    cross_dst_list.append(np.full(nbrs.shape, t_global, dtype=np.int64))
                    cross_src_list.append(nbrs.astype(np.int64))
                cross_dst_list.append(np.array([t_global], dtype=np.int64))
                cross_src_list.append(np.array([t_global], dtype=np.int64))

    if cross_dst_list:
        cross_dst = np.concatenate(cross_dst_list)
        cross_src = np.concatenate(cross_src_list)
    else:
        cross_dst = np.empty(0, dtype=np.int64)
        cross_src = np.empty(0, dtype=np.int64)

    all_dst = np.concatenate([train_edges[0], cross_dst])
    all_src = np.concatenate([train_edges[1], cross_src])
    test_edges = np.stack([all_dst, all_src], axis=0)

    stats = {
        "N": int(N),
        "N_train": int(N_train),
        "N_test": int(N_test),
        "n_train_edges": int(train_edges.shape[1]),
        "n_cross_edges": int(cross_dst.size),
        "n_test_edges_total": int(test_edges.shape[1]),
        "max_nbrs_per_node_cap": int(max_nbrs_per_node),
        "test_nbr_count_mean": float(test_nbr_counts.mean()) if N_test > 0 else 0.0,
        "test_nbr_count_min": int(test_nbr_counts.min()) if N_test > 0 else 0,
        "test_nbr_count_max": int(test_nbr_counts.max()) if N_test > 0 else 0,
        "n_test_orphan": int((test_nbr_counts == 0).sum()),
        "train_edge_stats": train_stats,
    }
    return train_edges, test_edges, stats


# --- per-task scoring ---
def score_task(dataset, task_key, seed, cached_bank):
    set_all_seeds(seed)
    split = load_split(dataset, task_key, seed)
    tr_smiles = list(split["train_normal"])
    te_smiles = list(split["test"])
    te_labels = np.asarray(split["test_labels"], dtype=int)
    N_train, N_test = len(tr_smiles), len(te_smiles)

    rules_kept = cached_bank["rules"]
    phi_cf_tr, X_cf_tr, dirs = compute_rule_firings_and_values(tr_smiles, rules_kept)
    phi_cf_te, X_cf_te, _ = compute_rule_firings_and_values(te_smiles, rules_kept)
    phi_cf_nodes = np.concatenate([phi_cf_tr, phi_cf_te], axis=0).astype(np.float32)

    tail = TailScorer(topk_fraction=0.4).fit(X_cf_tr, dirs)
    s_rule_te = tail.score(X_cf_te)

    X_std, K, keep_mask = filter_and_standardize(X_cf_tr, X_cf_te)
    phi_cf_kept = phi_cf_nodes[:, keep_mask].astype(np.float32, copy=True)

    train_edges_np, test_edges_np, _ = build_inductive_test_edges(
        phi_cf_kept, N_train, max_edges=MAX_EDGES,
    )

    phi_t = torch.tensor(X_std, device=DEVICE, dtype=torch.float32)
    phi_train_t = phi_t[:N_train]
    train_edges_t = torch.tensor(train_edges_np, device=DEVICE, dtype=torch.long)
    test_edges_t = torch.tensor(test_edges_np, device=DEVICE, dtype=torch.long)

    model = GraphAttentionAutoencoder(
        K, HIDDEN, LATENT,
        n_layers=N_LAYERS, n_heads=N_HEADS, dropout=DROPOUT,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_loss = float("inf"); best_state = None; patience = 0
    for epoch in range(1, EPOCHS + 1):
        cur_lr = cosine_lr(epoch, EPOCHS, LR, LR_DECAY_LAST)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        model.train()
        recon_train, _ = model(phi_train_t, train_edges_t)
        loss = F.mse_loss(recon_train, phi_train_t)
        opt.zero_grad(); loss.backward(); opt.step()
        lval = float(loss.item())
        if lval < best_loss - 1e-5:
            best_loss = lval
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= ES_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        recon_full, _ = model(phi_t, test_edges_t)
        sq_err_full = ((recon_full - phi_t) ** 2).sum(dim=1).cpu().numpy()
    s_graph_te = sq_err_full[N_train:]

    y = te_labels
    s_craft = rankify(s_graph_te) + rankify(s_rule_te)
    out = {
        "dataset": dataset, "task": task_key, "seed": int(seed),
        "auroc_graph_only": safe_auroc(y, s_graph_te),
        "auroc_rule_only": safe_auroc(y, s_rule_te),
        "auroc_ranksum": safe_auroc(y, s_craft),
    }
    del model, phi_t, train_edges_t, test_edges_t, recon_full
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out
