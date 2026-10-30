"""
recsys_lib.py
Core library for the comparative analysis of recommender systems.
Implements data handling, 11 recommenders (from scratch with NumPy/SciPy) and
the evaluation protocol (rating-prediction + top-N ranking metrics).
"""
import html
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def load_data(path_ratings, path_movies, k_core=20):
    r = pd.read_csv(path_ratings, sep="::", engine="python",
                    names=["user", "item", "rating", "ts"], dtype={"item": str})
    r["item"] = r["item"].astype(int)
    m = pd.read_csv(path_movies, sep="::", engine="python",
                    names=["item", "title", "genres"], dtype={"item": str},
                    encoding="utf-8")
    m["item"] = m["item"].astype(int)
    m = m.drop_duplicates("item")
    raw_stats = dict(n_ratings=len(r), n_users=r.user.nunique(), n_items=r.item.nunique())
    # iterative k-core filtering (users and items with >= k ratings)
    d = r.copy()
    while True:
        n = len(d)
        d = d[d.groupby("user")["item"].transform("size") >= k_core]
        d = d[d.groupby("item")["user"].transform("size") >= k_core]
        if len(d) == n:
            break
    d = d.drop_duplicates(["user", "item"]).reset_index(drop=True)
    u_codes, u_idx = np.unique(d.user.values, return_inverse=True)
    i_codes, i_idx = np.unique(d.item.values, return_inverse=True)
    d["u"], d["i"] = u_idx, i_idx
    meta = pd.DataFrame({"item": i_codes}).merge(m, on="item", how="left")
    meta["title"] = meta["title"].fillna("Unknown").map(html.unescape)
    meta["genres"] = meta["genres"].fillna("")
    return d[["u", "i", "rating", "ts"]], meta, raw_stats


def per_user_split(df, frac, seed):
    """Random per-user split: `frac` of each user's ratings go to the held-out part."""
    rng = np.random.default_rng(seed)
    rnd = rng.random(len(df))
    tmp = df.assign(_r=rnd)
    rk = tmp.groupby("u")["_r"].rank(method="first", pct=True)
    mask = rk.values > (1 - frac)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


class Split:
    """Holds a train / held-out pair and the matrices every model needs."""
    def __init__(self, train, test, n_users, n_items, meta, like_thr=8):
        self.n_users, self.n_items, self.meta = n_users, n_items, meta
        self.train, self.test = train, test
        self.like_thr = like_thr
        u, i, r = train.u.values, train.i.values, train.rating.values.astype(np.float64)
        self.R = sp.csr_matrix((r, (u, i)), shape=(n_users, n_items))        # ratings
        self.M = sp.csr_matrix((np.ones(len(u)), (u, i)), shape=(n_users, n_items))  # mask
        # held-out pairs sorted by user (for fast batch lookup)
        t = test.sort_values("u").reset_index(drop=True)
        self.t_u, self.t_i, self.t_r = t.u.values, t.i.values, t.rating.values.astype(float)
        rel = t[t.rating >= like_thr]
        self.Rel = sp.csr_matrix((np.ones(len(rel)), (rel.u.values, rel.i.values)),
                                 shape=(n_users, n_items))
        self.n_rel = np.asarray(self.Rel.sum(1)).ravel()
        self.eval_users = np.where(self.n_rel > 0)[0]
        self.item_pop = np.asarray(self.M.sum(0)).ravel()
        self.user_cnt = np.asarray(self.M.sum(1)).ravel()


# ----------------------------------------------------------------------------
# 2. MODELS  (API: fit(split) ; score(users) -> dense (len(users), n_items))
# ----------------------------------------------------------------------------
class Random:
    name, is_rating = "Random", False
    def __init__(self, seed=0): self.rng = np.random.default_rng(seed)
    def fit(self, s): self.n = s.n_items; return self
    def score(self, users): return self.rng.random((len(users), self.n))


class Popularity:
    name, is_rating = "Popularity", False
    def fit(self, s): self.pop = s.item_pop.astype(float); return self
    def score(self, users): return np.tile(self.pop, (len(users), 1))


class BiasBaseline:
    """r_ui ~ mu + b_u + b_i with damped (regularised) bias estimates."""
    name, is_rating = "Bias baseline", True
    def __init__(self, lam_i=10, lam_u=10): self.lam_i, self.lam_u = lam_i, lam_u
    def fit(self, s):
        R, M = s.R, s.M
        self.mu = R.sum() / M.sum()
        n_i = np.asarray(M.sum(0)).ravel(); n_u = np.asarray(M.sum(1)).ravel()
        self.bi = (np.asarray(R.sum(0)).ravel() - self.mu * n_i) / (n_i + self.lam_i)
        Rb = np.asarray(R.sum(1)).ravel() - self.mu * n_u - np.asarray(M.multiply(self.bi[None, :]).sum(1)).ravel()
        self.bu = Rb / (n_u + self.lam_u)
        # residual matrix on observed entries (used by neighbourhood / MF models)
        Rc = R.copy().tocoo()
        Rc.data = Rc.data - (self.mu + self.bu[Rc.row] + self.bi[Rc.col])
        self.Rc = Rc.tocsr()
        return self
    def base(self, users): return self.mu + self.bu[users][:, None] + self.bi[None, :]
    def score(self, users): return self.base(users)


def _topk_rows(sim_block, k, exclude_diag_offset=None):
    """Return indices & values of the k largest entries per row, sorted descending."""
    if exclude_diag_offset is not None:
        r = np.arange(sim_block.shape[0])
        sim_block[r, r + exclude_diag_offset] = -np.inf
    idx = np.argpartition(-sim_block, k, axis=1)[:, :k]
    val = np.take_along_axis(sim_block, idx, 1)
    order = np.argsort(-val, axis=1)
    return np.take_along_axis(idx, order, 1), np.take_along_axis(val, order, 1)


def _neighbours(X, M, kmax, shrink, block=1000):
    """Cosine similarity between rows of X (shrunk by co-rating support), top-kmax per row."""
    n = X.shape[0]
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-9
    Xd = (sp.diags(1 / norms) @ X).toarray().astype(np.float32)
    Md = M.toarray().astype(np.float32)
    I = np.zeros((n, kmax), np.int32); V = np.zeros((n, kmax), np.float32)
    for a in range(0, n, block):
        b = min(a + block, n)
        s = Xd[a:b] @ Xd.T
        c = Md[a:b] @ Md.T
        s = s * (c / (c + shrink))
        i_, v_ = _topk_rows(s, kmax, exclude_diag_offset=a)
        I[a:b], V[a:b] = i_, np.maximum(v_, 0)
    return I, V


def _to_sparse(I, V, k, n_cols):
    n = I.shape[0]
    rows = np.repeat(np.arange(n), k)
    return sp.csr_matrix((V[:, :k].ravel(), (rows, I[:, :k].ravel())), shape=(n, n_cols))


class UserKNN:
    is_rating = True
    def __init__(self, k=50, shrink=20, damp=0.5, kmax=200):
        self.k, self.shrink, self.damp, self.kmax = k, shrink, damp, kmax
        self.name = "User-KNN"
    def fit(self, s, bias=None):
        self.bias = bias or BiasBaseline().fit(s)
        self.Rc, self.M = self.bias.Rc, s.M
        self.I, self.V = _neighbours(self.Rc, s.M, self.kmax, self.shrink)
        self.set_k(self.k); return self
    def set_k(self, k):
        self.k = k; self.W = _to_sparse(self.I, self.V, k, self.Rc.shape[0])
    def score(self, users):
        W = self.W[users]
        num = (W @ self.Rc).toarray(); den = (W @ self.M).toarray()
        return self.bias.base(users) + num / (den + self.damp + 1e-9)


class ItemKNN:
    is_rating = True
    def __init__(self, k=50, shrink=20, damp=0.5, kmax=200):
        self.k, self.shrink, self.damp, self.kmax = k, shrink, damp, kmax
        self.name = "Item-KNN"
    def fit(self, s, bias=None):
        self.bias = bias or BiasBaseline().fit(s)
        self.RcT, self.MT = self.bias.Rc.T.tocsr(), s.M.T.tocsr()
        self.Rc, self.M = self.bias.Rc, s.M
        self.I, self.V = _neighbours(self.RcT, self.MT, self.kmax, self.shrink)
        self.set_k(self.k); return self
    def set_k(self, k):
        self.k = k; self.WT = _to_sparse(self.I, self.V, k, self.Rc.shape[1]).T.tocsr()
    def score(self, users):
        num = (self.Rc[users] @ self.WT).toarray()
        den = (self.M[users] @ self.WT).toarray()
        return self.bias.base(users) + num / (den + self.damp + 1e-9)


class ItemKNNImplicit:
    """Top-N item-based CF on binary interactions (cosine, no ratings)."""
    name, is_rating = "Item-KNN (implicit)", False
    def __init__(self, k=50, shrink=10, kmax=200): self.k, self.shrink, self.kmax = k, shrink, kmax
    def fit(self, s):
        self.B = s.M.tocsr(); BT = self.B.T.tocsr()
        self.I, self.V = _neighbours(BT, BT, self.kmax, self.shrink)
        self.set_k(self.k); return self
    def set_k(self, k):
        self.k = k; self.WT = _to_sparse(self.I, self.V, k, self.B.shape[1]).T.tocsr()
    def score(self, users): return (self.B[users] @ self.WT).toarray()


class SVDResidual:
    """Truncated SVD of the bias-residual rating matrix (rating prediction)."""
    is_rating = True
    def __init__(self, k=40): self.k = k; self.name = "SVD (residual)"
    def fit(self, s, bias=None):
        self.bias = bias or BiasBaseline().fit(s)
        U, S, Vt = svds(self.bias.Rc.astype(np.float64), k=self.k)
        self.US, self.Vt = U * S, Vt; return self
    def score(self, users): return self.bias.base(users) + self.US[users] @ self.Vt


class PureSVD:
    """Truncated SVD of the binary interaction matrix (Cremonesi et al., 2010) - top-N."""
    name, is_rating = "PureSVD (binary)", False
    def __init__(self, k=50): self.k = k
    def fit(self, s):
        U, S, Vt = svds(s.M.astype(np.float64), k=self.k)
        self.US, self.Vt = U * S, Vt; return self
    def score(self, users): return self.US[users] @ self.Vt


class ALSMF:
    """Regularised matrix factorisation trained with Alternating Least Squares
    (weighted-lambda regularisation) on the bias residuals."""
    is_rating = True
    def __init__(self, k=32, lam=0.1, iters=12, seed=0):
        self.k, self.lam, self.iters, self.seed = k, lam, iters, seed
        self.name = "ALS Matrix Factorisation"; self.history = []
    @staticmethod
    def _solve(indptr, indices, data, F, lam, k):
        out = np.zeros((len(indptr) - 1, k))
        eye = np.eye(k)
        for u in range(len(indptr) - 1):
            a, b = indptr[u], indptr[u + 1]
            if b == a: continue
            Fu = F[indices[a:b]]
            out[u] = np.linalg.solve(Fu.T @ Fu + lam * (b - a) * eye, Fu.T @ data[a:b])
        return out
    def fit(self, s, bias=None, track=None):
        self.bias = bias or BiasBaseline().fit(s)
        Rc = self.bias.Rc; RcT = Rc.T.tocsr()
        rng = np.random.default_rng(self.seed)
        P = rng.normal(0, 0.1, (Rc.shape[0], self.k)); Q = rng.normal(0, 0.1, (Rc.shape[1], self.k))
        self.history = []
        for it in range(self.iters):
            P = self._solve(Rc.indptr, Rc.indices, Rc.data, Q, self.lam, self.k)
            Q = self._solve(RcT.indptr, RcT.indices, RcT.data, P, self.lam, self.k)
            self.P, self.Q = P, Q
            if track is not None:
                self.history.append((it + 1, self._rmse(s, s.train.u.values, s.train.i.values, s.train.rating.values),
                                     self._rmse(s, s.t_u, s.t_i, s.t_r)))
        self.P, self.Q = P, Q
        return self
    def _rmse(self, s, u, i, r):
        pred = self.bias.mu + self.bias.bu[u] + self.bias.bi[i] + np.einsum("ij,ij->i", self.P[u], self.Q[i])
        return float(np.sqrt(np.mean((np.clip(pred, 0, 10) - r) ** 2)))
    def score(self, users): return self.bias.base(users) + self.P[users] @ self.Q.T


class ContentBased:
    """TF-IDF genre/decade item profiles; user profile = rating-weighted sum of item vectors."""
    name, is_rating = "Content-based (genres)", False
    def fit(self, s):
        meta = s.meta
        gl = [set(g.split("|")) - {""} for g in meta.genres]
        vocab = sorted(set().union(*gl))
        yr = meta.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
        dec = (yr // 10 * 10).fillna(0).astype(int)
        dec_vocab = sorted(set(dec) - {0})
        X = np.zeros((len(meta), len(vocab) + len(dec_vocab)), np.float32)
        for r, g in enumerate(gl):
            for t in g: X[r, vocab.index(t)] = 1
            if dec[r] in dec_vocab: X[r, len(vocab) + dec_vocab.index(dec[r])] = 1
        df_ = X.sum(0) + 1
        X = X * np.log(len(meta) / df_)[None, :]
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        self.X = X
        R, M = s.R, s.M
        mean_u = np.asarray(R.sum(1)).ravel() / np.maximum(np.asarray(M.sum(1)).ravel(), 1)
        Rc = R.copy().tocoo(); Rc.data = Rc.data - mean_u[Rc.row]
        self.prof = Rc.tocsr() @ X                       # users x features
        self.prof /= (np.linalg.norm(self.prof, axis=1, keepdims=True) + 1e-9)
        return self
    def score(self, users): return self.prof[users] @ self.X.T


class Hybrid:
    """Weighted blend of z-score-normalised component scores (seen items excluded)."""
    name, is_rating = "Hybrid (PureSVD+KNN+Content)", False
    def __init__(self, parts, weights): self.parts, self.weights = parts, weights
    def fit(self, s): self.M = s.M; return self
    def score(self, users):
        seen = self.M[users].toarray() > 0
        out = 0
        for p, w in zip(self.parts, self.weights):
            if w == 0: continue
            x = p.score(users).astype(np.float64); x[seen] = np.nan
            z = (x - np.nanmean(x, 1, keepdims=True)) / (np.nanstd(x, 1, keepdims=True) + 1e-9)
            out = out + w * np.nan_to_num(z, nan=0.0)
        return out


# ----------------------------------------------------------------------------
# 3. EVALUATION
# ----------------------------------------------------------------------------
def evaluate(model, s, ks=(10, 20), batch=1000, users=None, want_per_user=False, rec_store=None):
    """Rating metrics (RMSE/MAE on held-out ratings) and ranking metrics @K over all
    unseen items. Relevant = held-out items rated >= like_thr."""
    res = {}
    n_items = s.n_items
    # ---- rating prediction
    if model.is_rating:
        se, ae, cnt = 0.0, 0.0, 0
        for a in range(0, s.n_users, batch):
            b = min(a + batch, s.n_users)
            lo, hi = np.searchsorted(s.t_u, [a, b])
            if hi == lo: continue
            P = model.score(np.arange(a, b))
            pred = np.clip(P[s.t_u[lo:hi] - a, s.t_i[lo:hi]], 0, 10)
            err = pred - s.t_r[lo:hi]
            se += (err ** 2).sum(); ae += np.abs(err).sum(); cnt += len(err)
        res["RMSE"], res["MAE"] = float(np.sqrt(se / cnt)), float(ae / cnt)
    # ---- top-N ranking
    users = s.eval_users if users is None else users
    kmax = max(ks)
    disc = 1.0 / np.log2(np.arange(kmax) + 2)
    cum_idcg = np.cumsum(disc)
    acc = {f"{m}@{k}": [] for k in ks for m in ("P", "R", "NDCG", "HR")}
    rec_items = []
    for a in range(0, len(users), batch):
        ub = users[a:a + batch]
        S = np.asarray(model.score(ub), dtype=np.float32)
        seen = s.M[ub].toarray() > 0
        S[seen] = -np.inf
        top = np.argpartition(-S, kmax, axis=1)[:, :kmax]
        tv = np.take_along_axis(S, top, 1)
        top = np.take_along_axis(top, np.argsort(-tv, axis=1), 1)
        rel = s.Rel[ub].toarray()
        hit = np.take_along_axis(rel, top, 1)
        nrel = s.n_rel[ub]
        for k in ks:
            h = hit[:, :k]
            dcg = (h * disc[:k]).sum(1)
            idcg = cum_idcg[np.minimum(nrel, k).astype(int) - 1]
            acc[f"P@{k}"].append(h.sum(1) / k)
            acc[f"R@{k}"].append(h.sum(1) / nrel)
            acc[f"NDCG@{k}"].append(dcg / idcg)
            acc[f"HR@{k}"].append((h.sum(1) > 0).astype(float))
        rec_items.append(top[:, :10])
        if rec_store is not None: rec_store.append((ub, top[:, :10]))
    per_user = {k: np.concatenate(v) for k, v in acc.items()}
    for k, v in per_user.items(): res[k] = float(v.mean())
    rec = np.concatenate(rec_items)
    res["Coverage@10"] = float(len(np.unique(rec)) / n_items)
    pop = np.maximum(s.item_pop, 1) / s.n_users
    res["Novelty@10"] = float((-np.log2(pop[rec])).mean())
    res["AvgPop@10"] = float(s.item_pop[rec].mean())
    if want_per_user: res["_per_user"] = per_user
    return res
