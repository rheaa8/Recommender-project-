"""run_experiments.py - full pipeline: EDA, tuning on a validation split, final test
evaluation, segment analysis, sample recommendations, figures, results.json"""
import json, time, itertools, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from recsys_lib import *

SEED = 42
OUT = "results"; FIG = f"{OUT}/figs"
plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.grid": True, "grid.alpha": .25})
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ---------------- data ----------------
df, meta, raw = load_data("../data/ratings.dat", "../data/movies.dat", k_core=20)
n_u, n_i = df.u.max() + 1, df.i.max() + 1
R = {"raw": raw, "n_users": int(n_u), "n_items": int(n_i), "n_ratings": int(len(df)),
     "density": float(len(df) / n_u / n_i), "mean_rating": float(df.rating.mean()),
     "median_user_cnt": float(df.groupby("u").size().median()),
     "median_item_cnt": float(df.groupby("i").size().median())}
log("data", R)

# EDA figures
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3))
vc = df.rating.value_counts().sort_index()
ax[0].bar(vc.index, vc.values, color="#3b6ea5"); ax[0].set(title="Rating distribution", xlabel="Rating (0-10)", ylabel="Count")
uc = df.groupby("u").size().sort_values(ascending=False).values
ic = df.groupby("i").size().sort_values(ascending=False).values
ax[1].loglog(np.arange(1, len(uc) + 1), uc, color="#c0504d"); ax[1].set(title="Ratings per user (long tail)", xlabel="User rank", ylabel="# ratings")
ax[2].loglog(np.arange(1, len(ic) + 1), ic, color="#4f9d69"); ax[2].set(title="Ratings per item (long tail)", xlabel="Item rank", ylabel="# ratings")
plt.tight_layout(); plt.savefig(f"{FIG}/eda.png"); plt.close()
gen = meta.genres.str.split("|").explode().value_counts().drop("", errors="ignore")
R["top_genres"] = gen.head(8).to_dict()

# ---------------- splits ----------------
train_full, test = per_user_split(df, 0.20, SEED)
train_sub, val = per_user_split(train_full, 0.125, SEED + 1)   # 70% / 10% / 20% overall
S_val = Split(train_sub, val, n_u, n_i, meta)
S_te = Split(train_full, test, n_u, n_i, meta)
R["split"] = dict(train=len(train_sub), val=len(val), test=len(test), test_users=int(len(S_te.eval_users)))
log("splits", R["split"])
vu = S_val.eval_users[:3000]                               # speed: tune on 3000 users

# ---------------- tuning (validation set only) ----------------
TUNE = {}
bias_v = BiasBaseline().fit(S_val)
# k sensitivity for the neighbourhood models
for cls, key in ((UserKNN, "User-KNN"), (ItemKNN, "Item-KNN")):
    m = cls(k=50).fit(S_val, bias_v); rows = []
    for k in (5, 10, 20, 50, 100, 200):
        m.set_k(k); r = evaluate(m, S_val, users=vu, ks=(10,))
        rows.append(dict(k=k, RMSE=r["RMSE"], NDCG10=r["NDCG@10"])); log(key, rows[-1])
    TUNE[key] = rows
best_uk = min(TUNE["User-KNN"], key=lambda d: d["RMSE"])["k"]
best_ik = min(TUNE["Item-KNN"], key=lambda d: d["RMSE"])["k"]
# damping
m = ItemKNN(k=best_ik).fit(S_val, bias_v); TUNE["damp"] = []
for dmp in (0.05, 0.2, 0.5, 1.0, 2.0):
    m.damp = dmp; r = evaluate(m, S_val, users=vu, ks=(10,)); TUNE["damp"].append(dict(damp=dmp, RMSE=r["RMSE"], NDCG10=r["NDCG@10"])); log("damp", TUNE["damp"][-1])
best_damp = min(TUNE["damp"], key=lambda d: d["RMSE"])["damp"]
# implicit item-knn k
mi = ItemKNNImplicit(k=50).fit(S_val); TUNE["Item-KNN (implicit)"] = []
for k in (10, 20, 50, 100, 200):
    mi.set_k(k); r = evaluate(mi, S_val, users=vu, ks=(10,)); TUNE["Item-KNN (implicit)"].append(dict(k=k, NDCG10=r["NDCG@10"])); log("iknn-imp", TUNE["Item-KNN (implicit)"][-1])
best_imp_k = max(TUNE["Item-KNN (implicit)"], key=lambda d: d["NDCG10"])["k"]
# SVD / PureSVD factors
TUNE["SVD (residual)"], TUNE["PureSVD (binary)"] = [], []
for f in (5, 10, 20, 40, 80):
    r = evaluate(SVDResidual(f).fit(S_val, bias_v), S_val, users=vu, ks=(10,)); TUNE["SVD (residual)"].append(dict(k=f, RMSE=r["RMSE"], NDCG10=r["NDCG@10"])); log("svd", TUNE["SVD (residual)"][-1])
for f in (10, 25, 50, 100, 200):
    r = evaluate(PureSVD(f).fit(S_val), S_val, users=vu, ks=(10,)); TUNE["PureSVD (binary)"].append(dict(k=f, NDCG10=r["NDCG@10"])); log("puresvd", TUNE["PureSVD (binary)"][-1])
best_svd = min(TUNE["SVD (residual)"], key=lambda d: d["RMSE"])["k"]
best_psvd = max(TUNE["PureSVD (binary)"], key=lambda d: d["NDCG10"])["k"]
# ALS grid
TUNE["ALS"] = []
for f, lam in itertools.product((16, 32, 64), (0.15, 0.2, 0.3)):
    t = time.time(); m = ALSMF(k=f, lam=lam, iters=10).fit(S_val, bias_v)
    r = evaluate(m, S_val, users=vu, ks=(10,)); TUNE["ALS"].append(dict(k=f, lam=lam, RMSE=r["RMSE"], NDCG10=r["NDCG@10"])); log("als", TUNE["ALS"][-1], round(time.time() - t))
best_als = min(TUNE["ALS"], key=lambda d: d["RMSE"])
# hybrid weights (PureSVD, Item-KNN implicit, Content)
psv, iki, cb = PureSVD(best_psvd).fit(S_val), ItemKNNImplicit(k=best_imp_k).fit(S_val), ContentBased().fit(S_val)
TUNE["Hybrid"] = []
for w in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, .5, 0), (.7, .3, 0), (.6, .2, .2), (.5, .3, .2), (.4, .4, .2), (.5, .4, .1), (.45, .45, .1), (.7, 0, .3), (.8, 0, .2)]:
    h = Hybrid([psv, iki, cb], w).fit(S_val); r = evaluate(h, S_val, users=vu, ks=(10,))
    TUNE["Hybrid"].append(dict(w=list(w), NDCG10=r["NDCG@10"])); log("hybrid", TUNE["Hybrid"][-1])
best_w = max(TUNE["Hybrid"], key=lambda d: d["NDCG10"])["w"]
BEST = dict(user_k=best_uk, item_k=best_ik, damp=best_damp, imp_k=best_imp_k, svd_k=best_svd, psvd_k=best_psvd,
            als_k=best_als["k"], als_lam=best_als["lam"], hybrid_w=best_w)
log("BEST", BEST)

# ---------------- final training on the full training set, test evaluation ----------------
bias = BiasBaseline().fit(S_te)
uk = UserKNN(k=best_uk, damp=best_damp).fit(S_te, bias)
ik = ItemKNN(k=best_ik, damp=best_damp).fit(S_te, bias)
svd = SVDResidual(best_svd).fit(S_te, bias)
als = ALSMF(k=best_als["k"], lam=best_als["lam"], iters=15).fit(S_te, bias, track=True)
psv_t, iki_t, cb_t = PureSVD(best_psvd).fit(S_te), ItemKNNImplicit(k=best_imp_k).fit(S_te), ContentBased().fit(S_te)
hyb = Hybrid([psv_t, iki_t, cb_t], best_w).fit(S_te)
models = [Random().fit(S_te), Popularity().fit(S_te), bias, cb_t, uk, ik, svd, als, iki_t, psv_t, hyb]
bias.name = "Bias baseline"
FINAL, PER_USER = {}, {}
for m in models:
    t = time.time(); r = evaluate(m, S_te, want_per_user=True); PER_USER[m.name] = r.pop("_per_user"); FINAL[m.name] = r
    log("TEST", m.name, {k: round(v, 4) for k, v in r.items() if k in ("RMSE", "MAE", "NDCG@10", "R@10", "P@10")}, round(time.time() - t))
R["als_history"] = als.history

# bootstrap-style 95% CI for NDCG@10 (normal approx over users) & paired test vs best
for n, pu in PER_USER.items():
    x = pu["NDCG@10"]; FINAL[n]["NDCG@10_ci"] = float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
best_name = max(FINAL, key=lambda n: FINAL[n]["NDCG@10"])
R["best_name"] = best_name
paired = {}
for n in FINAL:
    if n == best_name: continue
    d = PER_USER[best_name]["NDCG@10"] - PER_USER[n]["NDCG@10"]
    paired[n] = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))     # t-statistic
R["paired_t_vs_best"] = paired

# ---------------- segment analysis (user activity) ----------------
eu = S_te.eval_users; cnt = S_te.user_cnt[eu]
q1, q2 = np.quantile(cnt, [1 / 3, 2 / 3])
seg = np.where(cnt <= q1, "Light", np.where(cnt <= q2, "Medium", "Heavy"))
R["segments"] = {"cuts": [float(q1), float(q2)], "n": {s_: int((seg == s_).sum()) for s_ in ("Light", "Medium", "Heavy")}, "ndcg": {}}
for n in ("Popularity", "Item-KNN", "ALS Matrix Factorisation", "PureSVD (binary)", "Content-based (genres)", best_name):
    R["segments"]["ndcg"][n] = {s_: float(PER_USER[n]["NDCG@10"][seg == s_].mean()) for s_ in ("Light", "Medium", "Heavy")}

# ---------------- sample recommendations ----------------
rng = np.random.default_rng(7)
cand = [u for u in S_te.eval_users if 40 <= S_te.user_cnt[u] <= 120]
demo_u = int(rng.choice(cand))
titles = meta.title.values
liked = S_te.train[(S_te.train.u == demo_u)].sort_values("rating", ascending=False).head(8)
R["demo"] = {"user": demo_u, "liked": [(titles[i], int(r)) for i, r in zip(liked.i, liked.rating)], "recs": {},
             "held_out_liked": [titles[i] for i in S_te.test[(S_te.test.u == demo_u) & (S_te.test.rating >= 8)].i]}
for m in (pop := models[1], cb_t, ik, als, psv_t, hyb):
    sc = np.asarray(m.score(np.array([demo_u])))[0].copy(); sc[S_te.M[demo_u].toarray()[0] > 0] = -np.inf
    R["demo"]["recs"][m.name] = [titles[i] for i in np.argsort(-sc)[:5]]

R["final"], R["tune"], R["best"] = FINAL, TUNE, BEST
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)

# ---------------- figures ----------------
C = {"Random": "#999999", "Popularity": "#8c6bb1", "Bias baseline": "#bdbdbd", "Content-based (genres)": "#e6a23c",
     "User-KNN": "#5b9bd5", "Item-KNN": "#2e75b6", "SVD (residual)": "#70ad47", "ALS Matrix Factorisation": "#548235",
     "Item-KNN (implicit)": "#c55a11", "PureSVD (binary)": "#a5442b", "Hybrid (PureSVD+KNN+Content)": "#7f1d1d"}
# (1) rating accuracy
rm = {n: v["RMSE"] for n, v in FINAL.items() if "RMSE" in v}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].barh(list(rm), list(rm.values()), color=[C[n] for n in rm]); ax[0].set(title="Rating prediction: RMSE (lower = better)", xlim=(min(rm.values()) - .05, max(rm.values()) + .03)); ax[0].invert_yaxis()
mae = {n: FINAL[n]["MAE"] for n in rm}
ax[1].barh(list(mae), list(mae.values()), color=[C[n] for n in mae]); ax[1].set(title="Rating prediction: MAE (lower = better)", xlim=(min(mae.values()) - .05, max(mae.values()) + .03)); ax[1].invert_yaxis(); ax[1].set_yticklabels([])
for a_, d_ in ((ax[0], rm), (ax[1], mae)):
    for y, v in enumerate(d_.values()): a_.text(v + .002, y, f"{v:.3f}", va="center", fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/rating_accuracy.png"); plt.close()
# (2) ranking
names = sorted(FINAL, key=lambda n: FINAL[n]["NDCG@10"])
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for a_, met, t in zip(ax, ("NDCG@10", "R@10", "HR@10"), ("NDCG@10", "Recall@10", "Hit Rate@10")):
    v = [FINAL[n][met] for n in names]; a_.barh(names, v, color=[C[n] for n in names]); a_.set_title(t)
    for y, x in enumerate(v): a_.text(x + .002, y, f"{x:.3f}", va="center", fontsize=7)
plt.tight_layout(); plt.savefig(f"{FIG}/ranking.png"); plt.close()
# (3) accuracy-vs-diversity
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for n in FINAL:
    ax[0].scatter(FINAL[n]["Coverage@10"] * 100, FINAL[n]["NDCG@10"], s=60, color=C[n]); ax[0].annotate(n.split(" (")[0], (FINAL[n]["Coverage@10"] * 100, FINAL[n]["NDCG@10"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax[1].scatter(FINAL[n]["Novelty@10"], FINAL[n]["NDCG@10"], s=60, color=C[n]); ax[1].annotate(n.split(" (")[0], (FINAL[n]["Novelty@10"], FINAL[n]["NDCG@10"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
ax[0].set(xlabel="Catalogue coverage@10 (%)", ylabel="NDCG@10", title="Accuracy vs coverage"); ax[1].set(xlabel="Novelty@10 (bits)", title="Accuracy vs novelty")
plt.tight_layout(); plt.savefig(f"{FIG}/tradeoff.png"); plt.close()
# (4) sensitivity
fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
for key, c in (("User-KNN", "#5b9bd5"), ("Item-KNN", "#2e75b6")):
    ax[0].plot([d["k"] for d in TUNE[key]], [d["RMSE"] for d in TUNE[key]], "o-", label=key, color=c)
ax[0].set(xscale="log", xlabel="Neighbourhood size k", ylabel="Validation RMSE", title="KNN: effect of k"); ax[0].legend()
ax[1].plot([d["k"] for d in TUNE["SVD (residual)"]], [d["RMSE"] for d in TUNE["SVD (residual)"]], "o-", color="#70ad47", label="SVD (residual)")
for lam, c in zip((0.15, 0.2, 0.3), ("#c6e0b4", "#548235", "#274e13")):
    g = [d for d in TUNE["ALS"] if d["lam"] == lam]; ax[1].plot([d["k"] for d in g], [d["RMSE"] for d in g], "s--", color=c, label=f"ALS λ={lam}")
ax[1].set(xlabel="Latent factors", ylabel="Validation RMSE", title="Matrix factorisation"); ax[1].legend(fontsize=6.5)
h = als.history; ax[2].plot([x[0] for x in h], [x[1] for x in h], "o-", label="train", color="#2e75b6"); ax[2].plot([x[0] for x in h], [x[2] for x in h], "o-", label="test", color="#c0504d")
ax[2].set(xlabel="ALS iteration", ylabel="RMSE", title="ALS convergence"); ax[2].legend()
plt.tight_layout(); plt.savefig(f"{FIG}/sensitivity.png"); plt.close()
# (5) segments
fig, ax = plt.subplots(figsize=(6.5, 3.2)); w = .13
for j, (n, d) in enumerate(R["segments"]["ndcg"].items()):
    ax.bar(np.arange(3) + j * w, [d["Light"], d["Medium"], d["Heavy"]], w, label=n.split(" (")[0], color=C[n])
ax.set_xticks(np.arange(3) + w * 2.5); ax.set_xticklabels([f"Light (≤{int(q1)})", f"Medium (≤{int(q2)})", f"Heavy (>{int(q2)})"]); ax.set(ylabel="NDCG@10", title="Performance by user activity level (train ratings)"); ax.legend(fontsize=6.5, ncol=2)
plt.tight_layout(); plt.savefig(f"{FIG}/segments.png"); plt.close()
log("DONE")
