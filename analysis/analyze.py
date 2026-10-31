"""Findings, the matchup model, and its validation. Reads data/serves.pkl, writes site/data/*.json.

1. Baselines: how often each aim wins, by tour / serve / side / surface.
2. Minimax screen (Walker & Wooders 2001): game theory says a pro's win rate should be the same
   whichever way they aim. We test that per player and side, with Benjamini-Hochberg control.
3. Predictability: do servers repeat or switch their aim more than chance? Permutation test
   within each match and side, so the null keeps each match's own mix.
4. Pressure: does the mix change on break points, relative to the same player's normal mix?
5. Matchup model: penalized logistic regression (a MAP random-effects model) for
   P(server wins point | serve, side, aim, surface, server, returner). Penalties picked on 2024,
   then validated on 2025-26 matches the model never saw.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "site" / "data"
AIMS = ["wide", "body", "T"]
SIDES = ["deuce", "ad"]
SURFACES = ["Hard", "Clay", "Grass", "Other"]
RNG = np.random.default_rng(7)

MIN_FIT = 100        # serve (or return) attempts before a player gets their own coefficients
MIN_EXPORT = 1000    # attempts before a player is listed in the dashboard
MIN_TEST_SIDE = 200  # first serves on one side before we run the minimax test for that player


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def load():
    s = pd.read_pickle(ROOT / "data" / "serves.pkl")
    s["aim_i"] = s["aim"].map({a: i for i, a in enumerate(AIMS)}).astype(np.int8)
    s["side_i"] = (s["side"] == "ad").astype(np.int8)
    s["serve_i"] = (s["serve"] - 1).astype(np.int8)
    s["surf_i"] = s["surface"].map({x: i for i, x in enumerate(SURFACES)}).astype(np.int8)
    s["year"] = s["date"].dt.year
    return s


# ---------------------------------------------------------------- 1. baselines

def baselines(s):
    g = s.groupby(["tour", "serve", "side", "aim"]).agg(n=("won", "size"), won=("won", "mean"),
                                                         land=("in", "mean"), free=("unreturned", "mean"))
    by_surface = s[s["serve"] == 1].groupby(["tour", "surface", "side", "aim"]).agg(n=("won", "size"), won=("won", "mean"))
    mix = s[s["serve"] == 1].groupby(["tour", "side"])["aim"].value_counts(normalize=True)
    out = {"cells": [], "surface": [], "mix": []}
    for (tour, serve, side, aim), row in g.iterrows():
        out["cells"].append({"tour": tour, "serve": int(serve), "side": side, "aim": aim, "n": int(row.n),
                             "won": r(row.won), "land": r(row.land), "free": r(row.free)})
    for (tour, surf, side, aim), row in by_surface.iterrows():
        if surf != "Other":
            out["surface"].append({"tour": tour, "surface": surf, "side": side, "aim": aim, "n": int(row.n), "won": r(row.won)})
    for (tour, side, aim), v in mix.items():
        out["mix"].append({"tour": tour, "side": side, "aim": aim, "share": r(v)})
    return out


# ---------------------------------------------------------------- 2. minimax screen

def bh(p, q=0.10):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    passed = np.zeros(len(p), bool)
    below = np.nonzero(ranked <= q)[0]
    if len(below):
        passed[order[: below.max() + 1]] = True
    return passed


def minimax(s):
    f = s[s["serve"] == 1]
    rows = []
    for (tour, server, side), g in f.groupby(["tour", "server", "side"]):
        if len(g) < MIN_TEST_SIDE:
            continue
        # The two main targets only: body first serves are rare and partly mis-hits (see finding 02).
        t = g[g["aim"].isin(["T", "wide"])].groupby("aim")["won"].agg(["size", "sum"])
        t = t[t["size"] >= 30]
        if len(t) < 2:
            continue
        table = np.vstack([t["sum"].values, (t["size"] - t["sum"]).values])
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        rates = (t["sum"] / t["size"])
        rows.append({"tour": tour, "player": server, "side": side, "n": int(len(g)), "p": float(p),
                     "best": rates.idxmax(), "worst": rates.idxmin(),
                     "best_rate": float(rates.max()), "worst_rate": float(rates.min()),
                     "gap": float(rates.max() - rates.min()),
                     "share_best": float(g["aim"].eq(rates.idxmax()).mean())})
    df = pd.DataFrame(rows)
    out = {"by_tour": [], "flagged": []}
    for tour, d in df.groupby("tour"):
        flags = bh(d["p"].values)
        d = d.assign(flag=flags)
        ks = stats.kstest(d["p"], "uniform")
        hist, _ = np.histogram(d["p"], bins=20, range=(0, 1))
        out["by_tour"].append({"tour": tour, "tests": int(len(d)), "players": int(d["player"].nunique()),
                               "share_p05": r((d["p"] < 0.05).mean()), "bh_flagged": int(flags.sum()),
                               "ks_p": float(ks.pvalue), "hist": hist.tolist(),
                               "median_gap": r(d["gap"].median())})
        for _, row in d[d["flag"]].sort_values("p").iterrows():
            out["flagged"].append({k: (r(v) if isinstance(v, float) else v) for k, v in row.items() if k != "flag"})
    return out


# ---------------------------------------------------------------- 3. predictability

def serial(s, n_perm=200):
    f = s[s["serve"] == 1][["tour", "match_id", "server", "side", "aim_i"]].reset_index(drop=True)
    grp = f.groupby(["match_id", "server", "side"], sort=False).ngroup().values
    order = np.argsort(grp, kind="stable")  # rows are already in point order inside each group
    f, grp = f.iloc[order].reset_index(drop=True), grp[order]
    aim = f["aim_i"].values
    same_grp = np.r_[False, grp[1:] == grp[:-1]]

    def repeats(a):
        return np.r_[False, a[1:] == a[:-1]] & same_grp

    pid = f.groupby(["tour", "server"]).ngroup().values
    n_players = pid.max() + 1
    obs = np.bincount(pid, weights=repeats(aim), minlength=n_players)
    pairs = np.bincount(pid, weights=same_grp, minlength=n_players)
    null = np.empty((n_perm, n_players))
    for k in range(n_perm):
        perm = np.lexsort((RNG.random(len(aim)), grp))
        null[k] = np.bincount(pid, weights=repeats(aim[perm]), minlength=n_players)
    exp = null.mean(0)
    sd = null.std(0, ddof=1)
    names = f.groupby(["tour", "server"]).ngroup().drop_duplicates()
    keys = f.loc[names.index, ["tour", "server"]].values
    players = []
    for i, (tour, server) in enumerate(keys):
        j = names.values[i]
        if pairs[j] < 300:
            continue
        players.append({"tour": tour, "player": server, "pairs": int(pairs[j]), "obs": r(obs[j] / pairs[j]),
                        "exp": r(exp[j] / pairs[j]), "z": r((obs[j] - exp[j]) / sd[j] if sd[j] > 0 else 0, 2)})
    out = {"players": players, "by_tour": []}
    tour_of = pd.Series(keys[:, 0], index=names.values)
    for tour in ("ATP", "WTA"):
        m = (tour_of.reindex(range(n_players)) == tour).values
        o, e = obs[m].sum(), null[:, m].sum(1)
        pl = [p for p in players if p["tour"] == tour]
        out["by_tour"].append({"tour": tour, "pairs": int(pairs[m].sum()), "obs": r(o / pairs[m].sum()),
                               "exp": r(e.mean() / pairs[m].sum()), "perm_p": r(((np.abs(e - e.mean()) >= abs(o - e.mean())).sum() + 1) / (n_perm + 1)),
                               "z": r((o - e.mean()) / e.std(ddof=1), 1),
                               "share_switch_more": r(np.mean([p["obs"] < p["exp"] for p in pl])),
                               "share_sig_switch": r(np.mean([p["z"] < -1.96 for p in pl])),
                               "share_sig_repeat": r(np.mean([p["z"] > 1.96 for p in pl])), "players": len(pl)})
    return out


SHRINK_K = 100  # pseudo-serves pulling a "what comes after X" mix toward the player's usual mix


def anticipation(s):
    """Can a returner use the last serve to that side to guess the next one? Learn on <=2024, score on 2025-26."""
    f = s[(s["serve"] == 1) & s["prev_aim_same_side"].notna()].copy()
    f["prev_i"] = f["prev_aim_same_side"].map({a: i for i, a in enumerate(AIMS)})
    f["repeat"] = (f["aim_i"] == f["prev_i"]).astype(int)
    out = []
    for tour, d in f.groupby("tour"):
        tr, te = d[d["year"] <= 2024], d[d["year"] >= 2025]
        marg = tr.groupby(["server", "side_i", "aim_i"]).size().unstack(fill_value=0).reindex(columns=range(3), fill_value=0)
        mprob = (marg + 1).div((marg + 1).sum(axis=1), axis=0)
        cond = tr.groupby(["server", "side_i", "prev_i", "aim_i"]).size().unstack(fill_value=0).reindex(columns=range(3), fill_value=0)
        m_for_c = mprob.reindex(cond.index.droplevel("prev_i")).values
        cprob = pd.DataFrame((cond.values + SHRINK_K * m_for_c) / (cond.values.sum(1, keepdims=True) + SHRINK_K), index=cond.index)
        t = te.join(mprob, on=["server", "side_i"], how="inner").join(cprob.add_prefix("c"), on=["server", "side_i", "prev_i"], how="inner")
        ai = t["aim_i"].values.astype(int)
        pm, pc = t[[0, 1, 2]].values, t[["c0", "c1", "c2"]].values
        idx = np.arange(len(t))
        # Do servers win more when they repeat (returner guessed wrong) than when they switch? Strata: server x side x aim.
        g = d.groupby(["server", "side_i", "aim_i", "repeat"])["won"].agg(["sum", "size"]).unstack("repeat").dropna()
        g = g[(g[("size", 0)] >= 20) & (g[("size", 1)] >= 20)]
        diff = g[("sum", 1)] / g[("size", 1)] - g[("sum", 0)] / g[("size", 0)]
        w = 1 / (1 / g[("size", 1)] + 1 / g[("size", 0)])
        pbar = d["won"].mean()
        out.append({"tour": tour, "test_serves": int(len(t)),
                    "acc_usual": r((pm.argmax(1) == ai).mean()), "acc_tell": r((pc.argmax(1) == ai).mean()),
                    "ll_usual": r(-np.log(pm[idx, ai]).mean()), "ll_tell": r(-np.log(pc[idx, ai]).mean()),
                    "repeat_minus_switch": r(np.average(diff, weights=w)), "repeat_minus_switch_se": r(np.sqrt(pbar * (1 - pbar) / w.sum()))})
    return out


# ---------------------------------------------------------------- 4. pressure

def pressure(s):
    f = s[(s["serve"] == 1) & ~s["tiebreak"]]
    out = []
    for tour, d in f.groupby("tour"):
        normal = d[~d["break_point"]].groupby(["server", "side"])["aim"].value_counts(normalize=True).rename("p").reset_index()
        fav = normal.sort_values("p").groupby(["server", "side"]).tail(1).rename(columns={"aim": "fav", "p": "p_fav"})
        bp = d[d["break_point"]].merge(fav, on=["server", "side"])
        normal_n = d[~d["break_point"]].groupby(["server", "side"]).size().rename("n_normal").reset_index()
        bp = bp.merge(normal_n, on=["server", "side"])
        bp = bp[bp["n_normal"] >= 100]
        hit = bp["aim"].eq(bp["fav"])
        exp = bp["p_fav"]
        z = (hit.sum() - exp.sum()) / np.sqrt((exp * (1 - exp)).sum())
        # T share on break points vs the same servers' normal T share
        t_norm = normal[normal["aim"] == "T"].rename(columns={"p": "pT"})[["server", "side", "pT"]]
        bpt = d[d["break_point"]].merge(t_norm, on=["server", "side"], how="left").fillna({"pT": 0})
        bpt = bpt.merge(normal_n, on=["server", "side"])
        bpt = bpt[bpt["n_normal"] >= 100]
        won_bp = d[d["break_point"]].groupby("aim")["won"].mean()
        won_norm = d[~d["break_point"]].groupby("aim")["won"].mean()
        out.append({"tour": tour, "bp_serves": int(len(bp)), "fav_obs": r(hit.mean()), "fav_exp": r(exp.mean()),
                    "fav_z": r(z, 1), "t_obs": r(bpt["aim"].eq("T").mean()), "t_exp": r(bpt["pT"].mean()),
                    "won_bp": {a: r(won_bp.get(a)) for a in AIMS}, "won_normal": {a: r(won_norm.get(a)) for a in AIMS}})
    return out


# ---------------------------------------------------------------- 5. matchup model

class Design:
    """Column layout: 48 base cells, then per-server and per-returner effects (overall, by serve, by serve x side x aim)."""

    def __init__(self, servers, returners):
        self.srv = {p: i for i, p in enumerate(servers)}
        self.ret = {p: i for i, p in enumerate(returners)}
        ns, nr = len(servers), len(returners)
        self.blocks = {}
        o = 0
        for name, size in [("base", 48), ("su", ns), ("ss", ns * 2), ("sv", ns * 12), ("ru", nr), ("rs", nr * 2), ("rv", nr * 12)]:
            self.blocks[name] = (o, o + size)
            o += size
        self.p = o

    def matrix(self, a):
        n = len(a)
        cell = a["serve_i"].values * 6 + a["side_i"].values * 3 + a["aim_i"].values
        cols, rows = [], []
        base = cell * 4 + a["surf_i"].values
        rows.append(np.arange(n)); cols.append(self.blocks["base"][0] + base)
        for role, key in (("s", "server"), ("r", "returner")):
            idx = a[key].map(self.srv if role == "s" else self.ret)
            has = idx.notna().values
            i = idx[has].astype(int).values
            rr = np.arange(n)[has]
            for blk, c in ((role + "u", i), (role + "s", i * 2 + a["serve_i"].values[has]), (role + "v", i * 12 + cell[has])):
                rows.append(rr); cols.append(self.blocks[blk][0] + c)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, self.p))

    def penalty(self, lam_u, lam_s, lam_v):
        lam = np.empty(self.p)
        for blk, (a, b) in self.blocks.items():
            lam[a:b] = {"base": 1e-2, "su": lam_u, "ru": lam_u, "ss": lam_s, "rs": lam_s, "sv": lam_v, "rv": lam_v}[blk]
        return lam


def aggregate(s):
    keys = ["server", "returner", "serve_i", "side_i", "aim_i", "surf_i"]
    return s.groupby(keys, observed=True)["won"].agg(n="size", k="sum").reset_index()


def fit(agg, design, lam):
    X = design.matrix(agg)
    n, k = agg["n"].values.astype(float), agg["k"].values.astype(float)
    theta0 = np.zeros(design.p)
    a, b = design.blocks["base"]
    # Warm start the base cells at their empirical log-odds.
    cell = X[:, a:b]
    nb, kb = cell.T @ n, cell.T @ k
    theta0[a:b] = np.log((kb + 1) / (nb - kb + 1))

    def f(theta):
        eta = X @ theta
        loss = (n * np.logaddexp(0, eta) - k * eta).sum() + 0.5 * (lam * theta ** 2).sum()
        grad = X.T @ (n * expit(eta) - k) + lam * theta
        return loss, grad

    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-6})
    theta = res.x
    p = expit(X @ theta)
    hdiag = X.T @ (n * p * (1 - p)) + lam  # X is 0/1, so X**2 == X
    return theta, 1.0 / hdiag


def predict(design, theta, a):
    return expit(design.matrix(a) @ theta)


def logloss(p, a):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(a["k"] * np.log(p) + (a["n"] - a["k"]) * np.log(1 - p)).sum() / a["n"].sum())


def brier(p, a):
    return float(((a["k"] * (1 - p) ** 2 + (a["n"] - a["k"]) * p ** 2).sum()) / a["n"].sum())


def vocab(s):
    sv = s["server"].value_counts()
    rv = s["returner"].value_counts()
    return sorted(sv[sv >= MIN_FIT].index), sorted(rv[rv >= MIN_FIT].index)


GRID = [(3, 10, 30), (10, 30, 100), (30, 100, 300), (30, 100, 1000), (30, 100, 3000), (30, 100, 10000),
        (100, 300, 3000), (100, 1000, 3000), (300, 300, 3000), (300, 1000, 3000), (1000, 1000, 3000), (300, 3000, 3000)]


def tune(s):
    tr, va = s[s["year"] <= 2023], s[s["year"] == 2024]
    srv, ret = vocab(tr)
    d = Design(srv, ret)
    atr, ava = aggregate(tr), aggregate(va)
    scores = []
    for lam in GRID:
        theta, _ = fit(atr, d, d.penalty(*lam))
        scores.append((logloss(predict(d, theta, ava), ava), lam))
    scores.sort()
    return scores[0][1], [{"lam": list(l), "logloss": r(v, 5)} for v, l in scores]


def holdout(s, lam):
    tr, te = s[s["year"] <= 2024], s[s["year"] >= 2025]
    srv, ret = vocab(tr)
    atr, ate = aggregate(tr), aggregate(te)
    out = {"train_serves": int(len(tr)), "test_serves": int(len(te)), "test_matches": int(te["match_id"].nunique())}
    models = {}
    variants = {"base": ([], []), "server": (srv, []), "full": (srv, ret)}
    for name, (sv, rv) in variants.items():
        d = Design(sv, rv)
        theta, _ = fit(atr, d, d.penalty(*lam))
        p = predict(d, theta, ate)
        models[name] = (d, theta, p)
        out[name] = {"logloss": r(logloss(p, ate), 5), "brier": r(brier(p, ate), 5)}

    d, theta, p = models["full"]
    # Calibration in 10 bins of predicted probability, weighted by serves.
    ate = ate.assign(p=p)
    bins = pd.qcut(np.repeat(ate["p"].values, ate["n"].values), 10, retbins=True, duplicates="drop")[1]
    ate["bin"] = np.clip(np.searchsorted(bins, ate["p"], side="right") - 1, 0, len(bins) - 2)
    cal = ate.groupby("bin").apply(lambda g: pd.Series({"pred": np.average(g["p"], weights=g["n"]),
                                                       "obs": g["k"].sum() / g["n"].sum(), "n": g["n"].sum()}), include_groups=False)
    out["calibration"] = [{"pred": r(x.pred), "obs": r(x.obs), "n": int(x.n)} for x in cal.itertuples()]

    # Do the direction-specific edges persist? Compare predicted vs realized T-minus-wide gaps
    # for each server and side on 2025-26 first serves, against a no-shrinkage raw-rate forecast.
    te1 = te[te["serve"] == 1].copy()
    te1["p"] = predict(d, theta, te1.assign(n=1, k=0))
    raw = tr[tr["serve"] == 1].groupby(["server", "side", "aim"])["won"].agg(["sum", "size"])
    rows = []
    for (server, side), g in te1.groupby(["server", "side"]):
        a = g.groupby("aim").agg(n=("won", "size"), won=("won", "mean"))
        if not {"T", "wide"} <= set(a.index) or a.loc[["T", "wide"], "n"].min() < 30:
            continue
        # Predicted gap: what the model expects for these same returners and surfaces if aimed T vs wide.
        base = g.assign(n=1, k=0)
        pt = predict(d, theta, base.assign(aim_i=2)).mean()
        pw = predict(d, theta, base.assign(aim_i=0)).mean()
        try:
            rt = raw.loc[(server, side, "T")]
            rw = raw.loc[(server, side, "wide")]
            raw_gap = rt["sum"] / rt["size"] - rw["sum"] / rw["size"] if min(rt["size"], rw["size"]) >= 30 else np.nan
        except KeyError:
            raw_gap = np.nan
        obs_gap = a.loc["T", "won"] - a.loc["wide", "won"]
        var = sum(a.loc[x, "won"] * (1 - a.loc[x, "won"]) / a.loc[x, "n"] for x in ("T", "wide"))
        rows.append({"server": server, "side": side, "pred": pt - pw, "raw": raw_gap, "obs": obs_gap, "w": 1 / var})
    gaps = pd.DataFrame(rows).dropna()

    def slope(x, y, w):
        X = np.c_[np.ones(len(x)), x]
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        resid = y - X @ beta
        cov = np.linalg.inv(X.T @ W @ X) * (resid ** 2 * w).sum() / (len(x) - 2)
        return float(beta[1]), float(np.sqrt(cov[1, 1]))

    sm, sm_se = slope(gaps["pred"].values, gaps["obs"].values, gaps["w"].values)
    sr, sr_se = slope(gaps["raw"].values, gaps["obs"].values, gaps["w"].values)
    mse = lambda c: float(np.average((gaps[c] - gaps["obs"]) ** 2, weights=gaps["w"]))
    gaps["q"] = pd.qcut(gaps["pred"], 5, labels=False)
    binned = gaps.groupby("q").apply(lambda g: pd.Series({"pred": np.average(g["pred"], weights=g["w"]),
                                                         "obs": np.average(g["obs"], weights=g["w"]),
                                                         "se": 1 / np.sqrt(g["w"].sum()), "cells": len(g)}), include_groups=False)
    out["direction_check"] = {
        "cells": int(len(gaps)),
        "model_slope": r(sm, 3), "model_slope_se": r(sm_se, 3),
        "raw_slope": r(sr, 3), "raw_slope_se": r(sr_se, 3),
        "model_mse": r(mse("pred"), 6), "raw_mse": r(mse("raw"), 6), "zero_mse": r(float(np.average(gaps["obs"] ** 2, weights=gaps["w"])), 6),
        "bins": [{"pred": r(x.pred), "obs": r(x.obs), "se": r(x.se), "cells": int(x.cells)} for x in binned.itertuples()],
        "scatter": [{"player": x.server, "side": x.side, "raw": r(x.raw), "obs": r(x.obs), "w": r(x.w, 1)} for x in gaps.itertuples()],
        "raw_intercept": r(float(np.average(gaps["obs"] - sr * gaps["raw"], weights=gaps["w"]))),
    }
    return out


def export_model(s, tour, lam, extras):
    t = s[s["tour"] == tour]
    srv, ret = vocab(t)
    d = Design(srv, ret)
    theta, var = fit(aggregate(t), d, d.penalty(*lam))

    def blk(name):
        a, b = d.blocks[name]
        return theta[a:b], var[a:b]

    base, base_var = blk("base")
    players = {}
    hands = pd.concat([t[["server", "server_hand"]].set_axis(["p", "h"], axis=1),
                       t[["returner", "returner_hand"]].set_axis(["p", "h"], axis=1)]).groupby("p")["h"].agg(lambda x: x.mode().iat[0] if len(x.mode()) else "")
    n_srv = t["server"].value_counts()
    n_ret = t["returner"].value_counts()
    matches = pd.concat([t[["match_id", "server"]].set_axis(["m", "p"], axis=1), t[["match_id", "returner"]].set_axis(["m", "p"], axis=1)]).drop_duplicates().groupby("p").size()
    last = pd.concat([t[["date", "server"]].set_axis(["d", "p"], axis=1), t[["date", "returner"]].set_axis(["d", "p"], axis=1)]).groupby("p")["d"].max()

    f = t[t["serve"] == 1]
    mix = t.groupby(["server", "serve_i", "side_i", "aim_i"]).size()
    wins = t.groupby(["server", "serve_i", "side_i", "aim_i"])["won"].sum()
    bp = f[f["break_point"] & ~f["tiebreak"]].groupby(["server", "side_i", "aim_i"]).size()
    tr = f.dropna(subset=["prev_aim_same_side"]).assign(prev_i=lambda x: x["prev_aim_same_side"].map({a: i for i, a in enumerate(AIMS)}))
    trans = tr.groupby(["server", "side_i", "prev_i", "aim_i"]).size()
    per_match = f.groupby(["server", "side_i"]).size().div(f.groupby("server")["match_id"].nunique(), level="server")

    def arr(series, p, shape):
        out = np.zeros(int(np.prod(shape)), int)
        if p in series.index.get_level_values(0):
            sub = series.loc[p]
            for idx, v in sub.items():
                idx = idx if isinstance(idx, tuple) else (idx,)
                out[np.ravel_multi_index(idx, shape)] = int(v)
        return out.tolist()

    for p in sorted(set(srv) | set(ret)):
        if n_srv.get(p, 0) < MIN_EXPORT or n_ret.get(p, 0) < MIN_EXPORT:
            continue
        si, ri = d.srv.get(p), d.ret.get(p)
        rec = {"hand": hands.get(p, ""), "matches": int(matches.get(p, 0)), "last": str(last.get(p).date()),
               "n_srv": int(n_srv.get(p, 0)), "n_ret": int(n_ret.get(p, 0))}
        for role, i, prefix in (("srv", si, "s"), ("ret", ri, "r")):
            if i is None:
                continue
            u, uv = blk(prefix + "u")
            ss, ssv = blk(prefix + "s")
            v, vv = blk(prefix + "v")
            rec[role] = {"u": r(u[i]), "s": [r(x) for x in ss[i * 2:i * 2 + 2]], "v": [r(x) for x in v[i * 12:i * 12 + 12]],
                         "uv": r(uv[i], 5), "sv": [r(x, 5) for x in ssv[i * 2:i * 2 + 2]], "vv": [r(x, 5) for x in vv[i * 12:i * 12 + 12]]}
        rec["mix"] = arr(mix, p, (2, 2, 3))
        rec["wins"] = arr(wins, p, (2, 2, 3))
        rec["bp"] = arr(bp, p, (2, 3))
        rec["trans"] = arr(trans, p, (2, 3, 3))
        rec["per_match"] = [r(per_match.get((p, k), 0), 1) for k in (0, 1)]
        rec.update(extras.get((tour, p), {}))
        players[p] = rec
    return {"tour": tour, "base": [r(x) for x in base], "base_var": [r(x, 6) for x in base_var],
            "lam": list(lam), "players": players}


def two_first_serves(s):
    """Would a player win more second-serve points by hitting another first serve? (p1_in * w1_in vs W2)"""
    rows = []
    for (tour, p), g in s.groupby(["tour", "server"]):
        f, sec = g[g["serve"] == 1], g[g["serve"] == 2]
        if len(sec) < 400:
            continue
        land = f["in"].mean()
        w_in = f.loc[f["in"] == 1, "won"].mean()
        risky = land * w_in
        safe = sec["won"].mean()
        se = np.sqrt(risky * (1 - risky) / len(f) + safe * (1 - safe) / len(sec))
        rows.append({"tour": tour, "player": p, "risky": risky, "safe": safe, "diff": risky - safe, "se": se})
    df = pd.DataFrame(rows)
    out = []
    for tour, d in df.groupby("tour"):
        better = d[d["diff"] > 1.96 * d["se"]].sort_values("diff", ascending=False)
        out.append({"tour": tour, "players": int(len(d)), "median_diff": r(d["diff"].median()),
                    "share_gain": r((d["diff"] > 0).mean()), "sig_gain": int(len(better)),
                    "top": [{"player": x.player, "risky": r(x.risky), "safe": r(x.safe)} for x in better.head(8).itertuples()],
                    "points": [{"player": x.player, "risky": r(x.risky), "safe": r(x.safe)} for x in d.itertuples()]})
    return out, df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = load()
    meta = {"serves": int(len(s)), "points": int(s[s["serve"] == 1].shape[0]), "matches": int(s["match_id"].nunique()),
            "first": str(s["date"].min().date()), "last": str(s["date"].max().date()),
            "by_tour": {t: {"matches": int(g["match_id"].nunique()), "servers": int(g["server"].nunique())} for t, g in s.groupby("tour")}}
    print("baselines"); base = baselines(s)
    print("minimax"); mm = minimax(s)
    print("serial"); se = serial(s)
    print("pressure"); pr = pressure(s)
    print("two first serves"); tf, tf_players = two_first_serves(s)
    print("anticipation"); an = anticipation(s)
    extras = {}
    for x in se["players"]:
        extras.setdefault((x["tour"], x["player"]), {})["switch"] = {"obs": x["obs"], "exp": x["exp"], "z": x["z"], "pairs": x["pairs"]}
    for x in tf_players.itertuples():
        extras.setdefault((x.tour, x.player), {})["second"] = {"risky": r(x.risky), "safe": r(x.safe), "se": r(x.se)}
    validation = {}
    lams = {}
    for tour in ("ATP", "WTA"):
        t = s[s["tour"] == tour]
        print("tune", tour); lam, grid = tune(t)
        print("holdout", tour, lam); validation[tour] = holdout(t, lam) | {"grid": grid, "lam": list(lam)}
        lams[tour] = lam
    findings = {"meta": meta, "baselines": base, "minimax": mm, "serial": {"by_tour": se["by_tour"]}, "anticipation": an,
                "serial_players": se["players"], "pressure": pr, "two_first": tf, "validation": validation}
    (OUT / "findings.json").write_text(json.dumps(findings, separators=(",", ":")))
    for tour in ("ATP", "WTA"):
        print("export", tour)
        m = export_model(s, tour, lams[tour], extras)
        (OUT / f"model-{tour.lower()}.json").write_text(json.dumps(m, separators=(",", ":")))
        print(tour, len(m["players"]), "players exported")
    print(json.dumps(an, indent=1))
    for tour, v in validation.items():
        print(tour, v["lam"], {k: v[k] for k in ("base", "server", "full")}, {k: x for k, x in v["direction_check"].items() if k != "bins"})


if __name__ == "__main__":
    main()
