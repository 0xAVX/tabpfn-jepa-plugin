"""Eval: JEPA plug ablations. Saves figs/jepa.csv (+ figs/arch.csv).
A. NATICUSdroid wide (86): TabPFN all vs JEPA-top24 vs random-24.
B. S6E9 narrow: (i) raw vs raw+JEPA latents; (ii) all13 vs JEPA-top6 vs random-6.
C. Label-free check: JEPA relevance rank vs supervised SHAP rank (Spearman).
D. Architecture ablation: encoder depth x mask ratio -> top-24 AUC (NATICUS).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

sys.path.insert(0, "/home/dead/tabpfn-jepa-plug/src")
sys.path.insert(0, "/home/dead/playground-series-s6e9")
from jepa import embed, prep, prep_apply, prep_fit, relevance, train_plug
from src.ev import load, stratified_subsample, tabpfn_predict_proba

SEED = 0
rng = np.random.RandomState(0)


def load_naticus():
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name="NATICUSdroid", as_frame=True, parser="auto")
    X, y = d.data.copy(), d.target
    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
    num = [c for c in X.columns if c not in cat]
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat] = enc.fit_transform(X[cat].astype(str))
    X[num] = X[num].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(-1)
    y = LabelEncoder().fit_transform(y.astype(str))
    if y.mean() > 0.5:
        y = 1 - y
    return X, y


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    rows, arch = [], []

    print("== A. NATICUSdroid: all86 vs JEPA-top24 vs random-24 ==", flush=True)
    Xn, yn = load_naticus()
    Xtr, Xte, ytr, yte = train_test_split(Xn, yn, test_size=0.25, stratify=yn,
                                           random_state=SEED)
    t = prep_fit(pd.DataFrame(Xtr, columns=Xn.columns))
    Xp_tr, cols = prep_apply(pd.DataFrame(Xtr, columns=Xn.columns), t)
    net, dev = train_plug(Xp_tr, epochs=15, verbose=True)
    rel = relevance(net, dev, Xp_tr)
    order = np.argsort(-rel)
    print("top8:", [cols[i] for i in order[:8]], flush=True)
    print("(inductive: JEPA fit + relevance on Xtr only; Xte unseen)", flush=True)
    k = 24
    rk = rng.choice(len(cols), k, replace=False)
    for name, idx in [("all86", None), ("jepa24", order[:k]), ("random24", rk)]:
        p = tabpfn_predict_proba(Xtr if idx is None else Xtr.iloc[:, idx], ytr,
                                 Xte if idx is None else Xte.iloc[:, idx], seed=SEED)
        a = roc_auc_score(yte, p)
        rows.append(("naticus-select", name, a))
        print(f"  {name}: {a:.4f}", flush=True)

    print("== D. arch ablation: depth x mask -> top-24 AUC ==", flush=True)
    for depth, mr in [(1, 0.4), (2, 0.4), (3, 0.4), (2, 0.6)]:
        n2, d2 = train_plug(Xp_tr, epochs=8, depth=depth, mask_ratio=mr)
        o2 = np.argsort(-relevance(n2, d2, Xp_tr, mask_ratio=mr))
        p = tabpfn_predict_proba(Xtr.iloc[:, o2[:k]], ytr, Xte.iloc[:, o2[:k]],
                                 seed=SEED)
        a = roc_auc_score(yte, p)
        arch.append((depth, mr, a))
        print(f"  depth={depth} mask={mr}: {a:.4f}", flush=True)
    pd.DataFrame(arch, columns=["depth", "mask", "auc"]).to_csv("figs/arch.csv",
                                                                index=False)

    print("== B+C. S6E9: augment parity, top-6 select, SHAP agreement ==", flush=True)
    X, y, _, _, feats = load("/home/dead/playground-series-s6e9/data")
    Xs, ys = stratified_subsample(X, y, 5_000, seed=2)
    rest, rest_y = X.drop(Xs.index), np.delete(y, Xs.index.values)
    _, Xv, _, yv = train_test_split(rest, rest_y, test_size=10_000,
                                    stratify=rest_y, random_state=1)
    t2 = prep_fit(Xs)
    Xp2tr, cols2 = prep_apply(Xs, t2)
    net2, dev2 = train_plug(Xp2tr, epochs=5)
    rel2 = relevance(net2, dev2, Xp2tr)
    order2 = np.argsort(-rel2)
    print("s6e9 top6:", [cols2[i] for i in order2[:6]], flush=True)
    print("(inductive: JEPA fit + relevance on train ctx only)", flush=True)
    Xp2, _ = prep_apply(X, t2)

    a_raw = roc_auc_score(yv, tabpfn_predict_proba(Xs, ys, Xv, seed=SEED))
    rows.append(("s6e9-augment", "raw", a_raw))
    E_tr = embed(net2, dev2, Xp2[Xs.index.values])
    E_va = embed(net2, dev2, Xp2[Xv.index.values])
    Xa, Xva = Xs.copy(), Xv.copy()
    for j in range(E_tr.shape[1]):
        Xa[f"jepa{j}"] = E_tr[:, j]
        Xva[f"jepa{j}"] = E_va[:, j]
    a_aug = roc_auc_score(yv, tabpfn_predict_proba(Xa, ys, Xva, seed=SEED))
    rows.append(("s6e9-augment", "raw+jepa", a_aug))
    print(f"  raw {a_raw:.4f} vs raw+jepa {a_aug:.4f}", flush=True)

    rk6 = rng.choice(len(feats), 6, replace=False)
    idx_map = [feats.index(c) for c in cols2]
    for name, sel in [("all13", None), ("jepa6", order2[:6]), ("random6", rk6)]:
        ii = None if sel is None else [idx_map[i] for i in sel]
        p = tabpfn_predict_proba(Xs if ii is None else Xs.iloc[:, ii], ys,
                                 Xv if ii is None else Xv.iloc[:, ii], seed=SEED)
        a = roc_auc_score(yv, p)
        rows.append(("s6e9-select", name, a))
        print(f"  {name}: {a:.4f}", flush=True)

    shap = pd.read_csv("/home/dead/playground-series-s6e9/figs/shap_importance.csv")
    shap_rank = {f: r for r, f in enumerate(shap["feature"])}
    jepa_rank = {c: r for r, c in enumerate(c for c in cols2)}
    common = [c for c in cols2 if c in shap_rank]
    rho, _ = spearmanr([jepa_rank[c] for c in common], [shap_rank[c] for c in common])
    rows.append(("s6e9-shap-agree", f"spearman_rho", float(rho)))
    print(f"  JEPA-relevance vs SHAP rank rho={rho:.3f} (label-free vs supervised)",
          flush=True)

    pd.DataFrame(rows, columns=["exp", "setup", "auc"]).to_csv("figs/jepa.csv",
                                                               index=False)
    print(f"saved figs/jepa.csv + figs/arch.csv ({(time.time()-t0)/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
