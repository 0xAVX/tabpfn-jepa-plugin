"""TabPFN-JEPA plug: label-free dataset adapter for frozen TabPFN-3.5.

Idea (novel combo): T-JEPA-style self-supervised encoder (Thimonier et al. 2024,
mask-predict in latent space, no labels, no augmentations) trained per-dataset,
used two ways:
  select(): rank features by label-free relevance -> fit wide data into TabPFN.
  embed():  append JEPA latents as extra TabPFN features (parity/augment check).

Anti-collapse: VICReg variance + covariance terms (standard, cheap).
Inspired by Gen-Verse/JEPA-Anything's factorized predictive-core framing:
context/target pathways + predictor, domain adapter stays outside the FM.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def prep_fit(df: pd.DataFrame) -> dict:
    t = {"means": {}, "stds": {}, "maps": {}, "cols": df.columns.tolist()}
    for c in t["cols"]:
        v = df[c]
        if str(v.dtype) in ("category", "object"):
            t["maps"][c] = pd.Index(v.astype(str).unique())
        else:
            v = pd.to_numeric(v, errors="coerce")
            t["means"][c], t["stds"][c] = float(v.mean()), float(v.std()) + 1e-6
    return t


def prep_apply(df: pd.DataFrame, t: dict) -> tuple[np.ndarray, list[str]]:
    cols = t["cols"]
    out = np.zeros((len(df), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        v = df[c]
        if c in t["maps"]:
            out[:, j] = pd.Categorical(v.astype(str),
                                       categories=t["maps"][c]).codes.astype(np.float32)
        else:
            v = pd.to_numeric(v, errors="coerce").fillna(0).astype(np.float32)
            out[:, j] = (v - t["means"][c]) / t["stds"][c]
    return out, cols


def prep(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    return prep_apply(df, prep_fit(df))


class Plug(nn.Module):
    """depth/width-configurable context encoder + predictor (target = EMA copy)."""
    def __init__(self, d_in: int, d_lat: int = 128, depth: int = 2, width: int = 256):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, d_lat))
        self.ctx = nn.Sequential(*layers)
        self.pred = nn.Sequential(nn.Linear(d_lat, width), nn.GELU(),
                                  nn.Linear(width, d_lat))

    def forward(self, x_ctx, x_tgt, mask_ctx, mask_tgt):
        z_ctx = self.ctx(x_ctx * mask_ctx)
        with torch.no_grad():
            z_tgt = self.tgt(x_tgt * mask_tgt)
        return self.pred(z_ctx), z_tgt

    def attach_target(self):
        self.tgt = copy.deepcopy(self.ctx)
        for p in self.tgt.parameters():
            p.requires_grad = False


def vicreg(z: torch.Tensor, var_w=1.0, cov_w=0.04):
    std = z.std(dim=0) + 1e-4
    var_loss = torch.relu(1 - std).mean()
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_w * var_loss + cov_w * (off ** 2).mean()


def train_plug(X: np.ndarray, epochs=15, batch=2048, d_lat=128, depth=2, width=256,
               mask_ratio=0.4, seed=0, device="auto", verbose=False):
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else torch.device(device)
    net = Plug(X.shape[1], d_lat, depth, width).to(dev)
    net.attach_target()
    opt = torch.optim.AdamW(list(net.ctx.parameters()) + list(net.pred.parameters()),
                            lr=3e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            xb = Xt[perm[i:i + batch]].to(dev)
            m1 = (torch.rand(xb.shape, generator=g) > mask_ratio).float().to(dev)
            m2 = 1 - m1
            pred, tgt = net(xb, xb, m1, m2)
            loss = nn.functional.mse_loss(pred, tgt) + vicreg(pred) \
                + vicreg(net.ctx(xb * m1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pc, pt in zip(net.ctx.parameters(), net.tgt.parameters()):
                    pt.mul_(0.99).add_(pc, alpha=0.01)
            tot += float(loss.detach()); nb += 1
        if verbose:
            print(f"ep {ep}: {tot / nb:.4f}", flush=True)
    net.eval()
    return net, dev


@torch.no_grad()
def relevance(net: Plug, dev, X: np.ndarray, batch=4096, seed=0, mask_ratio=0.4,
              ) -> np.ndarray:
    """Label-free relevance: loss increase when feature j is masked. Higher = matters."""
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X).to(dev)
    d = X.shape[1]
    base, rel = [], np.zeros(d)
    allm = (torch.rand(min(len(X), batch * 4), d, generator=g) > mask_ratio).float()
    xb0 = Xt[:len(allm)]
    p0, t0 = net(xb0, xb0, allm.to(dev), (1 - allm).to(dev))
    base = float(nn.functional.mse_loss(p0, t0))
    for j in range(d):
        m = allm.clone()
        m[:, j] = 0
        p, t = net(xb0, xb0, m.to(dev), (1 - m).to(dev))
        rel[j] = float(nn.functional.mse_loss(p, t)) - base
    return np.maximum(rel, 0)


@torch.no_grad()
def embed(net: Plug, dev, X: np.ndarray, batch=8192) -> np.ndarray:
    Xt = torch.from_numpy(X).to(dev)
    ones = torch.ones(1, X.shape[1]).to(dev)
    return torch.cat([net.ctx(Xt[i:i + batch] * ones) for i in range(0, len(Xt), batch)]
                     ).cpu().numpy()
