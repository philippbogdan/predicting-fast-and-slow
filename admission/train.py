"""Train Heimdall's network (nnK.py recipe) and Bayesian/ensemble variants.

nnK.py (Keras): Dense(128, relu) -> Dense(16, relu) -> Dense(1, sigmoid),
Glorot-uniform weights, zero biases, Adam(lr=1e-3), binary cross-entropy,
batch 32, 20 epochs, validation_split=0.2 (last 20% of the training rows),
EarlyStopping(monitor=val_loss, patience=3, min_delta=0.01), MinMaxScaler fit
on the training half of a 50/50 train_test_split(random_state=42).

``train_mlps`` trains K such networks at once (one per seed) with batched
matrix products so the GPU is used efficiently; each member keeps its own
initialisation, shuffling, Adam state and early-stopping state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYERS = (12, 128, 16, 1)


@dataclass
class MLP:
    """One trained network in ArtefactModel's layout (float64 numpy)."""
    x_min: np.ndarray
    x_range: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    epochs: int = 0

    def hidden(self, X):
        Z = (X - self.x_min) / self.x_range
        H1 = np.maximum(Z @ self.W1 + self.b1, 0)
        H2 = np.maximum(H1 @ self.W2 + self.b2, 0)
        return H1, H2

    def logit(self, X):
        return (self.hidden(X)[1] @ self.W3 + self.b3)[:, 0]


def heimdall_split(X: np.ndarray, y: np.ndarray):
    """nnK.py's train/eval split and scaler."""
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)
    x_min = x_tr.min(0)
    x_range = x_tr.max(0) - x_min
    x_range[x_range == 0] = 1.0  # sklearn MinMaxScaler convention
    return x_tr, x_te, y_tr, y_te, x_min, x_range


def _glorot(gen, k, fan_in, fan_out):
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return (torch.rand((k, fan_in, fan_out), generator=gen, dtype=torch.float32) * 2 - 1) * lim


def train_mlps(X: np.ndarray, y: np.ndarray, seeds, max_epochs=20, batch=32,
               patience=3, min_delta=0.01, lr=1e-3) -> list[MLP]:
    """Train len(seeds) Heimdall networks on the same data (nnK.py recipe)."""
    x_tr, _, y_tr, _, x_min, x_range = heimdall_split(X, y)
    Z = (x_tr - x_min) / x_range
    n_val = int(len(Z) * 0.2)          # Keras takes the *last* 20% before shuffling
    Zt, yt = Z[: len(Z) - n_val], y_tr[: len(Z) - n_val]
    Zv, yv = Z[len(Z) - n_val:], y_tr[len(Z) - n_val:]
    K = len(seeds)
    Zt = torch.tensor(Zt, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(yt, dtype=torch.float32, device=DEVICE)
    Zv = torch.tensor(Zv, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yv, dtype=torch.float32, device=DEVICE)
    params = []
    gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
    layers = (X.shape[1],) + LAYERS[1:]
    for fi, fo in zip(layers[:-1], layers[1:]):
        W = torch.cat([_glorot(g, 1, fi, fo) for g in gens]).to(DEVICE)
        b = torch.zeros((K, 1, fo), device=DEVICE)
        params += [W, b]
    params = [p.requires_grad_(True) for p in params]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-7
    step = 0

    def forward(Zb):  # Zb: [K, B, 12]
        h = Zb
        for li in range(3):
            h = torch.baddbmm(params[2 * li + 1], h, params[2 * li])
            if li < 2:
                h = torch.relu(h)
        return h[..., 0]

    def val_loss():
        with torch.no_grad():
            out = []
            for s in range(0, len(Zv), 65536):
                zb = Zv[s:s + 65536].unsqueeze(0).expand(K, -1, -1)
                lo = forward(zb)
                out.append(torch.nn.functional.binary_cross_entropy_with_logits(
                    lo, yv[s:s + 65536].unsqueeze(0).expand(K, -1), reduction="none").sum(1))
            return (torch.stack(out).sum(0) / len(Zv)).cpu().numpy()

    best = np.full(K, np.inf)
    wait = np.zeros(K, int)
    active = np.ones(K, bool)
    epochs = np.zeros(K, int)
    perm_gens = [torch.Generator(device=DEVICE).manual_seed(int(s) + 7919) for s in seeds]
    nt = len(Zt)
    for ep in range(max_epochs):
        if not active.any():
            break
        perms = torch.stack([torch.randperm(nt, generator=g, device=DEVICE) for g in perm_gens])
        act = torch.tensor(active, device=DEVICE, dtype=torch.float32).view(K, 1, 1)
        for s in range(0, nt, batch):
            idx = perms[:, s:s + batch]
            zb = Zt[idx]            # [K, B, 12]
            yb = yt[idx]            # [K, B]
            lo = forward(zb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, yb, reduction="none").mean(1).sum()
            grads = torch.autograd.grad(loss, params)
            step += 1
            with torch.no_grad():
                bc1 = 1 - b1 ** step
                bc2 = 1 - b2 ** step
                for p, g, mm, vv in zip(params, grads, m, v):
                    g = g * act if g.dim() == 3 else g
                    mm.mul_(b1).add_(g, alpha=1 - b1)
                    vv.mul_(b2).addcmul_(g, g, value=1 - b2)
                    upd = lr * (mm / bc1) / ((vv / bc2).sqrt() + eps)
                    p.sub_(upd * act)
        vl = val_loss()
        for k in range(K):
            if not active[k]:
                continue
            epochs[k] = ep + 1
            if vl[k] < best[k] - min_delta:
                best[k] = vl[k]
                wait[k] = 0
            else:
                wait[k] += 1
                if wait[k] >= patience:
                    active[k] = False
    P = [p.detach().cpu().numpy().astype(np.float64) for p in params]
    return [MLP(x_min, x_range, P[0][k], P[1][k][0], P[2][k], P[3][k][0], P[4][k], P[5][k][0],
                int(epochs[k])) for k in range(K)]


@dataclass
class BNN:
    """Mean-field Gaussian Bayesian version of Heimdall's network (Bayes by Backprop)."""
    x_min: np.ndarray
    x_range: np.ndarray
    mu: list
    sigma: list

    def sample_logits(self, X, n_samples=32, seed=0, chunk=1 << 13):
        """Monte Carlo predictive logits, shape [n_samples, n]."""
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        mu = [torch.tensor(a, dtype=torch.float32, device=DEVICE) for a in self.mu]
        sg = [torch.tensor(a, dtype=torch.float32, device=DEVICE) for a in self.sigma]
        Ws = []
        for li in range(3):
            eW = torch.randn((n_samples,) + mu[2 * li].shape, generator=g, device=DEVICE)
            eb = torch.randn((n_samples,) + mu[2 * li + 1].shape, generator=g, device=DEVICE)
            Ws.append((mu[2 * li] + sg[2 * li] * eW, mu[2 * li + 1] + sg[2 * li + 1] * eb))
        out = []
        Zall = (X - self.x_min) / self.x_range
        with torch.no_grad():
            for s in range(0, len(Zall), chunk):
                h = torch.tensor(Zall[s:s + chunk], dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(n_samples, -1, -1)
                for li, (W, b) in enumerate(Ws):
                    h = torch.baddbmm(b.unsqueeze(1), h, W)
                    if li < 2:
                        h = torch.relu(h)
                out.append(h[..., 0].cpu().numpy())
        return np.concatenate(out, axis=1)


def train_bnn(X: np.ndarray, y: np.ndarray, seed=0, max_epochs=20, batch=256, lr=1e-3,
              prior_sigma=1.0, patience=3, min_delta=0.01) -> BNN:
    """Bayes by Backprop with local reparameterisation, same architecture."""
    x_tr, _, y_tr, _, x_min, x_range = heimdall_split(X, y)
    Z = (x_tr - x_min) / x_range
    n_val = int(len(Z) * 0.2)
    Zt = torch.tensor(Z[: len(Z) - n_val], dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr[: len(Z) - n_val], dtype=torch.float32, device=DEVICE)
    Zv = torch.tensor(Z[len(Z) - n_val:], dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_tr[len(Z) - n_val:], dtype=torch.float32, device=DEVICE)
    gen = torch.Generator().manual_seed(seed)
    mus, rhos = [], []
    layers = (X.shape[1],) + LAYERS[1:]
    for fi, fo in zip(layers[:-1], layers[1:]):
        mus += [_glorot(gen, 1, fi, fo)[0].to(DEVICE), torch.zeros(fo, device=DEVICE)]
        rhos += [torch.full((fi, fo), -5.0, device=DEVICE), torch.full((fo,), -5.0, device=DEVICE)]
    params = [p.requires_grad_(True) for p in mus + rhos]
    opt = torch.optim.Adam(params, lr=lr, eps=1e-7)
    n = len(Zt)
    pgen = torch.Generator(device=DEVICE).manual_seed(seed + 11)

    def forward(z, sample=True):
        h = z
        for li in range(3):
            W, b = mus[2 * li], mus[2 * li + 1]
            sW = torch.nn.functional.softplus(rhos[2 * li])
            sb = torch.nn.functional.softplus(rhos[2 * li + 1])
            mean = h @ W + b
            if sample:
                var = (h * h) @ (sW * sW) + sb * sb
                mean = mean + var.clamp_min(1e-12).sqrt() * torch.randn(mean.shape, device=DEVICE, generator=pgen)
            h = torch.relu(mean) if li < 2 else mean
        return h[:, 0]

    def kl():
        tot = 0.0
        for mu_, rho_ in zip(mus, rhos):
            s = torch.nn.functional.softplus(rho_)
            tot = tot + (torch.log(prior_sigma / s) + (s * s + mu_ * mu_) / (2 * prior_sigma ** 2) - 0.5).sum()
        return tot

    best, wait = np.inf, 0
    for ep in range(max_epochs):
        perm = torch.randperm(n, generator=pgen, device=DEVICE)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            lo = forward(Zt[idx])
            nll = torch.nn.functional.binary_cross_entropy_with_logits(lo, yt[idx])
            loss = nll + kl() / n
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(torch.nn.functional.binary_cross_entropy_with_logits(forward(Zv, sample=False), yv))
        if vl < best - min_delta:
            best, wait = vl, 0
        else:
            wait += 1
            if wait >= patience:
                break
    with torch.no_grad():
        mu_np = [p.detach().cpu().numpy() for p in mus]
        sg_np = [torch.nn.functional.softplus(r).detach().cpu().numpy() for r in rhos]
    return BNN(x_min, x_range, mu_np, sg_np)


def save_models(path, mlps: list, bnn: BNN | None):
    d = {}
    for k, m in enumerate(mlps):
        for a in ("x_min", "x_range", "W1", "b1", "W2", "b2", "W3", "b3"):
            d[f"m{k}_{a}"] = getattr(m, a)
        d[f"m{k}_epochs"] = np.array(m.epochs)
    if bnn is not None:
        d["bnn_x_min"], d["bnn_x_range"] = bnn.x_min, bnn.x_range
        for i, (mu, sg) in enumerate(zip(bnn.mu, bnn.sigma)):
            d[f"bnn_mu{i}"], d[f"bnn_sg{i}"] = mu, sg
    np.savez_compressed(path, n_mlp=len(mlps), **d)


def load_models(path):
    z = np.load(path)
    mlps = [MLP(*(z[f"m{k}_{a}"] for a in ("x_min", "x_range", "W1", "b1", "W2", "b2", "W3", "b3")),
                epochs=int(z[f"m{k}_epochs"])) for k in range(int(z["n_mlp"]))]
    bnn = None
    if "bnn_x_min" in z:
        bnn = BNN(z["bnn_x_min"], z["bnn_x_range"], [z[f"bnn_mu{i}"] for i in range(6)],
                  [z[f"bnn_sg{i}"] for i in range(6)])
    return mlps, bnn
