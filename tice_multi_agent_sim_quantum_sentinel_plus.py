import math, random, io, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn

# -------- optional deps (graceful) --------
try:
    import networkx as nx
except Exception:
    nx = None
try:
    import statsmodels.api as sm
except Exception:
    sm = None
try:
    from scipy import integrate
except Exception:
    integrate = None
try:
    import ot  # POT
except Exception:
    ot = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from fastapi import FastAPI
except Exception:
    FastAPI = None
try:
    from prometheus_client import Gauge
except Exception:
    Gauge = None

# ------------------------------ reproducibility ------------------------------
SEED = 42
def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------ synthetic task -------------------------------
def make_batch(n: int = 96, img_hw: int = 28, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    h = w = img_hw
    imgs, labels = [], []
    yy_full, xx_full = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    for i in range(n):
        img = torch.zeros((h, w))
        if i % 2 == 0:  # blobs
            centers = np.random.randint(4, 24, size=(3, 2))
            for (cy, cx) in centers:
                blob = torch.exp(-((yy_full - cy)**2 + (xx_full - cx)**2) / (2 * (np.random.uniform(2.5, 5.0)**2)))
                img += blob
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            label = np.random.randint(0, 5)
        else:  # stripes
            freq = np.random.choice([3,4,5,6]); phase = np.random.rand() * 2 * math.pi
            yy = yy_full.float()
            img = 0.5 * (1 + torch.sin(2 * math.pi * yy / freq + phase))
            label = 5 + np.random.randint(0, 5)
        imgs.append(img.unsqueeze(0)); labels.append(label)
    X = torch.stack(imgs, dim=0).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    X = torch.clamp(X * (0.9 + 0.2 * torch.rand_like(X)), 0.0, 1.0)
    return X, y

def flip_labels(y: torch.Tensor, rate: float, num_classes: int = 10) -> torch.Tensor:
    if rate <= 0: return y
    y = y.clone(); n = y.shape[0]; k = int(rate * n)
    if k == 0: return y
    idx = torch.randperm(n)[:k]
    rand = torch.randint(0, num_classes, (k,), device=y.device)
    rand = (rand + (rand == y[idx]).long()) % num_classes
    y[idx] = rand
    return y

# ------------------------------ model ----------------------------------------
class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.head = nn.Linear(16 * 14 * 14, 10)
        self._reset()
    def _reset(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.head(x)  # logits
    def embedding(self) -> torch.Tensor:
        return self.head.weight.mean(dim=0)

# ------------------------------ metrics --------------------------------------
@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()

def quantum_entropy_bits(p: np.ndarray, gamma: float = 0.25, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0); p = p / p.sum()
    H = float(-(p * np.log2(p)).sum())
    # interference-like term
    s = 0.0
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            s += (p[i]*p[j])**0.5
    Hq = H - 2.0 * gamma * s
    return max(0.0, Hq)

def omega_from_entropy(xi_bits: float, T: float = 2.0) -> float:
    return float(math.exp(-xi_bits / max(T, 1e-9)))

def build_phi_from_embeddings(embs: np.ndarray) -> np.ndarray:
    E = embs
    E_center = E - E.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(E_center, axis=1, keepdims=True) + 1e-8
    Phi = (E_center @ E_center.T) / (norms @ norms.T)
    return Phi

def multi_agent_lambda(embs: np.ndarray) -> float:
    eps = 1e-8
    E = embs
    E_center = E - E.mean(axis=0, keepdims=True)
    delta_sq = np.sum(E_center**2, axis=1)
    eta = np.var(E, axis=1) + eps
    norms = np.linalg.norm(E_center, axis=1, keepdims=True) + eps
    Phi = (E_center @ E_center.T) / (norms @ norms.T)
    A = Phi.shape[0]
    off = (Phi.sum() - np.trace(Phi)) / (A * (A - 1)) if A > 1 else 1.0
    return float(((delta_sq / eta).mean()) * off)

def forman_ricci_orc(Phi: np.ndarray, thr: float = 0.15) -> float:
    A = (np.abs(Phi) >= thr).astype(float) * np.sign(Phi)
    np.fill_diagonal(A, 0.0)
    deg = np.sum(np.abs(A), axis=1)
    curvs = []
    for u in range(A.shape[0]):
        for v in range(u + 1, A.shape[1]):
            if A[u, v] != 0.0:
                kappa = 0.5 * (4.0 - deg[u] - deg[v])
                curvs.append(float(np.sign(A[u, v]) * kappa))
    return float(np.mean(curvs)) if curvs else 0.0

def orc_ollivier_sinkhorn(Phi: np.ndarray, thr: float = 0.15, reg_eps: float = 0.01) -> float:
    # Full POT if available; else fallback to light Sinkhorn
    A = (np.abs(Phi) >= thr).astype(float)
    np.fill_diagonal(A, 0.0)
    n = Phi.shape[0]
    if n < 2 or A.sum() == 0: return 0.0
    P = (Phi - Phi.min()) / (Phi.max() - Phi.min() + 1e-12)
    C = 1.0 - P
    curvs = []
    for u in range(n):
        Nu = np.where(A[u] > 0)[0]
        if Nu.size == 0: continue
        mu = np.ones(Nu.size) / Nu.size
        for v in range(u+1, n):
            if A[u,v] == 0: continue
            Nv = np.where(A[v] > 0)[0]
            if Nv.size == 0: continue
            mv = np.ones(Nv.size) / Nv.size
            cost = C[np.ix_(Nu, Nv)]
            if ot is not None:
                T = ot.sinkhorn(mu, mv, cost, reg_eps)
            else:
                # light sinkhorn kernel
                K = np.exp(-cost / max(reg_eps, 1e-12))
                uvec = np.ones_like(mu); vvec = np.ones_like(mv)
                for _ in range(60):
                    uvec = mu / (K @ vvec + 1e-12)
                    vvec = mv / (K.T @ uvec + 1e-12)
                T = np.outer(uvec, vvec) * K
            W = float((T * cost).sum())
            base = C[u, v] + 1e-9
            kappa = 1.0 - (W / base)
            curvs.append(kappa)
    return float(np.mean(curvs)) if curvs else 0.0

def compute_curve_index(Phi: np.ndarray, threshold: float = 0.15, signed: bool = True) -> float:
    if nx is None: return 0.0
    n = Phi.shape[0]
    if n < 2: return 0.0
    A = (np.abs(Phi) >= threshold)
    np.fill_diagonal(A, False)
    G = nx.from_numpy_array(A)
    if G.number_of_edges() == 0: return 0.0
    deg = np.array([G.degree(i) for i in range(n)])
    edges = np.array(G.edges())
    u, v = edges[:, 0], edges[:, 1]
    ricci_e = 4 - deg[u] - deg[v]
    node_sum = np.zeros(n); node_cnt = np.zeros(n)
    np.add.at(node_sum, u, ricci_e); np.add.at(node_sum, v, ricci_e)
    np.add.at(node_cnt, u, 1);      np.add.at(node_cnt, v, 1)
    node_ricci = node_sum / (node_cnt + 1e-9)
    w = Phi.mean(axis=1) if signed else np.clip(Phi, 0, None).mean(axis=1)
    return float(np.dot(w, node_ricci))

def arima_or_linear_forecast(series: List[float], steps: int = 5) -> List[float]:
    x = np.asarray(series, dtype=float)
    if len(x) < 3:
        # mirror last value
        return [float(x[-1])] * steps
    if sm is not None:
        try:
            model = sm.tsa.ARIMA(x, order=(1,1,1))
            fit = model.fit()
            fc = fit.forecast(steps=steps)
            return [float(v) for v in fc]
        except Exception:
            pass
    # linear fallback
    t = np.arange(len(x))
    A = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    t_future = np.arange(len(x), len(x)+steps)
    return [float(m*tf + b) for tf in t_future]

def graph_laplacian_from_phi(Phi: np.ndarray, thr: float = 0.15) -> np.ndarray:
    A = (np.abs(Phi) >= thr).astype(float)
    np.fill_diagonal(A, 0.0)
    D = np.diag(A.sum(axis=1))
    L = D - A
    return L

def evolve_psi_field(Phi: np.ndarray, eta_mean: float, steps: int = 10, dt: float = 0.1,
                     gamma: float = 0.1, D: float = 0.05) -> np.ndarray:
    n = Phi.shape[0]
    L = graph_laplacian_from_phi(Phi)
    I_inj = 1.0 / max(eta_mean, 1e-9)
    psi0 = np.ones(n) * 0.5
    def rhs(_t, psi): return -gamma*psi + D * ( - (L @ psi) ) + I_inj
    if integrate is not None:
        t = np.linspace(0, steps*dt, steps+1)
        sol = integrate.odeint(lambda y, _t: rhs(_t, y), psi0, t)
        return sol[-1]
    # Euler fallback
    psi = psi0.copy()
    for _ in range(steps):
        psi = psi + dt * rhs(0.0, psi)
    return psi

# ------------------------------ run loop -------------------------------------
@dataclass
class RunResult:
    proof_coherence: List[float]
    omega_mean: float
    lambdas: List[float]
    scg_diffs: List[float]
    scg_slope: float
    orc_forman: float
    orc_ollivier: float
    curve_index_C: float
    xi_mean_bits: float
    phi_last: np.ndarray
    lambda_forecast: List[float]
    dao_ledger: Dict[str, float]
    psi_last: np.ndarray

def run_sim(epochs: int = 5, agents: int = 3, adversarial_flip_rate: float = 0.2,
            thr: float = 0.15) -> RunResult:
    set_seed(); device = "cpu"
    nets = [Agent().to(device) for _ in range(agents)]
    optimizers = [torch.optim.AdamW(n.parameters(), lr=1e-3, weight_decay=1e-4) for n in nets]
    ce = nn.CrossEntropyLoss()

    acc_hist, omega_hist, lambda_hist, xi_hist = [], [], [], []
    dao_xp = np.zeros(agents, dtype=float)  # DAO ledger

    for ep in range(epochs):
        X, y = make_batch(n=96, device=device)
        y_adv = flip_labels(y, adversarial_flip_rate, 10) if (ep % 2 == 0) else y

        embs = []; xi_bits_epoch = []

        for ai, (net, opt) in enumerate(zip(nets, optimizers)):
            net.train()
            labels = y_adv if (ai == 0 and ep % 2 == 0) else y
            logits = net(X)
            loss = ce(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()

            embs.append(net.embedding().detach().cpu().numpy())
            pmean = torch.softmax(logits, dim=-1).mean(dim=0).detach().cpu().numpy()
            xi_bits_epoch.append(quantum_entropy_bits(pmean))

        # epoch metrics (eval, consistent BN)
        with torch.no_grad():
            for n in nets: n.eval()
            logits_stack = torch.stack([n(X) for n in nets], dim=0)
            acc_epoch = float((logits_stack.argmax(-1) == y).float().mean().item())

        xi_mean = float(np.mean(xi_bits_epoch))
        omega = omega_from_entropy(xi_mean, T=2.0)
        E = np.stack(embs, axis=0)
        lam = multi_agent_lambda(E)

        # DAO XP update (reward low entropy & positive curvature increment)
        xp_gain = max(0.0, lam - (lambda_hist[-1] if lambda_hist else 0.0)) + max(0.0, 0.5 - xi_mean)
        dao_xp += xp_gain

        acc_hist.append(acc_epoch); omega_hist.append(omega)
        lambda_hist.append(lam);    xi_hist.append(xi_mean)

        # Trust-weighted FedAvg each epoch (stabilization)
        Phi_tmp = build_phi_from_embeddings(E)
        xiq_agents = np.array(xi_bits_epoch)
        w_trust = np.exp(-xiq_agents)
        orc_f = forman_ricci_orc(Phi_tmp, thr=thr)
        w_curv = max(0.0, 1.0 + orc_f)
        # DAO weight
        w_dao = (dao_xp + 1e-6) / (dao_xp.sum() + 1e-6)
        w = (w_trust / (w_trust.sum() + 1e-9)) * 0.6 + w_dao * 0.4
        w = w * w_curv; w = w / (w.sum() + 1e-9)
        with torch.no_grad():
            avg_state = {}
            for k in nets[0].state_dict():
                stacked = torch.stack([nets[i].state_dict()[k].float() * float(w[i]) for i in range(agents)], dim=0)
                avg_state[k] = stacked.sum(dim=0)
            for n in nets:
                n.load_state_dict(avg_state, strict=True)

    # SCG diffs & slope
    scg_diffs = [0.0] + [lambda_hist[i] - lambda_hist[i-1] for i in range(1, len(lambda_hist))]
    t = np.arange(len(lambda_hist)); A = np.vstack([t, np.ones_like(t)]).T
    scg_slope = float(np.linalg.lstsq(A, np.array(lambda_hist), rcond=None)[0][0])

    # Final Φ, curvature, C
    E_final = np.stack([n.embedding().detach().cpu().numpy() for n in nets], axis=0)
    Phi = build_phi_from_embeddings(E_final)
    orc_forman = forman_ricci_orc(Phi, thr=thr)
    orc_ollivier = orc_ollivier_sinkhorn(Phi, thr=thr)
    C_val = compute_curve_index(Phi, threshold=thr, signed=True)

    # Ψ field
    eta_mean = float(np.mean(np.var(E_final, axis=1) + 1e-8))
    psi_last = evolve_psi_field(Phi, eta_mean=eta_mean, steps=20, dt=0.1)
    # Modulate last lambda by Ψ resonance (for downstream use)
    lambda_hist[-1] = float(lambda_hist[-1] * (np.mean(psi_last)**2))

    # Forecast next Λ
    lambda_fc = arima_or_linear_forecast(lambda_hist, steps=5)

    return RunResult(
        proof_coherence=acc_hist,
        omega_mean=float(np.mean(omega_hist)),
        lambdas=[float(x) for x in lambda_hist],
        scg_diffs=[float(x) for x in scg_diffs],
        scg_slope=scg_slope,
        orc_forman=float(orc_forman),
        orc_ollivier=float(orc_ollivier),
        curve_index_C=float(C_val),
        xi_mean_bits=float(np.mean(xi_hist)),
        phi_last=Phi,
        lambda_forecast=lambda_fc,
        dao_ledger={f"agent_{i}": float(v) for i, v in enumerate(dao_xp)},
        psi_last=psi_last
    )

# --------------- optional glyph/dashboard (if matplotlib) --------------------
def save_dashboard_png(res: RunResult, path: str = "tice_dashboard.png"):
    if plt is None:
        return None
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(res.lambdas); ax1.set_title("Λ over epochs"); ax1.set_xlabel("epoch"); ax1.set_ylabel("Λ")
    ax2 = fig.add_subplot(2,2,2)
    im = ax2.imshow(res.phi_last, vmin=-1, vmax=1)
    ax2.set_title("Φ (final)"); fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax3 = fig.add_subplot(2,2,3)
    ax3.bar(["Ω̄","Ξχ̄","ORCᶠ","ORCᴼ","C"], [res.omega_mean, res.xi_mean_bits, res.orc_forman, res.orc_ollivier, res.curve_index_C])
    ax3.set_title("Glyphs"); ax3.set_ylabel("value")
    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(res.lambda_forecast); ax4.set_title("Λ forecast"); ax4.set_xlabel("step"); ax4.set_ylabel("Λ")
    fig.suptitle("TICE Quantum Sentinel Dashboard", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path

# ----------------------------- optional API ----------------------------------
def create_app() -> Optional["FastAPI"]:
    if FastAPI is None:
        return None
    app = FastAPI(title="TICE Quantum Sentinel API")
    gauge_lambda = Gauge("tice_lambda_last", "Last observed curvature Λ") if Gauge else None
    gauge_orc = Gauge("tice_orc_forman", "Forman ORC") if Gauge else None
    @app.post("/simulate")
    def simulate(epochs: int = 5, agents: int = 3, adversarial: float = 0.2):
        res = run_sim(epochs=epochs, agents=agents, adversarial_flip_rate=adversarial)
        if gauge_lambda: gauge_lambda.set(res.lambdas[-1])
        if gauge_orc:    gauge_orc.set(res.orc_forman)
        dash = save_dashboard_png(res) or ""
        return {
            "proof_coherence": res.proof_coherence,
            "omega_mean": res.omega_mean,
            "lambdas": res.lambdas,
            "scg_diffs": res.scg_diffs,
            "scg_slope": res.scg_slope,
            "orc_forman": res.orc_forman,
            "orc_ollivier": res.orc_ollivier,
            "curve_index_C": res.curve_index_C,
            "xi_mean_bits": res.xi_mean_bits,
            "lambda_forecast": res.lambda_forecast,
            "dao_ledger": res.dao_ledger,
            "dashboard_png": dash
        }
    return app

if __name__ == "__main__":
    res = run_sim()
    print("Proof coherence (acc):", res.proof_coherence)
    print("Ω (mean):", res.omega_mean)
    print("Λ values:", res.lambdas)
    print("SCG diffs:", res.scg_diffs)
    print("SCG slope:", res.scg_slope)
    print("ORC Forman:", res.orc_forman)
    print("ORC Ollivier:", res.orc_ollivier)
    print("Curve Index C:", res.curve_index_C)
    print("Ξχ (mean bits):", res.xi_mean_bits)
    print("Λ forecast:", res.lambda_forecast)
    print("DAO Ledger:", res.dao_ledger)
    print("Ψ last:", res.psi_last)
    print("Φ last:\n", res.phi_last)
    path = save_dashboard_png(res)
    if path: print("Dashboard saved to:", path)
