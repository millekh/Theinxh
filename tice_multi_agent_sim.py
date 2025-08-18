import math, random, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn

# Optional dependencies -------------------------------------------------------
try:  # networkx for graph operations
    import networkx as nx
    _HAS_NX = True
except Exception:  # pragma: no cover - optional
    nx = None
    _HAS_NX = False

try:  # POT for optimal transport
    import ot
    _HAS_POT = True
except Exception:  # pragma: no cover - optional
    ot = None
    _HAS_POT = False

try:  # matplotlib for dashboard image
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # pragma: no cover - optional
    plt = None
    _HAS_MPL = False

try:  # ARIMA forecasting
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_ARIMA = True
except Exception:  # pragma: no cover - optional
    ARIMA = None
    _HAS_ARIMA = False

# ------------------------------ reproducibility ------------------------------
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------ synthetic task -------------------------------
def make_batch(n: int = 96, img_hw: int = 28, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a toy 10-class vision dataset of blobs and stripes."""
    h = w = img_hw
    imgs, labels = [], []
    for i in range(n):
        img = torch.zeros((h, w))
        if i % 2 == 0:  # blobs
            centers = np.random.randint(4, 24, size=(3, 2))
            for (cy, cx) in centers:
                yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
                blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (np.random.uniform(2.5, 5.0) ** 2)))
                img += blob
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            label = np.random.randint(0, 5)
        else:  # stripes
            freq = np.random.choice([3, 4, 5, 6])
            phase = np.random.rand() * 2 * math.pi
            yy = torch.arange(h).float().unsqueeze(1).repeat(1, w)
            img = 0.5 * (1 + torch.sin(2 * math.pi * yy / freq + phase))
            label = 5 + np.random.randint(0, 5)
        imgs.append(img.unsqueeze(0))
        labels.append(label)
    X = torch.stack(imgs, dim=0).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    X = torch.clamp(X * (0.9 + 0.2 * torch.rand_like(X)), 0.0, 1.0)
    return X, y

def flip_labels(y: torch.Tensor, rate: float, num_classes: int = 10) -> torch.Tensor:
    if rate <= 0:
        return y
    y = y.clone()
    n = y.shape[0]
    k = int(rate * n)
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
        return self.head(x)

    def embedding(self) -> torch.Tensor:
        return self.head.weight.mean(dim=0)

# ------------------------------ metrics --------------------------------------
@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()

def quantum_entropy_bits(p: np.ndarray, gamma: float = 0.25, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    H = float(-(p * np.log2(p)).sum())
    s = np.sum(np.sqrt(p)[:, None] * np.sqrt(p)[None, :]) - np.sum(np.sqrt(p) ** 2)
    s *= 0.5
    Hq = H - 2.0 * gamma * s
    return max(0.0, Hq)

def omega_from_entropy(xi_bits: float, T: float = 2.0) -> float:
    return float(math.exp(-xi_bits / max(T, 1e-9)))

def _phi_from_embeddings(E: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    E_center = E - E.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(E_center, axis=1, keepdims=True) + eps
    return (E_center @ E_center.T) / (norms @ norms.T)

def multi_agent_lambda(embs: np.ndarray) -> float:
    eps = 1e-8
    E = embs
    E_center = E - E.mean(axis=0, keepdims=True)
    delta_sq = np.sum(E_center ** 2, axis=1)
    eta = np.var(E, axis=1) + eps
    Phi = _phi_from_embeddings(E, eps=eps)
    A = Phi.shape[0]
    off = (Phi.sum() - np.trace(Phi)) / (A * (A - 1)) if A > 1 else 1.0
    return float(((delta_sq / eta).mean()) * off)

def forman_ricci_orc(Phi: np.ndarray, thr: float = 0.15) -> float:
    A = (np.abs(Phi) >= thr).astype(float) * np.sign(Phi)
    np.fill_diagonal(A, 0.0)
    deg = np.sum(np.abs(A), axis=1)
    curvs = []
    n = A.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if A[u, v] != 0.0:
                kappa = 0.5 * (4.0 - deg[u] - deg[v])
                curvs.append(float(np.sign(A[u, v]) * kappa))
    return float(np.mean(curvs)) if curvs else 0.0

def orc_ollivier(Phi: np.ndarray, thr: float = 0.15, sink_eps: float = 0.01, iters: int = 60) -> float:
    n = Phi.shape[0]
    A = (np.abs(Phi) >= thr).astype(float)
    np.fill_diagonal(A, 0.0)
    if n < 2 or A.sum() == 0:
        return 0.0
    P = (Phi - Phi.min()) / (Phi.max() - Phi.min() + 1e-12)
    C = 1.0 - P

    def _sinkhorn(mu, mv, cost):
        if _HAS_POT:
            return ot.sinkhorn(mu, mv, cost, reg=sink_eps)
        K = np.exp(-cost / max(sink_eps, 1e-12))
        u = np.ones_like(mu)
        v = np.ones_like(mv)
        for _ in range(iters):
            u = mu / (K @ v + 1e-12)
            v = mv / (K.T @ u + 1e-12)
        return np.outer(u, v) * K

    curvs = []
    for u in range(n):
        Nu = np.where(A[u] > 0)[0]
        if Nu.size == 0:
            continue
        mu = np.ones(Nu.size) / Nu.size
        for v in range(u + 1, n):
            if A[u, v] == 0:
                continue
            Nv = np.where(A[v] > 0)[0]
            if Nv.size == 0:
                continue
            mv = np.ones(Nv.size) / Nv.size
            cost = C[np.ix_(Nu, Nv)]
            T = _sinkhorn(mu, mv, cost)
            W = float(np.sum(T * cost))
            base = C[u, v] + 1e-9
            kappa = 1.0 - (W / base)
            curvs.append(kappa)
    return float(np.mean(curvs)) if curvs else 0.0

def curve_index_C(Phi: np.ndarray, threshold: float = 0.15, signed: bool = True) -> float:
    if not _HAS_NX:
        A = (np.abs(Phi) >= threshold).astype(int)
        np.fill_diagonal(A, 0)
        deg = A.sum(axis=1)
        cur = 0.0
        cnt = 0
        n = A.shape[0]
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 1:
                    ricci_e = 4 - deg[u] - deg[v]
                    cur += ricci_e
                    cnt += 1
        if cnt == 0:
            return 0.0
        node_ricci = np.full(n, cur / cnt)
        w = Phi.mean(axis=1) if signed else np.clip(Phi, 0, None).mean(axis=1)
        return float(np.dot(w, node_ricci))

    A = (np.abs(Phi) >= threshold)
    np.fill_diagonal(A, False)
    G = nx.from_numpy_array(A)
    if G.number_of_edges() == 0:
        return 0.0
    n = Phi.shape[0]
    deg = np.array([G.degree(i) for i in range(n)])
    edges = np.array(G.edges())
    u, v = edges[:, 0], edges[:, 1]
    ricci_e = 4 - deg[u] - deg[v]
    node_sum = np.zeros(n)
    node_cnt = np.zeros(n)
    np.add.at(node_sum, u, ricci_e)
    np.add.at(node_sum, v, ricci_e)
    np.add.at(node_cnt, u, 1)
    np.add.at(node_cnt, v, 1)
    node_ricci = node_sum / (node_cnt + 1e-9)
    w = Phi.mean(axis=1) if signed else np.clip(Phi, 0, None).mean(axis=1)
    return float(np.dot(w, node_ricci))

# ------------------------------ Ψτ oscillator --------------------------------
def psi_tau_oscillator(Phi: np.ndarray, steps: int = 8, gamma: float = 0.1, D: float = 0.05) -> np.ndarray:
    thr = 0.15
    A = (np.abs(Phi) >= thr).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = np.sum(A, axis=1)
    L = np.diag(deg) - A
    Aagents = Phi.shape[0]
    psi = np.full(Aagents, 0.5, dtype=float)
    I = np.ones(Aagents, dtype=float)
    dt = 1.0
    for _ in range(steps):
        dpsi = -gamma * psi + D * (-(L @ psi)) + I
        psi = psi + dt * dpsi
    return psi

# ------------------------------ forecasting ----------------------------------
def forecast_lambda(lambdas: List[float], steps: int = 5) -> List[float]:
    y = np.asarray(lambdas, dtype=float)
    if len(y) < 3 or not _HAS_ARIMA:
        t = np.arange(len(y))
        a, b = np.polyfit(t, y, 1)
        t_future = np.arange(len(y), len(y) + steps)
        return (a * t_future + b).tolist()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y, order=(1, 1, 1))
            fit = model.fit()
            pred = fit.forecast(steps=steps)
        return pred.tolist()
    except Exception:
        t = np.arange(len(y))
        a, b = np.polyfit(t, y, 1)
        t_future = np.arange(len(y), len(y) + steps)
        return (a * t_future + b).tolist()

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
    lambda_forecast: List[float]
    dao_ledger: Dict[str, float]
    psi_last: np.ndarray
    phi_last: np.ndarray
    dashboard_path: Optional[str]

def run_sim(
    epochs: int = 5,
    agents: int = 3,
    adversarial_flip_rate: float = 0.2,
    quantum_gamma: float = 0.25,
    psi_steps: int = 6,
    psi_gamma: float = 0.1,
    psi_D: float = 0.05,
    mk_dashboard: bool = True,
) -> RunResult:
    set_seed()
    device = "cpu"
    nets = [Agent().to(device) for _ in range(agents)]
    optimizers = [torch.optim.AdamW(n.parameters(), lr=1e-3, weight_decay=1e-4) for n in nets]
    ce = nn.CrossEntropyLoss()

    acc_hist, omega_hist, lambda_hist, xi_hist = [], [], [], []
    dao_ledger = {f"agent_{i}": 0.0 for i in range(agents)}
    prev_lam = 0.0

    for ep in range(epochs):
        X, y = make_batch(n=96, device=device)
        y_adv = flip_labels(y, adversarial_flip_rate, 10) if (ep % 2 == 0) else y

        embs = []
        xi_bits_epoch = []
        for ai, (net, opt) in enumerate(zip(nets, optimizers)):
            net.train()
            labels = y_adv if (ai == 0 and ep % 2 == 0) else y
            logits = net(X)
            loss = ce(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            embs.append(net.embedding().detach().cpu().numpy())
            prob_mean = torch.softmax(logits, dim=-1).mean(dim=0).detach().cpu().numpy()
            xi_bits_epoch.append(quantum_entropy_bits(prob_mean, gamma=quantum_gamma))

        with torch.no_grad():
            for n in nets:
                n.eval()
            logits_stack = torch.stack([n(X) for n in nets], dim=0)
            acc_epoch = float((logits_stack.argmax(-1) == y).float().mean().item())

        xi_mean = float(np.mean(xi_bits_epoch))
        omega = omega_from_entropy(xi_mean, T=2.0)
        E = np.stack(embs, axis=0)
        lam = multi_agent_lambda(E)

        scg_gain = lam - prev_lam
        if scg_gain > 0:
            for k in dao_ledger.keys():
                dao_ledger[k] += scg_gain
        prev_lam = lam

        acc_hist.append(acc_epoch)
        omega_hist.append(omega)
        lambda_hist.append(lam)
        xi_hist.append(xi_mean)

    scg_diffs = [0.0] + [lambda_hist[i] - lambda_hist[i - 1] for i in range(1, len(lambda_hist))]
    t = np.arange(len(lambda_hist))
    A = np.vstack([t, np.ones_like(t)]).T
    scg_slope = float(np.linalg.lstsq(A, np.array(lambda_hist), rcond=None)[0][0])

    E_final = np.stack([n.embedding().detach().cpu().numpy() for n in nets], axis=0)
    Phi = _phi_from_embeddings(E_final)
    orc_f = forman_ricci_orc(Phi, thr=0.15)
    orc_o = orc_ollivier(Phi, thr=0.15)
    C = curve_index_C(Phi, threshold=0.15, signed=True)

    psi_last = psi_tau_oscillator(Phi, steps=psi_steps, gamma=psi_gamma, D=psi_D)
    mod_factor = float(np.clip(np.mean(psi_last), 0.5, 2.0))
    lambda_hist[-1] *= min(1.10, max(0.90, mod_factor))

    with torch.no_grad():
        probs = [torch.softmax(n(X), dim=-1).mean(dim=0).cpu().numpy() for n in nets]
        xiq = np.array([quantum_entropy_bits(p, gamma=quantum_gamma) for p in probs])
        w_trust = np.exp(-xiq)
        w_curv = max(0.0, 1.0 + orc_f)
        w = w_trust * w_curv
        w = w / (w.sum() + 1e-9)
        avg_state = {}
        for k in nets[0].state_dict().keys():
            stacked = torch.stack([nets[i].state_dict()[k].float() * float(w[i]) for i in range(agents)], dim=0)
            avg_state[k] = stacked.sum(dim=0)
        for n in nets:
            n.load_state_dict(avg_state, strict=True)

    lambda_fc = forecast_lambda(lambda_hist, steps=5)

    dashboard_path = None
    if mk_dashboard and _HAS_MPL:
        try:
            fig = plt.figure(figsize=(7, 6))
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(lambda_hist, label="Λ")
            ax1.plot(range(len(lambda_hist), len(lambda_hist) + len(lambda_fc)), lambda_fc, linestyle="--", label="Λ forecast")
            ax1.set_title("Λ & Forecast")
            ax1.legend()
            ax2 = fig.add_subplot(2, 1, 2)
            im = ax2.imshow(Phi, aspect="auto")
            ax2.set_title("Φ (cosine trust)")
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            fig.tight_layout()
            dashboard_path = "tice_dashboard.png"
            fig.savefig(dashboard_path, dpi=160)
            plt.close(fig)
        except Exception:
            dashboard_path = None

    return RunResult(
        proof_coherence=acc_hist,
        omega_mean=float(np.mean(omega_hist)),
        lambdas=list(map(float, lambda_hist)),
        scg_diffs=list(map(float, scg_diffs)),
        scg_slope=float(scg_slope),
        orc_forman=float(orc_f),
        orc_ollivier=float(orc_o),
        curve_index_C=float(C),
        xi_mean_bits=float(np.mean(xi_hist)),
        lambda_forecast=list(map(float, lambda_fc)),
        dao_ledger={k: float(v) for k, v in dao_ledger.items()},
        psi_last=psi_last,
        phi_last=Phi,
        dashboard_path=dashboard_path,
    )

if __name__ == "__main__":
    res_default = run_sim(epochs=5, agents=3, adversarial_flip_rate=0.2)
    print("Default (agents=3, epochs=5):")
    print("Proof coherence (acc):", res_default.proof_coherence)
    print("Ω (mean):", res_default.omega_mean)
    print("Λ values:", res_default.lambdas)
    print("SCG diffs:", res_default.scg_diffs)
    print("SCG slope:", res_default.scg_slope)
    print("ORC Forman:", res_default.orc_forman)
    print("ORC Ollivier:", res_default.orc_ollivier)
    print("Curve Index C:", res_default.curve_index_C)
    print("Ξχ (mean bits):", res_default.xi_mean_bits)
    print("Λ forecast:", res_default.lambda_forecast)
    print("DAO Ledger:", res_default.dao_ledger)
    print("Ψ last:", res_default.psi_last)
    print("Φ last:\n", res_default.phi_last)
    print("Dashboard:", res_default.dashboard_path)

    res_med = run_sim(epochs=7, agents=5, adversarial_flip_rate=0.3)
    print("\nMedium difficulty (agents=5, epochs=7, flip=0.3):")
    print("Proof coherence (acc):", res_med.proof_coherence)
    print("Ω (mean):", res_med.omega_mean)
    print("Λ values:", res_med.lambdas)
    print("SCG diffs:", res_med.scg_diffs)
    print("SCG slope:", res_med.scg_slope)
    print("ORC Forman:", res_med.orc_forman)
    print("ORC Ollivier:", res_med.orc_ollivier)
    print("Curve Index C:", res_med.curve_index_C)
    print("Ξχ (mean bits):", res_med.xi_mean_bits)
    print("Λ forecast:", res_med.lambda_forecast)
    print("DAO Ledger:", res_med.dao_ledger)
    print("Ψ last:", res_med.psi_last)
    print("Φ last:\n", res_med.phi_last)
    print("Dashboard:", res_med.dashboard_path)

    res_high = run_sim(epochs=10, agents=10, adversarial_flip_rate=0.4)
    print("\nHigh difficulty (agents=10, epochs=10, flip=0.4):")
    print("Proof coherence (acc):", res_high.proof_coherence)
    print("Ω (mean):", res_high.omega_mean)
    print("Λ values:", res_high.lambdas)
    print("SCG diffs:", res_high.scg_diffs)
    print("SCG slope:", res_high.scg_slope)
    print("ORC Forman:", res_high.orc_forman)
    print("ORC Ollivier:", res_high.orc_ollivier)
    print("Curve Index C:", res_high.curve_index_C)
    print("Ξχ (mean bits):", res_high.xi_mean_bits)
    print("Λ forecast:", res_high.lambda_forecast)
    print("DAO Ledger:", res_high.dao_ledger)
    print("Ψ last:", res_high.psi_last)
    print("Φ last:\n", res_high.phi_last)
    print("Dashboard:", res_high.dashboard_path)
