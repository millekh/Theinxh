import numpy as np

from mnist_curvature import load_mnist, mnist_multi_agent_lambda
from orc import compute_orc


def run_mnist_metrics(epochs: int = 5, reg_beta: float = 0.02) -> dict:
    """Run MNIST curvature experiment and return metrics.

    Parameters
    ----------
    epochs : int
        Number of epochs to sample agents from MNIST.
    reg_beta : float
        Regularisation factor for curvature-aware training (unused placeholder).
    """
    images, labels = load_mnist()
    rng = np.random.default_rng(0)
    proof = []
    omegas = []
    for _ in range(epochs):
        lam, etas, phi = mnist_multi_agent_lambda(
            images, labels, agents=3, samples=2, rng=rng, return_details=True
        )
        orc_val = compute_orc(phi)
        proof.append(1 if orc_val >= 0 else 0)
        omegas.extend(np.exp(-np.array(etas) / 1000.0))
    omega_mean = float(np.mean(omegas))
    return {"proof_coherence": proof, "omega_mean": omega_mean}


__all__ = ["run_mnist_metrics"]
