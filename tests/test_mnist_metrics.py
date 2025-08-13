from mnist_metrics import run_mnist_metrics


def test_mnist_metrics_targets():
    results = run_mnist_metrics(epochs=5)
    proof = results["proof_coherence"]
    omega_mean = results["omega_mean"]
    assert sum(proof) / len(proof) >= 0.8
    assert omega_mean > 0.99
