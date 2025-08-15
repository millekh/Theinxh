import numpy as np
import pytest

try:
    from tice_multi_agent_sim_quantum_sentinel_plus import run_sim
except Exception:  # pragma: no cover - missing heavy deps
    pytest.skip("torch or other deps not available", allow_module_level=True)


def test_run_sim_basic():
    res = run_sim(epochs=1, agents=2, adversarial_flip_rate=0.0)
    assert isinstance(res.omega_mean, float)
    assert len(res.lambdas) == 1
    assert np.isfinite(res.lambdas[0])
    assert len(res.lambda_forecast) == 5
