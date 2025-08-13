import numpy as np
import pytest

from orc import compute_orc


def test_compute_orc_edge():
    phi = np.array([[0.0, 1.0], [1.0, 0.0]])
    val = compute_orc(phi)
    assert isinstance(val, float)
    assert val == pytest.approx(1.0)
