import numpy as np
import pytest
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics import preference_condition_embeddings, theta_human_score


def test_preference_condition_embeddings():
    emb = np.array([[1.0, 2.0], [3.0, 4.0]])
    prefs = [0.5, 2.0]
    cond = preference_condition_embeddings(emb, prefs)
    expected = emb * np.array(prefs)
    assert np.allclose(cond, expected)


def test_theta_human_score():
    ratings = [5, 4, 5]
    score = theta_human_score(ratings, scale=5.0)
    assert score == pytest.approx(np.mean(ratings) / 5.0)
