"""Auxiliary metrics for the TICE curvature system."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def curve_index(trust: np.ndarray, weights: Sequence[float]) -> float:
    """Compute a simple Curve Index ``C`` from a trust graph.

    Parameters
    ----------
    trust : np.ndarray
        Symmetric trust adjacency matrix ``Phi`` with values in ``[0, 1]``.
    weights : Sequence[float]
        Importance weights ``w_i`` for each agent.

    Returns
    -------
    float
        Aggregate curve index.
    """
    trust = np.asarray(trust, dtype=float)
    weights = np.asarray(weights, dtype=float)
    degrees = trust.sum(axis=1)
    n = trust.shape[0]
    c_val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if trust[i, j] > 0:
                g = 4 - degrees[i] - degrees[j]
                w = 0.5 * (weights[i] + weights[j])
                c_val += w * g * trust[i, j]
    return float(c_val)


def compute_xi_chi(probabilities: Sequence[float]) -> float:
    """Compute ``XiChi`` negentropy metric for a probability distribution."""
    p = np.asarray(probabilities, dtype=float)
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log2(p + 1e-12)))


def forecast_scg(lambdas: Sequence[float], dt: float = 1.0) -> float:
    """Forecast Symbolic Curvature Gain (SCG) from a series of ``Lambda`` values."""
    lam = np.asarray(lambdas, dtype=float)
    if len(lam) < 2:
        raise ValueError("Need at least two lambda values")
    return float((lam[-1] - lam[0]) / (dt * (len(lam) - 1)))


def preference_condition_embeddings(embeddings: np.ndarray, preferences: Sequence[float]) -> np.ndarray:
    """Modulate agent embeddings by human or IRL preferences.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape ``[agents, dim]`` containing agent embeddings.
    preferences : Sequence[float]
        Preference weights for each embedding dimension.

    Returns
    -------
    np.ndarray
        Preference‑conditioned embeddings.
    """
    emb = np.asarray(embeddings, dtype=float)
    pref = np.asarray(preferences, dtype=float)
    if pref.shape[-1] != emb.shape[-1]:
        raise ValueError("Preference length must match embedding dimension")
    return emb * pref


def theta_human_score(ratings: Sequence[float], scale: float = 5.0) -> float:
    """Compute the Θ^human interpretability score.

    The score is the mean of human ratings normalised by the rating ``scale``
    (e.g., 5 for a 1–5 Likert scale).
    """
    r = np.asarray(ratings, dtype=float)
    if r.size == 0:
        raise ValueError("ratings must be non-empty")
    return float(np.clip(r.mean() / scale, 0.0, 1.0))


__all__ = [
    "curve_index",
    "compute_xi_chi",
    "forecast_scg",
    "preference_condition_embeddings",
    "theta_human_score",
]
