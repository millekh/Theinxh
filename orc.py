import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci


def compute_orc(phi: np.ndarray) -> float:
    """Compute mean Ollivier-Ricci curvature for trust matrix ``phi``.

    Parameters
    ----------
    phi : np.ndarray
        Symmetric trust matrix with zeros on the diagonal.
    """
    G = nx.from_numpy_array(phi)
    orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()
    values = [d.get("ricciCurvature", 0.0) for _, _, d in orc.G.edges(data=True)]
    if not values:
        return 0.0
    return float(np.mean(values))


__all__ = ["compute_orc"]
