import pytest
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from psi_agi import compute_psi_agi


def test_compute_psi_agi_basic():
    price_df = pd.DataFrame({
        "Normalized": [1.0, 1.2, 1.1, 1.3, 1.4, 1.5, 1.3, 1.2, 1.1, 1.0],
        "Quantum_Output": [0.5] * 10,
    })
    prediction_df = pd.DataFrame({
        "Predicted_Normalized_Price": [1.1, 0.9, 1.0]
    })

    result = compute_psi_agi(price_df, prediction_df, window=3)

    assert "Ψ_AGI_Score" in result.columns
    expected_vol = price_df["Normalized"].rolling(window=3).std().fillna(method="bfill").iloc[-1]
    latest_quantum = price_df["Quantum_Output"].iloc[-1]
    expected_scores = [
        (latest_quantum * pred) / (1 + expected_vol)
        for pred in prediction_df["Predicted_Normalized_Price"].values
    ]
    assert result["Ψ_AGI_Score"].tolist() == pytest.approx(expected_scores)


def test_missing_columns():
    price_df = pd.DataFrame({"Normalized": [1], "Quantum_Output": [0.5]})
    pred_df = pd.DataFrame({"Other": [1]})
    with pytest.raises(KeyError):
        compute_psi_agi(price_df, pred_df)

