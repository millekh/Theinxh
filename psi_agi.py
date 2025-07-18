import pandas as pd


def compute_psi_agi(price_df: pd.DataFrame, prediction_df: pd.DataFrame, *, window: int = 10) -> pd.DataFrame:
    """Compute Ψ_AGI scores for predicted prices.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame containing at least ``Normalized`` and ``Quantum_Output`` columns.
    prediction_df : pd.DataFrame
        DataFrame containing ``Predicted_Normalized_Price`` column.
    window : int, optional
        Window size used for volatility estimation, by default 10.

    Returns
    -------
    pd.DataFrame
        ``prediction_df`` with an added ``Ψ_AGI_Score`` column.
    """

    if "Normalized" not in price_df or "Quantum_Output" not in price_df:
        raise KeyError("price_df must contain 'Normalized' and 'Quantum_Output' columns")
    if "Predicted_Normalized_Price" not in prediction_df:
        raise KeyError("prediction_df must contain 'Predicted_Normalized_Price' column")

    rolling_volatility = price_df["Normalized"].rolling(window=window).std().bfill()

    latest_quantum = price_df["Quantum_Output"].iloc[-1]
    latest_volatility = rolling_volatility.iloc[-1]

    prediction_df = prediction_df.copy()
    prediction_df["Ψ_AGI_Score"] = (
        latest_quantum
        * prediction_df["Predicted_Normalized_Price"]
        / (1 + latest_volatility)
    )
    return prediction_df


__all__ = ["compute_psi_agi"]

