import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_crisis_probability(
    dates: pd.Index,
    returns: np.ndarray,
    probs_crisis: np.ndarray,
    output_path: str | Path | None = None,
    crisis_regime_index: int = 1,
):
    """
    Plot returns and probability of crisis regime over time.

    Parameters
    ----------
    dates : pd.Index
        Date index corresponding to returns.
    returns : np.ndarray
        Return series.
    probs_crisis : np.ndarray
        Probability of crisis regime (T,).
    output_path : str or Path or None
        Path to save PNG; if None, just show the plot.
    crisis_regime_index : int
        Which regime is "crisis" (1 by default).
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(dates, returns, label="Returns")
    ax1.set_ylabel("Returns")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(dates, probs_crisis, label=f"P(Regime={crisis_regime_index})", linestyle="--")
    ax2.set_ylabel("Crisis Probability")

    fig.suptitle("Regime-Switching: Returns and Crisis Probability")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
