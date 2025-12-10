import pandas as pd
import numpy as np
from pathlib import Path


def load_returns_from_prices(
    path: str | Path,
    date_col: str = "date",
    price_col: str = "close",
    use_log_returns: bool = True,
) -> pd.Series:
    """
    Load price data from CSV and compute returns.

    Parameters
    ----------
    path : str or Path
        Path to CSV containing at least [date_col, price_col].
    date_col : str
        Name of the date column.
    price_col : str
        Name of the price column.
    use_log_returns : bool
        If True, compute log returns; otherwise simple returns.

    Returns
    -------
    pd.Series
        Returns indexed by datetime.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    prices = df[price_col].astype(float).values

    if use_log_returns:
        rets = np.log(prices[1:] / prices[:-1])
    else:
        rets = prices[1:] / prices[:-1] - 1.0

    idx = df[date_col].iloc[1:]
    return pd.Series(rets, index=idx, name="returns")


def load_precomputed_returns(
    path: str | Path,
    date_col: str = "date",
    returns_col: str = "log_return",
) -> pd.Series:
    """
    Load precomputed returns from CSV.

    Parameters
    ----------
    path : str or Path
        Path to CSV containing at least [date_col, returns_col].
    date_col : str
        Name of the date column.
    returns_col : str
        Name of the returns column.

    Returns
    -------
    pd.Series
        Returns indexed by datetime.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    return df.set_index(date_col)[returns_col].astype(float)


def simulate_two_regime_returns(
    n_obs: int = 5000,
    mu_calm: float = 0.05,
    sigma_calm: float = 0.15,
    mu_crisis: float = -0.10,
    sigma_crisis: float = 0.40,
    p: float = 0.98,
    q: float = 0.95,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate returns from a two-state Markov regime-switching model.

    Parameters
    ----------
    n_obs : int
        Number of observations to simulate.
    mu_calm, sigma_calm : float
        Mean and vol in calm regime (state 0).
    mu_crisis, sigma_crisis : float
        Mean and vol in crisis regime (state 1).
    p : float
        Persistence of calm regime: P(s_t=0 | s_{t-1}=0).
    q : float
        Persistence of crisis regime: P(s_t=1 | s_{t-1}=1).
    seed : int or None
        Random seed.

    Returns
    -------
    states : np.ndarray
        Simulated hidden states (0/1).
    returns : np.ndarray
        Simulated returns.
    """
    rng = np.random.default_rng(seed)
    states = np.zeros(n_obs, dtype=int)
    rets = np.zeros(n_obs)

    # Transition matrix
    P = np.array([[p, 1 - p],
                  [1 - q, q]])

    # Start in calm regime with prob 0.5
    states[0] = rng.choice([0, 1], p=[0.5, 0.5])
    for t in range(1, n_obs):
        states[t] = rng.choice([0, 1], p=P[states[t - 1]])

    mu = np.array([mu_calm, mu_crisis])
    sigma = np.array([sigma_calm, sigma_crisis])

    eps = rng.standard_normal(n_obs)
    rets = mu[states] + sigma[states] * eps
    return states, rets
