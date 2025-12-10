from __future__ import annotations

import numpy as np
from scipy.stats import norm
from .regime_switching_model import RegimeSwitchingModel


def black_scholes_call(
    S0: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    """
    Black–Scholes European call price.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    sigma : float
        Volatility (annualized).
    r : float
        Risk-free rate (continuously compounded).
    q : float
        Dividend yield (continuously compounded).

    Returns
    -------
    float
        Call option price.
    """
    if T <= 0 or sigma <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_S = np.exp(-q * T)
    disc_K = np.exp(-r * T)

    return S0 * disc_S * norm.cdf(d1) - K * disc_K * norm.cdf(d2)


def regime_switching_call_price(
    model: RegimeSwitchingModel,
    spot: float,
    maturity: float,
    strike: float | None = None,
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    terminal_probs: np.ndarray | None = None,
) -> float:
    """
    Price a call as a mixture of Black–Scholes prices across regimes.

    Parameters
    ----------
    model : RegimeSwitchingModel
        Fitted model with `params` set.
    spot : float
        Current spot price.
    maturity : float
        Time to maturity (years).
    strike : float or None
        Strike price. If None, use ATM: K = spot.
    risk_free_rate : float
        Risk-free rate (continuously compounded).
    dividend_yield : float
        Dividend yield.
    terminal_probs : np.ndarray or None
        2-element array of regime probabilities at expiry. If None,
        uses stationary distribution implied by P.

    Returns
    -------
    float
        Regime-switching call price.
    """
    if model.params is None:
        raise ValueError("Model must be fitted before option pricing.")

    P = model.params.P
    sigma = model.params.sigma

    if strike is None:
        strike = spot

    # If terminal_probs not provided, use stationary distribution
    if terminal_probs is None:
        # Solve pi P = pi, sum(pi) = 1
        A = np.vstack([P.T - np.eye(2), np.ones(2)])
        b = np.array([0.0, 0.0, 1.0])
        # Least squares solution
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        terminal_probs = pi

    prices = [
        black_scholes_call(
            S0=spot,
            K=strike,
            T=maturity,
            sigma=float(sigma[i]),
            r=risk_free_rate,
            q=dividend_yield,
        )
        for i in range(2)
    ]

    return float(np.dot(terminal_probs, np.array(prices)))
