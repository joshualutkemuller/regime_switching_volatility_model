from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass, asdict


@dataclass
class RegimeSwitchingParams:
    """
    Container for two-regime Gaussian returns model parameters.

    Attributes
    ----------
    mu : np.ndarray, shape (2,)
        Regime-specific means.
    sigma : np.ndarray, shape (2,)
        Regime-specific standard deviations (>0).
    P : np.ndarray, shape (2, 2)
        Transition matrix, rows sum to 1.
    """
    mu: np.ndarray
    sigma: np.ndarray
    P: np.ndarray


class RegimeSwitchingModel:
    """
    Two-state Markov regime-switching model with Gaussian emissions.

    r_t = mu_{s_t} + sigma_{s_t} * epsilon_t,   epsilon_t ~ N(0,1)

    Estimation is via maximum likelihood using Hamilton's filter.
    """

    def __init__(self, n_regimes: int = 2):
        if n_regimes != 2:
            raise NotImplementedError("This implementation currently supports 2 regimes only.")
        self.n_regimes = n_regimes
        self.params: RegimeSwitchingParams | None = None
        self.filtered_probs: np.ndarray | None = None
        self.predicted_probs: np.ndarray | None = None
        self.smoothed_probs: np.ndarray | None = None
        self.log_likelihood: float = -np.inf

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize_parameters(self, returns: np.ndarray) -> RegimeSwitchingParams:
        """
        Initialize parameters using sample moments split at the median.

        Parameters
        ----------
        returns : np.ndarray
            1D array of returns.

        Returns
        -------
        RegimeSwitchingParams
        """
        sorted_ret = np.sort(returns)
        n = len(returns)
        low_regime = sorted_ret[: n // 2]
        high_regime = sorted_ret[n // 2 :]

        mu = np.array([low_regime.mean(), high_regime.mean()])
        sigma = np.array([low_regime.std(ddof=1), high_regime.std(ddof=1)])

        # Start with symmetric, persistent transition matrix
        P = np.array([[0.95, 0.05],
                      [0.05, 0.95]])

        self.params = RegimeSwitchingParams(mu=mu, sigma=sigma, P=P)
        return self.params

    # ------------------------------------------------------------------
    # Hamilton filter
    # ------------------------------------------------------------------
    def hamilton_filter(
        self,
        returns: np.ndarray,
        params: RegimeSwitchingParams | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Hamilton filter: compute filtered and predicted state probabilities.

        Parameters
        ----------
        returns : np.ndarray
            1D array of returns.
        params : RegimeSwitchingParams or None
            Model parameters. If None, uses self.params.

        Returns
        -------
        filtered_probs : np.ndarray, shape (T, 2)
        predicted_probs : np.ndarray, shape (T, 2)
        log_likelihood : float
        """
        if params is None:
            if self.params is None:
                raise ValueError("Parameters must be initialized before filtering.")
            params = self.params

        T = len(returns)
        n = self.n_regimes

        mu, sigma, P = params.mu, params.sigma, params.P

        xi_filt = np.zeros((T, n))
        xi_pred = np.zeros((T, n))
        log_likelihood = 0.0

        # Initialize with ergodic probabilities (simplified: uniform)
        xi_filt[0] = np.array([1.0 / n] * n)

        for t in range(T):
            # Prediction step
            if t == 0:
                xi_pred[t] = P.T @ xi_filt[0]
            else:
                xi_pred[t] = P.T @ xi_filt[t - 1]

            # Conditional densities
            eta = np.array(
                [norm.pdf(returns[t], loc=mu[i], scale=sigma[i]) for i in range(n)]
            )

            numerator = xi_pred[t] * eta
            denom = np.sum(numerator)

            # Numerical safety
            if denom <= 0 or not np.isfinite(denom):
                denom = 1e-300

            xi_filt[t] = numerator / denom
            log_likelihood += np.log(denom)

        return xi_filt, xi_pred, log_likelihood

    # ------------------------------------------------------------------
    # Negative log-likelihood and MLE
    # ------------------------------------------------------------------
    def _theta_to_params(self, theta: np.ndarray) -> RegimeSwitchingParams:
        """
        Map unconstrained parameter vector to model parameters.

        theta = [mu1, mu2, log10_sigma1, log10_sigma2, a, b]
        where a, b are logits for p11, p22.
        """
        mu1, mu2, log10_s1, log10_s2, a, b = theta
        sigma1 = 10.0 ** log10_s1
        sigma2 = 10.0 ** log10_s2

        def logistic(x: float) -> float:
            return 1.0 / (1.0 + np.exp(-x))

        p11 = logistic(a)
        p22 = logistic(b)

        P = np.array(
            [
                [p11, 1.0 - p11],
                [1.0 - p22, p22],
            ]
        )

        return RegimeSwitchingParams(
            mu=np.array([mu1, mu2]),
            sigma=np.array([sigma1, sigma2]),
            P=P,
        )

    def negative_log_likelihood(self, theta: np.ndarray, returns: np.ndarray) -> float:
        """
        Objective function for optimization: -log L(theta).

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector.
        returns : np.ndarray
            1D array of returns.

        Returns
        -------
        float
            Negative log-likelihood.
        """
        params = self._theta_to_params(theta)
        _, _, logl = self.hamilton_filter(returns, params)
        return -logl

    def fit(
        self,
        returns: np.ndarray,
        bounds: dict | None = None,
        method: str = "L-BFGS-B",
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> RegimeSwitchingParams:
        """
        Maximum likelihood estimation of model parameters.

        Parameters
        ----------
        returns : np.ndarray
            1D array of returns.
        bounds : dict or None
            Dict with keys {mu_low, mu_high, log10_sigma_low, log10_sigma_high,
            logit_low, logit_high}. If None, uses generic defaults.
        method : str
            Optimization method (passed to scipy.optimize.minimize).
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        RegimeSwitchingParams
            Estimated parameters.
        """
        if bounds is None:
            bounds = {
                "mu_low": -0.5,
                "mu_high": 0.5,
                "log10_sigma_low": -3.0,
                "log10_sigma_high": 0.0,
                "logit_low": -4.0,
                "logit_high": 4.0,
            }

        init = self.initialize_parameters(returns)
        theta0 = np.array(
            [
                init.mu[0],
                init.mu[1],
                np.log10(init.sigma[0]),
                np.log10(init.sigma[1]),
                2.0,
                2.0,
            ]
        )

        bnds = [
            (bounds["mu_low"], bounds["mu_high"]),           # mu1
            (bounds["mu_low"], bounds["mu_high"]),           # mu2
            (bounds["log10_sigma_low"], bounds["log10_sigma_high"]),  # log10 sigma1
            (bounds["log10_sigma_low"], bounds["log10_sigma_high"]),  # log10 sigma2
            (bounds["logit_low"], bounds["logit_high"]),     # a
            (bounds["logit_low"], bounds["logit_high"]),     # b
        ]

        result = minimize(
            self.negative_log_likelihood,
            theta0,
            args=(returns,),
            method=method,
            bounds=bnds,
            options={"maxiter": max_iter, "ftol": tol},
        )

        if not result.success:
            print(f"Warning: optimization did not converge: {result.message}")

        self.params = self._theta_to_params(result.x)
        self.filtered_probs, self.predicted_probs, self.log_likelihood = \
            self.hamilton_filter(returns, self.params)

        return self.params

    # ------------------------------------------------------------------
    # Smoothing (Kim smoother) and Viterbi
    # ------------------------------------------------------------------
    def smooth_probabilities(self, returns: np.ndarray) -> np.ndarray:
        """
        Kim smoother: compute xi_{t|T} from filtered and predicted probs.

        Parameters
        ----------
        returns : np.ndarray
            1D array of returns (only used to recompute filter if necessary).

        Returns
        -------
        smoothed_probs : np.ndarray, shape (T, 2)
        """
        if self.params is None:
            raise ValueError("Model must be fitted before smoothing.")

        if self.filtered_probs is None or self.predicted_probs is None:
            self.filtered_probs, self.predicted_probs, _ = self.hamilton_filter(
                returns, self.params
            )

        P = self.params.P
        T, n = self.filtered_probs.shape

        xi_smooth = np.zeros_like(self.filtered_probs)
        xi_smooth[-1] = self.filtered_probs[-1]

        # Backward recursion
        for t in range(T - 2, -1, -1):
            # Avoid division by zero
            denom = self.predicted_probs[t + 1]
            denom = np.where(denom <= 0, 1e-300, denom)

            # Element-wise ratio
            ratio = xi_smooth[t + 1] / denom
            # Backward smoothing step
            xi_smooth[t] = self.filtered_probs[t] * (P @ ratio)
            xi_smooth[t] /= xi_smooth[t].sum()

        self.smoothed_probs = xi_smooth
        return xi_smooth

    def most_likely_regime_sequence(self, use_smoothed: bool = True) -> np.ndarray:
        """
        Classify each time step to the most likely regime.

        Parameters
        ----------
        use_smoothed : bool
            If True, use smoothed probabilities; else filtered.

        Returns
        -------
        np.ndarray
            Regime indices (0 or 1).
        """
        if use_smoothed:
            if self.smoothed_probs is None:
                raise ValueError("Smoothed probabilities not available. Run smooth_probabilities.")
            probs = self.smoothed_probs
        else:
            if self.filtered_probs is None:
                raise ValueError("Filtered probabilities not available. Fit the model first.")
            probs = self.filtered_probs

        return probs.argmax(axis=1)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return parameters and log-likelihood as a serializable dict."""
        if self.params is None:
            raise ValueError("Model has not been fitted.")
        d = asdict(self.params)
        d["log_likelihood"] = float(self.log_likelihood)
        return d
