import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import json

from src.regime_switching.data_utils import (
    load_returns_from_prices,
    load_precomputed_returns,
)
from src.regime_switching.regime_switching_model import RegimeSwitchingModel
from src.regime_switching.regime_inference import infer_regimes
from src.regime_switching.options_pricing import regime_switching_call_price
from src.regime_switching.plotting import plot_crisis_probability


def main(config_path: str = "config.yaml"):
    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimization"]
    optn_cfg = cfg["options_pricing"]
    out_cfg = cfg["output"]

    results_dir = Path(out_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    price_file = data_cfg["price_file"]
    use_log = data_cfg.get("use_log_returns", True)

    if use_log:
        returns = load_returns_from_prices(
            price_file,
            date_col=data_cfg["date_column"],
            price_col=data_cfg["price_column"],
            use_log_returns=True,
        )
    else:
        returns = load_precomputed_returns(
            price_file,
            date_col=data_cfg["date_column"],
            returns_col=data_cfg["returns_column"],
        )

    ret_values = returns.values.astype(float)
    dates = returns.index

    # ------------------------------------------------------------------
    # Fit regime-switching model
    # ------------------------------------------------------------------
    model = RegimeSwitchingModel(n_regimes=model_cfg["n_regimes"])

    bounds = {
        "mu_low": model_cfg["bounds"]["mu_low"],
        "mu_high": model_cfg["bounds"]["mu_high"],
        "log10_sigma_low": model_cfg["bounds"]["log10_sigma_low"],
        "log10_sigma_high": model_cfg["bounds"]["log10_sigma_high"],
        "logit_low": model_cfg["bounds"]["logit_low"],
        "logit_high": model_cfg["bounds"]["logit_high"],
    }

    model.fit(
        ret_values,
        bounds=bounds,
        method=opt_cfg.get("method", "L-BFGS-B"),
        max_iter=opt_cfg.get("max_iter", 500),
        tol=opt_cfg.get("tol", 1e-6),
    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    inf = infer_regimes(model, ret_values, smooth=True)
    smoothed_probs = inf["smoothed_probs"]
    crisis_prob = smoothed_probs[:, 1]  # regime 1 as "crisis"

    # ------------------------------------------------------------------
    # Option pricing example
    # ------------------------------------------------------------------
    S0 = float(returns.index.size)  # totally arbitrary; replace with real spot
    S0 = 100.0
    T = optn_cfg["maturity_years"]
    r = optn_cfg["risk_free_rate"]
    q = optn_cfg["dividend_yield"]

    call_price = regime_switching_call_price(
        model=model,
        spot=S0,
        maturity=T,
        strike=S0 * optn_cfg["strike_moneyness"],
        risk_free_rate=r,
        dividend_yield=q,
    )

    # ------------------------------------------------------------------
    # Output: plot & summary
    # ------------------------------------------------------------------
    if out_cfg.get("save_plots", True):
        plot_path = results_dir / "regime_probabilities.png"
        plot_crisis_probability(
            dates=dates,
            returns=ret_values,
            probs_crisis=crisis_prob,
            output_path=plot_path,
        )

    if out_cfg.get("save_model_summary", True):
        summary = {
            "params": model.to_dict(),
            "call_price_example": call_price,
        }
        with open(results_dir / "model_fit_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("=== Regime-switching model summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
