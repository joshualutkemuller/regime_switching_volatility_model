# Regime-Switching Volatility Model (Hamilton Filter)

This repository implements a full **two-state Markov Regime-Switching Volatility Model** using Hamilton’s filter, maximum-likelihood estimation, and optional regime-mixing option pricing. The model is designed for quantitative finance applications including volatility forecasting, stress-regime detection, and tactical derivatives analytics.

The project is modular, configuration-driven, and suitable for both academic research and production-grade quant workflows.

## Overview

Financial markets rarely behave as a single homogeneous process. Volatility clusters into **calm** and **crisis** regimes, and the transition between these states carries information valuable for forecasting, hedging, and risk management.

This model:

- Estimates regime-dependent means and volatilities  
- Learns transition probabilities via maximum likelihood  
- Produces filtered and smoothed regime probabilities  
- Generates most-likely regime classifications  
- Computes short- and long-horizon volatility forecasts  
- Optionally prices options using a regime-mixture of Black–Scholes models  

## Repository Structure

```
regime_switching_vol_model/
├─ config.yaml
├─ main.py
├─ src/
│  └─ regime_switching/
│     ├─ __init__.py
│     ├─ data_utils.py
│     ├─ regime_switching_model.py
│     ├─ regime_inference.py
│     ├─ options_pricing.py
│     └─ plotting.py
├─ data/
│  ├─ sp500_prices.csv
│  └─ vix_data.csv
├─ results/
│  ├─ model_fit_summary.json
│  └─ regime_probabilities.png
└─ notebooks/
   └─ regime_analysis.ipynb
```

## Installation

```bash
git clone https://github.com/<your_username>/regime_switching_vol_model.git
cd regime_switching_vol_model
pip install -r requirements.txt
```

## Running the Model

```bash
python main.py
```

Outputs are saved to `results/`, including:

- `regime_probabilities.png`  
- `model_fit_summary.json`  

## Example Output

```json
{
  "params": {
    "mu": [0.0004, -0.0012],
    "sigma": [0.007, 0.019],
    "P": [[0.985, 0.015], [0.035, 0.965]],
    "log_likelihood": -14230.1
  },
  "call_price_example": 2.66
}
```

## License

MIT License
