import numpy as np
from .regime_switching_model import RegimeSwitchingModel


def infer_regimes(
    model: RegimeSwitchingModel,
    returns: np.ndarray,
    smooth: bool = True,
) -> dict:
    """
    Run filtering and smoothing (if requested) and return regime information.

    Parameters
    ----------
    model : RegimeSwitchingModel
        Fitted model.
    returns : np.ndarray
        Returns used for inference.
    smooth : bool
        If True, compute smoothed probabilities.

    Returns
    -------
    dict
        {
          "filtered_probs": ...,
          "smoothed_probs": ... or None,
          "regime_sequence_filtered": ...,
          "regime_sequence_smoothed": ... or None
        }
    """
    if model.params is None:
        raise ValueError("Model must be fitted before inference.")

    filt, pred, logl = model.hamilton_filter(returns, model.params)
    model.filtered_probs = filt
    model.predicted_probs = pred
    model.log_likelihood = logl

    smoothed = None
    if smooth:
        smoothed = model.smooth_probabilities(returns)

    seq_filt = filt.argmax(axis=1)
    seq_smooth = smoothed.argmax(axis=1) if smoothed is not None else None

    return {
        "filtered_probs": filt,
        "smoothed_probs": smoothed,
        "regime_sequence_filtered": seq_filt,
        "regime_sequence_smoothed": seq_smooth,
    }
