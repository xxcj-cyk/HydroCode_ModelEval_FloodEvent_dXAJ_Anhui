import numpy as np

from hydromodels_pbm.workflow.metrics import METRIC_REGISTRY


# ---------------------------------------------------------------------------
# Calibration loss
# ---------------------------------------------------------------------------


def rmse_loss(observed, simulated):
    value = METRIC_REGISTRY['RMSE'](simulated, observed)
    if np.isnan(value):
        raise ValueError('RMSE is nan; check input or simulation.')
    return value


def nse_loss(observed, simulated):
    value = METRIC_REGISTRY['NSE'](simulated, observed)
    if np.isnan(value):
        raise ValueError('NSE is nan; check input or simulation.')
    return 1.0 - value


LOSS_FUNCTIONS = {
    'RMSE': rmse_loss,
    'NSE': nse_loss,
}
