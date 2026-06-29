import numpy as np

from hydromodels_pbm.model.GR4J_CemaNeige import cemaneige_series, gthreshold_from_warmup
from hydromodels_pbm.model.XAJ_mz import simulate_xaj_mz
from hydromodels_pbm.utils.normalization import to_array_2d


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def xaj_mz_cemaneige(drivers, parameters, warmup_length=365, kernel_size=15):
    if drivers.shape[2] != 3:
        raise ValueError(
            'XAJ-mz-CemaNeige expects drivers [time, basin, 3]: P, PET, T '
            f'(got feature dim {drivers.shape[2]})'
        )
    params = to_array_2d(parameters)
    p = drivers[:, :, 0]
    pet = drivers[:, :, 1]
    temp = drivers[:, :, 2]
    gthreshold = gthreshold_from_warmup(p, temp, warmup_length)
    pliq = cemaneige_series(p, temp, params[:, 15], params[:, 16], gthreshold)
    p_and_e = np.empty((drivers.shape[0], drivers.shape[1], 2), dtype=np.float64)
    p_and_e[:, :, 0] = pliq
    p_and_e[:, :, 1] = pet
    qsim, ets = simulate_xaj_mz(
        p_and_e,
        params[:, :15],
        warmup_length=warmup_length,
        kernel_size=kernel_size,
    )
    w = max(0, int(warmup_length))
    return qsim[w:, :, np.newaxis], ets[w:, :]
