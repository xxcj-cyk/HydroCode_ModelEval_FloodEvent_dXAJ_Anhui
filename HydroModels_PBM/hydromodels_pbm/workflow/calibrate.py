from pathlib import Path

from hydromodels_pbm.dataset.data_loader import FloodEventDataLoader, LongTermDataLoader
from hydromodels_pbm.workflow.evaluate import write_best_params
from hydromodels_pbm.workflow.optimizers import (
    ALGORITHM_RUNNERS,
    LongTermCalibrateSetup,
    FloodEventCalibrateSetup,
)
from hydromodels_pbm.workflow.simulator import denormalize_param_dict


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def create_calibrate_setup(
    loader,
    basin_id,
    model_name,
    objective_function,
    algorithm_name,
    output_dir,
):
    load_kw = {
        'basin_id': basin_id,
        'model': model_name,
        'algorithm': algorithm_name,
        'output_dir': output_dir,
    }
    if isinstance(loader, FloodEventDataLoader):
        events = loader.load_calibration(**load_kw)
        return FloodEventCalibrateSetup(
            events,
            loader.warmup_length,
            model_name,
            objective_function,
            loader.input_keys,
        )
    if isinstance(loader, LongTermDataLoader):
        p_and_e, qobs, calib_idx = loader.load_calibration(**load_kw)
        return LongTermCalibrateSetup(
            p_and_e,
            qobs,
            loader.warmup_length,
            calib_idx,
            model_name,
            objective_function,
            loader.input_keys,
        )
    raise TypeError(f'unsupported loader type: {type(loader)!r}')


# ---------------------------------------------------------------------------
# Basin calibration
# ---------------------------------------------------------------------------


def calibrate_basin(
    basin_id,
    out_dir,
    model_name,
    algorithm_name,
    objective_function,
    algo_params,
    loader,
    init_norm=None,
):
    basin_dir = Path(out_dir) / str(basin_id)
    basin_dir.mkdir(parents=True, exist_ok=True)

    setup = create_calibrate_setup(
        loader,
        basin_id,
        model_name,
        objective_function,
        algorithm_name,
        out_dir,
    )
    search = ALGORITHM_RUNNERS[algorithm_name](
        setup, str(basin_dir), algo_params, init_norm=init_norm
    )
    params = write_best_params(
        basin_dir,
        basin_id,
        model_name,
        denormalize_param_dict(search['norm'], model_name),
    )
    return {
        'basin_id': str(basin_id),
        'model_name': model_name,
        'objective_function': objective_function,
        'objective_value': round(search['objective'], 3),
        'params_physical': params,
    }
