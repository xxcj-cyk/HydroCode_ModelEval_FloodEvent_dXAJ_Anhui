"""HydroModels_PBM — classical hydrological model calibration (import as hydromodels_pbm)."""

__version__ = '0.2.0'

__all__ = ['__version__', 'run_script_experiment']


def run_script_experiment(*args, **kwargs):
    from hydromodels_pbm.workflow.pipeline import run_script_experiment as run

    return run(*args, **kwargs)
