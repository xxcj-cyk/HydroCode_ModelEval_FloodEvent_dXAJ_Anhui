"""HydroModels_DLM — deep learning hydrological models (import as hydromodels_dlm)."""

import os
import sys

__version__ = '0.1.0'


def configure_runtime():
    if sys.platform == 'win32':
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
