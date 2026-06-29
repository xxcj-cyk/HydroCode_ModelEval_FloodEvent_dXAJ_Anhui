import json
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Basin log buffer
# ---------------------------------------------------------------------------


class BasinRunLog(logging.Handler):
    active = None

    def __init__(self):
        super().__init__()
        self.lines = []
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):
        self.lines.append(self.format(record))

    def start(self):
        self.lines = []
        get_logger().addHandler(self)

    def stop(self):
        get_logger().removeHandler(self)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(self.lines) + '\n', encoding='utf-8')


def start_basin_log():
    BasinRunLog.active = BasinRunLog()
    BasinRunLog.active.start()


def append_basin_log(line):
    if BasinRunLog.active is not None:
        BasinRunLog.active.lines.append(line)


def save_basin_log(path):
    if BasinRunLog.active is not None:
        BasinRunLog.active.save(path)


def stop_basin_log():
    if BasinRunLog.active is not None:
        BasinRunLog.active.stop()
        BasinRunLog.active = None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_elapsed(seconds):
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f'{minutes}m{secs:.0f}s'
    hours = minutes // 60
    minutes = minutes % 60
    return f'{hours}h{minutes}m{secs:.0f}s'


def log_section(log, title):
    log.info('─' * 48)
    log.info(title)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def setup_logging():
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    logger = logging.getLogger('hydromodels_pbm')
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    return logger


def get_logger(name=None):
    if name:
        return logging.getLogger(f'hydromodels_pbm.{name}')
    return logging.getLogger('hydromodels_pbm')


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
