import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def get_logger(name=None):
    if name:
        return logging.getLogger(f'hydromodels_dlm.{name}')
    return logging.getLogger('hydromodels_dlm')


def ensure_log_level(level=logging.INFO):
    root = get_logger()
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)


def setup_logging(level=None, log_file=None):
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    logger = get_logger()
    if logger.handlers:
        return logger

    if level is None:
        env_level = os.environ.get('HYDRO_LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, env_level, logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if log_file is None:
        log_file = os.environ.get('HYDRO_LOG_FILE') or None

    ensure_log_level(level)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)

    if log_file:
        path = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        file_handler = logging.FileHandler(path, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        )
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Log layout
# ---------------------------------------------------------------------------


LOG_INDENT = '   '


def log_detail(log, message, *args, **kwargs):
    log.info(LOG_INDENT + message, *args, **kwargs)


def log_section(log, title):
    log.info('─' * 48)
    log.info(title)


def format_elapsed_mmss(elapsed_s):
    total = int(round(float(elapsed_s)))
    minutes, seconds = divmod(total, 60)
    return f'{minutes:02d}:{seconds:02d}'


def format_elapsed_hms(elapsed_s):
    total = int(round(float(elapsed_s)))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours}h{minutes}m{seconds}s'


def format_epoch_progress(epoch, epochs, phase, steps, elapsed_s, bar_width=40):
    rate = steps / elapsed_s if elapsed_s > 0 else 0.0
    bar = '#' * bar_width
    elapsed = format_elapsed_mmss(elapsed_s)
    phase_label = str(phase).capitalize()
    return (
        f'Epoch {epoch}/{epochs} {phase_label}: '
        f'100%|{bar}| {steps}/{steps} [{elapsed}<00:00, {rate:.2f}batch/s]'
    )


def log_epoch_progress(log, epoch, epochs, phase, steps, elapsed_s):
    log_detail(log, format_epoch_progress(epoch, epochs, phase, steps, elapsed_s))


# ---------------------------------------------------------------------------
# Basin log
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
    stop_basin_log()
    ensure_log_level()
    BasinRunLog.active = BasinRunLog()
    BasinRunLog.active.start()


def save_basin_log(path):
    if BasinRunLog.active is not None:
        BasinRunLog.active.save(path)


def stop_basin_log():
    active = BasinRunLog.active
    if active is None:
        return
    active.stop()
    if BasinRunLog.active is active:
        BasinRunLog.active = None


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


class LoggingTqdm(tqdm):
    def __init__(self, iterable, *, desc, unit, leave, total):
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        super().__init__(
            iterable,
            desc=desc,
            unit=unit,
            total=total,
            leave=leave,
            dynamic_ncols=True,
            ascii=True,
            file=sys.stderr,
        )


def progress_iter(
    iterable,
    *,
    desc,
    enabled=True,
    unit=None,
    leave=False,
    total=None,
):
    if not enabled:
        return iterable
    return LoggingTqdm(
        iterable,
        desc=desc,
        unit=unit or 'it',
        leave=leave,
        total=total,
    )


def manual_progress_bar(
    *,
    total,
    desc,
    enabled=True,
    unit=None,
    leave=False,
):
    if not enabled:
        return None
    return LoggingTqdm(
        range(0),
        desc=desc,
        unit=unit or 'it',
        leave=leave,
        total=total,
    )
