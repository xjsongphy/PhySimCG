from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

try:
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover - fallback for environments without rich
    RichHandler = None


class FSyncFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())


def create_lab3_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("lab3")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    log_dir = Path("lab3/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"runtime_{timestamp}.log"
    latest_log_path = log_dir / "runtime.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = FSyncFileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    latest_fh = FSyncFileHandler(latest_log_path, mode="w", encoding="utf-8")
    latest_fh.setLevel(logging.DEBUG)
    latest_fh.setFormatter(fmt)
    logger.addHandler(latest_fh)

    if RichHandler is not None:
        rh = RichHandler(
            show_time=True,
            show_level=True,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
        )
        rh.setLevel(logging.DEBUG if debug else logging.INFO)
        rh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rh)
    else:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG if debug else logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    logger.info("Logging to %s", log_path.resolve())
    logger.info("Latest log mirror: %s", latest_log_path.resolve())
    return logger
