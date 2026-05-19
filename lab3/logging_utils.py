from __future__ import annotations

import logging
import os
from pathlib import Path


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
    log_path = log_dir / "runtime.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = FSyncFileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Logging to %s", log_path.resolve())
    return logger
