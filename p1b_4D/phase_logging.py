"""Lightweight structured phase logging without scientific computation."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterator


def create_phase_logger(log_directory: Path, name: str = "stackelberg") -> logging.Logger:
    """Create an idempotent project logger with console and file handlers.

    Inputs
    ------
    log_directory:
        Existing directory for the UTF-8 phase log.
    name:
        Stable logger name.

    Outputs
    -------
    logging.Logger
        Configured non-propagating logger.

    Assumptions
    -----------
    The project path manager created log_directory.

    Notes
    -----
    Handler creation is idempotent for a logger name and log path.
    """
    if not isinstance(log_directory, Path):
        raise TypeError("log_directory must be a pathlib.Path")
    if not log_directory.is_dir():
        raise ValueError(f"log_directory does not exist: {log_directory}")
    if not name:
        raise ValueError("logger name must not be empty")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_path = (log_directory / "stackelberg.log").resolve()
    existing_paths = {
        Path(handler.baseFilename).resolve()
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    }
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    if log_path not in existing_paths:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if not any(type(handler) is logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def close_phase_logger(logger: logging.Logger) -> None:
    """Flush, close, and detach every handler owned by a phase logger.

    Inputs
    ------
    logger:
        Logger returned by create_phase_logger.

    Outputs
    -------
    None

    Assumptions
    -----------
    The caller no longer needs the attached handlers.

    Notes
    -----
    Closing logging resources is this function's primary side effect.
    """
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger")
    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


@contextmanager
def log_phase(logger: logging.Logger, phase_name: str) -> Iterator[dict[str, object]]:
    """Log phase name, elapsed time, success, warnings, and errors.

    Inputs
    ------
    logger:
        Logger returned by create_phase_logger.
    phase_name:
        Human-readable phase identifier.

    Outputs
    -------
    Iterator[dict[str, object]]
        Status record finalized when the context exits.

    Assumptions
    -----------
    Exceptions indicate failure and must remain visible to the caller.

    Notes
    -----
    Logging is the primary side effect. Exceptions are logged and re-raised.
    """
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger")
    if not phase_name:
        raise ValueError("phase_name must not be empty")

    record: dict[str, object] = {
        "phase_name": phase_name,
        "success": False,
        "warnings": [],
        "errors": [],
        "execution_time_seconds": None,
    }
    started = perf_counter()
    logger.info("phase=%s status=started", phase_name)
    try:
        yield record
    except Exception as exc:
        record["errors"] = [str(exc)]
        record["execution_time_seconds"] = perf_counter() - started
        logger.exception(
            "phase=%s status=failed elapsed_seconds=%.6f",
            phase_name,
            record["execution_time_seconds"],
        )
        raise
    else:
        record["success"] = True
        record["execution_time_seconds"] = perf_counter() - started
        for warning in record["warnings"]:
            logger.warning("phase=%s warning=%s", phase_name, warning)
        logger.info(
            "phase=%s status=success elapsed_seconds=%.6f",
            phase_name,
            record["execution_time_seconds"],
        )
