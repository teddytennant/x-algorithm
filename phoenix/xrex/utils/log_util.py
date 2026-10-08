# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import os
import socket
import sys
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

from xai_logging_utils.structured import format_extra_kv
from xrex.utils import cluster

if TYPE_CHECKING:
    from typing import TextIO


class ForceRotateOnStartHandler(RotatingFileHandler):
    def __init__(self, filename, maxBytes=0, backupCount=0, encoding=None, delay=False):
        super().__init__(
            filename, maxBytes=maxBytes, backupCount=backupCount, encoding=encoding, delay=delay
        )
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            self.doRollover()


_LOG_DATASET: ContextVar[str] = ContextVar("log_dataset", default="")
_LOG_ROW: ContextVar[str] = ContextVar("log_row", default="")


def set_log_dataset(name: str | None) -> None:
    _LOG_DATASET.set(name or "")


def set_log_row(record_id: int | None) -> None:
    if record_id is None or record_id < 0:
        _LOG_ROW.set("")
    else:
        _LOG_ROW.set(str(record_id))


def get_formatter(
    rank: int | None, prefix: str = "", worker_name: str | None = None
) -> logging.Formatter:
    cyan = "\033[2;36m"
    faint = "\033[2m"
    reset = "\033[0m"
    bold_yellow = "\033[1;33m"
    bold_red = "\033[1;31m"
    level_char = "%(levelname).1s"

    placement = (cluster.get_node_name(), cluster.get_pod_name(), cluster.get_cluster())
    if all(placement):
        details = ":".join(placement)
    else:
        details = socket.gethostname()

    worker_details = f"/{worker_name.replace('%', '%%')}" if worker_name else ""
    rank_details = f"/r={rank}" if rank is not None else ""
    loc = "%(module)s:%(lineno)d"

    _PREFIX = f"{faint}[{prefix}%(asctime)s {level_char}] {cyan}[{details}{worker_details}%(rl_attr)s{rank_details}/pid=%(process)d] {loc}:{reset} "

    LEVEL_COLORS = {
        logging.WARNING: bold_yellow,
        logging.ERROR: bold_red,
        logging.CRITICAL: bold_red,
    }

    class MicrosecondFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created)
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        def format(self, record):
            ds = _LOG_DATASET.get()
            row = _LOG_ROW.get()
            attr = ""
            if ds:
                attr += f"/ds={ds}"
            if row:
                attr += f"/row={row}"
            record.rl_attr = attr
            extra_kv = format_extra_kv(record)
            msg = super().format(record)
            if extra_kv:
                head, sep, tail = msg.partition("\n")
                msg = f"{head}  {extra_kv}{sep}{tail}"
            color = LEVEL_COLORS.get(record.levelno)
            if color:
                if msg.startswith(faint):
                    msg = msg[len(faint) :]
                msg = msg.replace(cyan, "")
                return f"{color}{msg}{reset}"
            return msg

    formatter = MicrosecondFormatter(f"{_PREFIX}%(message)s")
    return formatter


def configure_logging(
    level: int | str = logging.INFO,
    rank: int | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    extra_quiet_loggers: list[str] | None = None,
    root_logger_stream_handler_level: int | str | None = None,
    log_rotate: bool = False,
    log_rotate_max_bytes: int = 10 * 1024 * 1024,
    log_rotate_backup_count: int = 10000000,
    log_rotate_force_rollover: bool = True,
    rank_logger_all_ranks: bool = False,
    worker_name: str | None = None,
) -> None:
    if worker_name is None:
        worker_name = os.environ.get("XAI_WORKER_NAME")

    def _is_telemetry_capture(h: logging.Handler) -> bool:
        return type(h).__name__ in ("DebugLogCaptureHandler", "TelemetryLogHandler")

    for logger_name, logger_obj in logging.root.manager.loggerDict.items():
        if logger_name == "rank":
            continue
        if not isinstance(logger_obj, logging.Logger):
            continue
        logger_obj.handlers = [h for h in logger_obj.handlers if _is_telemetry_capture(h)]

    root_logger = logging.getLogger()
    root_logger.handlers = [h for h in root_logger.handlers if _is_telemetry_capture(h)]
    root_logger.setLevel(level)
    handler = add_stream_handler(root_logger, rank=rank, worker_name=worker_name)
    handler.setLevel(root_logger_stream_handler_level or level)

    quiet_loggers = ["botocore", "jax", "absl", "matplotlib"]
    for name in quiet_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARN)

    if extra_quiet_loggers:
        for name in extra_quiet_loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.ERROR)

    rank_logger = None
    if rank is not None:
        rank_logger = logging.getLogger("rank")
        rank_logger.handlers.clear()
        rank_logger.propagate = False

        if rank == 0 or rank_logger_all_ranks:
            add_stream_handler(rank_logger, rank=rank, worker_name=worker_name)

    for name in ("repl", "checkpointing"):
        add_stream_handler(logging.getLogger(name), worker_name=worker_name)

    if log_dir is not None and rank_logger is not None:
        path = Path(log_dir) / f"rank_{rank}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        if log_rotate:
            if log_rotate_force_rollover:
                Handler = ForceRotateOnStartHandler
            else:
                Handler = RotatingFileHandler

            file_handler = Handler(
                path,
                maxBytes=log_rotate_max_bytes,
                backupCount=log_rotate_backup_count,
            )
        else:
            file_handler = logging.FileHandler(path)
        file_handler.setFormatter(get_formatter(rank, worker_name=worker_name))
        root_logger.addHandler(file_handler)
        rank_logger.addHandler(file_handler)
    setup_axiom_only_logger(worker_name=worker_name)


class MultiLineStreamHandler(logging.StreamHandler):
    def emit(self, record):
        orig_exc_info = record.exc_info
        orig_exc_text = record.exc_text
        try:
            dummy_formatter = self.formatter or logging.Formatter()
            full_msg = dummy_formatter.format(record)
            messages = full_msg.split("\n")
            record.exc_info = None
            record.exc_text = None
            for message in messages:
                if message.strip():
                    record.msg = message
                    super().emit(record)
        finally:
            record.exc_info = orig_exc_info
            record.exc_text = orig_exc_text


def setup_axiom_only_logger(worker_name: str | None = None):
    if worker_name is None:
        worker_name = os.environ.get("XAI_WORKER_NAME")
    axiom_logger = logging.getLogger("axiom")
    axiom_logger.handlers.clear()
    axiom_logger.propagate = False
    handler = MultiLineStreamHandler(sys.stdout)
    handler.setFormatter(get_formatter(None, prefix="AXIOM_LOG ", worker_name=worker_name))
    axiom_logger.addHandler(handler)
    axiom_logger.setLevel(logging.INFO)
    return axiom_logger


class _BrokenPipeSafeStreamHandler(logging.StreamHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, BrokenPipeError):
            try:
                msg = self.format(record) + "\n"
                os.write(2, msg.encode("utf-8", errors="replace"))
            except BaseException:
                pass
            return
        super().handleError(record)


def add_stream_handler(
    logger: logging.Logger, rank: int | None = None, worker_name: str | None = None
) -> "logging.StreamHandler[TextIO]":
    target = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    handler = _BrokenPipeSafeStreamHandler(target)
    handler.setFormatter(get_formatter(rank, worker_name=worker_name))
    logger.addHandler(handler)
    return handler
