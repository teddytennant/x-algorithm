# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging
import os
import random
import threading
import time

rank_logger = logging.getLogger("rank")

_TRANSIENT_CONNECT_MARKERS = ("identity-server agent",)
_CONNECT_ATTEMPTS = 4
_CONNECT_BACKOFF_CAP_SECS = 4.0


def connect_cluster_kms_client(**kwargs):
    import xai_kms

    slept = 0.0
    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        try:
            return xai_kms.KmsClient.from_cluster_env(**kwargs)
        except Exception as error:
            transient = any(marker in str(error) for marker in _TRANSIENT_CONNECT_MARKERS)
            if not transient or attempt == _CONNECT_ATTEMPTS:
                raise
            delay = min(0.5 * 2 ** (attempt - 1), _CONNECT_BACKOFF_CAP_SECS) * (
                0.5 + random.random()
            )
            rank_logger.warning(
                "KMS client bootstrap failed transiently (%s); retrying in %.1fs "
                "(attempt %d/%d, slept %.1fs so far)",
                error,
                delay,
                attempt,
                _CONNECT_ATTEMPTS,
                slept,
            )
            time.sleep(delay)
            slept += delay
    raise AssertionError("unreachable")


_shared_lock = threading.Lock()
_shared_decrypt_client = None


def shared_decrypt_client():
    global _shared_decrypt_client
    with _shared_lock:
        if _shared_decrypt_client is None:
            _shared_decrypt_client = connect_cluster_kms_client()
        return _shared_decrypt_client


def reset_shared_decrypt_client() -> None:
    global _shared_decrypt_client
    with _shared_lock:
        _shared_decrypt_client = None


_fork_inherited_clients = []


def _forget_in_fork_child() -> None:
    global _shared_decrypt_client, _shared_lock
    _shared_lock = threading.Lock()
    if _shared_decrypt_client is not None:
        _fork_inherited_clients.append(_shared_decrypt_client)
        _shared_decrypt_client = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_in_fork_child)
