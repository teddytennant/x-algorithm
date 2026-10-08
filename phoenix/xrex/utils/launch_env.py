# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import os
from pathlib import Path

from xrex.utils import cluster

IS_LOCAL: bool = "HOST_NODE_ADDR" not in os.environ

DATA_DIR: str = "/tmp/xai_data" if IS_LOCAL else "/data"

XAI_USER: str = cluster.get_user() or "unknown_user"

JOB_NAME: str = cluster.get_job_name() or "local"

CHECKPOINT_DIR: str = os.path.join(DATA_DIR, "checkpoints", XAI_USER, JOB_NAME)

NUM_REPLICAS: int = int(os.environ.get("EXPLORER_COUNT", "1"))
REPLICA_INDEX: int = int(os.environ.get("EXPLORER_INDEX", "0"))

CONFIG: str | None = os.environ.get("EXPLORER_CONFIG")

GIT_COMMIT_HASH_ANNOTATION = "x.ai/git-commit-hash"
PARTIAL_OBJECT_METADATA_ACCEPT = "application/json;as=PartialObjectMetadata;g=meta.k8s.io;v=v1"


def read_experiment_git_commit_hash(
    experiment: str | None = None,
    namespace: str | None = None,
) -> str:
    experiment = experiment or JOB_NAME
    namespace = namespace or XAI_USER
    if not experiment or experiment == "local":
        return ""
    try:
        from kubernetes import client, config
    except ImportError:
        return ""
    try:
        config.load_incluster_config()
    except Exception:
        return ""
    try:
        api_client = client.ApiClient()
        api_client.set_default_header("Accept", PARTIAL_OBJECT_METADATA_ACCEPT)
        exp = client.CustomObjectsApi(api_client).get_namespaced_custom_object(
            group="x.ai",
            version="v1",
            namespace=namespace,
            plural="experiments",
            name=experiment,
            _request_timeout=2,
        )
    except Exception:
        return ""
    annotations = (exp.get("metadata") or {}).get("annotations") or {}
    value = annotations.get(GIT_COMMIT_HASH_ANNOTATION, "")
    value = value.strip() if isinstance(value, str) else ""
    return "" if value == "<unknown>" else value


CLUSTER: str | None = cluster.get_cluster() or None
CLOUD: str | None = cluster.get_cloud() or None

TRAIN_DOCKER_IMAGE: str = os.environ.get("XAI_TRAIN_DOCKER_IMAGE", "")
CPU_DOCKER_IMAGE: str = os.environ.get("XAI_CPU_DOCKER_IMAGE", "")


def run_files_note(run_dir: str | Path) -> str | None:
    del run_dir
    return None
