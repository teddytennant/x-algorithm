import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass

import boto3
from botocore.config import Config as BotoConfig
from grox.config.config import grox_config
from grox.core.lm.convo import Video as ConvoVideo
from monitor.metrics import Metrics
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

O2_ENDPOINT = "https://o2-https.o2-prod.atla-prod-storage.cluster.kube.int-x.ai"
_O2_CA_BUNDLE = "/etc/ssl/internal-ca/ca-bundle.crt"
_RETRY_WHILE_EMPTY_S = 60.0
_FETCH_TIMEOUT_S = 15.0
_MAX_VIDEOS = 10
_MAX_FRAMES_PER_VIDEO = 10
_MAX_FRAME_BYTES = 2 * 1024 * 1024


class MediaReferenceVideo(BaseModel):
    name: str
    source_post_id: str | None = None
    total_duration: float = 0.0
    frames: list[str] = Field(min_length=1, max_length=_MAX_FRAMES_PER_VIDEO)


class MediaReferenceManifest(BaseModel):
    version: str
    videos: list[MediaReferenceVideo] = Field(max_length=_MAX_VIDEOS)


@dataclass(frozen=True)
class MediaReferenceBundle:
    version: str
    digest: str
    videos: list[ConvoVideo]


class S3Store:
    def __init__(self, bucket: str, prefix: str):
        self._bucket = bucket
        self._prefix = f"{prefix}/" if prefix else ""
        self._client = boto3.client(
            "s3",
            endpoint_url=O2_ENDPOINT,
            verify=_O2_CA_BUNDLE if os.path.exists(_O2_CA_BUNDLE) else None,
            config=BotoConfig(
                connect_timeout=5,
                read_timeout=10,
                retries={"total_max_attempts": 1},
                s3={"addressing_style": "path"},
            ),
        )

    async def get(self, key: str) -> bytes:
        try:
            return await asyncio.to_thread(self._read, self._prefix + key)
        except self._client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"missing bundle object {key!r}") from None

    def _read(self, key: str) -> bytes:
        return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=self._prefix + key,
            Body=data,
            ContentType=content_type,
        )


def store_for(uri: str) -> S3Store:
    scheme, _, rest = uri.partition("://")
    bucket, _, prefix = rest.partition("/")
    if scheme != "s3" or not bucket:
        raise ValueError(
            f"media reference bundle uri must be an s3://<bucket>/<prefix> URI, got {uri!r}"
        )
    return S3Store(bucket=bucket, prefix=prefix.strip("/"))


async def _get_bounded(store: S3Store, key: str) -> bytes:
    data = await store.get(key)
    if len(data) > _MAX_FRAME_BYTES:
        raise ValueError(
            f"bundle object {key!r} is {len(data)} bytes, over the {_MAX_FRAME_BYTES} limit"
        )
    return data


async def _fetch_bundle(uri: str) -> MediaReferenceBundle:
    store = store_for(uri)
    manifest_bytes = await _get_bounded(store, "manifest.json")
    manifest = MediaReferenceManifest.model_validate_json(manifest_bytes)
    if not manifest.videos:
        raise ValueError("manifest lists no reference videos")

    digest = hashlib.sha256(manifest_bytes)
    videos: list[ConvoVideo] = []
    for video in manifest.videos:
        frames = list(
            await asyncio.gather(*(_get_bounded(store, name) for name in video.frames))
        )
        for data in frames:
            digest.update(data)
        videos.append(
            ConvoVideo(
                frames=frames,
                duration=video.total_duration / len(frames),
                total_duration=video.total_duration,
            )
        )
    return MediaReferenceBundle(
        version=manifest.version, digest=digest.hexdigest()[:12], videos=videos
    )


class _BundleState:
    def __init__(self) -> None:
        self.bundle: MediaReferenceBundle | None = None
        self.last_attempt: float | None = None
        self.lock = asyncio.Lock()

    def due(self, refresh_interval_s: float) -> bool:
        if self.last_attempt is None:
            return True
        wait = refresh_interval_s if self.bundle is not None else _RETRY_WHILE_EMPTY_S
        return time.monotonic() - self.last_attempt >= wait


class MediaReferenceBundleLoader:
    _states: dict[str, _BundleState] = {}

    @classmethod
    async def current(cls, name: str) -> MediaReferenceBundle | None:
        cfg = grox_config.media_reference_bundles.get(name)
        if cfg is None:
            return None
        state = cls._states.setdefault(name, _BundleState())
        if state.bundle is not None and not state.due(cfg.refresh_interval_s):
            return state.bundle
        async with state.lock:
            if not state.due(cfg.refresh_interval_s):
                return state.bundle
            state.last_attempt = time.monotonic()
            metric = Metrics.counter("media_reference_bundle.load.count")
            try:
                bundle = await asyncio.wait_for(
                    _fetch_bundle(cfg.uri), timeout=_FETCH_TIMEOUT_S
                )
            except Exception as e:
                metric.add(1, attributes={"bundle": name, "result": "error"})
                logger.error(
                    f"media reference bundle {name!r} load failed from {cfg.uri}: {e!r}"
                )
                return state.bundle
            changed = state.bundle is None or bundle.digest != state.bundle.digest
            state.bundle = bundle
            metric.add(
                1,
                attributes={
                    "bundle": name,
                    "result": "changed" if changed else "unchanged",
                },
            )
            if changed:
                logger.info(
                    f"media reference bundle {name!r} loaded: version={bundle.version} digest={bundle.digest} "
                    f"videos={len(bundle.videos)} frames={sum(len(v.frames) for v in bundle.videos)} from {cfg.uri}"
                )
            return state.bundle
