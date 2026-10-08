import asyncio
import logging
import random
import time

import numpy as np
from embed.embed_http import ChatTemplate, XaiEmbeddingClientHttp
from monitor.metrics import Metrics
from video_tools.frame_sampler import FrameSampler
from video_tools.video_frames import VideoFrame

from grox.config.config import grox_config
from grox.core.data_loaders.data_types import RankingQuery

logger = logging.getLogger(__name__)

COARSE_FPS = 3.0
FINE_FPS = 15.0
TOP_PEAKS = 2
PEAK_MIN_GAP_SEC = 1.0
PEAK_RADIUS_SEC = 0.5
MAX_FRAMES = 3
PICK_MIN_GAP_SEC = 0.1
_EMBED_CONCURRENCY = 16
_TRUNCATE_DIM = 1024
_QUERY_INSTRUCTION = "Represent the user's input."
_EMBED_LATENCY_BUCKETS_MS = [
    25,
    50,
    100,
    200,
    300,
    500,
    750,
    1000,
    1500,
    2500,
    5000,
    10000,
]


def pick_peaks(
    times: np.ndarray, scores: np.ndarray, k: int, min_gap: float
) -> list[int]:
    picks: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if all(abs(times[i] - times[p]) >= min_gap for p in picks):
            picks.append(int(i))
        if len(picks) == k:
            break
    return picks


class EmbeddingRankedKeyFrames:
    _clients: dict[str, XaiEmbeddingClientHttp] = {}
    _directions: dict[RankingQuery, np.ndarray] = {}
    _direction_lock = asyncio.Lock()

    @classmethod
    def _client(cls, instruction: str) -> XaiEmbeddingClientHttp:
        if instruction not in cls._clients:
            cfg = grox_config.get_embedding_model(
                grox_config.media_hydration.embedding_ranked_key_frames_model
            )
            if cfg.timeout_seconds is None:
                cfg = cfg.model_copy(update={"timeout_seconds": 30.0})
            cls._clients[instruction] = XaiEmbeddingClientHttp(
                config=cfg, chat_template=ChatTemplate(system_prompt=instruction)
            )
        return cls._clients[instruction]

    @staticmethod
    def _normalize(vector) -> np.ndarray:
        v = np.asarray(vector[:_TRUNCATE_DIM], dtype=np.float64)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    @classmethod
    async def _direction(cls, query: RankingQuery) -> np.ndarray:
        async with cls._direction_lock:
            if query not in cls._directions:
                client = cls._client(_QUERY_INSTRUCTION)
                positive, negative = await asyncio.gather(
                    client.encode_async(query.positive_query, []),
                    client.encode_async(query.negative_query, []),
                )
                cls._directions[query] = cls._normalize(positive) - cls._normalize(
                    negative
                )
            return cls._directions[query]

    @classmethod
    async def _score(
        cls,
        frames: list[VideoFrame],
        indices: list[int],
        query: RankingQuery,
        direction: np.ndarray,
    ) -> dict[int, float]:
        client = cls._client(query.frame_instruction)
        semaphore = asyncio.Semaphore(_EMBED_CONCURRENCY)

        async def one(i: int) -> float | None:
            async with semaphore:
                start = time.perf_counter()
                try:
                    embedding = await client.encode_async("", [frames[i].frame])
                    outcome = "ok"
                except Exception:
                    logger.debug(
                        f"embedding frame at {frames[i].time_sec}s failed",
                        exc_info=True,
                    )
                    embedding, outcome = None, "error"
                attributes = {"query": query.name, "outcome": outcome}
                Metrics.counter("embedding_ranked_key_frames.embed.count").add(
                    1, attributes=attributes
                )
                Metrics.histogram(
                    "embedding_ranked_key_frames.embed.duration_ms",
                    _EMBED_LATENCY_BUCKETS_MS,
                ).record((time.perf_counter() - start) * 1000, attributes=attributes)
                return (
                    None
                    if embedding is None
                    else float(cls._normalize(embedding) @ direction)
                )

        scores = await asyncio.gather(*(one(i) for i in indices))
        return {
            i: score
            for i, score in zip(indices, scores, strict=True)
            if score is not None
        }

    @classmethod
    async def select(
        cls, video_bytes: bytes, tile_size: int | None, query: RankingQuery
    ) -> list[bytes]:
        cfg = grox_config.media_hydration
        frames = await FrameSampler.sample(
            video_bytes,
            FINE_FPS,
            cfg.embedding_ranked_key_frames_max_seconds,
            cfg.embedding_ranked_key_frames_embed_size,
            offset_sec=random.uniform(0.0, 1.0 / FINE_FPS),
        )
        if not frames:
            return []
        times = np.array([f.time_sec for f in frames])
        direction = await cls._direction(query)

        slots = np.floor(times * COARSE_FPS + 1e-6)
        coarse_indices = [
            int(i) for i in np.flatnonzero(np.r_[True, np.diff(slots) > 0])
        ]
        scores = await cls._score(frames, coarse_indices, query, direction)
        attempted = len(coarse_indices)
        coarse = sorted(scores)
        peaks = pick_peaks(
            times[coarse],
            np.array([scores[i] for i in coarse]),
            TOP_PEAKS,
            PEAK_MIN_GAP_SEC,
        )
        near = {
            i
            for p in peaks
            for i in range(len(frames))
            if abs(times[i] - times[coarse[p]]) <= PEAK_RADIUS_SEC
        }
        fine_indices = sorted(near - scores.keys())
        scores |= await cls._score(frames, fine_indices, query, direction)
        attempted += len(fine_indices)

        min_score = cfg.embedding_ranked_key_frames_min_score
        scored = sorted(
            i for i in scores if min_score is None or scores[i] >= min_score
        )
        picks = pick_peaks(
            times[scored],
            np.array([scores[i] for i in scored]),
            MAX_FRAMES,
            PICK_MIN_GAP_SEC,
        )
        chosen = sorted(scored[p] for p in picks)
        logger.info(
            f"embedding-ranked key frames ({query.name}): {len(frames)} frames over {times[-1]:.1f}s, "
            f"embedded {len(scores)} (failed {attempted - len(scores)}), "
            f"best {round(max(scores.values()), 4) if scores else None}, min score {min_score}, "
            f"picked {[(float(times[i]), round(scores[i], 4)) for i in chosen]}"
        )
        if not chosen:
            return []
        chosen_frames = await FrameSampler.frames_at(
            video_bytes, [frames[i].time_sec for i in chosen], tile_size
        )
        return [f.frame for f in chosen_frames]
