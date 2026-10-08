import asyncio
import io
import logging

import av
import cv2
import numpy as np
from av.container import InputContainer

from video_tools.video_frames import VideoFramesExtractor

logger = logging.getLogger(__name__)

_SIGNATURE_SIDE = 32
_SHOT_CUT_RATIO = 4.0
_SHOT_CUT_MIN = 6.0
_BLANK_STDDEV = 8.0


class ShotKeyFramesExtractor:
    @classmethod
    async def extract(
        cls,
        video_bytes: bytes,
        window_sec: float = 3.0,
        max_fps: float = 15.0,
        max_key_frames: int = 3,
        tile_size: int | None = None,
    ) -> list[bytes]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            cls._extract,
            video_bytes,
            window_sec,
            max_fps,
            max_key_frames,
            tile_size,
        )

    @classmethod
    def _extract(
        cls,
        video_bytes: bytes,
        window_sec: float,
        max_fps: float,
        max_key_frames: int,
        tile_size: int | None,
    ) -> list[bytes]:
        if window_sec <= 0 or max_fps <= 0 or max_key_frames <= 0:
            raise ValueError("window_sec, max_fps and max_key_frames must be positive")
        with av.open(io.BytesIO(video_bytes)) as container:
            if not isinstance(container, InputContainer):
                raise TypeError(
                    f"Container is not an InputContainer: {type(container).__name__}"
                )
            total_duration = (
                float(container.duration / av.time_base) if container.duration else None
            )
            window = min(window_sec, total_duration) if total_duration else window_sec
            fps, ptss, times, signatures = cls._scan(container, window, max_fps)
            if not times:
                logger.warning("No decodable frames in the opening window")
                return []
            shots = cls._segment(signatures)
            picks = cls._rank(shots, signatures)[:max_key_frames]
            mids = [(a + b) // 2 for a, b, _ in picks]
            jpegs = cls._decode_at(container, [ptss[m] for m in mids])
        kept = sorted(
            (times[m], VideoFramesExtractor._process_frame(jpeg, tile_size))
            for m, jpeg in zip(mids, jpegs, strict=True)
            if jpeg is not None
        )
        logger.info(
            f"Scanned {len(times)} frames in {window:.2f}s at {fps}fps: {len(shots)} shots, key frames at {[round(t, 2) for t, _ in kept]}"
        )
        return [frame for _, frame in kept]

    @classmethod
    def _scan(
        cls, container: InputContainer, window: float, max_fps: float
    ) -> tuple[float | None, list[int], list[float], np.ndarray]:
        stream = next(s for s in container.streams if s.type == "video")
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else None
        min_gap = 1.0 / max_fps
        container.seek(0, stream=stream, backward=True)
        ptss: list[int] = []
        times: list[float] = []
        sigs: list[np.ndarray] = []
        last_kept = -1.0
        start = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            t = float(frame.pts * frame.time_base)
            if start is None:
                start = t
            t -= start
            if t > window:
                break
            if t - last_kept < min_gap - 1e-6:
                continue
            last_kept = t
            gray = frame.to_ndarray(format="gray")
            small = cv2.resize(
                gray, (_SIGNATURE_SIDE, _SIGNATURE_SIDE), interpolation=cv2.INTER_AREA
            )
            sigs.append(small.astype(np.float32).ravel())
            ptss.append(frame.pts)
            times.append(round(t, 4))
        return (
            fps,
            ptss,
            times,
            (
                np.stack(sigs)
                if sigs
                else np.empty((0, _SIGNATURE_SIDE * _SIGNATURE_SIDE), np.float32)
            ),
        )

    @classmethod
    def _decode_at(
        cls, container: InputContainer, ptss: list[int]
    ) -> list[bytes | None]:
        stream = next(s for s in container.streams if s.type == "video")
        out: dict[int, bytes | None] = {pts: None for pts in ptss}
        for pts in sorted(out):
            container.seek(pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None or frame.pts < pts:
                    continue
                if frame.pts == pts:
                    ok, jpeg = cv2.imencode(".jpeg", frame.to_ndarray(format="bgr24"))
                    out[pts] = jpeg.tobytes() if ok else None
                break
        return [out[pts] for pts in ptss]

    @classmethod
    def _segment(cls, signatures: np.ndarray) -> list[tuple[int, int]]:
        n = len(signatures)
        if n == 1:
            return [(0, 0)]
        deltas = np.abs(np.diff(signatures, axis=0)).mean(axis=1)
        threshold = max(_SHOT_CUT_MIN, _SHOT_CUT_RATIO * float(np.median(deltas)))
        cuts = [i + 1 for i, d in enumerate(deltas) if d > threshold]
        bounds, start = [], 0
        for cut in cuts:
            bounds.append((start, cut - 1))
            start = cut
        bounds.append((start, n - 1))
        return bounds

    @classmethod
    def _rank(
        cls, shots: list[tuple[int, int]], signatures: np.ndarray
    ) -> list[tuple[int, int, float]]:
        median = np.median(signatures, axis=0)
        ranked, blanks = [], []
        for a, b in shots:
            mid = signatures[(a + b) // 2]
            entry = (a, b, float(np.abs(mid - median).mean()))
            (blanks if float(mid.std()) < _BLANK_STDDEV else ranked).append(entry)
        ranked = ranked or blanks[:1]
        ranked.sort(key=lambda r: (-r[2], r[0]))
        return ranked
