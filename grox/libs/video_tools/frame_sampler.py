import asyncio
import io
import logging

import av
import cv2
from av.container import InputContainer

from video_tools.video_frames import VideoFrame, VideoFramesExtractor

logger = logging.getLogger(__name__)


class FrameSampler:
    @classmethod
    async def sample(
        cls,
        video_bytes: bytes,
        fps: float,
        max_sec: float,
        tile_size: int | None = None,
        offset_sec: float = 0.0,
    ) -> list[VideoFrame]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, cls._sample, video_bytes, fps, max_sec, tile_size, offset_sec
        )

    @classmethod
    def _sample(
        cls,
        video_bytes: bytes,
        fps: float,
        max_sec: float,
        tile_size: int | None,
        offset_sec: float,
    ) -> list[VideoFrame]:
        if fps <= 0 or max_sec <= 0:
            raise ValueError("fps and max_sec must be positive")
        step = 1.0 / fps
        frames: list[VideoFrame] = []
        with av.open(io.BytesIO(video_bytes)) as container:
            if not isinstance(container, InputContainer):
                raise TypeError(
                    f"Container is not an InputContainer: {type(container).__name__}"
                )
            stream = next(s for s in container.streams if s.type == "video")
            stream.thread_type = "AUTO"
            next_point = offset_sec
            start = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    continue
                t = float(frame.pts * frame.time_base)
                if start is None:
                    start = t
                t -= start
                if t > max_sec:
                    break
                if t < next_point - 1e-6:
                    continue
                next_point += step * (int((t - next_point) / step + 1e-6) + 1)
                ok, jpeg = cv2.imencode(".jpeg", frame.to_ndarray(format="bgr24"))
                if ok:
                    frames.append(
                        VideoFrame(
                            time_sec=round(t, 4),
                            frame=VideoFramesExtractor._process_frame(
                                jpeg.tobytes(), tile_size
                            ),
                        )
                    )
        logger.info(
            f"Sampled {len(frames)} frames at {fps}fps over the first {max_sec}s (grid offset {offset_sec:.3f}s)"
        )
        return frames

    @classmethod
    async def frames_at(
        cls, video_bytes: bytes, times: list[float], tile_size: int | None = None
    ) -> list[VideoFrame]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, cls._frames_at, video_bytes, times, tile_size
        )

    @classmethod
    def _frames_at(
        cls, video_bytes: bytes, times: list[float], tile_size: int | None
    ) -> list[VideoFrame]:
        pending = sorted(times)
        frames: list[VideoFrame] = []
        with av.open(io.BytesIO(video_bytes)) as container:
            if not isinstance(container, InputContainer):
                raise TypeError(
                    f"Container is not an InputContainer: {type(container).__name__}"
                )
            stream = next(s for s in container.streams if s.type == "video")
            stream.thread_type = "AUTO"
            start = None
            for frame in container.decode(stream):
                if not pending:
                    break
                if frame.pts is None or frame.time_base is None:
                    continue
                t = float(frame.pts * frame.time_base)
                if start is None:
                    start = t
                t -= start
                if t < pending[0] - 1e-3:
                    continue
                ok, jpeg = cv2.imencode(".jpeg", frame.to_ndarray(format="bgr24"))
                while pending and t >= pending[0] - 1e-3:
                    pending.pop(0)
                    if ok:
                        frames.append(
                            VideoFrame(
                                time_sec=round(t, 4),
                                frame=VideoFramesExtractor._process_frame(
                                    jpeg.tobytes(), tile_size
                                ),
                            )
                        )
        return frames
