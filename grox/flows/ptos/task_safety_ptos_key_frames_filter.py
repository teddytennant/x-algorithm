import logging
from typing import override

from monitor.metrics import Metrics
from strato_http.queries.safety_post_annotations_result import (
    StratoSafetyPostAnnotationsResultDirectMh,
)

from grox.core.data_loaders.data_types import Post, RankingQuery, Video
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import Task
from grox.core.tasks.task_filters import TaskFilterWithPost
from grox.flows.ptos.constants import HIGH_FAV_THRESHOLD, KEY_FRAMES_UNCONDITIONAL_FAV
from grox.flows.ptos.state import (
    SafetyPolicyCategory,
    SafetyPostAnnotations,
    SafetyPtosState,
)

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "task.safety_ptos_key_frames_filter"
NUDITY_RANKING = RankingQuery(
    name="nudity",
    frame_instruction="Represent this video frame for detecting nudity and exposed intimate body parts",
    positive_query="a naked person with exposed intimate body parts",
    negative_query="a fully clothed person",
)


class TaskSafetyPtosKeyFramesFilter(TaskFilterWithPost):
    _result_direct_mh = StratoSafetyPostAnnotationsResultDirectMh()

    @override
    @classmethod
    async def _eligible_with_post(cls, post: Post, ctx: TaskContext) -> bool:
        if not post.user:
            return cls._skip(post, "no_user")
        media = [
            *(post.media or []),
            *(post.quoted_post.media or [] if post.quoted_post else []),
        ]
        if not any(isinstance(m, Video) for m in media):
            return cls._skip(post, "no_video")
        favs = post.get_fav_count()
        if favs < HIGH_FAV_THRESHOLD:
            return cls._skip(post, "below_fav")
        prior = await cls._result_direct_mh.fetch(int(post.id))
        if (
            prior is not None
            and prior.safetyBoolMetadata
            and prior.safetyBoolMetadata.isNsfw
        ):
            return cls._skip(post, "already_nsfw")
        prior_adult = prior is not None and any(
            v.category == SafetyPolicyCategory.AdultContent
            for a in prior.safetyPostAnnotations or []
            for v in a.violatedPolicies or []
        )
        if favs < KEY_FRAMES_UNCONDITIONAL_FAV and not prior_adult:
            return cls._skip(
                post, "no_prior_ptos" if prior is None else "no_adult_prior"
            )
        for medium in media:
            if isinstance(medium, Video):
                medium.key_frames_ranking = NUDITY_RANKING
        Metrics.counter(f"{_METRIC_PREFIX}.eligible.count").add(
            1,
            attributes={
                "forced": str(favs >= KEY_FRAMES_UNCONDITIONAL_FAV).lower(),
                "prior_adult": str(prior_adult).lower(),
            },
        )
        return True

    @classmethod
    def _skip(cls, post: Post, reason: str) -> bool:
        Metrics.counter(f"{_METRIC_PREFIX}.skipped.count").add(
            1, attributes={"reason": reason}
        )
        logger.info(f"Post {post.id}: skipped ({reason})")
        return False


class TaskSafetyPtosKeyFramesAdultOnly(Task):
    @classmethod
    async def _exec(cls, ctx: TaskContext) -> None:
        ctx.state(SafetyPtosState).annotations = SafetyPostAnnotations(
            violatedPolicies=[]
        )
