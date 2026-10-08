from grox.core.plans.plan import Plan
from grox.core.registry import register
from grox.core.tasks.task_media import TaskMediaHydration
from grox.flows.ptos.task_rate_limit import TaskRateLimitSafetyPtosKeyFrames
from grox.flows.ptos.task_safety_ptos_adult_content_cross_validation import (
    TaskSafetyPtosAdultContentCrossValidation,
)
from grox.flows.ptos.task_safety_ptos_key_frames_filter import (
    TaskSafetyPtosKeyFramesAdultOnly,
    TaskSafetyPtosKeyFramesFilter,
)
from grox.flows.ptos.task_safety_ptos_policy import TaskSafetyPtosPolicyDetection
from grox.flows.ptos.task_safety_ptos_safemodel_sex_nudity import (
    TaskSafetyPtosSafemodelSexNudity,
)
from grox.flows.ptos.task_write_safety_post_annotations_result_sink import (
    TaskWriteSafetyPostAnnotationsResultSink,
)


@register
class PlanSafetyPtosKeyFrames(Plan):
    KEY = "safety_ptos_key_frames"

    TASKS = {
        "task_safety_ptos_annotation_rate_limit": TaskRateLimitSafetyPtosKeyFrames,
        "task_safety_ptos_key_frames_filter": TaskSafetyPtosKeyFramesFilter,
        "task_media_hydration": TaskMediaHydration,
        "task_safety_ptos_key_frames_adult_only": TaskSafetyPtosKeyFramesAdultOnly,
        "task_safety_ptos_policy_detection": TaskSafetyPtosPolicyDetection,
        "task_safety_ptos_safemodel_sex_nudity": TaskSafetyPtosSafemodelSexNudity,
        "task_safety_ptos_adult_content_cross_validation": TaskSafetyPtosAdultContentCrossValidation,
        "task_write_safety_post_annotations_result_sink": TaskWriteSafetyPostAnnotationsResultSink,
    }

    TASK_DEPENDENCIES = {
        "task_safety_ptos_annotation_rate_limit": {},
        "task_safety_ptos_key_frames_filter": {
            "task_safety_ptos_annotation_rate_limit"
        },
        "task_media_hydration": {"task_safety_ptos_key_frames_filter"},
        "task_safety_ptos_key_frames_adult_only": {"task_media_hydration"},
        "task_safety_ptos_policy_detection": {"task_safety_ptos_key_frames_adult_only"},
        "task_safety_ptos_safemodel_sex_nudity": {"task_safety_ptos_policy_detection"},
        "task_safety_ptos_adult_content_cross_validation": {
            "task_safety_ptos_policy_detection",
            "task_safety_ptos_safemodel_sex_nudity",
        },
        "task_write_safety_post_annotations_result_sink": {
            "task_safety_ptos_adult_content_cross_validation"
        },
    }
