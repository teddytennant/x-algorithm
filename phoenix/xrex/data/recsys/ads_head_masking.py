# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from xrex.data.recsys.ads_late_window import (
    ADS_LATE_WINDOW_TWIN_HEAD_INDICES,
    ADS_LATE_WINDOW_TWIN_SUPPRESSOR,
)
from xrex.data.recsys.constants import (
    CLICK_ACTION_INDEX,
    CLICK_CONDITIONED_ACTION_INDICES,
    CONVERSION_KEEP_APP,
    CONVERSION_KEEP_WEB,
    MMP_CLICK_ACTION_INDEX,
    MMP_CLICK_CONDITIONED_ACTION_INDICES,
    MMP_CLICK_VIEW_THROUGH_ACTION_INDICES,
    VIEW_THROUGH_ACTION_INDICES,
)

FRESH_STREAM_ID = 0
EARLY_RELABEL_STREAM_ID = 1
LATE_SPLIT_STREAM_ID = 2

ADS_FAMILY_SLICE_HEADS: dict[str, int] = {
    "delayed_website_clicked": CLICK_CONDITIONED_ACTION_INDICES[0],
    "delayed_website_non_clicked": VIEW_THROUGH_ACTION_INDICES[0],
    "delayed_app_clicked": MMP_CLICK_CONDITIONED_ACTION_INDICES[0],
    "delayed_app_non_clicked": MMP_CLICK_VIEW_THROUGH_ACTION_INDICES[0],
}


def _head_indicator(indices: Sequence[int], num_actions: int) -> npt.NDArray[np.bool_]:
    indicator = np.zeros(num_actions, dtype=np.bool_)
    indicator[list(indices)] = True
    return indicator


def build_trained_candidate_mask(
    actions: npt.NDArray[np.bool_],
    sample_source: npt.NDArray[np.integer] | None,
    trained_actions_bits: npt.NDArray[np.uint8],
    *,
    ads_head_masking: bool,
) -> npt.NDArray[np.bool_]:
    rows, _, num_actions = actions.shape
    keep_web = ((trained_actions_bits & CONVERSION_KEEP_WEB) != 0)[:, :, None]
    keep_app = ((trained_actions_bits & CONVERSION_KEEP_APP) != 0)[:, :, None]

    if not ads_head_masking:
        return np.broadcast_to(keep_web | keep_app, actions.shape).copy()

    if sample_source is None:
        source = np.full((rows, 1, 1), FRESH_STREAM_ID, dtype=np.int64)
    else:
        source = np.asarray(sample_source).reshape(rows, 1, 1)
    fresh = source == FRESH_STREAM_ID
    late = source == LATE_SPLIT_STREAM_ID
    early = ~fresh & ~late

    actions = actions.astype(np.bool_, copy=False)
    click = actions[:, :, CLICK_ACTION_INDEX][:, :, None]
    app_click = actions[:, :, MMP_CLICK_ACTION_INDEX][:, :, None]

    web_ct = _head_indicator(CLICK_CONDITIONED_ACTION_INDICES, num_actions)
    web_vt = _head_indicator(VIEW_THROUGH_ACTION_INDICES, num_actions)
    app_ct = _head_indicator(MMP_CLICK_CONDITIONED_ACTION_INDICES, num_actions)
    app_vt = _head_indicator(MMP_CLICK_VIEW_THROUGH_ACTION_INDICES, num_actions)
    twins = _head_indicator(ADS_LATE_WINDOW_TWIN_HEAD_INDICES, num_actions)
    engagement = ~(web_ct | web_vt | app_ct | app_vt | twins)

    suppressor_index = list(range(num_actions))
    for twin, suppressor in ADS_LATE_WINDOW_TWIN_SUPPRESSOR.items():
        suppressor_index[twin] = suppressor
    no_early_counterpart = ~actions[:, :, suppressor_index]

    kept = keep_web | keep_app
    return (
        (engagement & fresh & kept)
        | (web_ct & early & keep_web & click)
        | (web_vt & early & keep_web & ~click)
        | (app_ct & early & keep_app & app_click)
        | (app_vt & early & keep_app & ~app_click)
        | (twins & late & click & no_early_counterpart & kept)
    )
