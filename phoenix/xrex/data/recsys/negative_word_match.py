# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import numpy as np
from numpy import typing as npt

from xrex.data.recsys.feature_config import CategoricalFeature, FloatFeature


def clear_word_match(
    categorical_features: npt.NDArray,
    float_features: npt.NDArray,
    negative_slots: npt.NDArray[np.bool_],
) -> None:
    exact = CategoricalFeature.exactPhraseSeq.value
    fraction = FloatFeature.matchedWordFractionSeq.value
    if exact < categorical_features.shape[2]:
        categorical_features[:, :, exact][negative_slots] = 0
    if fraction < float_features.shape[2]:
        float_features[:, :, fraction][negative_slots] = 0.0


def compute_word_match(
    categorical_features: npt.NDArray,
    float_features: npt.NDArray,
    negative_slots: npt.NDArray[np.bool_],
    texts: np.ndarray,
    queries: np.ndarray,
    authors: np.ndarray,
) -> None:
    exact = CategoricalFeature.exactPhraseSeq.value
    fraction = FloatFeature.matchedWordFractionSeq.value
    if exact >= categorical_features.shape[2] or fraction >= float_features.shape[2]:
        return
    from xai_search_lexical_match import search_lexical_match

    rows, slots = np.nonzero(negative_slots & _is_text(texts) & _is_text(queries))
    valid, fractions, exact_phrase = search_lexical_match(
        queries[rows, slots].tolist(),
        texts[rows, slots].tolist(),
        [author if isinstance(author, str) else "" for author in authors[rows, slots]],
    )
    valid = np.asarray(valid, dtype=bool)
    rows, slots = rows[valid], slots[valid]
    float_features[rows, slots, fraction] = np.asarray(fractions, dtype=np.float32)[valid]
    categorical_features[rows, slots, exact] = np.asarray(exact_phrase, dtype=bool)[valid] + 1


def _is_text(values: np.ndarray) -> npt.NDArray[np.bool_]:
    return np.vectorize(lambda value: isinstance(value, str) and bool(value), otypes=[bool])(values)
