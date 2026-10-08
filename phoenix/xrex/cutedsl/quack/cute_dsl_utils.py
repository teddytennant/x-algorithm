# Copyright (c) 2025, Tri Dao.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from quack-kernels (https://github.com/Dao-AILab/quack);
# modified by X.AI Corp.

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from dataclasses import dataclass, fields

import cutlass
from cutlass.cutlass_dsl import NumericMeta

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


def _partition_fields(obj):
    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
    non_constexpr = {n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)}
    return constexpr, non_constexpr


def _new_from_mlir_values(self, values):
    constexpr_fields, non_constexpr_fields = _partition_fields(self)
    for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
        non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
        values = values[n_items:]
    return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        _, non_constexpr_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in non_constexpr_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    __new_from_mlir_values__ = _new_from_mlir_values
