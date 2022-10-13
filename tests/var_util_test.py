# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for var_util."""

from typing import Any, Optional

from absl.testing import absltest
import chex
import flax
import numpy as np

from bobbin import var_util

_FrozenDict = flax.core.FrozenDict


@flax.struct.dataclass
class _Pair:
    x: Any
    y: Any
    meta: Optional[Any] = flax.struct.field(pytree_node=False, default=None)


class AddressingTest(chex.TestCase):
    def test_addressing(self):
        var = dict(
            params=_FrozenDict(
                Layers=(
                    _FrozenDict(
                        weight=np.random.normal(size=(3, 3)),
                        bias=np.random.normal(size=(3,)),
                    ),
                    _FrozenDict(scale=np.random.gamma(1.0, size=(3,))),
                )
            ),
            dataclass=_Pair(x=[1.0, 2.0, 3.0], y="abc"),
        )
        expected_paths = dict(
            params=_FrozenDict(
                Layers=(
                    _FrozenDict(
                        weight="/params/Layers/0/weight",
                        bias="/params/Layers/0/bias",
                    ),
                    _FrozenDict(scale="/params/Layers/1/scale"),
                )
            ),
            dataclass=_Pair(
                x=["/dataclass/x/0", "/dataclass/x/1", "/dataclass/x/2"],
                y="/dataclass/y",
            ),
        )
        paths = var_util.nested_vars_to_paths(var)
        chex.assert_trees_all_equal_structs(var, paths)
        chex.assert_trees_all_equal(paths, expected_paths)


if __name__ == "__main__":
    absltest.main()
