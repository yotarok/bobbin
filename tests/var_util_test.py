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

import json
import tempfile
from typing import Any, Optional, TYPE_CHECKING

from absl.testing import absltest
import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bobbin import tensorboard
from bobbin import training
from bobbin import var_util

_FrozenDict = flax.core.FrozenDict


if TYPE_CHECKING:
    nn = Any  # noqa: F811


class _Pair(flax.struct.PyTreeNode):
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

    def test_addressing_with_sructured_leaves(self):
        var = dict(
            params=_FrozenDict(
                Layers=(
                    _FrozenDict(
                        summary=_Pair(x=1, y=2),
                        params=_FrozenDict(
                            weight=np.random.normal(size=(3,)),
                            bias=np.random.normal(size=(1,)),
                        ),
                    ),
                    _FrozenDict(
                        summary=_Pair(x=6, y=7),
                        params=np.random.normal(size=(3,)),
                    ),
                )
            ),
        )
        expected_paths = dict(
            params=_FrozenDict(
                Layers=(
                    _FrozenDict(
                        summary="/params/Layers/0/summary",
                        params=_FrozenDict(
                            weight="/params/Layers/0/params/weight",
                            bias="/params/Layers/0/params/bias",
                        ),
                    ),
                    _FrozenDict(
                        summary="/params/Layers/1/summary",
                        params="/params/Layers/1/params",
                    ),
                )
            ),
        )
        paths = var_util.nested_vars_to_paths(
            var, is_leaf=lambda x: isinstance(x, _Pair)
        )
        chex.assert_trees_all_equal(paths, expected_paths)

    def test_flatten_with_paths_with_structured_leaves(self):
        var = dict(
            elem1=_Pair(x=1, y=2),
            elem2=_Pair(x=3, y=4),
        )
        leaves_with_paths = var_util.flatten_with_paths(
            var, is_leaf=lambda x: isinstance(x, _Pair)
        )
        chex.assert_equal(
            [("/elem1", _Pair(x=1, y=2)), ("/elem2", _Pair(x=3, y=4))],
            list(leaves_with_paths),
        )

    def test_addressing_train_state(self):
        class TestModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                self.sow(
                    "tensorboard",
                    "random",
                    tensorboard.MplImageSummary(
                        jnp.zeros((3, 3)),
                        aspect="auto",
                        origin="lower",
                        with_colorbar=True,
                        h_paddings=jnp.zeros((3,)),
                    ),
                )
                return nn.Dense(features=1)(x)

        batch_size = 3
        dims = 5
        tx = optax.sgd(0.001)
        model = TestModel()
        init_model_vars = model.init(
            jax.random.PRNGKey(0), np.zeros((batch_size, dims))
        )
        train_state = training.initialize_train_state(model.apply, init_model_vars, tx)
        paths = var_util.nested_vars_to_paths(train_state)
        np.testing.assert_equal(
            paths.extra_vars["tensorboard"]["random"][0].image,
            "/extra_vars/tensorboard/random/0/image",
        )


class JsonTest(chex.TestCase):
    def test_serialize_and_deserialize_in_mem(self):
        var = dict(
            elem1=_Pair(x=np.array([1, 2]), y=np.array([3, 4])),
            elem2=_Pair(x=5, y=None),
        )

        json_str = var_util.dump_pytree_json(var)
        reconstructed_dict = json.loads(json_str)
        chex.assert_equal(
            reconstructed_dict,
            dict(
                elem1=dict(
                    x={"__array__": True, "data": [1, 2], "dtype": "int64"},
                    y={"__array__": True, "data": [3, 4], "dtype": "int64"},
                ),
                elem2=dict(x=5, y=None),
            ),
        )

        zeroed_var = jax.tree_map(lambda x: 0, var)
        reconstructed_var = var_util.parse_pytree_json(json_str, zeroed_var)
        jax.tree_util.tree_map(
            lambda x, y: np.testing.assert_allclose(x, y), var, reconstructed_var
        )

    def test_serialize_and_deserialize_in_fs(self):
        var = dict(
            elem1=_Pair(x=np.array([1, 2]), y=np.array([3, 4])),
            elem2=_Pair(x=5, y=None),
        )

        tmpf = tempfile.NamedTemporaryFile()
        var_util.write_pytree_json_file(tmpf.name, var)
        zeroed_var = jax.tree_map(lambda x: 0, var)
        reconstructed_var = var_util.read_pytree_json_file(tmpf.name, zeroed_var)

        jax.tree_util.tree_map(
            lambda x, y: np.testing.assert_allclose(x, y), var, reconstructed_var
        )


class VarInspectorTest(chex.TestCase):
    def test_total_dimensionality(self):
        var = dict(
            elem1=_Pair(x=np.array([1, 2]), y=np.array([3, 4])),
            elem2=_Pair(x=5, y=None),
        )
        np.testing.assert_equal(var_util.total_dimensionality(var), 5)

    def test_shape_summarizer(self):
        var = dict(
            elem1=_Pair(
                x=np.array([1, 2], dtype=np.int32), y=np.array([3, 4], dtype=np.float32)
            ),
            elem2=_Pair(
                x=[5, np.array([1, 2], dtype=np.float32)],
                y=np.zeros((3, 3, 2), dtype=np.float64),
            ),
            elem_none=None,
        )
        np.testing.assert_equal(
            var_util.summarize_shape(var).strip(),
            """
elem1:
  x: (2,) dtype=int32
  y: (2,) dtype=float32
elem2:
  x:
    0: () dtype=int64
    1: (2,) dtype=float32
  y: (3, 3, 2) dtype=float64
elem_none: None
""".strip(),
        )


if __name__ == "__main__":
    absltest.main()
