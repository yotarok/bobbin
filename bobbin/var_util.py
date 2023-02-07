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

"""Utility functions for handling variable collections."""

import os
import json
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import chex
from etils import epath
import flax
import jax
import jax.numpy as jnp
import numpy as np


_Array = chex.Array
_ArrayTree = chex.ArrayTree
_PRNGKey = chex.PRNGKey
_Scalar = chex.Scalar

_IsLeafFn = Callable[[Any], bool]


def _nested_structs_to_dicts(v: _ArrayTree) -> Dict[Any, Any]:
    return flax.serialization.to_state_dict(v)


def _is_supported_dict(a: Any) -> bool:
    return isinstance(a, (dict, flax.core.frozen_dict.FrozenDict))


def _nested_dicts_to_paths(
    node: _ArrayTree,
    prefix: str = "",
    *,
    pathsep: str = "/",
    seq_index_prefix: str = "",
) -> _ArrayTree:
    """Converts leaves in the nested leaves to the path names from the root node."""
    if _is_supported_dict(node):
        kwargs = dict()
        for k, v in node.items():
            kwargs[k] = _nested_dicts_to_paths(v, prefix + pathsep + k)
        return type(node)(**kwargs)
    else:
        return prefix


def nested_vars_to_paths(
    node: _ArrayTree, *, pathsep: str = "/", is_leaf: Optional[_IsLeafFn] = None
) -> _ArrayTree:
    """Constructs a tree with the same structure but containing path names as leaves."""
    if is_leaf is not None:
        # Fill placeholder in the places of deemed leaves for mimicking
        # `_nested_structs_to_dict` (that uses `flax.serialization.to_state_dict`)
        placeholder = -1
        node = jax.tree_util.tree_map(
            lambda x: placeholder if is_leaf(x) else x, node, is_leaf=is_leaf
        )

    node_dict = _nested_structs_to_dicts(node)
    paths_dicts = _nested_dicts_to_paths(node_dict, pathsep=pathsep)
    return flax.serialization.from_state_dict(node, paths_dicts)


def flatten_with_paths(
    node: _ArrayTree, *, is_leaf: Optional[_IsLeafFn] = None
) -> Iterator[Tuple[str, _Array]]:
    """Returns an iterator for leaves in the tree and their paths."""
    paths = nested_vars_to_paths(node, is_leaf=is_leaf)
    paths, unused_treedef = jax.tree_util.tree_flatten(paths)
    leaves, unused_treedef = jax.tree_util.tree_flatten(node, is_leaf=is_leaf)
    for path, leaf in zip(paths, leaves):
        yield path, leaf


def prng_keygen(seed: _PRNGKey) -> Iterator[_PRNGKey]:
    """Returns an iterator of unique RNGs."""
    while True:
        rng, seed = jax.random.split(seed)
        yield rng


def _json_object_hook_for_arrays(d: Dict[str, Any]) -> Any:
    if "__array__" in d and d["__array__"]:
        dtype = np.dtype(d.get("dtype", "float32"))
        return np.array(d["data"], dtype=dtype)
    return d


class _ArrayEncoder(json.JSONEncoder):
    """Internal JSON encoder that supports array encoding."""

    def default(self, obj: Any):
        if isinstance(obj, (np.ndarray, jnp.DeviceArray, jax.Array)):
            if obj.shape == ():
                # Scalar is serialized as normal scalar
                return obj.tolist()
            return dict(
                __array__=True,
                dtype=obj.dtype.name,
                data=obj.tolist(),
            )
        return super().default(obj)


def dump_pytree_json(
    tree: chex.ArrayTree,
) -> str:
    return json.dumps(flax.serialization.to_state_dict(tree), cls=_ArrayEncoder)


def write_pytree_json_file(path: Union[str, os.PathLike], tree: chex.ArrayTree) -> None:
    epath.Path(path).write_text(dump_pytree_json(tree))


def parse_pytree_json(
    json_str: Union[bytes, str], template: chex.ArrayTree
) -> chex.ArrayTree:
    state_dict = json.loads(json_str, object_hook=_json_object_hook_for_arrays)
    return flax.serialization.from_state_dict(template, state_dict)


def read_pytree_json_file(
    path: Union[str, os.PathLike], template: chex.ArrayTree
) -> Optional[chex.ArrayTree]:
    json = epath.Path(path).read_text()
    return parse_pytree_json(json, template)


def total_dimensionality(tree: chex.ArrayTree) -> int:
    """Returns total dimensionality of the variables in the given tree."""
    return jax.tree_util.tree_reduce(
        lambda n, arr: n + np.product(np.asarray(arr).shape), tree, 0
    )


def summarize_shape(tree: chex.ArrayTree) -> str:
    """Returns a string that summarizes shapes and dtypes of the tree."""
    indent_width = 2

    def metadata_to_str(x: chex.Array):
        return f"{str(x.shape)} dtype={str(x.dtype)}"

    def visit_node(subtree, indent_level: int):
        ret = ""
        for k, v in subtree.items():
            ret += " " * (indent_width * indent_level)
            ret += k + ":"
            if hasattr(v, "shape") and hasattr(v, "dtype"):
                ret += " " + metadata_to_str(v) + "\n"
            else:
                ret += "\n"
                ret += visit_node(v, indent_level + 1)
        return ret

    if hasattr(tree, "shape") and hasattr(tree, "dtype"):
        return metadata_to_str(tree)

    # normalize
    tree = jax.tree_util.tree_map(np.asarray, tree)
    state_dict = flax.serialization.to_state_dict(tree)
    return visit_node(state_dict, 0)
