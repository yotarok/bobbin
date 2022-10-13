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

from typing import Any, Dict, Iterator, Tuple

import chex
import flax
import jax


_Array = chex.Array
_ArrayTree = chex.ArrayTree
_PRNGKey = chex.PRNGKey
_Scalar = chex.Scalar


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


def nested_vars_to_paths(node: _ArrayTree, *, pathsep: str = "/") -> _ArrayTree:
    """Constructs a tree with the same structure but containing path names as leaves."""
    node_dict = _nested_structs_to_dicts(node)
    paths_dicts = _nested_dicts_to_paths(node_dict, pathsep=pathsep)
    paths, unused_treedef = jax.tree_util.tree_flatten(paths_dicts)
    treedef = jax.tree_util.tree_structure(node)
    return jax.tree_util.tree_unflatten(treedef, paths)


def flatten_with_paths(node: _ArrayTree) -> Iterator[Tuple[str, _Array]]:
    """Returns an iterator for leaves in the tree and their paths."""
    paths = nested_vars_to_paths(node)
    paths, unused_treedef = jax.tree_util.tree_flatten(paths)
    leaves, unused_treedef = jax.tree_util.tree_flatten(node)
    for path, leaf in zip(paths, leaves):
        yield path, leaf


def prng_keygen(seed: _PRNGKey) -> Iterator[_PRNGKey]:
    """Returns an iterator of unique RNGs."""
    while True:
        rng, seed = jax.random.split(seed)
        yield rng
