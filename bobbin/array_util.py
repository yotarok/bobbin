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
#
#
"""Utility functions for handling arrays."""


from typing import Optional

import chex
import jax


def split_leading_axis(
    n: int,
    tree: chex.ArrayTree,
    *,
    leading_axis_name: Optional[str] = None,
    split_group_name: Optional[str] = None,
):
    """Splits the leading axis into the specified groups.

    This function is intended to be used for constructing micro-batches for,
    e.g., `pmap`.

    Args:
      n: The number of groups.
      tree: The tree to be split.
      leading_axis_name: The display name for leading axis used in exception
        error message when both `leading_axis_name` and `split_group_name` are
        given. (e.g. "batch size")
      split_group_name: The display name for split groups used in exception
        error message. (e.g. "# of devices")

    Returns:
      A function `split(x)` that reshapes `(n * m, ...)`-shaped array into
      `(n, m, ...)` where `n` is the argument of this function.  The function
      raises `ValueError` when the leading axis of `x` is not a multiple of `n`.
    """

    def split(x: chex.Array):
        lead_dim, *rest_dims = x.shape
        if lead_dim % n != 0:
            detailed_msg = ""
            if leading_axis_name is not None:
                detailed_msg = (
                    f"\n{leading_axis_name} must be a multiple of {split_group_name}"
                )
            raise ValueError(
                f"Trying to split {lead_dim} rows into {n} groups." + detailed_msg
            )
        return x.reshape((n, lead_dim // n) + tuple(rest_dims))

    return jax.tree_util.tree_map(split, tree)
