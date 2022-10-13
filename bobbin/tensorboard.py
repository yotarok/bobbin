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

import abc
import os
import pathlib
import re
from typing import Callable, Dict, Iterator, Optional, Set, Tuple, Union

import chex
from flax.metrics import tensorboard as flax_tb

from .evaluation import EvalResults
from .training import TrainState
from .var_util import flatten_with_paths

_ArrayTree = chex.ArrayTree


class VarSelector(metaclass=abc.ABCMeta):
    """Abstract base class for variable selectors."""

    @abc.abstractmethod
    def enumerate(self, tree: _ArrayTree) -> Iterator[Tuple[str, chex.Array]]:
        ...


class PathRexpVarSelector(VarSelector):
    """Variable selector that finds tree-leaves based on regex over path names."""

    def __init__(self, path_rexp: Union[str, re.Pattern]):
        # `compile` does nothing if arg is already compiled.
        self.path_rexp = re.compile(path_rexp)

    def enumerate(self, tree: _ArrayTree) -> Iterator[Tuple[str, chex.Array]]:
        rexp = re.compile(self.path_rexp)
        for path, var in flatten_with_paths(tree):
            if rexp.match(path):
                yield path, var


def _make_rexp_sub(pattern: Union[str, re.Pattern], repl: str):
    pattern = re.compile(pattern)

    def sub(s: str) -> str:
        return re.sub(pattern, repl, s)

    return sub


def publish_train_intermediates(
    writer: flax_tb.SummaryWriter,
    tree: _ArrayTree,
    step: int,
    *,
    scalar_selector: VarSelector = PathRexpVarSelector(".*:scalar(/0)?"),
    scalar_tag_rewriter: Callable[[str], str] = _make_rexp_sub(":scalar(/0)?$", ""),
) -> None:
    """Writes variables specified in the args to SummaryWriter `writer`.

    Currently, this function only supports scalar summaries.

    Args:
        writer: Destination as `flax.metrics.tensorboard.SummaryWriter`.
        tree: Source of summary variables. Typically, a training state, or a
            variable collection.
        step: The step number of summary information.
        scalar_selector: Selector for extracting scalars to be published. By
            default, variables with ":scalar" suffix will be selected.
        scalar_tag_rewriter: A function that converts variable paths to a tag
            name used in TensorBoard. By default, tag names are defined as a
            name of last path component without ":scalar" suffix.
    """
    for varname, val in scalar_selector.enumerate(tree):
        tag = scalar_tag_rewriter(varname)
        writer.scalar(tag, val, step=step)


_EvalResultsWriteFn = Callable[[EvalResults, TrainState, flax_tb.SummaryWriter], None]


def make_eval_results_writer(
    summary_root_dir: os.PathLike[str],
    dataset_filter: Optional[Set[str]] = None,
    method: Optional[_EvalResultsWriteFn] = None,
):
    """Makes a function that publishes evaluation results from different datasets.

    Args:
        summary_root_dir: Root directory for summaries.
        dataset_filter: If set, only the results from datasets in `dataset_filter`
            will be processed.

    Returns:
        A function `tb_writer(results, state)` where `results` is
        `Dict[str, EvalResults]` where `results[name]` represents the evaluation
        result for the dataset with `name`.  Instances of `EvalResults` passed
        to this function must override`EvalResults.write_to_tensorboard` method.
        `state` is a `TrainState`.
    """
    summary_root_dir = pathlib.Path(summary_root_dir)
    writers = dict()

    def _tb_writer(res: Dict[str, EvalResults], st: TrainState):
        for name, result in res.items():
            if dataset_filter is not None and name not in dataset_filter:
                continue
            if name not in writers:
                writers[name] = flax_tb.SummaryWriter(summary_root_dir / name)
            if method is None:
                result.write_to_tensorboard(st, writers[name])
            else:
                method(result, st, writers[name])

    return _tb_writer
