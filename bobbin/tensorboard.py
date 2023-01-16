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


from __future__ import annotations

import abc
import os
import pathlib
from typing import Callable, Dict, Optional, Union

import chex
import flax
from flax import struct
from flax.metrics import tensorboard as flax_tb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import EvalResults
from .var_util import flatten_with_paths

_ArrayTree = chex.ArrayTree
_TrainState = flax.training.train_state.TrainState


class PublishableSow(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def publish(self, writer: flax_tb.SummaryWriter, tag: str, step: int) -> None:
        ...


def _is_publishable_sow(node: _ArrayTree) -> bool:
    return isinstance(node, PublishableSow)


@struct.dataclass
class ScalarSow(PublishableSow):
    value: chex.Array

    def publish(self, writer: flax_tb.SummaryWriter, tag: str, step: int) -> None:
        writer.scalar(tag, self.value, step=step)


@struct.dataclass
class ImageSow(PublishableSow):
    image: chex.Array

    def publish(self, writer: flax_tb.SummaryWriter, tag: str, step: int) -> None:
        writer.image(tag, self.image, step=step)


@struct.dataclass
class MplImageSow(PublishableSow):
    image: chex.Array
    h_paddings: Optional[chex.Array] = None
    v_paddings: Optional[chex.Array] = None
    cmap: Optional[str] = struct.field(pytree_node=False, default=None)
    interpolation: Optional[str] = struct.field(pytree_node=False, default=None)
    aspect: Optional[str] = struct.field(pytree_node=False, default=None)
    origin: Optional[str] = struct.field(pytree_node=False, default=None)
    with_colorbar: bool = struct.field(pytree_node=False, default=False)

    @property
    def trimmed_image(self):
        ret = self.image
        if self.v_paddings is not None:
            ret = np.delete(
                ret, [i for i, pad in enumerate(self.v_paddings) if pad > 0.5], axis=0
            )
        if self.h_paddings is not None:
            ret = np.delete(
                ret, [i for i, pad in enumerate(self.h_paddings) if pad > 0.5], axis=1
            )
        return ret

    def publish(self, writer: flax_tb.SummaryWriter, tag: str, step: int) -> None:
        matplotlib.use("agg")
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(
            self.trimmed_image,
            cmap=self.cmap,
            aspect=self.aspect,
            interpolation=self.interpolation,
            origin=self.origin,
        )
        if self.with_colorbar:
            fig.colorbar(im)
        fig.canvas.draw()

        imagedata = np.array(fig.canvas.renderer._renderer)
        plt.close(fig)
        imagedata = imagedata[::, ::, :3]  # delete alpha
        writer.image(tag, imagedata, step=step)


def publish_train_intermediates(
    writer: flax_tb.SummaryWriter,
    tree: _ArrayTree,
    step: int,
    *,
    prefix: str = "sow/",
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
    for path, val in flatten_with_paths(tree, is_leaf=_is_publishable_sow):
        if not _is_publishable_sow(val):
            continue
        val.publish(writer, prefix + path[1:], step=step)


_EvalResultsWriteFn = Callable[[EvalResults, _TrainState, flax_tb.SummaryWriter], None]


def make_eval_results_writer(
    summary_root_dir: Union[str, os.PathLike[str]],
    dataset_filter: Optional[set[str]] = None,
    method: Optional[_EvalResultsWriteFn] = None,
) -> Callable[[dict[str, EvalResults], _TrainState], None]:
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

    def _tb_writer(res: Dict[str, EvalResults], st: _TrainState):
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
