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
import logging
import os
import sys
from typing import Callable, Dict, Optional, Union

import chex
from etils import epath
import flax
from flax import struct
from flax.metrics import tensorboard as flax_tb
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .training import TrainState as BobbinTrainState
from .evaluation import EvalResults
from .var_util import flatten_with_paths
from .var_util import summarize_shape
from .var_util import total_dimensionality

_ArrayTree = chex.ArrayTree
_TrainState = flax.training.train_state.TrainState


class NullSummaryWriter:
    """Null-object counterpart of `flax.metrics.tensorboard.SummaryWriter`."""

    def close(self):
        pass

    def flush(self):
        pass

    def scalar(self, *args, **kwargs):
        pass

    def image(self, *args, **kwargs):
        pass

    def audio(self, *args, **kwargs):
        pass

    def histogram(self, *args, **kwargs):
        pass

    def text(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def hparams(self, *args, **kwargs):
        pass


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


def _format_argv():
    ret = ""
    for i, arg in enumerate(sys.argv):
        ret += f'- argv[{i}]: "{arg}"\n'
    return ret


def publish_trainer_env_info(
    writer: flax_tb.SummaryWriter,
    train_state: flax.training.TrainState,
    *,
    prefix: str = "trainer/diagnosis/",
    also_do_logging: bool = True,
    loglevel: int = logging.INFO,
) -> None:
    """Publishes environment information to Tensorboard "Text" section.

    Currently, this function publishes the following informations.

    - The numbers of parameters and extra variables in the model.
    - Shape information of the parameter tree.
    - `str(jax.local_devices())`
    - List of `sys.argv`s.

    Args:
      writer: An instance of `flax.metrics.tensorboard.SummaryWriter`.
      train_state: An instance of `flax.training.train_state.TrainState` that
        contains parameters.
      prefix: Prefix to tag names to be published.
      also_do_logging: If True (default), published text data is also logged.
      loglevel: Log level used when `also_do_logging == True`.
    """
    if (ndim := np.asarray(train_state.step).ndim) >= 1:
        if ndim > 1:
            raise ValueError("train_state is replicated more than twice.")
        train_state = flax.jax_utils.unreplicate(train_state)
    step = train_state.step

    def write_text(tag, s):
        writer.text(prefix + tag, s, step)
        if also_do_logging:
            logging.log(loglevel, "%s = %s", tag, s)

    num_params = total_dimensionality(train_state.params)
    s = f"Number of parameters = {num_params}"
    if isinstance(train_state, BobbinTrainState):
        num_extra_vars = total_dimensionality(train_state.extra_vars)
        s += f", number of extra variables = {num_extra_vars}"
    write_text(prefix + "total_num_params", s)

    param_shape_info = "```\n" + summarize_shape(train_state.params) + "```"
    write_text(prefix + "param_shape", param_shape_info)

    local_device_info = str(jax.local_devices())
    write_text(prefix + "local_device", local_device_info)

    cmdline = _format_argv()
    write_text(prefix + "trainer_argv", cmdline)

    process = f"Number of processes = {jax.process_count()}"
    write_text(prefix + "process", process)


_EvalResultsWriteFn = Callable[[EvalResults, _TrainState, flax_tb.SummaryWriter], None]


def make_eval_results_writer(
    summary_root_dir: Union[str, os.PathLike[str]],
    dataset_filter: Optional[set[str]] = None,
    method: Optional[_EvalResultsWriteFn] = None,
    write_from_all_processes: bool = False,
) -> Callable[[dict[str, EvalResults], _TrainState], None]:
    """Makes a function that publishes evaluation results from different datasets.

    Args:
        summary_root_dir: Root directory for summaries.
        dataset_filter: If set, only the results from datasets in `dataset_filter`
            will be processed.
        method: If set, use `method(results, train_state, writer)` to publish the
            evaluation results `results`. Otherwise, `results.write_to_tensorboard`
            will be used.
        write_from_all_processes: If False (default), only the first process of
            the distributed workers calls `write_to_tensorboard` (or `method`).

    Returns:
        A function `tb_writer(results, state)` where `results` is
        `Dict[str, EvalResults]` where `results[name]` represents the evaluation
        result for the dataset with `name`.  Instances of `EvalResults` passed
        to this function must override`EvalResults.write_to_tensorboard` method.
        `state` is a `TrainState`.
    """
    summary_root_dir = epath.Path(summary_root_dir)
    writers = dict()

    def _tb_writer(res: Dict[str, EvalResults], st: _TrainState):
        for name, result in res.items():
            if dataset_filter is not None and name not in dataset_filter:
                continue
            if name not in writers:
                should_write = jax.process_index() == 0 or write_from_all_processes
                writers[name] = (
                    flax_tb.SummaryWriter(summary_root_dir / name)
                    if should_write
                    else NullSummaryWriter()
                )
            if method is None:
                result.write_to_tensorboard(st, writers[name])
            else:
                method(result, st, writers[name])
            writers[name].flush()

    return _tb_writer
