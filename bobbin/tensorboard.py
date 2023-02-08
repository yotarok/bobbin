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
import functools
import logging
import sys
from typing import Callable, Dict, Iterable, Optional, Tuple

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
from .evaluation import EvalResultProcessorFn
from .var_util import flatten_with_paths
from .var_util import summarize_shape
from .var_util import total_dimensionality

ArrayTree = chex.ArrayTree
TrainState = flax.training.train_state.TrainState

WriteEvalResultsFn = Callable[[EvalResults, TrainState, flax_tb.SummaryWriter], None]


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

    def as_result_processor(self, *args, **kwargs):
        def f(*args, **kwargs):
            pass

        return f


def _default_key_from_tag(tag: str) -> str:
    key, rest = tag.split("/", maxsplit=1)
    return key, rest


def _default_dirname_from_key(key: str) -> str:
    return key


class MultiDirectorySummaryWriter(flax_tb.SummaryWriter):
    """SummaryWriter that changes the destination depending on the tag.

    Note that this class is currently not exposed to users, and there's no
    actual usecases for wrapping functions like `scalar`.
    """

    _writers: Dict[str, flax_tb.SummaryWriter]
    _key_fn: Callable[[str], Tuple[str, str]]
    _dirname_fn: Callable[[str], str]

    def __init__(
        self,
        log_dir_root,
        *,
        keys: Iterable[str] = (),
        allow_new_keys: bool = True,
        only_from_leader_process: bool = True,
        auto_flush: bool = True,
        tag_to_key: Callable[[str], str] = _default_key_from_tag,
        dirname: Callable[[str], str] = _default_dirname_from_key,
    ):
        """Constructs `MultiDirectorySummaryWriter`.

        Args:
          log_dir_root: root directory for this `MultiDirectorySummaryWriter`.
            it can be anything that can be passed to `epath.Path`.
          keys: pre-specified set of keys.
          only_on_leader_process: if True, (by default), writers do not
            actually write summaries in the processes with
            `jax.process_index() != 0`.
          auto_flush: if True, sub-writers are instantiated with
            `auto_flush=True` argument.
          tag_to_key: a function that extracts subwriter names and tag names
            used in the subwriter from the tags. default is a function
            equivalent to `lambda s: s.split('/', maxsplit=1)`.
          dirname: a function that converts keys to the directory names under
            the root directory.
        """
        self._log_dir_root = epath.Path(log_dir_root)
        self._key_fn = tag_to_key
        self._dirname_fn = dirname
        self._auto_flush = auto_flush
        self._use_null = only_from_leader_process and jax.process_index() != 0

        self._writers = dict()
        # Instantiate writers for pre-specified keys
        self._allow_new_keys = True
        for key in keys:
            self.subwriter(key)
        self._allow_new_keys = False  # It is forbidden once

    def subwriter(self, key: str) -> flax_tb.SummaryWriter:
        """Returns summary writer corresponding to the specific key.

        This function creates an instance of SummaryWriter if it is not created
        yet.

        Args:
          key: name of the subwriter.

        Returns:
          an instance of `SummaryWriter`.
        """
        if key not in self._writers:
            if self._allow_new_keys:
                if self._use_null:
                    writer = NullSummaryWriter()
                else:
                    writer = flax_tb.SummaryWriter(
                        self._log_dir_root / self._dirname_fn(key),
                        auto_flush=self._auto_flush,
                    )
                self._writers[key] = writer
            else:
                raise ValueError(
                    "Subwriter for `MultiDirectorySummaryWriter` with "
                    f"key={key} is requested but this instance is configured "
                    "to have only the pre-specified keys "
                    f"{list(self._writers.keys())}"
                )
        return self._writers[key]

    @functools.wraps(flax_tb.SummaryWriter.close)
    def close(self):
        for unused_name, writer in self._writers.items():
            writer.close()

    @functools.wraps(flax_tb.SummaryWriter.flush)
    def flush(self):
        for unused_name, writer in self._writers.items():
            writer.flush()

    def _find_writer_by_tag(self, tag: str) -> flax_tb.SummaryWriter:
        key, tag = self._key_fn(tag)
        return self.subwriter(key), tag

    @functools.wraps(flax_tb.SummaryWriter.scalar)
    def scalar(self, tag, value, step):
        writer, tag = self._find_writer_by_tag(tag)
        writer.scalar(tag, value, step)

    @functools.wraps(flax_tb.SummaryWriter.image)
    def image(self, tag, image, step, max_outputs=3):
        writer, tag = self._find_writer_by_tag(tag)
        writer.image(tag, image, step, max_outputs=max_outputs)

    @functools.wraps(flax_tb.SummaryWriter.audio)
    def audio(self, tag, audiodata, step, sample_rate=44100, max_outputs=3):
        writer, tag = self._find_writer_by_tag(tag)
        writer.audio(
            tag, audiodata, step, sample_rate=sample_rate, max_outputs=max_outputs
        )

    @functools.wraps(flax_tb.SummaryWriter.histogram)
    def histogram(self, tag, values, step, bins=None):
        writer, tag = self._find_writer_by_tag(tag)
        writer.histogram(tag, values, step, bins=bins)

    @functools.wraps(flax_tb.SummaryWriter.text)
    def text(self, tag, textdata, step):
        writer, tag = self._find_writer_by_tag(tag)
        writer.text(tag, textdata, step)

    @functools.wraps(flax_tb.SummaryWriter.write)
    def write(self, tag, tensor, step, metadata=None):
        writer, tag = self._find_writer_by_tag(tag)
        writer.write(tag, tensor, step, metadata=metadata)

    @functools.wraps(flax_tb.SummaryWriter.hparams)
    def hparams(self, hparams):
        for unused_name, writer in self._writers.items():
            writer.hparams(hparams)

    def as_result_processor(
        self, method: Optional[WriteEvalResultsFn] = None
    ) -> EvalResultProcessorFn:
        """Returns the function that writes eval results to TensorBoard.

        This method is for connecting `MultiDirectorySummaryWriter` to
        `crontab.RunEval`.
        """

        def f(result_per_dataset: Dict[str, EvalResults], train_state):
            for dataset_name, result in result_per_dataset.items():
                writer = self.subwriter(dataset_name)
                if method is None:
                    result.write_to_tensorboard(train_state, writer)
                else:
                    method(result, train_state, writer)
                writer.flush()

        return f


class PublishableSummary(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def publish(self, writer: flax_tb.SummaryWriter, tag: str, step: int) -> None:
        ...


def _is_publishable_summary(node: ArrayTree) -> bool:
    return isinstance(node, PublishableSummary)


@struct.dataclass
class ScalarSummary(PublishableSummary):
    value: chex.Array

    def publish(self, writer: flax_tb.SummaryWriter, tag: str, step: int) -> None:
        writer.scalar(tag, self.value, step=step)


@struct.dataclass
class ImageSummary(PublishableSummary):
    image: chex.Array

    def publish(self, writer: flax_tb.SummaryWriter, tag: str, step: int) -> None:
        writer.image(tag, self.image, step=step)


@struct.dataclass
class MplImageSummary(PublishableSummary):
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
    tree: ArrayTree,
    step: int,
    *,
    prefix: str = "summary/",
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
    for path, val in flatten_with_paths(tree, is_leaf=_is_publishable_summary):
        if not _is_publishable_summary(val):
            continue
        val.publish(writer, prefix + path[1:], step=step)


def _format_argv():
    ret = ""
    for i, arg in enumerate(sys.argv):
        ret += f'- argv[{i}]: "{arg}"\n'
    return ret


def publish_trainer_env_info(
    writer: flax_tb.SummaryWriter,
    train_state: TrainState,
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
