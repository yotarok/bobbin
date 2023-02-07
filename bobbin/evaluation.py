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

"""Evaluation utilities.
"""

from __future__ import annotations

import collections
import logging
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, Tuple, TypeVar

from flax import struct
from flax.metrics import tensorboard as flax_tb
import jax
import numpy as np

from .pmap_util import gather_from_jax_processes
from .pmap_util import unshard
from .pytypes import Batch
from .pytypes import BatchGen
from .training import TrainState


T = TypeVar("T")


@struct.dataclass
class EvalResults:
    """Evaluation results."""

    def to_log_message(self):
        """Format `EvalResult` to a logging-friendly string."""
        return str(self)

    def reduce(self, other: EvalResults) -> EvalResults:
        """Combines two `EvalResult`s."""
        raise NotImplementedError()

    def unshard_and_reduce(self) -> EvalResults:
        """Merges sharded results (typically obtained via `pmap`) by `reduce`."""
        ret, *tail = unshard(self)
        for y in tail:
            ret = ret.reduce(y)
        return ret

    def is_better_than(self, other: EvalResults) -> bool:
        """Returns True if `self` is better than `other`.

        This function is used for keeping track of the best checkpoint. If such
        comparison is not required, keep this unimplemented.

        Args:
            other: Another instance of the same EvalResults class.

        Returns:
            A boolean value.
        """
        raise NotImplementedError(
            "For comparing results, implement `EvalResults.is_better_than`"
        )

    def write_to_tensorboard(
        self,
        current_train_state: TrainState,
        writer: flax_tb.SummaryWriter,
    ):
        """Writes summary of this `EvalResults` to `writer`.

        This function is a default method for `.tensorboard.make_eval_results_writer`.
        If `make_eval_results_writer` isn't used, or the custom method is specified
        in this function, you can keep this unimplemented.

        Args:
            current_train_state: `TrainState` used to obtain this `EvalResults`.
            writer: destination `SummaryWriter`.
        """
        raise NotImplementedError(
            "`EvalResults` is instructed to publish to TensorBoard."
            " However, `EvalResults.write_to_tensorboard` is not overridden"
            f" in the subclass ({type(self)})"
        )


EvalResultProcessorFn = Callable[[Dict[str, EvalResults], TrainState], Any]


class EvalTask:
    """Base class defining evaluation task."""

    def create_eval_results(self, dataset_name: str) -> EvalResults:
        """Initializes evaluation result."""
        raise NotImplementedError()

    def evaluate(self, batch: Batch, *args, **kwargs) -> EvalResults:
        """Evaluate single batch and returns `EvalResults`."""
        raise NotImplementedError()

    def finalize_eval_results(self, metrics: EvalResults) -> EvalResults:
        """
        Finalize eval metrics before it is stored or published to tensorboard.
        """
        return metrics


def eval_batches(
    eval_task: EvalTask, dataset_name: str, batches: Iterable[Batch], *args, **kwargs
) -> EvalResults:
    """Evaluates the batches provided by the given iterator.

    Args:
      eval_task: `EvalTask` object for initializing evaluation results and
        running evaluation.
      dataset_name: the name of dataset to be evaluated over. this will be
        passed to `EvalTask.create_eval_results`.
      batch_gens: A dictionary containing pairs of dataset names and a nullary
        function that returns iterable of batches.
      *args, **kwargs: additional variable such as model variables. Those are
        verbatimly passed to `EvalTask.evaluate`.

    Returns:
      `EvalResult` for the given batches.
    """
    metrics = eval_task.create_eval_results(dataset_name)
    batches = iter(batches)

    multi_process = jax.process_count() > 1
    while True:
        active = True
        try:
            batch = next(batches)
        except StopIteration:
            active = False
            batch = None

        if multi_process:
            # If multi_process, check if another process is still active.
            if not np.asarray(gather_from_jax_processes(active)).any():
                break
        elif not active:  # Otherwise, just break the loop.
            break

        # This must be called even when the host is inactive for synchronized
        # parallel processing.
        new_result = eval_task.evaluate(batch, *args, **kwargs)
        if active:
            if not isinstance(new_result, type(metrics)):
                raise TypeError(
                    f"`eval_task.evaluate` returned {type(new_result)}. It must be"
                    " compatible with the return type of"
                    f" `eval_task.create_eval_results ({type(metrics)})."
                )
            metrics = metrics.reduce(new_result)

    # Reduce over parallel replicas
    if multi_process:
        all_metrics = gather_from_jax_processes(metrics)
        for process_id, other_metrics in enumerate(all_metrics):
            if process_id == jax.process_index():
                continue
            metrics = metrics.reduce(other_metrics)

    metrics = eval_task.finalize_eval_results(metrics)
    return metrics


def eval_datasets(
    eval_task: EvalTask, batch_gens: Dict[str, BatchGen], *args, **kwargs
) -> Dict[str, EvalResults]:
    """Evaluates the batches provided from multiple datasets.

    Args:
      eval_task: `EvalTask` object for initializing evaluation results and
        running evaluation.
      batch_gens: A dictionary containing pairs of dataset names and a nullary
        function that returns iterable of batches.
      *args, **kwargs: additional variable such as model variables. Those are
        verbatimly passed to `EvalTask.evaluate`.

    Returns:
      A dictionary containing pairs of the dataset names and `EvalResults`.
    """
    results = dict()
    for dsname, batch_gen in batch_gens.items():
        logging.info("Start evaluation process over %s", dsname)
        results[dsname] = eval_batches(
            eval_task,
            dsname,
            batch_gen(),
            *args,
            **kwargs,
        )
    return results


@struct.dataclass
class SampledSet(collections.abc.Collection, Generic[T]):
    """Immutable set containing the fixed number samples from the elements added."""

    max_size: int
    values: Tuple[T] = ()
    priorities: Tuple[float] = ()

    def __contains__(self, q: T) -> bool:
        return q in self.values

    def __iter__(self) -> Iterator[T]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def _draw_priority(self, x: T) -> float:
        del x
        return np.random.uniform()

    def add(self, x: T) -> SampledSet[T]:
        """Returns `SampledSet` with the given element `x` added to this set."""
        pairs = list(zip(self.values, self.priorities))
        pairs.append((x, self._draw_priority(x)))
        pairs.sort(key=lambda x: x[1])
        nvalues, npriorities = zip(*pairs[: self.max_size])
        return self.replace(values=tuple(nvalues), priorities=tuple(npriorities))

    def union(self, iterable: Iterable[T]) -> SampledSet[T]:
        """Returns the union of this set and the given set."""
        pairs = list(zip(self.values, self.priorities))
        if isinstance(iterable, SampledSet):
            other_pairs = list(zip(iterable.values, iterable.priorities))
        else:
            other_pairs = [(elem, self._draw_priority(elem)) for elem in iterable]
        pairs.extend(other_pairs)
        pairs.sort(key=lambda x: x[1])

        if len(pairs) == 0:
            nvalues = []
            npriorities = []
        else:
            nvalues, npriorities = zip(*pairs[: self.max_size])

        return self.replace(values=tuple(nvalues), priorities=tuple(npriorities))
