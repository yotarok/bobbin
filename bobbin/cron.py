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

"""Mechanism for triggering shcheduled actions in a training loop."""

from __future__ import annotations

import abc
import functools
import logging
import os
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from etils import epath
import flax
from flax import struct
from flax.metrics import tensorboard as flax_tb
from flax.training import checkpoints
import jax

from .pytypes import Batch
from .tensorboard import publish_train_intermediates
from .tensorboard import MultiDirectorySummaryWriter
from .training import TrainState as BobbinTrainState
from .training import StepInfo
from .evaluation import EvalResults
from .evaluation import EvalResultProcessorFn
from .evaluation import EvalTask
from .evaluation import eval_datasets
from .var_util import read_pytree_json_file
from .var_util import write_pytree_json_file

_TrainState = flax.training.train_state.TrainState

Action = Callable[..., Optional[Tuple[_TrainState, ...]]]


class Trigger(metaclass=abc.ABCMeta):
    """Base class for triggers."""

    def __or__(self, other: Trigger) -> OrTrigger:
        return OrTrigger(self, other)

    @abc.abstractmethod
    def check(self, train_state: _TrainState) -> bool:
        ...


class OrTrigger(Trigger):
    """Combined trigger representing "A or B"."""

    _triggers: Sequence[Trigger]

    def __init__(self, *triggers):
        super().__init__()
        self._triggers = triggers

    def check(self, train_state: _TrainState) -> bool:
        return any(trig.check(train_state) for trig in self._triggers)

    def __or__(self, other: Trigger) -> OrTrigger:
        # Custom definition for simplifying chained or-triggers.
        triggers = [trig for trig in self._triggers]
        triggers.append(other)
        return OrTrigger(*triggers)


class ForEachNSteps(Trigger):
    """Trigger that fires for each `N` steps."""

    def __init__(self, n: int):
        super().__init__()
        self._n = n

    def check(self, train_state: _TrainState) -> bool:
        return train_state.step % self._n == 0


class ForEachTSeconds(Trigger):
    """Trigger that fires if `T` seconds passed after the last fire."""

    def __init__(self, secs: float, fire_at_zero: bool = True):
        super().__init__()
        self._period_secs = secs
        self._last_fired = -secs if fire_at_zero else time.time()

    def check(self, train_state: _TrainState) -> bool:
        cur = time.time()
        fire = cur - self._last_fired > self._period_secs
        if fire:
            self._last_fired = cur
        return fire


class AtFirstNSteps(Trigger):
    """Trigger that fires at first `N` steps of a loop, or a process."""

    def __init__(self, n: int, of_process: bool = False):
        super().__init__()
        self._process_wise = of_process
        self._process_first_step = None
        self._n = n

    def check(self, train_state: _TrainState) -> bool:
        if self._process_wise:
            if self._process_first_step is None:
                self._process_first_step = train_state.step
            return (train_state.step - self._process_first_step) < (self._n - 1)
        else:
            return train_state.step <= self._n


class AtNthStep(Trigger):
    """Trigger that fires at `N`-th step."""

    def __init__(self, ns: Union[int, Iterable[int]]):
        super().__init__()
        if isinstance(ns, int):
            ns = [ns]
        self._ns = set(ns)

    def check(self, train_state: _TrainState) -> bool:
        return int(train_state.step) in self._ns


class RunEval:
    """Action that runs evaluation processes."""

    def __init__(
        self,
        eval_task: EvalTask,
        eval_batch_gens: Mapping[str, Iterator[Batch]],
        tensorboard_root_path: Union[None, str, os.PathLike[str]] = None,
    ):
        self._eval_task = eval_task
        self._eval_batch_gens = eval_batch_gens
        self._result_processors = []

        if tensorboard_root_path is not None:
            writer = MultiDirectorySummaryWriter(
                tensorboard_root_path, keys=eval_batch_gens.keys()
            )
            self.add_result_processor(writer.as_result_processor())

    def __call__(self, train_state: _TrainState, **unused_kwargs):
        if not isinstance(train_state, BobbinTrainState):
            raise ValueError("`RunEval` action must be used with `bobbin.TrainState`")

        eval_results = eval_datasets(
            self._eval_task, self._eval_batch_gens, train_state.model_vars
        )
        for dsname, results in eval_results.items():
            logging.info(
                "Evaluation results for dataset=%s @step=%d\n%s",
                dsname,
                train_state.step,
                results.to_log_message(),
            )

        for proc in self._result_processors:
            proc(eval_results, train_state)

        return eval_results

    def and_keep_best_checkpoint(self, tune_on: str, dest_path: str):
        return RunEvalKeepBest(self, tune_on, dest_path)

    def add_result_processor(self, f: EvalResultProcessorFn):
        self._result_processors.append(f)
        return self


def _try_deserialize_eval_results(
    path: os.PathLike, template: EvalResults
) -> Optional[EvalResults]:
    try:
        return read_pytree_json_file(path, template)
    except Exception:
        pass
    return None


@struct.dataclass
class RunEvalKeepBestResult:
    """Data class for return values of `RunEvalKeepBest` action."""

    eval_results: dict[str, EvalResults]
    current_best: Optional[EvalResults]
    saved_train_state: Optional[_TrainState]


class RunEvalKeepBest:
    """Action that runs evaluation and saves the best checkpoint."""

    def __init__(
        self, run_eval_action: Action, tune_on: str, dest_path: Union[str, os.PathLike]
    ):
        self._run_eval_action = run_eval_action
        self._tune_on = tune_on
        self._dest_path = epath.Path(dest_path)
        self._current_best = None
        self._results_path = self._dest_path / "results.json"

    def __call__(self, train_state, **kwargs):
        eval_results = self._run_eval_action(train_state, **kwargs)

        result = eval_results[self._tune_on]
        saved_train_state = None
        if self._current_best is None and jax.process_index() == 0:
            # Try loading
            self._current_best = _try_deserialize_eval_results(
                self._results_path, result
            )

        if self._current_best is None or result.is_better_than(self._current_best):
            self._current_best = result

            if jax.process_index() == 0:
                checkpoints.save_checkpoint(
                    self._dest_path, train_state, train_state.step, overwrite=True
                )
                write_pytree_json_file(self._results_path, result)
            saved_train_state = train_state
        return RunEvalKeepBestResult(
            eval_results=eval_results,
            current_best=self._current_best,
            saved_train_state=saved_train_state,
        )


class SaveCheckpoint:
    """Action that saves checkpoint."""

    def __init__(self, dest_path: str):
        self._dest_path = dest_path

    def __call__(self, train_state: _TrainState, **unused_kwargs):
        checkpoints.save_checkpoint(self._dest_path, train_state, train_state.step)


class WriteLog:
    """Action that outputs logs regarding training state."""

    def __init__(
        self, *, level: int = logging.INFO, logger: Optional[logging.Logger] = None
    ):
        self.logger = logging.root if logger is None else logger
        self.level = level

    def __call__(
        self, train_state: _TrainState, *, step_info: StepInfo, **unused_kwargs
    ):
        self.logger.log(
            self.level,
            "Training state @step=%d loss=%s",
            int(train_state.step),
            str(step_info.loss),
        )


class PublishTrainingProgress:
    """Action that publishes training SoWs and training throughput."""

    def __init__(
        self,
        writer: flax_tb.SummaryWriter,
        summary_collections: Sequence[str] = ("tensorboard",),
    ):
        self.writer = writer
        self.summary_collections = summary_collections
        self.last_fired_time = None
        self.last_fired_step = None

    def __call__(self, train_state: _TrainState, **unused_kwargs):
        if not isinstance(train_state, BobbinTrainState):
            raise ValueError(
                "`PublishTrainingProgress` action must be used with `bobbin.TrainState`"
            )
        cur_time = time.time()
        if self.last_fired_time is not None and self.last_fired_step is not None:
            wall_time = cur_time - self.last_fired_time
            nsteps = train_state.step - self.last_fired_step
            steps_per_sec = nsteps / wall_time
            self.writer.scalar(
                "trainer/steps_per_sec", steps_per_sec, step=train_state.step
            )

        for colname in self.summary_collections:
            if colname not in train_state.extra_vars:
                continue
            publish_train_intermediates(
                self.writer, train_state.extra_vars[colname], train_state.step
            )
        self.last_fired_time = cur_time
        self.last_fired_step = train_state.step


_SCHEDULE_ARG_PARSER = {
    "step_interval": ForEachNSteps,
    "at_step": AtNthStep,
    "at_first_steps": functools.partial(AtFirstNSteps, of_process=False),
    "at_first_steps_of_process": functools.partial(AtFirstNSteps, of_process=True),
    "time_interval": ForEachTSeconds,
}


class CronTab:
    """Table that contains pairs of triggers and actions."""

    def __init__(self):
        self._actions = []

    def schedule(
        self,
        action: Action,
        *,
        name: Optional[str] = None,
        **kwargs,
    ) -> CronTab:
        """Schedules an action for the trigger specified via `kwargs`.

        Args:
          action: a function `f(train_state, ...)` that takes `TrainState` and
            additional arguments specified in `CronTab.run`.
          name: name for the registered action-trigger pair.
          kwargs:
            the following trigger specifiers are currently supported:
              - `step_interval=N`: do action for each N-steps.
              - `at_step=N` or `at_step=(N1, N2, ...)`: do action at N, N1, or
                N2 steps
              - `at_first_steps=N`: do action for the first N steps of training.
              - `at_first_steps_of_process=N`: do action for the first N steps
                since the current training process started.
              - `time_interval=X`: do action if X (`float`) seconds passed since
                the last action from this trigger is invoked.

        Returns:
          `self`. updated crontab.
        """
        triggers = []

        for argname, argvalue in kwargs.items():
            if argname not in _SCHEDULE_ARG_PARSER:
                raise TypeError(
                    "`CronTag.schedule` got an unexpected keyword argument"
                    f"'{argname}'"
                )
            triggers.append(_SCHEDULE_ARG_PARSER[argname](argvalue))

        trigger = None
        if len(triggers) == 0:
            raise ValueError(
                "None of timing arguments is specified in `CronTab.schedule`. "
                "Specify at least one of: step_interval, at_step, at_first_steps, "
                "at_first_steps_of_process, time_interval."
            )
        elif len(triggers) == 1:
            trigger = triggers[0]
        else:
            trigger = OrTrigger(*triggers)

        if name is None:
            name = repr(action)
        return self.add(name, trigger, action)

    def add(self, name: str, trigger: Trigger, action: Action) -> CronTab:
        """Adds a trigger-action pair."""
        self._actions.append((name, trigger, action))
        return self

    def run(self, train_state: _TrainState, *args, **kwargs) -> Dict[str, Any]:
        """Checks triggers and run the corresponding actions if needed.

        Args:
          train_state: current training state.
          *args, **kwargs: extra parameters passed to the actions.
        """
        results = dict()
        for name, trig, act in self._actions:
            if trig.check(train_state):
                results[name] = act(train_state, *args, **kwargs)
        return results
