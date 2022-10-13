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
import json
import os
import pathlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from flax import struct
from flax.training import checkpoints
import jax.numpy as jnp
import numpy as np

from .training import StepInfo
from .training import TrainState
from .evaluation import EvalResults
from .evaluation import eval_datasets

Action = Callable[[TrainState, ...], Optional[Tuple[TrainState, ...]]]


class Trigger(metaclass=abc.ABCMeta):
    """Base class for triggers."""

    def __or__(self, other: Trigger) -> OrTrigger:
        return OrTrigger(self, other)


class OrTrigger(Trigger):
    """Combined trigger representing "A or B"."""

    def __init__(self, *triggers: List[Trigger]):
        super().__init__()
        self._triggers = triggers

    def check(self, train_state: TrainState) -> bool:
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

    def check(self, train_state: TrainState) -> bool:
        return train_state.step % self._n == 0


class ForEachTSeconds(Trigger):
    """Trigger that fires if `T` seconds passed after the last fire."""

    def __init__(self, secs: float, fire_at_zero: bool = True):
        super().__init__()
        self._period_secs = secs
        self._last_fired = -secs if fire_at_zero else time.time()

    def check(self, train_state: TrainState) -> bool:
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

    def check(self, train_state: TrainState) -> bool:
        if self._process_wise:
            if self._process_first_step is None:
                self._process_first_step = train_state.step
            return (train_state.step - self._process_first_step) < (self._n - 1)
        else:
            return train_state.step <= self._n


class AtNthStep(Trigger):
    """Trigger that fires at `N`-th step."""

    def __init__(self, n: int):
        super().__init__()
        self._n = n

    def check(self, train_state: TrainState) -> bool:
        return train_state.step == self._n


class RunEval:
    """Action that runs evaluation processes."""

    def __init__(self, eval_task, eval_batch_gens):
        self._eval_task = eval_task
        self._eval_batch_gens = eval_batch_gens
        self._result_processors = []

    def __call__(self, train_state: TrainState, **unused_kwargs):
        eval_results = eval_datasets(
            self._eval_task, self._eval_batch_gens, train_state.model_vars
        )
        for dsname, results in eval_results.items():
            print(f"Step={train_state.step}, Dataset={dsname}")
            print(results.to_log_message())

        for proc in self._result_processors:
            proc(eval_results, train_state)

        return eval_results

    def and_keep_best_checkpoint(self, tune_on: str, dest_path: str):
        return RunEvalKeepBest(self, tune_on, dest_path)

    def add_result_processor(
        self, f: Callable[[Dict[str, EvalResults], TrainState], Any]
    ):
        self._result_processors.append(f)
        return self


def _json_object_hook_for_arrays(d: Dict[str, Any]) -> Any:
    if "__array__" in d and d["__array__"]:
        dtype = np.dtype(d.get("dtype", "float32"))
        return np.array(d["data"], dtype=dtype)
    return d


class _ArrayEncoder(json.JSONEncoder):
    """Internal JSON encoder that supports array encoding."""

    def default(self, obj: Any):
        if isinstance(obj, np.ndarray) or isinstance(obj, jnp.DeviceArray):
            if obj.shape == ():
                # Scalar is serialized as normal scalar
                return obj.tolist()
            return dict(
                __array__=True,
                dtype=obj.dtype.name,
                data=obj.to_list(),
            )
        return super().default(obj)


def _try_deserialize_eval_results(
    path: os.PathLike, template: EvalResults
) -> Optional[EvalResults]:
    try:
        result_json = pathlib.Path(path).read_text()
        state_dict = json.loads(result_json, object_hook=_json_object_hook_for_arrays)
        ret = template.load_state_dict(state_dict)
        return ret
    except FileNotFoundError:
        pass
    return None


@struct.dataclass
class RunEvalKeepBestResult:
    """Data class for return values of `RunEvalKeepBest` action."""

    eval_results: Dict[str, EvalResults]
    current_best: Optional[EvalResults]
    saved_train_state: Optional[TrainState]


class RunEvalKeepBest:
    """Action that runs evaluation and saves the best checkpoint."""

    def __init__(self, run_eval_action: Action, tune_on: str, dest_path: os.PathLike):
        self._run_eval_action = run_eval_action
        self._tune_on = tune_on
        self._dest_path = dest_path
        self._current_best = None
        self._results_path = pathlib.Path(self._dest_path) / "results.json"

    def __call__(self, train_state, **kwargs):
        eval_results = self._run_eval_action(train_state, **kwargs)

        result = eval_results[self._tune_on]
        saved_train_state = None
        if self._current_best is None:
            # Try loading
            self._current_best = _try_deserialize_eval_results(
                self._results_path, result
            )

        if self._current_best is None or result.is_better_than(self._current_best):
            self._current_best = result

            checkpoints.save_checkpoint(
                self._dest_path, train_state, train_state.step, overwrite=True
            )
            results_json = json.dumps(result.to_state_dict(), cls=_ArrayEncoder)
            pathlib.Path(self._results_path).write_text(results_json)
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

    def __call__(self, train_state: TrainState, **unused_kwargs):
        checkpoints.save_checkpoint(self._dest_path, train_state, train_state.step)


class WriteLog:
    """Action that outputs logs regarding training state."""

    def __call__(
        self, train_state: TrainState, *, step_info: StepInfo, **unused_kwargs
    ):
        print(f"Step={train_state.step}, loss={step_info.loss}")
        print(train_state.extra_vars)


class CronTab:
    """Table that contains pairs of triggers and actions."""

    def __init__(self):
        self._actions = []

    def add(self, name: str, trigger: Trigger, action: Action) -> CronTab:
        self._actions.append((name, trigger, action))
        return self

    def run(self, train_state: TrainState, *args, **kwargs) -> Dict[str, Any]:
        results = dict()
        for name, trig, act in self._actions:
            if trig.check(train_state):
                results[name] = act(train_state, *args, **kwargs)
        return results
