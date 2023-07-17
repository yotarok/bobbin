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
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import flax
import flax.training.train_state
import jax
import numpy as np
import orbax.checkpoint

TrainState = flax.training.train_state.TrainState
Action = Callable[..., Optional[Tuple[TrainState, ...]]]


class Trigger(metaclass=abc.ABCMeta):
    """Base class for triggers."""

    def __or__(self, other: Trigger) -> OrTrigger:
        return OrTrigger(self, other)

    @abc.abstractmethod
    def check(self, train_state: TrainState) -> bool:
        ...


class OrTrigger(Trigger):
    """Combined trigger representing "A or B"."""

    _triggers: Sequence[Trigger]

    def __init__(self, *triggers):
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

    def __init__(self, ns: Union[int, Iterable[int]]):
        super().__init__()
        if isinstance(ns, int):
            ns = [ns]
        self._ns = set(ns)

    def check(self, train_state: TrainState) -> bool:
        return int(train_state.step) in self._ns


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
              - `step_interval=N` : do action for each N-steps.
              - `at_step=N` or `at_step=(N1, N2, ...)` : do action at N, N1, or
                N2 steps
              - `at_first_steps=N` : do action for the first N steps of training.
              - `at_first_steps_of_process=N` : do action for the first N steps
                since the current training process started.
              - `time_interval=X` : do action if X (`float`) seconds passed since
                the last action from this trigger is invoked.

        Returns:
          `self`. updated crontab.
        """
        triggers = []

        for argname, argvalue in kwargs.items():
            if argname not in _SCHEDULE_ARG_PARSER:
                raise TypeError(
                    "`CronTab.schedule` got an unexpected keyword argument"
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

    def run(
        self,
        train_state: TrainState,
        is_train_state_replicated: bool = True,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Checks triggers and run the corresponding actions if needed.

        Args:
          train_state: current training state.
          is_train_state_replicated: if True (by default), `train_state` will be
            unreplicated before given to actions.
          args, kwargs: extra parameters passed to the actions.
        """

        if is_train_state_replicated:
            train_state = flax.jax_utils.unreplicate(train_state)
        results = dict()
        for name, trig, act in self._actions:
            if trig.check(train_state):
                results[name] = act(train_state, *args, **kwargs)
        return results
