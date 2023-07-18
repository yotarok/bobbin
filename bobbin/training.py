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

"""Utilities for supporting implementation of training loop."""

from __future__ import annotations


import dataclasses
import functools
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import chex
from flax import linen as nn
from flax import struct
import flax.training.train_state
import flax.metrics.tensorboard as flax_tb
import jax
import jax.numpy as jnp
import logging
import numpy as np
import optax
import orbax.checkpoint
import time

from .pmap_util import tpmap

from .tensorboard import publish_train_intermediates

from .pytypes import Batch
from .pytypes import Parameter
from .pytypes import VarCollection


Action = Callable
ArrayTree = chex.ArrayTree
Scalar = chex.Scalar
PRNGKey = chex.PRNGKey


if TYPE_CHECKING:
    nn = Any  # noqa: F811


class TrainState(flax.training.train_state.TrainState):
    """Thin-wrapper to `flax.training.train_state.TrainState`.

    This class is introduced to accommodate `extra_vars` field for handling
    mutable (non-trainable) variables.
    """

    extra_vars: VarCollection

    @property
    def model_vars(self):
        """Returns model variable by merging parameters and extra_vars."""
        ret = self.extra_vars.copy()
        ret["params"] = self.params
        return ret

    def is_replicated_for_pmap(self):
        """Check if `TrainState` is replicated for `pmap`."""
        return jnp.array(self.step).ndim >= 1


class StepInfo(struct.PyTreeNode):
    """Output of each step containing loss and aux output of the loss function.

    Attributes:
      loss: Loss value computed in the step.
      loss_aux_out: Auxiliary output from the loss function.
    """

    loss: Scalar
    loss_aux_out: Any


TrainingStepFn = Callable[[TrainState, Batch, PRNGKey], Tuple[TrainState, StepInfo]]
LossFnSideOutput = Tuple[VarCollection, ArrayTree]  # mutated_vars and loss_aux
LossFnResult = Tuple[Scalar, LossFnSideOutput]
ValueAndGradResult = Tuple[LossFnResult, ArrayTree]
ValueAndGradFn = Callable[[Parameter, Batch], ValueAndGradResult]


@dataclasses.dataclass(frozen=True)
class TrainingStepFnBuilder(TrainingStepFn):
    """Lazy builder for training step function."""

    compute_loss_fn: Callable
    reduce_extra_vars_fn: Callable
    do_jit: bool = False
    static_argnums: Tuple[int, ...] = ()
    backend: Optional[str] = None
    devices: Optional[List[chex.Device]] = None
    donate_argnums: Tuple[int, ...] = ()
    pmap_axis_name: Optional[Any] = None

    def make_base_step_fn(self):
        def train_step_fn(
            train_state: TrainState,
            batch: Batch,
            prng_key: PRNGKey,
        ) -> Tuple[TrainState, StepInfo]:
            if self.pmap_axis_name is not None:
                try:
                    jax.lax.axis_index(self.pmap_axis_name)
                except NameError:
                    raise ValueError(
                        "training step function is configured to use "
                        f"pmap_axis_name={self.pmap_axis_name}, but is called "
                        "outside of a pmap context. If you do not intend to "
                        "perform multi-device parallelization, do not call "
                        "`TrainingStepFnBuilder.pmap`"
                    )

            value_and_grad = jax.value_and_grad(self.compute_loss_fn, has_aux=True)
            (loss, (mutated_vars, loss_aux)), grads = value_and_grad(
                train_state.params,
                batch,
                extra_vars=train_state.extra_vars,
                prng_key=prng_key,
                step=train_state.step,
            )

            if self.pmap_axis_name is not None:
                grads = jax.lax.pmean(grads, axis_name=self.pmap_axis_name)
            train_state = train_state.apply_gradients(grads=grads)
            updated_extra_vars = train_state.extra_vars.copy()
            for colname, tree in mutated_vars.items():
                if colname == "params":
                    continue

                if self.pmap_axis_name is not None:
                    updated_extra_vars[colname] = self.reduce_extra_vars_fn(
                        colname, tree, axis_name=self.pmap_axis_name
                    )

            train_state = train_state.replace(extra_vars=updated_extra_vars)

            return train_state, StepInfo(loss=loss, loss_aux_out=loss_aux)

        return train_step_fn

    def pmap(self, axis_name: Any) -> TrainingStepFnBuilder:
        if axis_name is None:
            raise ValueError("axis_name for `pmap` cannot be None.")
        return dataclasses.replace(self, pmap_axis_name=axis_name)

    def jit(self) -> TrainingStepFnBuilder:
        return dataclasses.replace(self, do_jit=True)

    @property
    @functools.lru_cache(maxsize=None)
    def compiled_fn(self):
        f = self.make_base_step_fn()
        if self.pmap_axis_name is not None:
            # pmap implies jit, so we don't perform JIT explicitly when pmap
            f = tpmap(
                f,
                axis_name=self.pmap_axis_name,
                argtypes=["thru", "shard", "rng"],
                devices=self.devices,
                backend=self.backend,
                donate_argnums=self.donate_argnums,
            )
        elif self.do_jit:
            device = None
            if self.devices:
                device = self.devices[0]
            f = jax.jit(
                f,
                static_argnums=self.static_argnums,
                backend=self.backend,
                device=device,
                donate_argnums=self.donate_argnums,
            )

        return f

    def __call__(self, *args, **kwargs):
        return self.compiled_fn(*args, **kwargs)


class PublishTrainingProgress:
    """Action that publishes training intermediates and training throughput."""

    def __init__(
        self,
        writer: flax_tb.SummaryWriter,
        summary_collections: Iterable[str] = ("tensorboard",),
    ):
        self.writer = writer
        self.summary_collections = tuple(summary_collections)
        self.last_fired_time = None
        self.last_fired_step = None

    def __call__(
        self, train_state: flax.training.train_state.TrainState, **unused_kwargs
    ):
        if not isinstance(train_state, TrainState):
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


class BaseTrainTask:
    """Base class defining training task."""

    def compute_loss(
        self,
        params: Parameter,
        batch: Batch,
        *,
        extra_vars: VarCollection,
        prng_key: PRNGKey,
        step: Scalar,
    ) -> LossFnResult:
        """Abstract method to be overridden for defining the loss function."""
        raise NotImplementedError()

    def make_training_step_fn(
        self,
    ) -> TrainingStepFnBuilder:
        """Creates training step function."""
        return TrainingStepFnBuilder(
            compute_loss_fn=self.compute_loss,
            reduce_extra_vars_fn=self.reduce_extra_vars,
        )

    def reduce_extra_vars(
        self, colname: str, tree: ArrayTree, *, axis_name: str
    ) -> ArrayTree:
        """Abstract method to be overridden for sync non-parameter variables."""
        return tree

    def write_trainer_log(
        self,
        train_state: flax.training.train_state.TrainState,
        *,
        step_info: StepInfo,
        logger: logging.Logger,
        loglevel: int,
    ):
        """Abstract method to be overridden for custom logging output."""
        # Note that mean can also work on scalars.
        mean_loss = np.mean(step_info.loss)
        logger.log(
            loglevel,
            "@step=%d, loss=%f",
            int(train_state.step),
            mean_loss,
        )

    def make_log_writer(
        self, *, logger: Optional[logging.Logger] = None, loglevel: int = logging.INFO
    ) -> Action:
        """Makes logging action that can be registered in `CronTab`."""
        if logger is None:
            logger = logging.root

        def write(train_state, *, step_info=None, **unused_kwargs):
            self.write_trainer_log(
                train_state, step_info=step_info, logger=logger, loglevel=loglevel
            )

        return write

    def make_training_progress_publisher(
        self,
        writer: flax_tb.SummaryWriter,
        summary_collections: Iterable[str] = ("tensorboard",),
    ) -> PublishTrainingProgress:
        return PublishTrainingProgress(writer, summary_collections)


class TrainTask(BaseTrainTask):
    """Task definition for training of parameters of `nn.Module`."""

    def __init__(
        self,
        model: nn.Module,
        example_args: Iterable[Any],
        example_kwargs: Optional[Mapping[str, Any]] = None,
        required_rngs: Iterable[str] = (),
    ):
        """Constructs the train task.

        Args:
          model: flax model to be trained.
          required_rngs: the sequence of RNG names required for training.
            the values provided here will be used in `TrainTask.get_rng_dict`
            for simplifying RNG handling in for example `compute_loss`.
        """
        super().__init__()
        self._model = model
        self._required_rngs = tuple(required_rngs)
        self._example_args = tuple(example_args)
        self._example_kwargs = dict() if example_kwargs is None else example_kwargs

    @property
    def model(self) -> nn.Module:
        """Returns `model` given in the constructor."""
        return self._model

    def get_rng_dict(
        self, rng_key: PRNGKey, extra_keys: Iterable[str] = ()
    ) -> Dict[str, PRNGKey]:
        """
        Splits `rng_key` and returns rngs for each key specified in constructor.

        Args:
          rng_key: base RNG seed to be split.
          extra_keys: if set, additional RNG seeds are generated and stored to
            the return value with the provided keys.
        """
        keys = self._required_rngs + tuple(extra_keys)
        rngs = jax.random.split(rng_key, len(keys))
        return {key: rng for key, rng in zip(keys, rngs)}

    def _initialize_vars(
        self, rng_key: PRNGKey, compile_init: Optional[Callable] = None
    ) -> VarCollection:
        """Initializes the model variables."""
        if compile_init is None:
            init_fn = self.model.init
        else:
            init_fn = compile_init(self.model.init)

        return init_fn(
            self.get_rng_dict(rng_key, ("params",)),
            *self._example_args,
            **self._example_kwargs,
        )

    def initialize_train_state(
        self,
        rng: PRNGKey,
        tx: optax.GradientTransformation,
        checkpoint_path: Union[str, bytes, os.PathLike, None] = None,
        compile_init: Optional[Callable] = None,
    ) -> TrainState:
        """Initializes `TrainState` for this task.

        Args:
          rng: RNG seed for variable initialization.
          tx: optax gradient transformer attached to `TrainState`.
          checkpoint_path: if set, this method first tries to deserialize from
            a checkpoint in the checkpoint_path.
          compile_init: if set, `self.model.init` is wrapped by the given
            function as `compile_init(self.model.init)`.

        Returns:
          initialized train state.
        """
        global initialize_train_state
        init_model_vars = self._initialize_vars(rng, compile_init=compile_init)
        return initialize_train_state(
            apply_fn=self.compute_loss,
            init_model_vars=init_model_vars,
            tx=tx,
            checkpoint_path=checkpoint_path,
        )


def initialize_train_state(
    apply_fn: Optional[Callable],
    init_model_vars: VarCollection,
    tx: optax.GradientTransformation,
    checkpoint_path: Union[str, bytes, os.PathLike, None] = None,
) -> TrainState:
    """Initializes `TrainState` by randomly or by loading from a checkpoint."""
    new_state = TrainState.create(
        apply_fn=apply_fn,
        params=init_model_vars["params"],
        tx=tx,
        extra_vars={k: v for k, v in init_model_vars.items() if k != "params"},
    )

    if checkpoint_path is not None:
        cpman = orbax.checkpoint.CheckpointManager(
            checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
        )
        if (latest := cpman.latest_step()) is not None:
            new_state = cpman.restore(latest, items=new_state)

    return new_state


def make_checkpointing_action(
    checkpoint_manager: orbax.checkpoint.CheckpointManager,
    experimental_use_distributed_mode: bool = False,
) -> Action:
    """Makes checkpointing action for invoking orbax checkpoint manager.

    Args:
        checkpoint_manager: Checkpoint manager
        experimental_use_distributed_mode: If False (by default), the array
            is gathered to a leader host and stored as numpy array, i.e. no
            multihost checkpointing will be used.
    """

    def invoke_checkpointer(train_state, *args, **kwargs):
        if not isinstance(train_state, flax.training.train_state.TrainState):
            raise TypeError(
                "Checkpointer is called with an argument that is not a TrainState"
            )

        if not experimental_use_distributed_mode:
            train_state = jax.tree_util.tree_map(np.asarray, train_state)
        checkpoint_manager.save(train_state.step, train_state)

    return invoke_checkpointer
