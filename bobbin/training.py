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

import os
from typing import Any, Callable, Optional, Tuple, Union

import chex
from flax import struct
from flax.training import checkpoints
import flax.training.train_state
import jax
import jax.numpy as jnp
import optax

from .pmap_util import wrapped_pmap
from .pmap_util import RngArg
from .pmap_util import ShardArg
from .pmap_util import ThruArg

from .pytypes import Batch
from .pytypes import Parameter
from .pytypes import VarCollection

_Device = Any
_ArrayTree = chex.ArrayTree
_Scalar = chex.Scalar
_PRNGKey = chex.PRNGKey


class TrainState(flax.training.train_state.TrainState):
    """Thin-wrapper to `flax.training.train_state.TrainState`.

    This class is introduced to accomodate `extra_vars` field for handling
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
        return jnp.array(self.step).ndim > 1


@struct.dataclass
class StepInfo:
    """Output of each step containing loss and aux output of the loss function.

    Attributes:
      loss: Loss value computed in the step.
      loss_aux_out: Auxiliary output from the loss function.
    """

    loss: _Scalar
    loss_aux_out: Any


TrainingStepFn = Callable[[TrainState, Batch, _PRNGKey], Tuple[TrainState, StepInfo]]


class TrainTask:
    """Base class defining training task."""

    def compute_loss(
        self,
        params: Parameter,
        batch: Batch,
        *,
        extra_vars: VarCollection,
        prng_key: _PRNGKey,
    ) -> Tuple[_Scalar, Tuple[VarCollection, _ArrayTree]]:
        """Abstract method to be overridden for defining the loss function."""
        raise NotImplementedError()

    def make_training_step_fn(
        self, pmap_axis_name: Optional[str] = "batch"
    ) -> TrainingStepFn:
        """Creates training step function."""

        def train_step_fn(
            train_state: TrainState,
            batch: Batch,
            prng_key: _PRNGKey,
        ) -> Tuple[TrainState, StepInfo]:
            (loss, (mutated_vars, loss_aux)), grads = jax.value_and_grad(
                self.compute_loss, has_aux=True
            )(
                train_state.params,
                batch,
                extra_vars=train_state.extra_vars,
                prng_key=prng_key,
            )

            if pmap_axis_name is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis_name)
            train_state = train_state.apply_gradients(grads=grads)

            updated_extra_vars = train_state.extra_vars.copy()
            for colname, tree in mutated_vars.items():
                if colname == "params":
                    continue

                if pmap_axis_name is not None:
                    updated_extra_vars[colname] = self.reduce_extra_vars(
                        colname, tree, axis_name=pmap_axis_name
                    )

            train_state = train_state.replace(extra_vars=updated_extra_vars)
            return train_state, StepInfo(loss=loss, loss_aux_out=loss_aux)

        train_step_fn = jax.named_call(train_step_fn, name="train_step_fn")
        return train_step_fn

    def reduce_extra_vars(
        self, colname: str, tree: _ArrayTree, *, axis_name: str
    ) -> _ArrayTree:
        """Abstract method to be overridden for sync non-parameter variables."""
        return tree


def initialize_train_state(
    apply_fn: Callable,
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

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        new_state = checkpoints.restore_checkpoint(checkpoint_path, new_state)

    return new_state


def pmap_for_train_step(
    train_step_fn: TrainingStepFn, axis_name="batch"
) -> TrainingStepFn:
    """Wraps `train_step_fn` with `pmap`."""
    return wrapped_pmap(
        train_step_fn, axis_name=axis_name, argtypes=[ThruArg(), ShardArg(), RngArg()]
    )
