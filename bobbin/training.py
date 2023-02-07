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

from .array_util import split_leading_axis

from .pmap_util import assert_replica_integrity
from .pmap_util import wrapped_pmap
from .pmap_util import RngArg
from .pmap_util import ShardArg
from .pmap_util import ThruArg

from .pytypes import Batch
from .pytypes import Parameter
from .pytypes import VarCollection

_Device = Any
_Array = chex.Array
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
        return jnp.array(self.step).ndim >= 1


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
LossFnSideOutput = Tuple[VarCollection, _ArrayTree]  # mutated_vars and loss_aux
LossFnResult = Tuple[_Scalar, LossFnSideOutput]
ValueAndGradResult = Tuple[LossFnResult, _ArrayTree]
ValueAndGradFn = Callable[[Parameter, Batch], ValueAndGradResult]


def _split_and_apply_value_and_grad(
    value_and_grad: ValueAndGradFn,
    params: Parameter,
    batch: Batch,
    *,
    split_steps: int,
    extra_vars: VarCollection,
    prng_key: _PRNGKey,
) -> ValueAndGradResult:
    """Applies value_and_grad function to split batches and merges results.

    Args:
      value_and_grad: Function that is obtained by applying
        `jax.value_and_grad(f, has_aux=True)` to `compute_loss`.
      params: Parameters passed to `compute_loss`
      batch: A batch that is split and given to `compute_loss` sequentially.
      split_steps: The number of splits used.
      extra_vars: Extra var collection given to `compute_loss`.
      prng_key: Base random-number generator.
    """
    if split_steps <= 0:
        raise ValueError(f"Invalid split step ({split_steps}) specified.")
    microbatches = split_leading_axis(
        split_steps,
        batch,
        leading_axis_name="per-device batch",
        split_group_name="split steps",
    )

    rng, prng_key = jax.random.split(prng_key)
    (loss, (mutated_vars, loss_aux_0)), grads = value_and_grad(
        params,
        jax.tree_util.tree_map(lambda x: x[0], microbatches),
        extra_vars=extra_vars,
        prng_key=rng,
    )  # pytype: disable=wrong-keyword-args
    loss_aux_list = [loss_aux_0]
    for step in range(1, split_steps):
        rng, prng_key = jax.random.split(prng_key)
        # Only the last step actually updates mutated_vars
        (split_loss, (mutated_vars, split_aux)), split_grads = value_and_grad(
            params,
            jax.tree_util.tree_map(lambda x: x[step], microbatches),
            extra_vars=extra_vars,
            prng_key=prng_key,
        )  # pytype: disable=wrong-keyword-args
        loss += split_loss
        loss_aux_list.append(split_aux)
        grads = jax.tree_util.tree_map(lambda x, y: x + y, grads, split_grads)

        # extra var mutation is applied sequentially.
        for colname, tree in mutated_vars.items():
            if colname not in extra_vars:
                continue
            extra_vars[colname] = mutated_vars[colname]

    loss /= split_steps
    grads = jax.tree_util.tree_map(lambda x: x / split_steps, grads)
    loss_aux = jax.tree_util.tree_map(lambda *xs: jnp.asarray([xs]), *loss_aux_list)

    return (loss, (mutated_vars, loss_aux)), grads


class TrainTask:
    """Base class defining training task."""

    def compute_loss(
        self,
        params: Parameter,
        batch: Batch,
        *,
        extra_vars: VarCollection,
        prng_key: _PRNGKey,
    ) -> LossFnResult:
        """Abstract method to be overridden for defining the loss function."""
        raise NotImplementedError()

    def make_training_step_fn(
        self, pmap_axis_name: Optional[str] = "batch", split_steps: Optional[int] = None
    ) -> TrainingStepFn:
        """Creates training step function."""

        def train_step_fn(
            train_state: TrainState,
            batch: Batch,
            prng_key: _PRNGKey,
        ) -> Tuple[TrainState, StepInfo]:
            if pmap_axis_name is not None:
                try:
                    jax.lax.axis_index(pmap_axis_name)
                except NameError:
                    raise ValueError(
                        "`make_training_step_fn` is called with "
                        f"pmap_axis_name={pmap_axis_name} but this function is"
                        "used outside of pmap context. If you do not intend to"
                        "perform multi-device parallelization, set "
                        "pmap_axis_name=None."
                    )

            value_and_grad = jax.value_and_grad(self.compute_loss, has_aux=True)
            if split_steps is None:
                (loss, (mutated_vars, loss_aux)), grads = value_and_grad(
                    train_state.params,
                    batch,
                    extra_vars=train_state.extra_vars,
                    prng_key=prng_key,
                )
            else:
                (
                    loss,
                    (mutated_vars, loss_aux),
                ), grads = _split_and_apply_value_and_grad(
                    value_and_grad,
                    train_state.params,
                    batch,
                    split_steps=split_steps,
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

    if jax.process_count() > 1:
        assert_replica_integrity(new_state, is_device_replicated=False)

    return new_state


def pmap_for_train_step(
    train_step_fn: TrainingStepFn, axis_name="batch"
) -> TrainingStepFn:
    """Wraps `train_step_fn` with `pmap`."""
    return wrapped_pmap(
        train_step_fn, axis_name=axis_name, argtypes=[ThruArg(), ShardArg(), RngArg()]
    )
