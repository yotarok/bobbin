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
"""Tests for training.
"""

import unittest

from absl.testing import absltest
import chex
import flax
import flax.linen as nn
import numpy as np
import jax
import jax.numpy as jnp
import optax

from bobbin import training


_N_CPU_DEVICES = 4
chex.set_n_cpu_devices(
    _N_CPU_DEVICES
)  # Just for making sure it is done before any jax ops


def setUpModule():
    chex.set_n_cpu_devices(_N_CPU_DEVICES)


def l2_distortion_loss(params: chex.Array, x: chex.Array):
    return jnp.mean((params[np.newaxis, :] - x) ** 2.0)


class SgdMeanEstimation(training.BaseTrainTask):
    """Sample of BaseTrainTask."""

    def compute_loss(self, params, batch, *, extra_vars, prng_key, step):
        # those are not used
        del extra_vars
        del prng_key
        del step

        return l2_distortion_loss(params, batch), (dict(), None)


class LogisticRegressionTask(training.TrainTask):
    """Sample of TrainTask (i.e. BaseTrainTask with flax module.)"""

    def __init__(self, input_dim: int = 5):
        mod = nn.Sequential(
            [nn.Dropout(0.5, deterministic=False), nn.Dense(features=1)]
        )
        example_args = (
            np.zeros(
                (
                    1,
                    input_dim,
                )
            ),
        )
        super().__init__(mod, example_args, required_rngs=("dropout",))

    def compute_loss(self, params, batch, *, extra_vars, prng_key, step):
        del prng_key
        del step

        inputs, labels = batch
        model_vars = extra_vars.copy()
        model_vars.update(params=params)
        pred = self.model.apply(model_vars, inputs, rngs=self.get_rng_dict())
        return -jax.nn.log_sigmoid(labels * pred)


class TrainStateTest(chex.TestCase):
    def test_is_replicated_property(self):
        tx = optax.sgd(0.001)
        train_state = training.initialize_train_state(
            l2_distortion_loss, {"params": np.ones((3,))}, tx=tx
        )
        train_state = flax.jax_utils.replicate(train_state, jax.local_devices())
        np.testing.assert_(train_state.is_replicated_for_pmap())


class StepFunctionTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_single_training_step(self):
        dims = 5
        batch_size = 3

        tx = optax.sgd(0.001)
        train_state = training.initialize_train_state(
            l2_distortion_loss, {"params": np.ones((dims,))}, tx=tx
        )

        task = SgdMeanEstimation()
        training_step_fn = task.make_training_step_fn(pmap_axis_name=None)

        batch = np.random.normal(size=(batch_size, dims))
        next_train_state, step_info = self.variant(training_step_fn)(
            train_state,
            batch,
            jax.random.PRNGKey(0),
        )

        # This holds because parameters are intialized with ones.
        np.testing.assert_allclose(
            step_info.loss, np.mean((batch - 1.0) ** 2), atol=1e-5, rtol=1e-5
        )

        np.testing.assert_equal(
            np.asarray(next_train_state.step).tolist(), train_state.step + 1
        )

    def test_pmap_training_step(self):
        dims = 5
        batch_size = 3 * jax.local_device_count()

        tx = optax.sgd(0.001)
        train_state = training.initialize_train_state(
            l2_distortion_loss, {"params": np.ones((dims,))}, tx=tx
        )
        train_state = flax.jax_utils.replicate(train_state, jax.local_devices())

        task = SgdMeanEstimation()
        training_step_fn = task.make_training_step_fn().pmap("batch")

        batch = np.random.normal(size=(batch_size, dims))
        next_train_state, step_info = training_step_fn(
            train_state,
            batch,
            jax.random.PRNGKey(0),
        )

        np.testing.assert_allclose(next_train_state.step, train_state.step + 1)

        np.testing.assert_allclose(
            np.mean(step_info.loss), np.mean((batch - 1.0) ** 2), atol=1e-5, rtol=1e-5
        )

    def test_pmap_split_step(self):
        nsteps = 7

        dims = 5
        batch_size = 3 * jax.local_device_count() * nsteps

        tx = optax.sgd(0.001)
        train_state = training.initialize_train_state(
            l2_distortion_loss, {"params": np.ones((dims,))}, tx=tx
        )
        train_state = flax.jax_utils.replicate(train_state, jax.local_devices())

        task = SgdMeanEstimation()
        training_step_fn = task.make_training_step_fn(split_steps=nsteps).pmap("batch")

        batch = np.random.normal(size=(batch_size, dims))
        next_train_state, step_info = training_step_fn(
            train_state,
            batch,
            jax.random.PRNGKey(0),
        )

        np.testing.assert_allclose(next_train_state.step, train_state.step + 1)

        np.testing.assert_allclose(
            np.mean(step_info.loss), np.mean((batch - 1.0) ** 2), atol=1e-5, rtol=1e-5
        )


class TrainTaskTest(chex.TestCase):
    def test_log_writer_creation(self):
        stub_logger = "unused_logger_object"
        loglevel = "unused_loglevel"
        task = SgdMeanEstimation()
        task.write_trainer_log = unittest.mock.MagicMock()
        writer = task.make_log_writer(logger=stub_logger, loglevel=loglevel)

        tx = optax.sgd(0.001)
        train_state = training.initialize_train_state(
            l2_distortion_loss, {"params": np.ones((3,))}, tx=tx
        )
        step_info = None
        writer(train_state, step_info=step_info)
        task.write_trainer_log.assert_called_once_with(
            train_state, step_info=step_info, logger=stub_logger, loglevel=loglevel
        )

    def test_initialization(self):
        task = LogisticRegressionTask(5)
        train_state = task.initialize_train_state(
            jax.random.PRNGKey(0), optax.sgd(0.001), checkpoint_path="none"
        )
        chex.assert_shape(train_state.params["layers_1"]["kernel"], (5, 1))
        chex.assert_shape(train_state.params["layers_1"]["bias"], (1,))


class TrainingProgressPublisherTest(chex.TestCase):
    @unittest.mock.patch("time.time")
    def test_speed_metrics(self, time_mock):
        state = training.TrainState.create(
            apply_fn=None, params=None, tx=optax.identity(), extra_vars=dict()
        )
        writer = unittest.mock.MagicMock()
        action = training.PublishTrainingProgress(writer)

        state = state.replace(step=10)
        time_mock.return_value = 1000.0
        action(state)

        state = state.replace(step=133)
        time_mock.return_value = 1010.0
        action(state)
        writer.scalar.assert_called_with("trainer/steps_per_sec", 12.3, step=133)


if __name__ == "__main__":
    absltest.main()
