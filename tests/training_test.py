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

from absl.testing import absltest
import chex
import flax
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


class SgdMeanEstimation(training.TrainTask):
    def compute_loss(self, params, batch, *, extra_vars, prng_key, step):
        # those are not used
        del extra_vars
        del prng_key

        return l2_distortion_loss(params, batch), (dict(), None)


class TrainStateTest(chex.TestCase):
    def test_is_replicated_property(self):
        tx = optax.sgd(0.001)
        train_state = training.initialize_train_state(
            l2_distortion_loss, {"params": np.ones((3,))}, tx=tx
        )
        train_state = flax.jax_utils.replicate(train_state, jax.local_devices())
        np.testing.assert_(train_state.is_replicated_for_pmap())

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

    @chex.variants(with_jit=True, without_jit=True)
    def test_pmap_training_step(self):
        dims = 5
        batch_size = 3 * jax.local_device_count()

        tx = optax.sgd(0.001)
        train_state = training.initialize_train_state(
            l2_distortion_loss, {"params": np.ones((dims,))}, tx=tx
        )
        train_state = flax.jax_utils.replicate(train_state, jax.local_devices())

        task = SgdMeanEstimation()
        training_step_fn = task.make_training_step_fn()
        training_step_fn = training.pmap_for_train_step(self.variant(training_step_fn))

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

    @chex.variants(with_jit=True, without_jit=True)
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
        training_step_fn = task.make_training_step_fn(split_steps=nsteps)
        training_step_fn = training.pmap_for_train_step(self.variant(training_step_fn))

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


if __name__ == "__main__":
    absltest.main()
