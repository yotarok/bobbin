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

"""Tests for pmap_util."""

from absl.testing import absltest
import chex
import jax
import jax.numpy as jnp
import numpy as np

from bobbin import pmap_util


def noisy_linear(
    weights: chex.Array,
    x: chex.Array,
    rng: chex.PRNGKey,
    noise_stddev: float,
    is_eval: bool,
) -> chex.Array:
    if is_eval:
        noised_weights = weights
    else:
        noised_weights = weights + noise_stddev * jax.random.normal(
            rng, shape=weights.shape
        )
    return jnp.einsum("...i, ij -> ...j", x, noised_weights)


_N_CPU_DEVICES = 4


def setUpModule():
    chex.set_n_cpu_devices(_N_CPU_DEVICES)


class WrappedPmapTest(chex.TestCase):
    def test_data_parallel(self):
        f = pmap_util.wrapped_pmap(
            noisy_linear,
            "batch",
            ("broadcast", "shard", "rng", "broadcast", "static"),
            backend="cpu",
        )
        weights = np.random.normal(size=(5, 7))
        data = np.random.normal(size=(8, 5))
        rng = jax.random.PRNGKey(123)
        result = f(weights, data, rng, 0.01, False)
        self.assertSequenceEqual(result.shape, (_N_CPU_DEVICES, 8 // _N_CPU_DEVICES, 7))

        # check result with zero-noise setting
        result = f(weights, data, rng, 0.0, False)
        reshaped_result = result.reshape(8, 7)
        np.testing.assert_allclose(
            reshaped_result, np.dot(data, weights), rtol=1e-6, atol=1e-6
        )

    def test_rng_split(self):
        f = pmap_util.wrapped_pmap(
            noisy_linear,
            "batch",
            ("broadcast", "shard", "rng", "broadcast", "static"),
            backend="cpu",
        )
        rng = jax.random.PRNGKey(123)
        randoms = f(np.zeros((4, 4)), np.identity(4), rng, 1.0, False)
        print(randoms)
        n_shards, unused_rows, unused_cols = randoms.shape
        self.assertEqual(n_shards, _N_CPU_DEVICES)
        for i in range(n_shards):
            for j in range(i + 1, n_shards):
                sq_dist = np.sum((randoms[i] - randoms[j]) ** 2)
                # TODO: Give some analysis on the comment
                self.assertGreater(sq_dist, 1.0)


if __name__ == "__main__":
    absltest.main()
