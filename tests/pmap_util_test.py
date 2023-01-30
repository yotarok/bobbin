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

import unittest

from absl.testing import absltest
from absl.testing import parameterized
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
chex.set_n_cpu_devices(
    _N_CPU_DEVICES
)  # Just for making sure it is done before any jax ops


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


class ProcessGatherTest(chex.TestCase):
    @parameterized.product(
        process_count=[7],
        process_index=[0, 3],
    )
    @unittest.mock.patch("jax.lax.all_gather")
    @unittest.mock.patch("jax.process_index")
    @unittest.mock.patch("jax.process_count")
    def test_gather(
        self,
        mock_process_count,
        mock_process_index,
        mock_all_gather,
        process_count,
        process_index,
    ):
        mock_process_count.return_value = process_count
        mock_process_index.return_value = process_index

        all_data = [n for n in range(process_count)]

        def all_gather_impl(
            unused_tree: chex.ArrayTree, axis_name: str
        ) -> chex.ArrayTree:
            del axis_name
            # this part is depending on internal behavior of
            # `gather_from_jax_processes` and need to be updated when the
            # internal behavior updated.
            ret_process_ids = []
            ret_device_ids = []
            ret_data = []
            for process_id in range(process_count):
                for device_id in range(jax.local_device_count()):
                    ret_process_ids.append(process_id)
                    ret_device_ids.append(device_id)
                    ret_data.append(all_data[process_id])
            return (
                jnp.asarray(ret_process_ids),
                jnp.asarray(ret_device_ids),
                jnp.asarray(ret_data),
            )

        mock_all_gather.side_effect = all_gather_impl
        with chex.fake_pmap():
            results = pmap_util.gather_from_jax_processes(all_data[process_index])
        np.testing.assert_array_equal(results, all_data)


if __name__ == "__main__":
    absltest.main()
