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
"""Tests for asrnn.

As mentioned in "bobbin/example_lib/README.md", those components are not
intended to be in "bobbin" core modules.  But, still it's better to have
it tested.
"""

from typing import Tuple
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from bobbin.example_lib import asrnn
import chex
import flax.linen as nn
import jax
import numpy as np


def _random_right_paddings(batch_size: int, max_length: int, *, min_length: int = 0):
    lengths = np.random.randint(min_length, max_length, size=(batch_size, 1))
    indices = np.arange(max_length)[np.newaxis, :]
    return (indices >= lengths).astype(np.float32)


def _assert_sanity(
    cls,
    test_args,
    test_kwargs=None,
    *,
    init_kwargs=None,
    required_rngs=("params", "dropout", "specaug")
) -> Tuple[nn.Module, chex.ArrayTree, chex.ArrayTree]:
    if test_kwargs is None:
        test_kwargs = dict()
    if init_kwargs is None:
        init_kwargs = dict()

    module = cls(**init_kwargs)

    rngs = {name: jax.random.PRNGKey(i) for i, name in enumerate(required_rngs)}
    init_vars = module.init(rngs, *test_args, **test_kwargs)
    results = module.apply(
        init_vars, *test_args, **test_kwargs, mutable=True, rngs=rngs
    )
    chex.assert_tree_all_finite(results)
    return module, init_vars, results


class LogMelFilterBankTest(chex.TestCase):
    def test_sanity(self):
        batch_size = 3
        max_length = 700
        _assert_sanity(
            asrnn.LogMelFilterBank,
            (
                np.random.uniform(
                    -(2**15.0), 2**15.0, size=(batch_size, max_length)
                ),
                _random_right_paddings(batch_size, max_length, min_length=300),
            ),
        )

    def test_with_zero_input(self):
        batch_size = 3
        max_length = 700
        module = asrnn.LogMelFilterBank(num_bins=5)
        np.testing.assert_array_less(0.0, module.output_log_floor)

        waveform = np.zeros((batch_size, max_length))
        inputs = (waveform, np.zeros_like(waveform, dtype=np.float32))

        # No parameters and no random variables,
        model_vars = module.init(jax.random.PRNGKey(0), *inputs)
        outputs, output_paddings = module.apply(model_vars, *inputs)
        for v in outputs:
            np.testing.assert_allclose(v, np.log(module.output_log_floor))


class CnnEncoderTest(chex.TestCase):
    @parameterized.named_parameters(("train", False), ("eval", True))
    def test_sanity(self, is_eval):
        batch_size = 3
        max_length = 17
        freq_bins = 13
        _assert_sanity(
            asrnn.CnnEncoder,
            (
                np.random.normal(size=(batch_size, max_length, freq_bins)),
                _random_right_paddings(batch_size, max_length, min_length=5),
            ),
            dict(is_eval=is_eval),
        )


class ConformerConvBlockTest(chex.TestCase):
    @parameterized.named_parameters(("normal", False), ("deterministic", True))
    def test_sanity(self, deterministic):
        batch_size = 3
        max_length = 5
        model_dims = 16
        _assert_sanity(
            asrnn.ConformerConvBlock,
            (
                np.random.normal(size=(batch_size, max_length, model_dims)),
                _random_right_paddings(batch_size, max_length, min_length=1),
            ),
            dict(deterministic=deterministic),
            init_kwargs=dict(batch_norm_axis_name=None),
        )


class ConformerFfnBlockTest(chex.TestCase):
    @parameterized.named_parameters(("normal", False), ("deterministic", True))
    def test_sanity(self, deterministic):
        batch_size = 3
        max_length = 5
        model_dims = 16
        _assert_sanity(
            asrnn.ConformerFfnBlock,
            (np.random.normal(size=(batch_size, max_length, model_dims)),),
            dict(deterministic=deterministic),
        )


class ConformerMhsaBlockTest(chex.TestCase):
    @parameterized.named_parameters(("normal", False), ("deterministic", True))
    def test_sanity(self, deterministic):
        batch_size = 3
        max_length = 5
        model_dims = 16
        _assert_sanity(
            asrnn.ConformerMhsaBlock,
            (
                np.random.normal(size=(batch_size, max_length, model_dims)),
                _random_right_paddings(batch_size, max_length, min_length=1),
            ),
            dict(deterministic=deterministic),
        )

    def test_mask(self):
        batch_size = 3
        max_length = 7
        paddings = (np.random.uniform(size=(batch_size, max_length)) > 0.5).astype(
            np.float32
        )
        mask = asrnn._paddings_to_mask(paddings)

        for i in range(batch_size):
            for query_t in range(max_length):
                for key_t in range(max_length):
                    expected = paddings[i, query_t] < 0.5 and paddings[i, key_t] < 0.5
                    np.testing.assert_equal(bool(mask[i, 0, query_t, key_t]), expected)


class ConformerBlockTest(chex.TestCase):
    @parameterized.named_parameters(("train", False), ("eval", True))
    def test_sanity(self, is_eval):
        batch_size = 3
        max_length = 5
        model_dims = 16

        conv_block_orig = asrnn.ConformerConvBlock

        def conv_block(**kwargs):
            kwargs["batch_norm_axis_name"] = None
            return conv_block_orig(**kwargs)

        # The use of `patch` here is a workaround for the fact that a lot of
        # sub-modules are hard-coded in asrnn library.
        with unittest.mock.patch(
            "bobbin.example_lib.asrnn.ConformerConvBlock", new=conv_block
        ):
            _assert_sanity(
                asrnn.ConformerBlock,
                (
                    np.random.normal(size=(batch_size, max_length, model_dims)),
                    _random_right_paddings(batch_size, max_length, min_length=1),
                ),
                dict(is_eval=is_eval),
            )


class SpecAugTest(chex.TestCase):
    @parameterized.named_parameters(("normal", False), ("deterministic", True))
    def test_sanity(self, deterministic):
        batch_size = 3
        freq_bins = 80
        time_steps = 100
        _assert_sanity(
            asrnn.SpecAug,
            (
                np.random.normal(size=(batch_size, time_steps, freq_bins)),
                _random_right_paddings(batch_size, time_steps, min_length=1),
            ),
            dict(deterministic=deterministic),
        )

    def test_mask_shape(self):
        batch_size = 1
        freq_bins = 80
        time_steps = 100

        inputs = (
            np.ones((batch_size, time_steps, freq_bins)),
            np.zeros((batch_size, time_steps)),
        )
        module = asrnn.SpecAug(
            freq_mask_count=1, time_mask_count=1, time_mask_max_ratio=0.3
        )
        model_vars = module.init(
            {"specaug": jax.random.PRNGKey(0)}, *inputs, deterministic=False
        )
        mask = module.apply(
            model_vars,
            *inputs,
            deterministic=False,
            rngs={"specaug": jax.random.PRNGKey(1)}
        )

        def check_masks(m):
            # Check if all the mask vectors are zeros, or identical to other
            # non-zero mask vectors.
            first_valid_mask = None
            mask_lengths = []
            current_mask_len = 0
            for row in mask:
                if (mask == 0.0).all():
                    current_mask_len += 1
                else:
                    if current_mask_len > 0:
                        mask_lengths.append(current_mask_len)
                    current_mask_len = 0
                    if first_valid_mask is None:
                        first_valid_mask = row
                    np.testing.assert_allclose(row, first_valid_mask)
            np.testing.assert_(first_valid_mask is not None)
            if current_mask_len > 0:
                mask_lengths.append(current_mask_len)
            return mask_lengths

        time_mask_lengths = check_masks(mask)
        freq_mask_lengths = check_masks(mask.T)

        np.testing.assert_(len(freq_mask_lengths) <= 1)
        np.testing.assert_array_less(freq_mask_lengths, module.freq_mask_max_bins)
        np.testing.assert_(len(time_mask_lengths) <= 1)
        np.testing.assert_array_less(
            time_mask_lengths, int(module.time_mask_max_ratio * time_steps)
        )


class CnnConformerEncoderTest(chex.TestCase):
    @parameterized.named_parameters(("train", False), ("eval", True))
    def test_sanity(self, is_eval):
        batch_size = 3
        max_length = 13
        model_dims = 16
        freq_bins = 80
        conv_block_orig = asrnn.ConformerConvBlock

        def conv_block(**kwargs):
            kwargs["batch_norm_axis_name"] = None
            return conv_block_orig(**kwargs)

        # The use of `patch` here is a workaround for the fact that a lot of
        # sub-modules are hard-coded in asrnn library.
        with unittest.mock.patch(
            "bobbin.example_lib.asrnn.ConformerConvBlock", new=conv_block
        ):
            _assert_sanity(
                asrnn.CnnConformerEncoder,
                (
                    np.random.normal(size=(batch_size, max_length, freq_bins)),
                    _random_right_paddings(batch_size, max_length, min_length=5),
                ),
                dict(is_eval=is_eval),
                init_kwargs=dict(
                    cnn_def=lambda: asrnn.CnnEncoder(num_outputs=model_dims),
                    num_outputs=model_dims,
                ),
            )


if __name__ == "__main__":
    absltest.main()
