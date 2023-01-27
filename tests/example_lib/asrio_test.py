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
"""Tests for asrio.

As mentioned in "bobbin/example_lib/README.md", those components are not
intended to be in "bobbin" core modules.  But, still it's better to have
it tested.
"""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from bobbin.example_lib import asrio
import chex
import numpy as np


_TEST_WPM = [
    "<unk>",
    "</s>",
    "▁",
    "a",
    "b",
    "ab",
    "ba",
    "▁b",
    "▁ab",
]


def _make_test_vocab_file():
    vocab_file = tempfile.NamedTemporaryFile("w")
    for line in _TEST_WPM:
        print(line, file=vocab_file)
    vocab_file.flush()
    return vocab_file


class WpmTest(absltest.TestCase):
    def test_load(self):
        vocab_file = _make_test_vocab_file()
        vocab = asrio.WpmVocab.load(vocab_file.name)
        for tok in _TEST_WPM:
            np.testing.assert_equal(vocab.id2str[vocab.str2id[tok]], tok)

    def test_encode(self):
        vocab_file = _make_test_vocab_file()
        vocab = asrio.WpmVocab.load(vocab_file.name)

        ids = asrio.wpm_encode(vocab, "abba")
        np.testing.assert_equal(ids, [8, 6])

        ids = asrio.wpm_encode(vocab, "abcbc")
        np.testing.assert_equal(ids, [8, 0, 4, 0])

    def test_encode_sentence(self):
        vocab_file = _make_test_vocab_file()
        vocab = asrio.WpmVocab.load(vocab_file.name)

        ids = asrio.wpm_encode_sentence(vocab, "abba babba abab")
        np.testing.assert_equal(ids, [8, 6, 7, 5, 6, 8, 5])


# Computed with TF as `tf.signal.linear_to_mel_weight_matrix(8, 23)`.
_REFERENCE_TF_MEL_WEIGHTS_23x8 = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.3533392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.6521328, 0.3478672, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.81493515, 0.18506487, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.09202316, 0.9079768, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.45590898, 0.544091, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.8879662, 0.11203378, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.37498763, 0.62501234, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.9072663, 0.0927337, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.47746074, 0.52253926, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.07988153, 0.92011845, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.7100313, 0.28996876, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.36429358, 0.6357064, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.03971606, 0.96028394, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7338552, 0.26614478],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.44467294, 0.55532706],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17044084, 0.82955915],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9096907],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.66115826],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42374954],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19651178],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]


class SignalTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(dtype=[np.float32, np.int16], pad_end=[True, False])
    def test_framing(self, dtype, pad_end):
        nsamples = 321
        frame_size = 13
        frame_step = 7
        waveform = np.arange(nsamples, dtype=dtype)
        pad_value = np.ones((), dtype=dtype)

        def test_f(x):
            return asrio.frame(
                x, frame_size, frame_step, pad_end=pad_end, pad_value=pad_value
            )

        frames = self.variant(test_f)(waveform)
        frames = np.asarray(frames).tolist()

        offset = 0
        for frame in frames:
            expected = waveform[offset : offset + frame_size].tolist()
            if pad_end and len(expected) < frame_size:
                expected.extend([pad_value] * (frame_size - len(expected)))
            np.testing.assert_allclose(frame, expected)
            offset += frame_step

    def test_mel_weight_matrix(self):
        actual = asrio.linear_to_mel_weight_matrix(8, 23)
        np.testing.assert_allclose(
            actual, _REFERENCE_TF_MEL_WEIGHTS_23x8, atol=1e-05, rtol=1e-05
        )

    def test_next_pow_of_two(self):
        np.testing.assert_equal(asrio.next_pow_of_two(123), 128)
        np.testing.assert_equal(asrio.next_pow_of_two(512), 512)


class EditDistanceTest(chex.TestCase):
    def test_edit_distance(self):
        hyp = "abba"
        ref = "abcd"
        s, d, i = asrio.compute_edit_distance(hyp, ref)

        np.testing.assert_equal(s, 2)
        np.testing.assert_equal(d, 0)
        np.testing.assert_equal(i, 0)


class MeanVarNormalizerTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_normalize(self):
        dims = 3
        batch_size = 5
        time_steps = 7
        normalizer = asrio.MeanVarNormalizer(
            mean=np.zeros((dims,)), stddev=np.ones((dims,)), n=None
        )

        def test_f(x):
            return normalizer.apply(x)

        inputs = np.random.normal(size=(batch_size, time_steps, dims))
        results = self.variant(test_f)(inputs)

        # Standard setting yields identity function.
        np.testing.assert_allclose(results, inputs)


def _padded_batch_from_token_lists(data, *, max_length):
    batch_size = len(data)
    padded_batch = np.zeros((batch_size, max_length), dtype=np.int32)
    paddings = np.ones((batch_size, max_length))

    for i, line in enumerate(data):
        padded_batch[i, : len(line)] = line
        paddings[i, : len(line)] = 0.0

    return padded_batch, paddings


class PaddedSequenceUtilTest(chex.TestCase):
    def test_recovering_from_padded_batch(self):
        data = [
            [1, 2, 3],
            [4, 5, 6, 7, 8],
        ]
        padded_batch, paddings = _padded_batch_from_token_lists(data, max_length=8)

        result = asrio.make_list_from_padded_batch(padded_batch, paddings)

        for actual, expected in zip(result, data):
            np.testing.assert_equal(actual, expected)

    def test_ctc_blank_removal(self):
        data = [
            [1, 1, 0, 2, 2, 3, 3, 3],
            [4, 4, 4, 5, 5, 5],
            [4, 0, 4, 5, 5, 0],
        ]
        expected_all = [[1, 2, 3], [4, 5], [4, 4, 5]]

        padded_batch, paddings = _padded_batch_from_token_lists(data, max_length=8)
        results = asrio.remove_ctc_blanks_and_repeats(padded_batch, paddings)

        for actual, expected in zip(results, expected_all):
            np.testing.assert_equal(actual, expected)


if __name__ == "__main__":
    absltest.main()
