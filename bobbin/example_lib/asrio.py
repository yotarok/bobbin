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
"""Peripheral data-structures and functions for ASR.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import difflib
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

Array = chex.Array

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


@dataclasses.dataclass
class WpmVocab:
    """A dataclass containing dictionary for subword decomposition."""

    str2id: Dict[str, int]
    id2str: Dict[int, str]

    @classmethod
    def load(cls, path: str, *, size_limit: Optional[int] = None):
        str2id = dict()
        id2str = dict()
        with tf.io.gfile.GFile(path) as f:
            for line in f.readlines():
                wp_str = line.strip().split(maxsplit=1)[0]
                wp_id = len(str2id)
                str2id[wp_str] = wp_id
                id2str[wp_id] = wp_str
                if size_limit is not None and len(str2id) >= size_limit:
                    break
        return cls(str2id=str2id, id2str=id2str)

    @property
    def no_token(self) -> int:
        return len(self.str2id)

    def __len__(self) -> int:
        return len(self.id2str)


def wpm_encode(
    vocab: WpmVocab, w: str, unk_symbol: str = "<unk>", bow_symbol: str = "▁"
) -> List[int]:
    """Encodes word into subword ids."""
    unk_id = vocab.str2id.get(unk_symbol, -1)
    ids = [vocab.str2id.get(ch, unk_id) for ch in [bow_symbol] + list(w)]

    def merge(left: int, right: int) -> int:
        merged_str = vocab.id2str[left] + vocab.id2str[right]
        return vocab.str2id.get(merged_str, vocab.no_token)

    def should_merge(candidates: list[int]) -> bool:
        return any(t != vocab.no_token for t in candidates)

    candidates = [merge(left, right) for left, right in zip(ids[:-1], ids[1:])]
    while should_merge(candidates):
        best_merge = np.argmin(candidates)
        ids = ids[:best_merge] + [candidates[best_merge]] + ids[best_merge + 2 :]

        left = (
            []
            if best_merge == 0
            else (
                candidates[: best_merge - 1]
                + [merge(ids[best_merge - 1], ids[best_merge])]
            )
        )
        right = (
            []
            if best_merge == len(ids) - 1
            else (
                [merge(ids[best_merge], ids[best_merge + 1])]
                + candidates[best_merge + 2 :]
            )
        )
        candidates = left + right
    return ids


def wpm_encode_sentence(vocab: WpmVocab, s: Union[str, bytes]) -> List[int]:
    if isinstance(s, np.ndarray):
        s = s.tolist()
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    words = s.lower().split(" ")
    ret = []
    for w in words:
        ret.extend(wpm_encode(vocab, w))
    return ret


def wpm_decode_sentence(
    wpm_vocab: WpmVocab,
    ids: Iterable[int],
    unk_text: str = "█",
    word_boundary_text: str = "▁",
) -> str:
    return (
        "".join(wpm_vocab.id2str.get(i, unk_text) for i in ids)
        .replace(word_boundary_text, " ")
        .strip()
    )


def _hertz_to_mel(frequencies_hertz: Union[float, int, Array]) -> Array:
    """Convert hertz to mel."""
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)
    )


def _pad_end_length(num_timesteps: int, frame_step: int, frame_size: int) -> int:
    """Returns how many sample needed to be padded for pad_end feature."""
    # The number of frames that can be extracted from the signal.
    num_frames = int(np.ceil(num_timesteps / frame_step))
    # Signal length required for computing `num_frames` frames.
    padded_length = frame_step * (num_frames - 1) + frame_size
    return padded_length - num_timesteps


def frame(
    x: Array,
    frame_length: int,
    frame_step: int,
    pad_end: bool = False,
    pad_value: Union[int, float] = 0.0,
) -> Array:
    """Slides a window and extract values inside the window.
    ]
        This function extracts `x[:, n:n+frame_length, :]` with sliding `n` with
        stride of `frame_step`, and returns an array `y` with the shape
        `(batch_size, num_frames, frame_length, num_channels)`. Unlike the
        counterpart in Tensorflow (`tf.signal.frame`), this function currently does
        not take `axis` argument, and the input tensor `x` is expected to have a
        shape of `(batch_size, timesteps)`.

        Args:
          x: An input array with `(*batch_size, timesteps)`-shape.
          frame_length: The frame length.
          frame_step: The frame hop size.
          pad_end: If True, the end of signal is padded so the window can continue
            sliding while the starting point of the window is in the valid range.
          pad_value: A scalar used as a padding value when `pad_end` is True.

        Returns:
          A tensor with shape `(*batch_size, num_frames, frame_length)`.
    """
    *batch_sizes, num_timesteps = x.shape

    if pad_end:
        num_extends = _pad_end_length(num_timesteps, frame_step, frame_length)
        pads = [(0, 0)] * len(batch_sizes) + [(0, num_extends)]
        x = jnp.pad(x, pads, "constant", constant_values=pad_value)
        num_timesteps += num_extends

    flat_x = x.reshape((-1, num_timesteps, 1))
    flat_y = jax.lax.conv_general_dilated_patches(
        flat_x,
        (frame_length,),
        (frame_step,),
        "VALID",
        dimension_numbers=("NTC", "OIT", "NTC"),
    )
    ret = flat_y.reshape((*batch_sizes, -1, frame_length))
    return ret


def linear_to_mel_weight_matrix(
    num_mel_bins: int = 20,
    num_spectrogram_bins: int = 129,
    sample_rate: Union[int, float] = 8000,
    lower_edge_hertz: Union[int, float] = 125.0,
    upper_edge_hertz: Union[int, float] = 3800.0,
    dtype: Any = np.float32,
) -> Array:
    r"""Numpy-port of `tf.signal.linear_to_mel_weight_matrix`.

    Note that this function works purely on numpy because mel-weights are
    shape-dependent constants that usually should not be computed in an
    accelerators.

    Args:
      num_mel_bins: Python int. How many bands in the resulting mel spectrum.
      num_spectrogram_bins: An integer `Tensor`. How many bins there are in the
        source spectrogram data, which is understood to be `fft_size // 2 + 1`,
        i.e. the spectrogram only contains the nonredundant FFT bins.
      sample_rate: An integer or float `Tensor`. Samples per second of the input
        signal used to create the spectrogram. Used to figure out the frequencies
        corresponding to each spectrogram bin, which dictates how they are mapped
        into the mel scale.
      lower_edge_hertz: Python float. Lower bound on the frequencies to be
        included in the mel spectrum. This corresponds to the lower edge of the
        lowest triangular band.
      upper_edge_hertz: Python float. The desired top edge of the highest
        frequency band.
      dtype: The `DType` of the result matrix. Must be a floating point type.

    Returns:
      An array of shape `[num_spectrogram_bins, num_mel_bins]`.
    Raises:
      ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not
        positive, `lower_edge_hertz` is negative, frequency edges are incorrectly
        ordered, `upper_edge_hertz` is larger than the Nyquist frequency.
    [mel]: https://en.wikipedia.org/wiki/Mel_scale
    """

    # Input validator from tensorflow/python/ops/signal/mel_ops.py#L71
    if num_mel_bins <= 0:
        raise ValueError("num_mel_bins must be positive. Got: %s" % num_mel_bins)
    if lower_edge_hertz < 0.0:
        raise ValueError(
            "lower_edge_hertz must be non-negative. Got: %s" % lower_edge_hertz
        )
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(
            "lower_edge_hertz %.1f >= upper_edge_hertz %.1f"
            % (lower_edge_hertz, upper_edge_hertz)
        )
    if sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive. Got: %s" % sample_rate)
    if upper_edge_hertz > sample_rate / 2:
        raise ValueError(
            "upper_edge_hertz must not be larger than the Nyquist "
            "frequency (sample_rate / 2). Got %s for sample_rate: %s"
            % (upper_edge_hertz, sample_rate)
        )

    # HTK excludes the spectrogram DC bin.
    bands_to_zero = 1
    nyquist_hertz = sample_rate / 2.0
    linear_frequencies = np.linspace(
        0.0, nyquist_hertz, num_spectrogram_bins, dtype=dtype
    )[bands_to_zero:]
    spectrogram_bins_mel = _hertz_to_mel(linear_frequencies)[:, np.newaxis]

    # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
    # center of each band is the lower and upper edge of the adjacent bands.
    # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
    # num_mel_bins + 2 pieces.
    edges = np.linspace(
        _hertz_to_mel(lower_edge_hertz),
        _hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2,
        dtype=dtype,
    )

    # Split the triples up and reshape them into [1, num_mel_bins] tensors.
    lower_edge_mel, center_mel, upper_edge_mel = (
        edges[:-2][np.newaxis, :],
        edges[1:-1][np.newaxis, :],
        edges[2:][np.newaxis, :],
    )

    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the mel domain, not Hertz.
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel
    )
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel
    )

    # Intersect the line segments with each other and zero.
    mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

    # Re-add the zeroed lower bins we sliced out above.
    return np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]])


def _hanning_nonzero(win_support, frame_size, dtype):
    """Computes modified version of Hanning window.

    This version of Hanning window doesn't have zero on the both tail, and
    therefore the `frame_size` parameter can be more accurate.

    Args:
      win_support: Number of samples for non-zero support in the window
      frame_size: Total size of the window (frame_size >= win_support)
      dtype: numpy data type of output and computation

    Returns:
      Numpy array of size frame_size with the window to apply.
    """
    assert frame_size >= win_support
    arg = np.pi * 2.0 / (win_support)
    hann = 0.5 - (0.5 * np.cos(arg * (np.arange(win_support, dtype=dtype) + 0.5)))
    zero_size = frame_size - win_support
    return np.pad(hann, [(0, zero_size)])


def next_pow_of_two(x: Union[int, float]) -> int:
    """Computes the minimum integer in `2**N`-form that is larger than `x`."""
    return int(2 ** np.ceil(np.log2(x)))


def compute_edit_distance(
    hyp: Sequence[Any], ref: Sequence[Any]
) -> Tuple[int, int, int]:
    """Computes edit distance between two sequences.

    Args:
      hyp: Hypothesis sequence to be tested.
      ref: Reference sequence to be compared.

    Returns:
      A tuple of three numbers as `(s, d, i)` where `s` is the number of
      substituition errors, `d` is the number of deletion error, and `i` is the
      number of insertion errors.
    """
    s = 0
    d = 0
    i = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp).get_opcodes():
        if tag == "delete":
            d += i2 - i1
            assert j1 == j2
        elif tag == "insert":
            i += j2 - j1
            assert i1 == i2
        elif tag == "replace":
            reflen = i2 - i1
            hyplen = j2 - j1
            if reflen > hyplen:
                d += reflen - hyplen
            elif hyplen > reflen:
                i += hyplen - reflen
            s += min(reflen, hyplen)
        else:
            assert tag == "equal"
    return s, d, i


class SequenceError(struct.PyTreeNode):
    """Dataclass containing edit-distance information."""

    refs: int = 0
    subs: int = 0
    dels: int = 0
    inss: int = 0

    @property
    def error_rate(self) -> float:
        return (self.subs + self.dels + self.inss) / self.refs

    @property
    def hyps(self) -> int:
        return self.refs + self.inss - self.dels

    def accumulate(self, hyp: Sequence[Any], ref: Sequence[Any]) -> SequenceError:
        """Increments error counts by computing errors from given hypothesis."""
        s, d, i = compute_edit_distance(hyp, ref)
        return type(self)(
            refs=self.refs + len(ref),
            subs=self.subs + s,
            dels=self.dels + d,
            inss=self.inss + i,
        )


class MeanVarNormalizer(struct.PyTreeNode):
    """A dataclass containing mean and variance (stddev) of input features.

    Attributes:
      mean: Means of the feature values.
      stddev: Standard deviations of the feature values.
      n: Optional field for storing the number of frames used for computing the
        statistics.  It is useful for updating the above fields.
    """

    mean: chex.Array
    stddev: chex.Array
    n: Optional[chex.Array]

    def apply(self, features: chex.Array) -> chex.Array:
        mean = self.mean.reshape((1,) * (features.ndim - 1) + (-1,))
        stddev = self.stddev.reshape((1,) * (features.ndim - 1) + (-1,))
        return (features - mean) / stddev

    @classmethod
    def empty(cls) -> MeanVarNormalizer:
        return cls(mean=np.zeros(1), stddev=np.zeros(1), n=0)


def make_list_from_padded_batch(
    data: np.ndarray, paddings: np.ndarray
) -> List[List[Any]]:
    """Recovers variable-length lists from the padded-batch.

    Args:
      data: An array with `(batch_size, max_length, ...)`-shape.
      paddings: Padding indicators stored in `(batch_size, max_length)`-shaped
        array.  `paddings[i, t] > 0.5` denotes that `t`-th element in `i`-th
        sequence is a padding.

    Returns:
      List of list with the length of `batch_size`.
    """
    data = data.tolist()
    paddings = paddings.tolist()
    return [
        [elem for pad, elem in zip(pads, row) if pad < 0.5]
        for row, pads in zip(data, paddings)
    ]


def remove_ctc_blanks_and_repeats(
    token_ids: chex.Array,
    token_paddings: chex.Array,
    *,
    blank_id: int = 0,
    use_jnp: bool = False,
) -> List[List[int]]:
    """Applies the CTC blank removal rule.

    Args:
      token_ids: Integer array with `(batch_size, max_length)`-shape.
      token_paddings: Padding indicators for `token_ids`.
      blank_id: ID for blank tokens.

    Returns:
      List (with length=`batch_size`) of variable-length lists containing
      tokens after CTC blank removal.
    """

    npmod = jnp if use_jnp else np

    batch_size, max_length = token_ids.shape
    is_repetition = npmod.concatenate(
        [npmod.zeros((batch_size, 1)), token_ids[:, 1:] == token_ids[:, :-1]], axis=1
    )
    is_blank = token_ids == blank_id
    remove_mask = npmod.logical_or(is_repetition, is_blank)

    updated_paddings = npmod.where(remove_mask, 1.0, token_paddings)
    return make_list_from_padded_batch(token_ids, updated_paddings)
