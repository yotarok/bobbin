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
"""
Linen modules for ASR
"""

from __future__ import annotations

from collections.abc import Sequence

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from bobbin.contrib import asrio

_Array = chex.Array


class LogMelFilterBank(nn.Module):
    sample_rate: float = 16_000.0
    frame_size_ms: float = 25.0
    frame_step_ms: float = 10.0
    num_bins: int = 80
    lower_edge_hz: float = 125.0
    upper_edge_hz: float = 7600.0
    preemph: float = 0.97
    input_scale_factor: float = 1.0
    output_log_floor: float = 1.0

    def setup(self):
        self._frame_step = int(round(self.sample_rate * self.frame_step_ms / 1000.0))
        stft_frame_size = int(round(self.sample_rate * self.frame_size_ms / 1000.0))
        self._frame_size = stft_frame_size + 1
        self._fft_size = asrio.next_pow_of_two(stft_frame_size)

    def _compute_feature_paddings(self, waveform_paddings: _Array) -> _Array:
        return jax.lax.reduce_window(
            waveform_paddings,
            init_value=1.0,
            computation=jax.lax.min,
            window_dimensions=[1, self._frame_size],
            window_strides=[1, self._frame_step],
            padding="valid",
        )

    def _preemphasize(self, frames: _Array):
        if self.preemph:
            return frames[..., 1:] - self.preemph * frames[..., :-1]
        else:
            return frames[..., :-1]

    def __call__(
        self, waveform: _Array, waveform_paddings: _Array
    ) -> tuple[_Array, _Array]:
        # STFT
        waveform = waveform.astype(np.float32) * self.input_scale_factor
        feature_paddings = self._compute_feature_paddings(waveform_paddings)
        frames = asrio.frame(waveform, self._frame_size, self._frame_step)
        frames = self._preemphasize(frames)

        window = jnp.hanning(self._frame_size - 1)
        while window.ndim < frames.ndim:
            window = window[np.newaxis, ...]
        frames = window * frames

        spectra = jnp.fft.rfft(frames, self._fft_size, axis=-1)
        magnitudes = jnp.abs(spectra)

        mel_weights = asrio.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_bins,
            num_spectrogram_bins=self._fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.lower_edge_hz,
            upper_edge_hertz=self.upper_edge_hz,
        )
        melfb_outs = jnp.einsum("fn, ...f -> ...n", mel_weights, magnitudes)
        logmelfb_outs = jnp.log(jnp.maximum(melfb_outs, self.output_log_floor))
        return logmelfb_outs, feature_paddings


class CnnEncoder(nn.Module):
    channels: Sequence[int] = (32, 32)
    kernel_sizes: Sequence[tuple[int, int]] = ((3, 3), (3, 3))
    strides: Sequence[tuple[int, int]] = ((2, 2), (2, 2))
    num_outputs: int = 256

    @nn.compact
    def __call__(
        self, features: _Array, feature_paddings: _Array
    ) -> tuple[_Array, _Array]:
        features = features[..., np.newaxis]

        depth = len(self.channels)
        assert depth == len(self.kernel_sizes)
        assert depth == len(self.strides)

        x = features
        pad = feature_paddings
        for depth, (features, kernel_size, strides) in enumerate(
            zip(self.channels, self.kernel_sizes, self.strides)
        ):
            x = nn.Conv(
                features=features,
                kernel_size=kernel_size,
                strides=strides,
                padding="VALID",
            )(x)
            x = nn.activation.relu(x)
            pad = asrio.frame(pad, kernel_size[0], strides[0]).max(axis=-1)
        x = x.reshape(pad.shape + (-1,))
        x = nn.Dense(features=self.num_outputs)(x)
        return x, pad
