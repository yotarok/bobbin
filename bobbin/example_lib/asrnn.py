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

from typing import Any, Sequence, Tuple, Union, Optional

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from bobbin.example_lib import asrio

_Array = chex.Array
ModuleDef = Any


def _input_padding_validation(
    funcname: str, x: _Array, x_paddings: _Array
) -> Tuple[int]:
    """Checks whether the leading axes of features and padding indicators match.

    Args:
      funcname: Function name used in description of the exception.
      x: Feature array to be validated.
      x_paddings: Padding indicators to be validated.

    Returns:
      feature_shape: Shape tuple after removing the leading axes for batch and
        timesteps.  This is expected to be used further for validation.

    Raises:
      `ValueError` when the leading axes do not match.
    """
    *batch_sizes, time_steps = x_paddings.shape
    batch_sizes = tuple(batch_sizes)
    check_batch_sizes = x.shape[: len(batch_sizes)]
    check_time_steps, *feature_shape = x.shape[len(batch_sizes) :]

    if batch_sizes != check_batch_sizes:
        raise ValueError(
            "Batch sizes of input features and padding indicators provided to "
            f"`{funcname}` do not match: {check_batch_sizes} vs {batch_sizes}."
        )

    if time_steps != check_time_steps:
        raise ValueError(
            "Length of input features and padding indicators provided to "
            f"`{funcname}` do not match: {check_time_steps} vs {time_steps}."
        )
    return tuple(feature_shape)


class LogMelFilterBank(nn.Module):
    """A module computing logarithm of Mel-FilterBank outputs.

    Attributes:
      sample_rate: Sampling rate of the input waveform signals
      frame_size_ms: Frame size for short-time Fourier transform (STFT) in ms.
      frame_step_ms: Frame-shift size for STFT in ms.
      num_bins: The number of filters used in the filterbank.
      lower_edge_hz: The lowest frequency covered by the filterbank in Hz.
      upper_edge_hz: The highest frequency covered by the filterbank in Hz.
      preemph: Preemphasis coeffcients.
      input_scale_factor: The scalefactor applied to the input waveform.
      output_log_floor: Floor value used when computing logarithmic output.
      use_energy: If True, the filterbank outputs sum of energies instead of
        sum of magnitudes.
    """

    sample_rate: float = 16_000.0
    frame_size_ms: float = 25.0
    frame_step_ms: float = 10.0
    num_bins: int = 80
    lower_edge_hz: float = 125.0
    upper_edge_hz: float = 7600.0
    preemph: float = 0.97
    input_scale_factor: float = 2.0 ** (-15.0)
    output_log_floor: float = 1e-5
    use_energy: bool = True

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
    ) -> Tuple[_Array, _Array]:
        _input_padding_validation("LogMelFilterBank", waveform, waveform_paddings)

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

        if self.use_energy:
            magnitudes = magnitudes**2.0

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
    """A module that encodes features using CNN.

    Attributes:
      channels: The numbers of output channels of each layer.
      kernel_sizes: 2-D kernel sizes of each layer.
      strides: 2-D subsampling strides of each layer.
      num_outputs: The output dimensionality.
      use_batch_norm: If True (default), BatchNorm is applied at the end of
        each CNN layer (convolution + ReLU).
      is_eval: Set True for switching the module to the evaluation mode.
    """

    channels: Sequence[int] = (
        256,
        256,
    )
    kernel_sizes: Sequence[Tuple[int, int]] = (
        (3, 3),
        (3, 3),
    )
    strides: Sequence[Tuple[int, int]] = (
        (2, 2),
        (2, 2),
    )
    num_outputs: int = 256
    use_batch_norm: bool = True
    is_eval: Optional[bool] = None

    @nn.compact
    def __call__(
        self,
        features: _Array,
        feature_paddings: _Array,
        *,
        is_eval: Optional[bool] = None,
    ) -> tuple[_Array, _Array]:
        _input_padding_validation("CnnEncoder", features, feature_paddings)
        is_eval = nn.merge_param("is_eval", self.is_eval, is_eval)
        features = features[..., np.newaxis]

        depth = len(self.channels)
        if depth != len(self.kernel_sizes) or depth != len(self.strides):
            raise ValueError(
                "`channels`, `kernel_sizes` and `strides` for `CnnEncoder` must"
                " have the same lengths."
            )

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
            pad = jax.lax.reduce_window(
                pad,
                init_value=1.0,
                computation=jax.lax.min,
                window_dimensions=[1, kernel_size[0]],
                window_strides=[1, strides[0]],
                padding="valid",
            )
            x = nn.activation.relu(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=is_eval)(x)
        x = x.reshape(pad.shape + (-1,))
        x = nn.Dense(features=self.num_outputs)(x)
        return x, pad


class ConformerConvBlock(nn.Module):
    """LConv block for Conformer block.

    Attributes:
      residual_dropout_prob: Dropout probability for dropout layer applied to
        the non-linear part of this module.
      kernel_size: Kernel size for internal convolution layer.
      deterministic: If True, the dropout layer is switched off.
    """

    residual_dropout_prob: float = 0.0
    kernel_size: int = 5
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(
        self, x: _Array, x_paddings: _Array, deterministic: Optional[bool] = None
    ) -> _Array:
        _input_padding_validation("ConformerConvBlock", x, x_paddings)
        inputs = x
        deterministic = nn.merge_param(
            "deterministic", deterministic, self.deterministic
        )

        *unused_batch_sizes, unused_time_steps, model_dims = x.shape
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=model_dims * 2)(x)
        x = nn.activation.glu(x)
        x = nn.Conv(
            features=model_dims,
            kernel_size=(self.kernel_size,),
            feature_group_count=model_dims,
        )(x)

        x = nn.BatchNorm(use_running_average=deterministic)(x)
        x = nn.activation.swish(x)
        x = nn.Dense(features=model_dims)(x)
        if self.residual_dropout_prob > 0:
            x = nn.Dropout(self.residual_dropout_prob)(x, deterministic=deterministic)
        x = x + inputs
        return x


class ConformerFfnBlock(nn.Module):
    """Feed-forward block used in Conformer block.

    Attributes:
      hidden_dims: The number of hidden units in the network.
      hidden_dropout_prob: Dropout probability applied to the hidden units.
      residual_dropout_prob: Dropout probability applied to the non-linear path
        of this block.
      residual_weight: Coeffcieint multiplied to the non-linear path when it is merged
        with the skip path.
      deterministic: If True, all dropout components are disabled.
    """

    hidden_dims: int = 1024
    hidden_dropout_prob: float = 0.0
    residual_dropout_prob: float = 0.0
    residual_weight: float = 0.5
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x: _Array, deterministic: Optional[bool] = None) -> _Array:
        inputs = x
        deterministic = nn.merge_param(
            "deterministic", deterministic, self.deterministic
        )

        *unused_batch_sizes, unused_time_steps, model_dims = x.shape
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.hidden_dims)(x)
        x = nn.activation.swish(x)
        if self.hidden_dropout_prob > 0:
            x = nn.Dropout(self.hidden_dropout_prob)(x, deterministic=deterministic)
        x = nn.Dense(features=model_dims)(x)
        if self.residual_dropout_prob > 0:
            x = nn.Dropout(self.residual_dropout_prob)(x, deterministic=deterministic)
        x = self.residual_weight * x + inputs
        return x


class ConformerMhsaBlock(nn.Module):
    """Multi-head self-attention module used in Conformer blocks.

    Attributes:
      num_heads: The number of heads in multi-head attention.
      residual_dropout_prob: Dropout probability applied to the non-linear path
        of this block.
      attention_dropout_prob: Dropout probability applied to the attention
        probabilities in multihead self-attention.
      deterministic: If True, all dropout components are disabled.
    """

    num_heads: int = 8
    residual_dropout_prob: float = 0
    attention_dropout_prob: float = 0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(
        self, x: _Array, x_paddings: _Array, deterministic: Optional[bool] = None
    ) -> _Array:
        _input_padding_validation("ConformerMhsaBlock", x, x_paddings)
        inputs = x
        deterministic = nn.merge_param(
            "deterministic", deterministic, self.deterministic
        )
        x = nn.LayerNorm()(x)

        *batch_sizes, time_steps = x_paddings.shape
        flat_batch_size = np.product(batch_sizes)
        flat_paddings = x_paddings.reshape((flat_batch_size, time_steps))
        flat_padding_mask = (
            jax.vmap(lambda p: p[:, np.newaxis] + p[np.newaxis, :])(flat_paddings) < 0.5
        )
        padding_mask = flat_padding_mask.reshape(
            tuple(batch_sizes) + (1, time_steps, time_steps)
        )
        x = nn.SelfAttention(self.num_heads, dropout_rate=self.attention_dropout_prob)(
            x,
            mask=padding_mask,
            deterministic=deterministic,
        )
        if self.residual_dropout_prob > 0:
            x = nn.Dropout(self.residual_dropout_prob)(x, deterministic=deterministic)
        x = x + inputs
        return x


class ConformerBlock(nn.Module):
    """Conformer block.

    Attributes:
      kernel_size: Kernel size for internal LConv module.
      ffn_width_multipler: A factor for defining the number of hidden units in the
        feed-forward modules.  The number is defined as
        `hidden_dim = int(ffn_width_multiplier * input_dim)`.
      num_heads: The number of heads in multi-head self-attention modules.
      conv_residual_dropout_prob: The dropout probability copied to
        `ConformerConvBlock.residual_dropout_prob`.
      mhsa_residual_dropout_prob: The dropout probability copied to
        `ConformerMhsaBlock.residual_dropout_prob`.
      mhsa_attention_dropout_prob: The dropout probability copied to
        `ConformerMhsaBlock.attention_dropout_prob`.
      ffn_residual_dropout_prob: The dropout probability copied to
        `ConformerFfnBlock.residual_dropout_prob`.
      ffn_hidden_dropout_prob: The dropout probability copied to
        `ConformerFfnBlock.hidden_dropout_prob`.
      is_eval: If True, all internal blocks are configured to be in evaluation mode.
    """

    kernel_size: int = 5
    ffn_width_multiplier: float = 4.0
    num_heads: int = 8
    conv_residual_dropout_prob: float = 0
    mhsa_residual_dropout_prob: float = 0.1
    mhsa_attention_dropout_prob: float = 0
    ffn_residual_dropout_prob: float = 0.1
    ffn_hidden_dropout_prob: float = 0
    is_eval: Optional[bool] = None

    @nn.compact
    def __call__(
        self, x: _Array, x_paddings: _Array, is_eval: Optional[bool] = None
    ) -> Tuple[_Array, _Array]:
        _input_padding_validation("ConformerBlock", x, x_paddings)
        is_eval = nn.merge_param("is_eval", is_eval, self.is_eval)

        *unused_batch_sizes, unused_time_stesps, model_dims = x.shape

        x = ConformerFfnBlock(
            hidden_dims=int(model_dims * self.ffn_width_multiplier),
            hidden_dropout_prob=self.ffn_hidden_dropout_prob,
            residual_dropout_prob=self.ffn_residual_dropout_prob,
        )(x, deterministic=is_eval)

        x = ConformerMhsaBlock(
            residual_dropout_prob=self.mhsa_residual_dropout_prob,
            attention_dropout_prob=self.mhsa_attention_dropout_prob,
        )(x, x_paddings, deterministic=is_eval)

        x = ConformerConvBlock(
            residual_dropout_prob=self.conv_residual_dropout_prob,
        )(x, x_paddings, deterministic=is_eval)

        x = ConformerFfnBlock(
            hidden_dims=int(model_dims * self.ffn_width_multiplier),
            hidden_dropout_prob=self.ffn_hidden_dropout_prob,
            residual_dropout_prob=self.ffn_residual_dropout_prob,
        )(x, deterministic=is_eval)

        x = nn.LayerNorm()(x)

        return x, x_paddings


def _build_1d_masks(
    rng: chex.PRNGKey,
    batch_size: int,
    target_length: int,
    multiplicity: int,
    *,
    widths: Union[int, _Array],
    limits: Optional[Union[int, _Array]] = None,
) -> _Array:
    rng_for_loc, rng_for_width = jax.random.split(rng)
    if limits is None:
        limits = target_length
    if not isinstance(limits, _Array):
        limits = jnp.full((batch_size,), limits, dtype=np.float32)
    if not isinstance(widths, _Array):
        widths = jnp.full((batch_size,), widths, dtype=np.float32)

    limits -= widths

    mask_lower_edges = (
        jax.random.uniform(key=rng_for_loc, shape=(batch_size, multiplicity))
        * limits[..., np.newaxis]
    )

    mask_widths = (
        jax.random.uniform(key=rng_for_width, shape=(batch_size, multiplicity))
        * widths[..., np.newaxis]
    )
    mask_upper_edges = mask_lower_edges + mask_widths

    bins = np.arange(target_length)[np.newaxis, :, np.newaxis]
    return jnp.logical_or(
        bins < mask_lower_edges[:, np.newaxis, :],
        mask_upper_edges[:, np.newaxis, :] < bins,
    ).astype(np.float32)


class SpecAug(nn.Module):
    freq_mask_max_bins: int = 27
    freq_mask_count: int = 2
    time_mask_count: int = 10
    time_mask_max_ratio: float = 0.05
    deterministic: Optional[bool] = None
    rng_collection: str = "specaug"

    def __call__(
        self,
        inputs: _Array,
        paddings: Optional[_Array] = None,
        *,
        deterministic: Optional[bool] = None,
    ) -> _Array:
        deterministic = nn.merge_param(
            "deterministic", deterministic, self.deterministic
        )
        if deterministic:
            return inputs
        x = inputs
        batch_size, time_steps, freq_bands, *channels = inputs.shape
        if self.freq_mask_max_bins > 0 and self.freq_mask_count > 0:
            mask = _build_1d_masks(
                self.make_rng(self.rng_collection),
                batch_size,
                freq_bands,
                self.freq_mask_count,
                widths=self.freq_mask_max_bins,
            )
            mask = jnp.product(mask, axis=-1)[:, np.newaxis, :]
            mask = mask.reshape(mask.shape + (1,) * len(channels))
            x = x * mask
        if self.time_mask_count > 0:
            if paddings is None:
                raise ValueError(
                    "Specify `paddings` when using `SpecAug` with time masks."
                )
            lengths = time_steps - paddings.sum(axis=-1)
            widths = lengths * self.time_mask_max_ratio
            mask = _build_1d_masks(
                self.make_rng(self.rng_collection),
                batch_size,
                time_steps,
                self.time_mask_count,
                widths=widths,
                limits=lengths,
            )
            mask = jnp.product(mask, axis=-1)
            mask = mask.reshape(mask.shape + (1,) + (1,) * len(channels))
            x = x * mask
        return x


def sinusoidal_positional_embeddings(timesteps: int, dims: int) -> _Array:
    """Returns positional embeddings."""

    positions = np.arange(0, timesteps)[:, np.newaxis]
    omega = np.exp(np.arange(0, dims, 2) * -(np.log(10000.0) / dims))[np.newaxis, :]
    pe = np.concatenate([np.sin(positions * omega), np.cos(positions * omega)], axis=-1)
    return pe


class CnnConformerEncoder(nn.Module):
    """Encoder module based on CNN followed by multiple Conformer blocks.

    Attrbiutes:
      cnn_def: A function that constructs `CnnEncoder` (or compatible) module.
      conformer_block_def: A function that constructs `ConformerBlock` (or compatible)
        module.
      conformer_input_dropout_prob: Dropout probability for the input of conformer
        blocks (including positional embeddings added).
      num_conformer_blocks: The number of Conformer blocks used.
      num_outputs: Output dimensionality.
      is_eval: If True, all internal blocks are configured to be in evaluation mode.
      use_fixed_pos_emb: If True, global positional embeddings are added to the input
        of conformer blocks.
    """

    cnn_def: ModuleDef = CnnEncoder
    conformer_block_def: ModuleDef = ConformerBlock
    conformer_input_dropout_prob: float = 0.1
    num_conformer_blocks: int = 4
    num_outputs: int = 256
    is_eval: Optional[bool] = None
    use_fixed_pos_emb: bool = True

    @nn.compact
    def __call__(
        self, features: _Array, feature_paddings: _Array, is_eval: Optional[bool]
    ) -> Tuple[_Array, _Array]:
        is_eval = nn.merge_param("is_eval", self.is_eval, is_eval)
        _input_padding_validation("CnnConformerEncoder", features, feature_paddings)

        features, feature_paddings = self.cnn_def()(
            features, feature_paddings, is_eval=is_eval
        )

        x = features
        if self.use_fixed_pos_emb:
            *batch_sizes, timesteps, model_dims = x.shape
            pe = sinusoidal_positional_embeddings(timesteps, model_dims)
            x = x + pe[np.newaxis, :]
        if self.conformer_input_dropout_prob > 0 and not is_eval:
            x = nn.Dropout(self.conformer_input_dropout_prob)(x, deterministic=is_eval)
        pad = feature_paddings
        for unused_conformer_depth in range(self.num_conformer_blocks):
            x, pad = self.conformer_block_def()(x, pad, is_eval=is_eval)

        x = nn.Dense(features=self.num_outputs)(x)
        return x, feature_paddings
