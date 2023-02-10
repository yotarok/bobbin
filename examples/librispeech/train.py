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

"""LibriSpeech example.
"""

# This sample is for demonstrating scalability to a large scale problem.
# See also "examples/gcp/README.md" for how to run this example script.
#
# For understanding how multi-process training work in Jax, please refer
# the document: https://jax.readthedocs.io/en/latest/multi_process.html

from __future__ import annotations

import argparse
import copy
import dataclasses
from etils import epath
import functools
import logging
import os
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import urllib.request

import chex
import fiddle as fdl
from fiddle import printing
import flax
from flax import struct
import flax.linen as nn
from flax.metrics import tensorboard as flax_tb
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import bobbin
from bobbin.example_lib import asrio
from bobbin.example_lib import asrnn

Array = chex.Array
Batch = bobbin.Batch
VarCollection = bobbin.VarCollection


# Similar to the MNIST example, the first part of this script is about data
# and resource handling. You can skip most of them; however, there's an
# important step for distributed training in `prepare_batch` function.

_DEFAULT_WPM_VOCAB_URL = "https://raw.githubusercontent.com/tensorflow/lingvo/master/lingvo/tasks/asr/wpm_16k_librispeech.vocab"  # noqa: E501


def tokenize_dataset(wpm_vocab: asrio.WpmVocab, ds: tf.data.Dataset) -> tf.data.Dataset:
    def tf_tokenizer(row: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        row["tokens"] = tf.numpy_function(
            lambda s: np.asarray(asrio.wpm_encode_sentence(wpm_vocab, s), np.int32),
            [row["text"]],
            tf.int32,
        )
        row["token_paddings"] = tf.zeros_like(row["tokens"], dtype=tf.float32)
        return row

    ds = ds.map(tf_tokenizer, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def pad_smaller_batch(
    ds: tf.data.Dataset,
    batch_size: int,
    padded_shapes: Dict[str, Sequence[int]],
    padding_values: Dict[str, Any],
) -> tf.data.Dataset:
    # pad partial batch at the last
    ds = ds.padded_batch(
        1,
        padded_shapes={k: [batch_size] + v for k, v in padded_shapes.items()},
        padding_values=padding_values,
    )
    return ds.map(
        lambda row: {k: v[0] for k, v in row.items()},
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def batch_dataset(
    ds: tf.data.Dataset, *, batch_size: int, is_train: bool
) -> tf.data.Dataset:
    max_speech_length = 273_600 if is_train else 576_000
    max_token_length = 256
    padding_values = dict(
        speech=tf.constant(0, np.int16),
        speech_paddings=1.0,
        tokens=0,
        token_paddings=1.0,
    )
    padded_shapes = dict(
        speech=[max_speech_length],
        speech_paddings=[max_speech_length],
        tokens=[max_token_length],
        token_paddings=[max_token_length],
    )

    def add_speech_paddings(row: Dict[str, tf.Tensor]):
        ret = row.copy()
        ret["speech_paddings"] = tf.zeros_like(ret["speech"], dtype=tf.float32)
        return ret

    def remove_unknown_fields(row: Dict[str, tf.Tensor]):
        return {k: v for k, v in row.items() if k in padded_shapes}

    def filter_oversized_samples(row: Dict[str, tf.Tensor]):
        fit = True
        for k, max_shape in padded_shapes.items():
            tf.debugging.assert_rank(row[k], len(max_shape))
            actual_shape = tf.shape(row[k])
            dim_fit = tf.math.less_equal(actual_shape, max_shape)
            fit = tf.logical_and(tf.math.reduce_all(dim_fit), fit)
        return fit

    ds = ds.map(add_speech_paddings, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(remove_unknown_fields, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.filter(filter_oversized_samples)
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        drop_remainder=is_train,
    )
    if not is_train:
        ds = pad_smaller_batch(ds, batch_size, padded_shapes, padding_values)
    return ds


def prepare_datasets(
    tfds_data_dir: Optional[str] = None,
    wpm_vocab_path: str = "",
    wpm_size_limit: Optional[int] = None,
    train_batch_size: int = 64,
    eval_batch_size: int = 8,
    max_train_batches: Optional[int] = None,
) -> Tuple[int, tf.data.Dataset, dict[str, tf.data.Dataset]]:
    builder_kwargs = dict(config="lazy_decode")
    read_config = tfds.ReadConfig(shuffle_seed=jax.process_index())
    train_dataset = tfds.load(
        "librispeech",
        split="train_clean100+train_clean360+train_other500",
        data_dir=tfds_data_dir,
        read_config=read_config,
        builder_kwargs=builder_kwargs,
        shuffle_files=True,
        download=False,
    )
    num_examples = int(train_dataset.cardinality())

    # Evaluation datasets for jax multi-process training should be evenly split
    # for each process.  For this purpose, `tfds.split_for_jax_process` can be
    # used as follows.  In theory, training data should also be split, but we
    # don't care because it is anyway repeated with shuffling, so we only need
    # to ensure that different random seeds are used in different processes.
    eval_dataset_names = ("dev", "test_clean", "test_other")
    eval_datasets = tfds.load(
        "librispeech",
        split=tuple(
            tfds.split_for_jax_process(s)
            for s in (
                "dev_clean+dev_other",
                "test_clean",
                "test_other",
            )
        ),
        data_dir=tfds_data_dir,
        builder_kwargs=builder_kwargs,
        download=False,
    )

    wpm_vocab = asrio.WpmVocab.load(wpm_vocab_path, size_limit=wpm_size_limit)
    train_dataset, *eval_datasets = [
        tokenize_dataset(wpm_vocab, ds) for ds in (train_dataset, *eval_datasets)
    ]

    # Here, we take the random seed from `jax.process_index()`.
    train_dataset = train_dataset.repeat().shuffle(
        buffer_size=1024, seed=jax.process_index()
    )
    train_dataset = batch_dataset(
        train_dataset, batch_size=train_batch_size, is_train=True
    )
    eval_datasets = [
        batch_dataset(ds, batch_size=eval_batch_size, is_train=False)
        for ds in eval_datasets
    ]

    if max_train_batches is not None:
        train_dataset = train_dataset.take(max_train_batches)

    train_dataset, *eval_datasets = [
        ds.prefetch(1) for ds in (train_dataset, *eval_datasets)
    ]

    return (
        num_examples,
        train_dataset,
        {name: ds for name, ds in zip(eval_dataset_names, eval_datasets)},
    )


# Similar to MNIST sample, we define a neural network to be optimized as
# `flax.linen.Module`.  We don't go details of this implementation. It is a
# simple module that takes `waveform` and `waveform_paddings` as inputs, and
# returns `logits` and `logit_paddings` as outputs.  Here, `*_paddings` is used
# for packing variable-length samples and logits in the fixed-size batches.
# `*_paddings` is used as a padding indicator, that means that if
# `logit_paddings[b, t] == 1`, `logits[b, t, ...]` is padded values, and should
# not be used. The padding indicators expected to have 0 or 1 values.
class CtcAsrModel(nn.Module):
    feature_normalizer: Optional[asrio.MeanVarNormalizer] = None
    frontend: nn.Module = asrnn.LogMelFilterBank()
    encoder: nn.Module = asrnn.CnnConformerEncoder()
    specaug: nn.Module = asrnn.SpecAug()
    classifier: nn.Module = nn.Dense(-1)

    @nn.compact
    def __call__(
        self, waveform: Array, waveform_paddings: Array, is_eval: bool = False
    ) -> Tuple[Array, Array]:
        features, feature_paddings = self.frontend(waveform, waveform_paddings)
        self.sow(
            "tensorboard",
            "features",
            bobbin.MplImageSummary(
                features[0].T,
                aspect="auto",
                origin="lower",
                with_colorbar=True,
                h_paddings=feature_paddings[0],
            ),
        )
        if self.feature_normalizer is not None:
            features = self.feature_normalizer.apply(features)
        features = self.specaug(features, feature_paddings, deterministic=is_eval)
        self.sow(
            "tensorboard",
            "masked_features",
            bobbin.MplImageSummary(
                features[0].T,
                aspect="auto",
                origin="lower",
                with_colorbar=True,
                h_paddings=feature_paddings[0],
            ),
        )

        encodes, encode_paddings = self.encoder(
            features, feature_paddings, is_eval=is_eval
        )
        logits = self.classifier(encodes)
        return logits, encode_paddings


# As similar to MNIST example, the auxiliary output for loss function is defined
# as follows.
class LossAuxOut(struct.PyTreeNode):
    logits: Array
    logit_paddings: Array
    per_sample_loss: Array


# Again similar to MNIST example, training task is defined by subclassing
# `bobbin.TrainTask`.
class CtcAsrTask(bobbin.TrainTask):
    def __init__(
        self,
        model: nn.Module,
        learn_rate_fn: optax.Schedule,
        example_shape: Sequence[int],
    ):
        example_args = (
            jnp.zeros((1, *example_shape), dtype=np.int16),
            jnp.ones((1, example_shape[0]), dtype=np.float32),
        )
        # As described in MNIST example, the constructor of `bobbin.TrainTask`
        # takes following arguments:
        #   - model to be trained
        #   - example inputs for determining the shapes of the variables.
        #   - the names of random-number generators (RNGs) required for
        #     computing the loss function.
        # For RNGs, it should be noted that this model requires two kinds of
        # RNGs; one for dropout, and one for SpecAug.
        super().__init__(
            model,
            example_args=example_args,
            required_rngs=("dropout", "specaug"),
        )
        # This TrainTask holds learning rate function for publishing learning
        # rates to TensorBoard.
        self._learn_rate_fn = learn_rate_fn

    # `bobbin.TrainTask.compute_loss` is the central part of the training.
    # This method must be overridden to define how to compute the loss function
    # from the given parameters and the input batch.
    def compute_loss(
        self, params, batch, *, extra_vars, prng_key, step
    ) -> Tuple[chex.Scalar, Tuple[VarCollection, LossAuxOut]]:
        # Prepare model variables for loss computation as we did in MNIST
        # example.
        model_vars = extra_vars.copy()
        model_vars.update(params=params, tensorboard=dict())

        # CTC loss is computed over logits computed.
        (logits, logit_paddings), updated_vars = self._model.apply(
            model_vars,
            batch["speech"],
            batch["speech_paddings"],
            is_eval=False,
            rngs=self.get_rng_dict(prng_key),
            mutable=flax.core.DenyList("params"),
        )
        per_sample_loss = optax.ctc_loss(
            logits, logit_paddings, batch["tokens"], batch["token_paddings"]
        )

        # CTC loss is normalized using the numbers of tokens. The numbers of
        # tokens can be obtained by using the pading indicators.
        num_tokens = jnp.maximum(jnp.sum(1.0 - batch["token_paddings"], axis=-1), 1.0)
        per_token_loss = per_sample_loss / num_tokens
        loss = jnp.mean(per_token_loss)

        # Update TensorBoard variables so it includes learning rate and loss
        # values.
        # The API for updating variables from `compute_loss` might be subjected
        # to change near future.
        tb_vars = updated_vars["tensorboard"].unfreeze()
        tb_vars["loss"] = bobbin.ScalarSummary(loss)
        tb_vars["learn_rate"] = bobbin.ScalarSummary(self._learn_rate_fn(step))
        updated_vars = updated_vars.copy(dict(tensorboard=tb_vars))

        return loss, (
            updated_vars,
            LossAuxOut(
                logits=logits,
                logit_paddings=logit_paddings,
                per_sample_loss=per_sample_loss,
            ),
        )


# Here we define two utility functions for publishing error informations.
def _error_to_log_message(error: asrio.SequenceError) -> str:
    return (
        f"S={error.subs}, D={error.dels}, I={error.inss}, "
        f"ER={error.error_rate * 100.0}%"
    )


def _write_error_to_tensorboard(
    error: asrio.SequenceError,
    writer: flax_tb.SummaryWriter,
    token_type: str,
    step: int,
    prefix: str = "",
) -> None:
    writer.scalar(prefix + f"error_rate_{token_type}", error.error_rate, step=step)
    writer.scalar(prefix + f"num_subs_{token_type}", error.subs, step=step)
    writer.scalar(prefix + f"num_dels_{token_type}", error.dels, step=step)
    writer.scalar(prefix + f"num_inss_{token_type}", error.inss, step=step)
    writer.scalar(prefix + f"num_refs_{token_type}", error.refs, step=step)
    writer.scalar(prefix + f"num_hyps_{token_type}", error.hyps, step=step)


# EvalResult for LibriSpeech task is composition of four kinds of errors,
# `token_error`, `word_error`, `char_error`, and sentence error.
# The first three kinds of errors are edit-distances, we use
# `asrio.SequenceError` to compute and hold edit-distances. The sentence errors
# are counted by using `num_sentences` and `num_sentence_errors` variables.
# In addition to that this metric holds timing information for measuring the
# decoding speed.  `start_time` is a timestamp for the moment when evaluation is
# started, and `end_time` is a timestamp for the moment when result is
# finalized. `sampled_results` field is for keeping some sampled results for
# monitoring the decoding results on TensorBoard.
class EvalResults(bobbin.EvalResults):
    token_error: asrio.SequenceError = struct.field(default_factory=asrio.SequenceError)
    word_error: asrio.SequenceError = struct.field(default_factory=asrio.SequenceError)
    char_error: asrio.SequenceError = struct.field(default_factory=asrio.SequenceError)
    num_sentences: int = 0
    num_sentence_errors: int = 0
    start_time: float = np.inf
    end_time: float = 0
    sampled_results: bobbin.SampledSet[Tuple[str, str]] = struct.field(
        pytree_node=False, default=bobbin.SampledSet(max_size=3)
    )

    # `end_time` will be populated when finalizing the result, so this property
    # needs to support the case when `end_time == 0` and `start_time` is an
    # actual start timestamp.
    @property
    def wall_time(self) -> Optional[float]:
        return (
            None if self.start_time > self.end_time else self.end_time - self.start_time
        )

    @property
    def sentences_per_second(self) -> Optional[float]:
        if self.wall_time is None:
            return None
        return self.num_sentences / self.wall_time

    @property
    def sentence_error_rate(self) -> float:
        return self.num_sentence_errors / self.num_sentences

    def to_log_message(self) -> str:
        sample_summary = ""
        for hyp, ref in self.sampled_results:
            sample_summary += f" REF: {ref}\n" f" HYP: {hyp}\n"
        return (
            "Token: "
            + _error_to_log_message(self.token_error)
            + "\n Char: "
            + _error_to_log_message(self.char_error)
            + "\n Word: "
            + _error_to_log_message(self.word_error)
            + f"\nSent.: ER={self.sentence_error_rate * 100}%"
            + f", N={self.num_sentences}"
            + f" ({self.sentences_per_second} sentences/sec)\n"
            + sample_summary
        )

    # As same as in the MNIST example, `reduce` is used to merge two
    # `EvalResults` from different batch.
    def reduce(self, other: EvalResults) -> EvalResults:
        kwargs = dict()
        for key in (
            "token_error",
            "word_error",
            "char_error",
            "num_sentences",
            "num_sentence_errors",
        ):
            kwargs[key] = jax.tree_util.tree_map(
                lambda x, y: x + y, getattr(self, key), getattr(other, key)
            )

        # reduction of timing information is a bit tricky.  Here, we take
        # minimum of `start_time` and maximum of `end_time` so we keep the
        # timing information from the first `EvalResults` created, and the last
        # `EvalResults` finalized.
        kwargs.update(
            start_time=min(self.start_time, other.start_time),
            end_time=max(self.end_time, other.end_time),
        )

        # TODO: This is not clean, but `gather_from_jax_processes` used in
        # `evaluate_batches` returns copy of the original fields for
        # non-pytree fields, and those copies dominates all N sampled
        # hypotheses. So, here, it takes union only if `other.sampled_results`
        # are not copies.
        sampled_results = (
            self.sampled_results.union(other.sampled_results)
            if self.sampled_results != other.sampled_results
            else self.sampled_results
        )
        kwargs.update(sampled_results=sampled_results)
        return EvalResults(**kwargs)

    # `is_better_than` will be used for keeping the "best" results and
    # checkpoint.
    def is_better_than(self, other: EvalResults) -> bool:
        return self.word_error.error_rate < other.word_error.error_rate

    # `write_to_tensorboard` in this example publishes some error rates, and the
    # number of processed tokens/ chars/ words.  In addition to those, it
    # outputs sampled decoding results as text summaries.
    def write_to_tensorboard(
        self,
        current_train_state: bobbin.TrainState,
        writer: flax_tb.SummaryWriter,
    ) -> None:
        step = current_train_state.step
        _write_error_to_tensorboard(self.token_error, writer, "token", step, "eval/")
        _write_error_to_tensorboard(self.char_error, writer, "char", step, "eval/")
        _write_error_to_tensorboard(self.word_error, writer, "word", step, "eval/")
        writer.scalar("eval/sent_error_rate", self.sentence_error_rate, step=step)
        writer.scalar("eval/sent_refs", self.num_sentences, step=step)
        if self.sentences_per_second is not None:
            logging.info("Sentences per sec (Eval) = %f", self.sentences_per_second)
            writer.scalar("eval/sent_per_sec", self.sentences_per_second, step=step)

        for hyp, ref in self.sampled_results:
            writer.text(
                "eval/sampled_results", f"```\nREF: {ref}\nHYP: {hyp}\n```", step=step
            )


# EvalTask is similar to what we saw in the MNIST example.
class EvalTask(bobbin.EvalTask):
    def __init__(self, model: nn.Module, wpm_vocab: asrio.WpmVocab):
        self.model = model
        self.wpm_vocab = wpm_vocab

    # Note that `start_time` is set here.
    def create_eval_results(self, unused_dataset_name):
        return EvalResults(start_time=time.time())

    # Here, we have an extra function that is called from `EvalResults.evaluate`.
    # This part of function is separated out from `evaluate` since we want to do
    # device-parallel JIT-ed computation only for this part of evaluation.
    @functools.partial(
        bobbin.tpmap,
        axis_name="batch",
        argtypes=["static", "shard", "broadcast"],
        donate_argnums=(1,),
    )
    def predict(self, batch: Batch, model_vars: VarCollection) -> Tuple[Array, Array]:
        logits, logit_paddings = self.model.apply(
            model_vars, batch["speech"], batch["speech_paddings"], is_eval=True
        )
        predicts = logits.argmax(axis=-1)
        return predicts, logit_paddings

    # Rest of the evaluation process is written in this `evaluate` function.
    def evaluate(self, batch: Batch, model_vars: VarCollection) -> EvalResults:
        start_time = time.time()
        if batch is None:
            # This is only allowed when we are sure that there's no parallel
            # reduction (e.g. `jax.lax.p*` functions) used in `parallel_predict`.
            # Otherwise, all reduction operators must be called in the same
            # order in each process. Usually this is done by feeding a dummy
            # batch.
            return EvalResults()
        results = self.predict(batch, model_vars)
        # It's important to ensure both are NumPy-array (and on CPU)
        results = jax.tree_util.tree_map(np.asarray, results)
        predicts, predict_paddings = bobbin.flatten_leading_axes(results)

        # Here, we remove CTC blanks and convert padded token-id batches into
        # a list-of-lists.
        batched_hyp_ids = asrio.remove_ctc_blanks_and_repeats(
            predicts, predict_paddings
        )
        batched_ref_ids = asrio.make_list_from_padded_batch(
            np.asarray(batch["tokens"]), np.asarray(batch["token_paddings"])
        )

        # Actual result is computed in the other function.
        results = self._build_results(batched_hyp_ids, batched_ref_ids)

        # Setting `start_time` here is actually redundant.
        # Evaluation loop runs as in the following pseudo-code:
        #    results = self.create_eval_results("dataset_name")
        #    for batch in batches:
        #      new_result = self.evaluate(batch, model_vars)
        #      results = results.reduce(new_results)
        # As shown above, if we have `start_time` at the first line of the code
        # we don't get `start_time` that is smaller than that in the `evaluate`
        # later. So, it is redundant but kept as it is for clarity.
        return results.replace(start_time=start_time, end_time=time.time())

    # This function build a result for the given sequences of token ids.
    def _build_results(
        self,
        batched_hyp_ids: Iterable[Sequence[int]],
        batched_ref_ids: Iterable[Sequence[int]],
    ) -> EvalResults:
        results = EvalResults()
        acc_token_error = results.token_error
        acc_word_error = results.word_error
        acc_char_error = results.char_error
        sent_err = 0
        num_sentences = 0
        sampled_results = bobbin.SampledSet(max_size=3)

        # For each sequence in the batch...
        for hyp_ids, ref_ids in zip(batched_hyp_ids, batched_ref_ids):
            if not ref_ids:
                continue  # all-pad sequences are treated as invalid sequence.

            # Compute token errors first.
            acc_token_error = acc_token_error.accumulate(hyp_ids, ref_ids)

            # Then, detokenize for comparing texts.
            hyp_text = asrio.wpm_decode_sentence(self.wpm_vocab, hyp_ids)
            ref_text = asrio.wpm_decode_sentence(self.wpm_vocab, ref_ids)

            # Word errors and character errors are computed in the same way.
            acc_word_error = acc_word_error.accumulate(
                hyp_text.split(" "), ref_text.split(" ")
            )
            acc_char_error = acc_char_error.accumulate(list(hyp_text), list(ref_text))

            # Sentence error can be computed by directly compare texts.
            if hyp_text != ref_text:
                sent_err += 1
            num_sentences += 1

            # Finally, add some samples from pairs of hypothesis and reference
            # strings.
            sampled_results = sampled_results.add((hyp_text, ref_text))

        return dataclasses.replace(
            results,
            token_error=acc_token_error,
            word_error=acc_word_error,
            char_error=acc_char_error,
            num_sentences=results.num_sentences + num_sentences,
            num_sentence_errors=results.num_sentence_errors + sent_err,
            sampled_results=sampled_results,
        )


# This function makes Transformer schedule function.
def transformer_schedule(
    base_learn_rate: float = 5.0, warmup_steps: int = 10000, model_dims: int = 256
) -> optax.Schedule:
    def schedule_fn(count):
        return base_learn_rate * (
            model_dims**-0.5
            * jnp.minimum((count + 1) * warmup_steps**-1.5, (count + 1) ** -0.5)
        )

    return schedule_fn


# Here, we have an additional structure for holding learning rate function
# separately in addition to `GradientTransformation`.  This is for publishing
# learning rates to TensorBoard.
class Optimizer(struct.PyTreeNode):
    """Pair of gradient transformation and learning rate scheuduler."""

    tx: optax.GradientTransformation
    learn_rate: optax.Schedule


# Fiddle configurator for `Optimizer`.
def make_optimizer_config() -> fdl.Config[Optimizer]:
    learn_rate_cfg = fdl.Config(transformer_schedule)
    tx_cfg = fdl.Config(optax.adamw, learn_rate_cfg, weight_decay=1e-6)
    return fdl.Config(Optimizer, tx=tx_cfg, learn_rate=learn_rate_cfg)


# This example doesn't use "autoconfig" feature of Fiddle. So, config building
# is explicitly separated here as a function.
def make_train_task_config(
    num_outputs: int,
    feature_normalizer: asrio.MeanVarNormalizer,
    speech_shape: Sequence[int],
    opt_config: fdl.Config[Optimizer],
) -> fdl.Config[CtcAsrModel]:
    default_depth = 4
    model_cfg = fdl.Config(
        CtcAsrModel,
        feature_normalizer=feature_normalizer,
        frontend=fdl.Config(asrnn.LogMelFilterBank),
        encoder=fdl.Config(
            asrnn.CnnConformerEncoder,
            cnn=fdl.Config(asrnn.CnnEncoder),
            conformer_blocks=tuple(
                fdl.Config(asrnn.ConformerBlock)
                for unused_depth in range(default_depth)
            ),
        ),
        specaug=fdl.Config(asrnn.SpecAug),
        classifier=fdl.Config(nn.Dense, features=num_outputs),
    )
    return fdl.Config(CtcAsrTask, model_cfg, opt_config.learn_rate, speech_shape)


# The following two utility functions are for simplifying command-line
# arguments.
#
# This function fills default arguments by downloading WPM definition from the
# web, and by finding the path for mean/ stddev statistics as a relative path
# from this script path.
def fill_default_arguments(args: argparse.Namespace):
    if args.wpm_vocab is None:
        f = tempfile.NamedTemporaryFile()
        f.write(urllib.request.urlopen(_DEFAULT_WPM_VOCAB_URL).read())
        f.flush()
        args.wpm_vocab = f.name
        args.wpm_vocab_file = f
    if args.feature_normalizer is None:
        p = epath.Path(__file__).parent / "librispeech.meanstddev.logmelfb80.json"
        args.feature_normalizer = str(p)


# If the script is running on Cloud TPU, we can safely call
# `jax.distributed.initialize()` as all required configuration is automatically
# done even when the program is running on a single process. So, this function
# is used to enable "--multi_process" flag if it is running on Cloud TPU.
def _is_running_on_cloud_tpu_vm() -> bool:
    libtpu_found = False
    try:
        # pytype: disable=import-error
        import libtpu

        del libtpu
        # pytype: enable=import-error
        libtpu_found = True
    except ImportError:
        pass
    return os.environ.get("CLOUDSDK_PYTHON", None) is not None and libtpu_found


# Finally, main function is here and it is not so different from that of the
# MNIST example.
def main(args: argparse.Namespace):
    fill_default_arguments(args)

    # For multi-process training, we first need to call
    # `jax.distributed.initialize()`.  On cloud TPUs, it is safe to call this
    # function without explicit configurations.
    # Here, we don't specify the configuration arguments here, so if you want to
    # run multi-process training without cloud TPUs, you may need to set some
    # environment variables.
    if _is_running_on_cloud_tpu_vm() or args.multi_process:
        jax.distributed.initialize()

    if jax.process_index() == 0:
        logging.basicConfig(stream=sys.stderr)
        logging.root.setLevel(logging.INFO)

    num_train_samples, train_ds, eval_dss = prepare_datasets(
        tfds_data_dir=args.tfds_data_dir,
        wpm_vocab_path=args.wpm_vocab,
        wpm_size_limit=args.wpm_size_limit,
        train_batch_size=args.per_device_batch_size * jax.local_device_count(),
        eval_batch_size=args.per_device_batch_size * jax.local_device_count(),
        max_train_batches=args.max_steps,
    )
    batch_size, *speech_shape = train_ds.element_spec["speech"].shape
    global_batch_size = batch_size * jax.process_count()
    eval_batch_gens = {dsname: ds.as_numpy_iterator for dsname, ds in eval_dss.items()}

    all_checkpoint_path = args.log_dir_path / "all_ckpts"
    best_checkpoint_path = args.log_dir_path / "best_ckpts"
    tensorboard_path = args.log_dir_path / "tensorboard"

    # Some differences from the MNIST example is that we need to handle external
    # resources like WPM vocabulary and feature normalizers.
    wpm_vocab = asrio.WpmVocab.load(args.wpm_vocab, size_limit=args.wpm_size_limit)
    normalizer = None
    if args.feature_normalizer is not None:
        with open(args.feature_normalizer) as f:
            # One can use `bobbin.parse_pytree_json` for reading some artifacts
            # serialized using `bobbin.dump_pyree_json`.
            normalizer = bobbin.parse_pytree_json(
                f.read(), asrio.MeanVarNormalizer.empty()
            )

    # This example is also for demonstrating Fiddle, and Fiddle first
    # constructs config object before actual instances being built.
    opt_cfg = make_optimizer_config()
    train_task_cfg = make_train_task_config(
        len(wpm_vocab), normalizer, speech_shape, opt_cfg
    )

    # Fiddle provides us freedom to modify some hyperparemeter here.
    if args.model_size.upper() == "100M":
        train_task_cfg.model.encoder.cnn.num_outputs = 512
        block_cfg = copy.deepcopy(train_task_cfg.model.encoder.conformer_blocks[0])
        block_cfg.kernel_size = 32
        train_task_cfg.model.encoder.conformer_blocks = tuple(
            copy.deepcopy(block_cfg) for unused_d in range(17)
        )
    elif args.model_size.upper() == "DEBUG":
        pass
    elif args.model_size.upper() == "UNITTEST":
        train_task_cfg.model.encoder.cnn.channels = (5, 5)
        train_task_cfg.model.encoder.cnn.num_outputs = 16
        block_cfg = copy.deepcopy(train_task_cfg.model.encoder.conformer_blocks[0])
        block_cfg.kernel_size = 3
        train_task_cfg.model.encoder.conformer_blocks = tuple(
            copy.deepcopy(block_cfg) for unused_d in range(2)
        )
    else:
        raise ValueError("model_size should be one of: '100m' or 'debug")

    # Actual optimizer and task are built with `fdl.build` as follows
    opt = fdl.build(opt_cfg)
    task = fdl.build(train_task_cfg)

    # Here, we configure model, optimizer, and tasks.
    evaler = EvalTask(task.model, wpm_vocab)

    # init must be deterministic for multi-host training
    train_state = task.initialize_train_state(
        jax.random.PRNGKey(0), opt.tx, checkpoint_path=all_checkpoint_path
    )
    train_state = flax.jax_utils.replicate(train_state, jax.local_devices())
    # raise SystemExit

    # As explained in Jax's multi-process training document
    # (https://jax.readthedocs.io/en/latest/multi_process.html), multi-process
    # training doesn't need any modification on the model if the function is
    # pmapped and `jax.lax.p*` functions are properly used.
    # So, here `pmap` is necessary.
    train_step_fn = task.make_training_step_fn(
        split_steps=args.split_training_batch
    ).pmap("batch")

    prng_key = jax.random.PRNGKey(jax.process_index() + 3)

    train_writer = (
        flax_tb.SummaryWriter(tensorboard_path / "train")
        if jax.process_index() == 0
        else bobbin.NullSummaryWriter()
    )
    bobbin.publish_trainer_env_info(train_writer, train_state)
    train_writer.text(
        "fiddle_hparams", printing.as_str_flattened(train_task_cfg), step=0
    )
    # Seeting up crontab (auxiliary actions periodically executed during the training)
    eval_freq = num_train_samples // global_batch_size
    warmup = 10
    crontab = bobbin.CronTab()

    # Here, it is important to know which process does what. For evaluation, we
    # use multiple processes and therefore it involves "pmap". For such action,
    # we need to register the same action in all the host involved. For other
    # I/O related actions, we shouldn't do that in the follower processes for
    # avoiding overwrite.
    crontab.schedule(
        evaler.make_cron_action(
            eval_batch_gens, tensorboard_root_path=tensorboard_path
        ).keep_best_checkpoint(
            "dev",
            best_checkpoint_path,
        ),
        step_interval=eval_freq,
    )
    # So, except for evaluation action above, every other action is registered
    # only if `jax.process_index() == 0`.
    if jax.process_index() == 0:
        crontab.schedule(
            task.make_checkpoint_saver(all_checkpoint_path),
            step_interval=1000,
            at_step=warmup,
        )
        crontab.schedule(
            task.make_log_writer(), time_interval=30.0, at_first_steps_of_process=warmup
        )
        crontab.schedule(
            task.make_training_progress_publisher(train_writer),
            step_interval=100,
        )

    # So we have N copies of train_state when we run this program with N
    # processes. However, those should have exactly same values if
    # `compute_loss` is properly written with using `jax.lax.p*` functions.
    logging.info("MAIN LOOP STARTS with devices %s", str(jax.local_devices()))
    for batch in train_ds.as_numpy_iterator():
        rng, prng_key = jax.random.split(prng_key)
        train_state, step_info = train_step_fn(train_state, batch, rng)
        crontab.run(train_state, step_info=step_info)


if __name__ == "__main__":
    # Disable TF's memory preallocation if TF is built with CUDA.
    tf.config.experimental.set_visible_devices([], "GPU")

    argparser = argparse.ArgumentParser(description="LibriSpeech training")
    argparser.add_argument("--tfds_data_dir", type=str, default=None)
    argparser.add_argument("--feature_normalizer", type=str, default=None)
    argparser.add_argument("--per_device_batch_size", type=int, default=8)
    argparser.add_argument("--wpm_vocab", type=str, default=None)
    argparser.add_argument("--wpm_size_limit", type=int, default=1024)
    argparser.add_argument("--log_dir_path", type=epath.Path, default="log")
    argparser.add_argument("--split_training_batch", type=int, default=None)
    argparser.add_argument("--multi_process", type=bool, default=None)
    argparser.add_argument("--max_steps", type=int, default=None)
    argparser.add_argument("--model_size", type=str, default="DEBUG")

    args = argparser.parse_args()
    main(args)
