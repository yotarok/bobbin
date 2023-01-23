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

"""Sample with LibriSpeech.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import pathlib
import tempfile
import time
from typing import Any, Dict, Optional, Sequence, Tuple
import urllib.request

import chex
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

_Array = chex.Array
_Batch = bobbin.Batch
_Parameter = bobbin.Parameter
_PRNGKey = chex.PRNGKey
_Scalar = chex.Scalar
_TrainState = bobbin.TrainState
_VarCollection = bobbin.VarCollection
_ModuleDef = Any

_AtFirstNSteps = bobbin.AtFirstNSteps
_AtNthStep = bobbin.AtNthStep
_ForEachNSteps = bobbin.ForEachNSteps
_ForEachTSeconds = bobbin.ForEachTSeconds
_PublishTrainingProgress = bobbin.PublishTrainingProgress
_RunEval = bobbin.RunEval
_SaveCheckpoint = bobbin.SaveCheckpoint
_WriteLog = bobbin.WriteLog


_DEFAULT_WPM_VOCAB_URL = "https://raw.githubusercontent.com/tensorflow/lingvo/master/lingvo/tasks/asr/wpm_16k_librispeech.vocab"  # noqa: E501


# Disable TF's memory preallocation if TF is built with CUDA.
tf.config.experimental.set_visible_devices([], "GPU")


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
    return ds.padded_batch(
        batch_size,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        drop_remainder=is_train,
    )


def prepare_datasets(
    tfds_data_dir: Optional[str] = None,
    wpm_vocab_path: str = "",
    wpm_size_limit: Optional[int] = None,
    train_batch_size: int = 64,
    eval_batch_size: int = 8,
) -> tuple[tf.data.Dataset, dict[str, tf.data.Dataset]]:
    builder_kwargs = dict(config="lazy_decode")
    train_dataset = tfds.load(
        "librispeech",
        split="train_clean100+train_clean360+train_other500",
        data_dir=tfds_data_dir,
        builder_kwargs=builder_kwargs,
        shuffle_files=True,
        download=False,
    )
    eval_dataset_names = ("dev", "test_clean", "test_other")
    eval_datasets = tfds.load(
        "librispeech",
        split=(
            "dev_clean+dev_other",
            "test_clean",
            "test_other",
        ),
        data_dir=tfds_data_dir,
        builder_kwargs=builder_kwargs,
        download=False,
    )

    wpm_vocab = asrio.WpmVocab.load(wpm_vocab_path, size_limit=wpm_size_limit)
    train_dataset, *eval_datasets = [
        tokenize_dataset(wpm_vocab, ds) for ds in (train_dataset, *eval_datasets)
    ]

    train_dataset = train_dataset.repeat().shuffle(buffer_size=1024)

    train_dataset = batch_dataset(
        train_dataset, batch_size=train_batch_size, is_train=True
    )
    eval_datasets = [
        batch_dataset(ds, batch_size=eval_batch_size, is_train=False)
        for ds in eval_datasets
    ]

    train_dataset, *eval_datasets = [
        ds.prefetch(1) for ds in (train_dataset, *eval_datasets)
    ]

    return train_dataset, {
        name: ds for name, ds in zip(eval_dataset_names, eval_datasets)
    }


class CtcAsrModel(nn.Module):
    num_outputs: int
    feature_normalizer: Optional[asrio.MeanVarNormalizer] = None

    def setup(self):
        self._encoder = asrnn.CnnConformerEncoder()
        self._frontend = asrnn.LogMelFilterBank()
        self._specaug = asrnn.SpecAug()
        self._classifier = nn.Dense(features=self.num_outputs)

    def __call__(
        self, waveform: _Array, waveform_paddings: _Array, is_eval: bool = False
    ) -> Tuple[_Array, _Array]:
        features, feature_paddings = self._frontend(waveform, waveform_paddings)
        self.sow(
            "tensorboard",
            "features",
            bobbin.MplImageSow(
                features[0].T,
                aspect="auto",
                origin="lower",
                with_colorbar=True,
                h_paddings=feature_paddings[0],
            ),
        )
        if self.feature_normalizer is not None:
            features = self.feature_normalizer.apply(features)
        features = self._specaug(features, feature_paddings, deterministic=is_eval)
        self.sow(
            "tensorboard",
            "masked_features",
            bobbin.MplImageSow(
                features[0].T,
                aspect="auto",
                origin="lower",
                with_colorbar=True,
                h_paddings=feature_paddings[0],
            ),
        )

        encodes, encode_paddings = self._encoder(
            features, feature_paddings, is_eval=is_eval
        )
        logits = self._classifier(encodes)
        return logits, encode_paddings


@struct.dataclass
class LossAuxOut:
    logits: _Array
    logit_paddings: _Array
    per_sample_loss: _Array


class CtcAsrTask(bobbin.TrainTask):
    def __init__(self, model: nn.Module):
        self._model = model

    def compute_loss(
        self, params, batch, *, extra_vars, prng_key
    ) -> Tuple[_Scalar, Tuple[_VarCollection, LossAuxOut]]:
        model_vars = extra_vars.copy()
        model_vars.update(params=params, intermediates=dict())
        rng_dropout, rng_specaug = jax.random.split(prng_key)
        (logits, logit_paddings), updated_vars = self._model.apply(
            model_vars,
            batch["speech"],
            batch["speech_paddings"],
            is_eval=False,
            rngs=dict(dropout=rng_dropout, specaug=rng_specaug),
            mutable=flax.core.DenyList("params"),
        )
        per_sample_loss = optax.ctc_loss(
            logits, logit_paddings, batch["tokens"], batch["token_paddings"]
        )

        num_tokens = jnp.maximum(jnp.sum(1.0 - batch["token_paddings"], axis=-1), 1.0)
        per_token_loss = per_sample_loss / num_tokens

        loss = jnp.mean(per_token_loss)
        tb_vars = updated_vars["tensorboard"].unfreeze()
        tb_vars["loss"] = bobbin.ScalarSow(loss)
        updated_vars = updated_vars.copy(dict(tensorboard=tb_vars))
        return loss, (
            updated_vars,
            LossAuxOut(
                logits=logits,
                logit_paddings=logit_paddings,
                per_sample_loss=per_sample_loss,
            ),
        )


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


@struct.dataclass
class EvalResults(bobbin.EvalResults):
    token_error: asrio.SequenceError = struct.field(default_factory=asrio.SequenceError)
    word_error: asrio.SequenceError = struct.field(default_factory=asrio.SequenceError)
    char_error: asrio.SequenceError = struct.field(default_factory=asrio.SequenceError)
    num_sentences: int = 0
    num_sentence_errors: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    num_result_samples: int = 3
    sampled_hyps: Sequence[tuple[str, str, float]] = ()

    _sum_fields: Sequence[str] = struct.field(
        pytree_node=False,
        default=(
            "token_error",
            "word_error",
            "char_error",
            "num_sentences",
            "num_sentence_errors",
        ),
    )

    @property
    def wall_time(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        assert self.end_time > self.start_time
        return self.end_time - self.start_time

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
        for hyp, ref, unused_sort_order in self.sampled_hyps:
            sample_summary += f" REF: {ref}\n" f" HYP: {hyp}\n"
        return (
            "Token: "
            + _error_to_log_message(self.token_error)
            + "\n Char: "
            + _error_to_log_message(self.char_error)
            + "\n Word: "
            + _error_to_log_message(self.word_error)
            + f"\nSent.: ER= {self.sentence_error_rate * 100}%"
            + f" ({self.sentences_per_second} sentences/sec)\n"
            + sample_summary
        )

    def preduce(self, axis_name: str) -> EvalResults:
        kwargs = dict()
        for key in self._sum_fields:
            kwargs[key] = jax.tree_util.tree_map(
                lambda x: jax.lax.psum(x, axis_name=axis_name), getattr(self, key)
            )
        # Sampled senteces and time metrics are okay not to reduce.
        kwargs.update(
            start_time=self.start_time,
            end_time=self.end_time,
            num_result_samples=self.num_result_samples,
            sampled_hyps=self.sampled_hyps,
        )
        return EvalResults(**kwargs)

    def reduce(self, other: EvalResults) -> EvalResults:
        kwargs = dict()
        for key in self._sum_fields:
            kwargs[key] = jax.tree_util.tree_map(
                lambda x, y: x + y, getattr(self, key), getattr(other, key)
            )

        if self.start_time is None or other.start_time is None:
            start_time = self.start_time or other.start_time
        else:
            start_time = min(self.start_time, other.start_time)
        if self.end_time is None or other.end_time is None:
            end_time = self.end_time or other.end_time
        else:
            end_time = max(self.end_time, other.end_time)
        kwargs.update(start_time=start_time, end_time=end_time)

        sampled_hyps = list(self.sampled_hyps) + list(other.sampled_hyps)
        sampled_hyps.sort(key=lambda x: x[2])
        sampled_hyps = tuple(sampled_hyps[: self.num_result_samples])
        kwargs.update(
            num_result_samples=self.num_result_samples, sampled_hyps=sampled_hyps
        )
        return EvalResults(**kwargs)

    def is_better_than(self, other: EvalResults) -> bool:
        return self.word_error.error_rate < other.word_error.error_rate

    def write_to_tensorboard(
        self, current_train_state: _TrainState, writer: flax_tb.SummaryWriter
    ) -> None:
        step = current_train_state.step
        _write_error_to_tensorboard(self.token_error, writer, "token", step, "eval/")
        _write_error_to_tensorboard(self.char_error, writer, "char", step, "eval/")
        _write_error_to_tensorboard(self.word_error, writer, "word", step, "eval/")
        writer.scalar("eval/sent_error_rate", self.sentence_error_rate, step=step)
        writer.scalar("eval/sent_refs", self.num_sentences, step=step)
        if self.sentences_per_second is not None:
            print(f"Sentences per sec (Eval) = {self.sentences_per_second}")
            writer.scalar("eval/sent_per_sec", self.sentences_per_second, step=step)

        for hyp, ref, unused_sort_order in self.sampled_hyps:
            writer.text(
                "eval/sampled_results", f"```\nREF: {ref}\nHYP: {hyp}\n```", step=step
            )


def _recover_padded_tokens(token_ids: np.ndarray, paddings: np.ndarray) -> np.ndarray:
    token_ids = np.asarray(token_ids).tolist()
    return [
        [lab for pad, lab in zip(pads, padded_ids) if pad < 0.5]
        for padded_ids, pads in zip(token_ids, paddings)
    ]


def _resolve_ctc_repetition(
    token_ids: Sequence[int], *, blank_id: int = 0
) -> Sequence[int]:
    prev = None
    unduped = []
    for tok in token_ids:
        if tok != prev:
            unduped.append(tok)
        prev = tok
    return [tok for tok in unduped if tok != blank_id]


class EvalTask(bobbin.EvalTask):
    def __init__(self, model: nn.Module, wpm_vocab: asrio.WpmVocab):
        self.model = model
        self.wpm_vocab = wpm_vocab

    def create_eval_results(self):
        return EvalResults(start_time=time.time())

    def finalize_eval_results(self, metrics: EvalResults) -> EvalResults:
        return metrics.replace(end_time=time.time())

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def predict(
        self, batch: _Batch, model_vars: _VarCollection
    ) -> Tuple[_Array, _Array]:
        logits, logit_paddings = self.model.apply(
            model_vars, batch["speech"], batch["speech_paddings"], is_eval=True
        )
        predicts = logits.argmax(axis=-1)
        return predicts, logit_paddings

    def evaluate(self, batch: _Batch, model_vars: _VarCollection) -> EvalResults:
        predicts, predict_paddings = self.predict(batch, model_vars)

        # It's important to ensure both are NumPy-array (and on CPU)
        predicts = np.asarray(predicts)
        predict_paddings = np.asarray(predict_paddings)

        sampled_hyps = []

        results = EvalResults()
        batched_hyp_ids = _recover_padded_tokens(predicts, predict_paddings)
        batched_ref_ids = _recover_padded_tokens(
            batch["tokens"], batch["token_paddings"]
        )

        acc_token_error = results.token_error
        acc_word_error = results.word_error
        acc_char_error = results.char_error
        sent_err = 0
        for hyp_ids, ref_ids in zip(batched_hyp_ids, batched_ref_ids):
            hyp_ids = _resolve_ctc_repetition(hyp_ids)
            acc_token_error = acc_token_error.accumulate(hyp_ids, ref_ids)
            hyp_text = (
                "".join(self.wpm_vocab.id2str.get(i, "█") for i in hyp_ids)
                .replace("▁", " ")
                .strip()
            )
            ref_text = (
                "".join(self.wpm_vocab.id2str.get(i, "█") for i in ref_ids)
                .replace("▁", " ")
                .strip()
            )
            acc_word_error = acc_word_error.accumulate(
                hyp_text.split(" "), ref_text.split(" ")
            )
            acc_char_error = acc_char_error.accumulate(list(hyp_text), list(ref_text))
            if hyp_text != ref_text:
                sent_err += 1

            sampled_hyps.append((hyp_text, ref_text, np.random.uniform()))

        return dataclasses.replace(
            results,
            token_error=acc_token_error,
            word_error=acc_word_error,
            char_error=acc_char_error,
            num_sentences=results.num_sentences + len(batched_hyp_ids),
            num_sentence_errors=results.num_sentence_errors + sent_err,
            sampled_hyps=tuple(sampled_hyps),
        )


def make_schedule(
    base_learn_rate: float = 5.0, warmup_steps: int = 10000, model_dims: int = 256
) -> optax.Schedule:
    def schedule(count):
        return base_learn_rate * (
            model_dims**-0.5
            * jnp.minimum((count + 1) * warmup_steps**-1.5, (count + 1) ** -0.5)
        )

    return schedule


def make_tx() -> tuple[optax.GradientTransformation, optax.Schedule]:
    schedule = make_schedule()
    return optax.adamw(schedule, weight_decay=1e-6), schedule


def fill_default_arguments(args: argparse.Namespace):
    if args.wpm_vocab is None:
        f = tempfile.NamedTemporaryFile()
        f.write(urllib.request.urlopen(_DEFAULT_WPM_VOCAB_URL).read())
        f.flush()
        args.wpm_vocab = f.name
        args.wpm_vocab_file = f
    if args.feature_normalizer is None:
        p = pathlib.Path(__file__).parent / "librispeech.meanstddev.logmelfb80.json"
        args.feature_normalizer = str(p)


def main(args: argparse.Namespace):
    fill_default_arguments(args)

    prng_keys = bobbin.prng_keygen(jax.random.PRNGKey(0))
    train_ds, eval_dss = prepare_datasets(
        tfds_data_dir=args.tfds_data_dir,
        wpm_vocab_path=args.wpm_vocab,
        wpm_size_limit=args.wpm_size_limit,
        train_batch_size=args.per_device_batch_size * jax.local_device_count(),
    )
    num_train_samples = 28_539 + 104_014 + 148_688

    all_checkpoint_path = args.log_dir_path / "all_ckpts"
    best_checkpoint_path = args.log_dir_path / "best_ckpts"
    tensorboard_path = args.log_dir_path / "tensorboard"

    wpm_vocab = asrio.WpmVocab.load(args.wpm_vocab, size_limit=args.wpm_size_limit)
    wpm_size = len(wpm_vocab.id2str)
    batch_size, *speech_shape = train_ds.element_spec["speech"].shape
    init_inputs = (
        np.zeros((1, *speech_shape), dtype=np.int16),
        np.ones((1, speech_shape[0]), dtype=np.float32),
    )
    normalizer = None
    if args.feature_normalizer is not None:
        normalizer = bobbin.parse_pytree_json(
            open(args.feature_normalizer).read(), asrio.MeanVarNormalizer.empty()
        )
    model = CtcAsrModel(num_outputs=wpm_size, feature_normalizer=normalizer)
    init_model_vars = jax.jit(model.init)(
        {
            "dropout": next(prng_keys),
            "params": next(prng_keys),
            "specaug": next(prng_keys),
        },
        *init_inputs,
    )

    task = CtcAsrTask(model)
    evaler = EvalTask(model, wpm_vocab)
    eval_batch_gens = {dsname: ds.as_numpy_iterator for dsname, ds in eval_dss.items()}
    tx, learn_rate_fn = make_tx()
    train_state = bobbin.initialize_train_state(
        model.apply, init_model_vars, tx, checkpoint_path=all_checkpoint_path
    )
    init_train_state = train_state
    train_state = flax.jax_utils.replicate(train_state, jax.local_devices())
    train_step_fn = bobbin.pmap_for_train_step(
        jax.jit(task.make_training_step_fn(), donate_argnums=(1,))
    )

    train_writer = flax_tb.SummaryWriter(tensorboard_path / "train")
    eval_writers = bobbin.make_eval_results_writer(tensorboard_path)

    eval_freq = num_train_samples // batch_size
    warmup = 10
    crontab = bobbin.CronTab()
    crontab.add(
        "eval_and_keep_best",
        _ForEachNSteps(eval_freq),
        _RunEval(evaler, eval_batch_gens)
        .add_result_processor(eval_writers)
        .and_keep_best_checkpoint(
            "dev",
            best_checkpoint_path,
        ),
    )
    crontab.add(
        "save_job_checkpoint",
        _ForEachNSteps(1000) | _AtNthStep(warmup),
        _SaveCheckpoint(all_checkpoint_path),
    )
    crontab.add(
        "heartbeat",
        _ForEachTSeconds(30.0) | _AtFirstNSteps(warmup, of_process=True),
        _WriteLog(),
    )
    crontab.add(
        "publish_tb", _ForEachNSteps(100), _PublishTrainingProgress(train_writer)
    )

    print(f"Total #Params = {bobbin.total_dimensionality(init_train_state.params)}")
    train_writer.text(
        "trainer/log/total_num_params",
        f"Number of parameters: {bobbin.total_dimensionality(init_train_state.params)}",
        step=init_train_state.step,
    )
    print("START", time.time())
    for batch in train_ds.as_numpy_iterator():
        train_state, step_info = train_step_fn(train_state, batch, next(prng_keys))
        train_state_0 = flax.jax_utils.unreplicate(train_state)
        extra_vars = train_state_0.extra_vars
        extra_vars["tensorboard"] = extra_vars["tensorboard"].unfreeze()
        extra_vars["tensorboard"]["learn_rate"] = bobbin.ScalarSow(
            learn_rate_fn(train_state_0.step)
        )
        train_state_0 = train_state_0.replace(extra_vars=extra_vars)
        crontab.run(train_state_0, step_info=step_info)
        del train_state.extra_vars["tensorboard"]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="MNIST training")
    argparser.add_argument("--tfds_data_dir", type=str, default=None)
    argparser.add_argument("--feature_normalizer", type=str, default=None)
    argparser.add_argument("--per_device_batch_size", type=int, default=8)
    argparser.add_argument("--wpm_vocab", type=str, default=None)
    argparser.add_argument("--wpm_size_limit", type=int, default=1024)
    argparser.add_argument("--log_dir_path", type=pathlib.Path, default=None)

    args = argparser.parse_args()
    main(args)
