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
_VarCollection = bobbin.VarCollection


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
) -> tuple[tf.data.Dataset, dict[str, tf.data.Dataset]]:
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
    ) -> Tuple[chex.Scalar, Tuple[_VarCollection, LossAuxOut]]:
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
    start_time: float = np.inf
    end_time: float = 0
    sampled_results: bobbin.SampledSet[Tuple[str, str]] = struct.field(
        pytree_node=False, default=bobbin.SampledSet(max_size=3)
    )

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

    def is_better_than(self, other: EvalResults) -> bool:
        return self.word_error.error_rate < other.word_error.error_rate

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


class EvalTask(bobbin.EvalTask):
    def __init__(self, model: nn.Module, wpm_vocab: asrio.WpmVocab):
        self.model = model
        self.wpm_vocab = wpm_vocab

    def create_eval_results(self, unused_dataset_name):
        return EvalResults(start_time=time.time())

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def predict(
        self, batch: _Batch, model_vars: _VarCollection
    ) -> Tuple[_Array, _Array]:
        logits, logit_paddings = self.model.apply(
            model_vars, batch["speech"], batch["speech_paddings"], is_eval=True
        )
        predicts = logits.argmax(axis=-1)
        return predicts, logit_paddings

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
        for hyp_ids, ref_ids in zip(batched_hyp_ids, batched_ref_ids):
            if not ref_ids:
                continue  # all-pad sequences are treated as invalid sequence.
            acc_token_error = acc_token_error.accumulate(hyp_ids, ref_ids)
            hyp_text = asrio.wpm_decode_sentence(self.wpm_vocab, hyp_ids)
            ref_text = asrio.wpm_decode_sentence(self.wpm_vocab, ref_ids)
            acc_word_error = acc_word_error.accumulate(
                hyp_text.split(" "), ref_text.split(" ")
            )
            acc_char_error = acc_char_error.accumulate(list(hyp_text), list(ref_text))
            if hyp_text != ref_text:
                sent_err += 1
            num_sentences += 1

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

    @functools.partial(
        bobbin.wrapped_pmap,
        axis_name="batch",
        argtypes=["static", "shard", "broadcast"],
        donate_argnums=(1,),
    )
    def parallel_predict(self, b, mvar):
        return self.predict(b, mvar)

    def evaluate(self, batch: _Batch, model_vars: _VarCollection) -> EvalResults:
        start_time = time.time()
        if batch is None:
            # This is only allowed when we are sure that there's no parallel
            # reduction (e.g. `jax.lax.p*` functions) used in `parallel_predict`.
            # Otherwise, all reduction operators must be called in the same
            # order in each process. Usually this is done by feeding a dummy
            # batch.
            return EvalResults()
        results = self.parallel_predict(batch, model_vars)
        # It's important to ensure both are NumPy-array (and on CPU)
        results = jax.tree_util.tree_map(np.asarray, results)
        predicts, predict_paddings = bobbin.flatten_leading_axes(results)

        batched_hyp_ids = asrio.remove_ctc_blanks_and_repeats(
            predicts, predict_paddings
        )
        batched_ref_ids = asrio.make_list_from_padded_batch(
            np.asarray(batch["tokens"]), np.asarray(batch["token_paddings"])
        )
        results = self._build_results(batched_hyp_ids, batched_ref_ids)
        return results.replace(start_time=start_time, end_time=time.time())


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
        p = epath.Path(__file__).parent / "librispeech.meanstddev.logmelfb80.json"
        args.feature_normalizer = str(p)


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


def main(args: argparse.Namespace):
    fill_default_arguments(args)

    if _is_running_on_cloud_tpu_vm() or args.multi_process:
        jax.distributed.initialize()

    if jax.process_index() == 0:
        logging.basicConfig(stream=sys.stderr)
        logging.root.setLevel(logging.INFO)

    train_ds, eval_dss = prepare_datasets(
        tfds_data_dir=args.tfds_data_dir,
        wpm_vocab_path=args.wpm_vocab,
        wpm_size_limit=args.wpm_size_limit,
        train_batch_size=args.per_device_batch_size * jax.local_device_count(),
        eval_batch_size=args.per_device_batch_size * jax.local_device_count(),
        max_train_batches=args.max_steps,
    )
    num_train_samples = 28_539 + 104_014 + 148_688

    all_checkpoint_path = args.log_dir_path / "all_ckpts"
    best_checkpoint_path = args.log_dir_path / "best_ckpts"
    tensorboard_path = args.log_dir_path / "tensorboard"

    wpm_vocab = asrio.WpmVocab.load(args.wpm_vocab, size_limit=args.wpm_size_limit)
    wpm_size = len(wpm_vocab.id2str)

    batch_size, *speech_shape = train_ds.element_spec["speech"].shape
    global_batch_size = batch_size * jax.process_count()
    init_inputs = (
        jnp.zeros((1, *speech_shape), dtype=np.int16),
        jnp.ones((1, speech_shape[0]), dtype=np.float32),
    )
    normalizer = None
    if args.feature_normalizer is not None:
        with open(args.feature_normalizer) as f:
            normalizer = bobbin.parse_pytree_json(
                f.read(), asrio.MeanVarNormalizer.empty()
            )
    model = CtcAsrModel(num_outputs=wpm_size, feature_normalizer=normalizer)

    # init must be deterministic for multi-host training
    prng_keys = bobbin.prng_keygen(jax.random.PRNGKey(0))
    init_model_vars = jax.jit(model.init)(
        {
            "dropout": next(prng_keys),
            "params": next(prng_keys),
            "specaug": next(prng_keys),
        },
        *init_inputs,
    )

    prng_keys = bobbin.prng_keygen(jax.random.PRNGKey(jax.process_index() * 123))

    task = CtcAsrTask(model)
    evaler = EvalTask(model, wpm_vocab)
    eval_batch_gens = {dsname: ds.as_numpy_iterator for dsname, ds in eval_dss.items()}
    tx, learn_rate_fn = make_tx()
    train_state = bobbin.initialize_train_state(
        model.apply, init_model_vars, tx, checkpoint_path=all_checkpoint_path
    )
    train_state = flax.jax_utils.replicate(train_state, jax.local_devices())
    if jax.process_count() > 1:
        bobbin.assert_replica_integrity(train_state)
    train_step_fn = bobbin.pmap_for_train_step(
        task.make_training_step_fn(split_steps=args.split_training_batch),
    )

    train_writer = (
        flax_tb.SummaryWriter(tensorboard_path / "train")
        if jax.process_index() == 0
        else bobbin.NullSummaryWriter()
    )
    bobbin.publish_trainer_env_info(train_writer, train_state)

    # Seeting up crontab (auxiliary actions periodically executed during the training)
    eval_freq = num_train_samples // global_batch_size
    warmup = 10
    crontab = bobbin.CronTab()
    crontab.schedule(
        bobbin.RunEval(
            evaler, eval_batch_gens, tensorboard_root_path=tensorboard_path
        ).and_keep_best_checkpoint(
            "dev",
            best_checkpoint_path,
        ),
        step_interval=eval_freq,
    )
    if jax.process_index() == 0:
        crontab.schedule(
            bobbin.SaveCheckpoint(all_checkpoint_path),
            step_interval=1000,
            at_step=warmup,
        )
        crontab.schedule(
            bobbin.WriteLog(), time_interval=30.0, at_first_steps_of_process=warmup
        )
        crontab.schedule(
            bobbin.PublishTrainingProgress(train_writer), step_interval=100
        )

    bobbin.publish_trainer_env_info(train_writer, train_state)
    logging.info("MAIN LOOP STARTS with devices %s", str(jax.local_devices()))
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

    args = argparser.parse_args()
    main(args)
