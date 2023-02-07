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

"""Sample with MNIST.
"""

from __future__ import annotations

import argparse
import functools
import logging
import sys
from typing import Dict, Optional, Sequence, Tuple

import chex
from etils import epath
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

_VarCollection = bobbin.VarCollection


class CNNDenseClassifier(nn.Module):
    num_classes: int = 10
    cnn_features: Sequence[int] = (32, 64)
    cnn_kernel_sizes: Sequence[Sequence[int]] = ((3, 3), (3, 3))
    cnn_pool_shapes: Sequence[Sequence[int]] = ((2, 2), (2, 2))
    cnn_pool_strides: Sequence[Sequence[int]] = ((2, 2), (2, 2))
    cnn_dropout_rates: Sequence[float] = (0.2, 0.2)
    normalize_cnn_out: bool = True  # Performs poorly, just for demonstration
    dense_features: Sequence[int] = (256,)
    dense_dropout_rates: Sequence[float] = (0.0,)

    @nn.compact
    def __call__(self, x: np.ndarray, *, is_eval: bool = False):
        cnn_layers = len(self.cnn_features)
        if (
            len(self.cnn_kernel_sizes) != cnn_layers
            or len(self.cnn_pool_shapes) != cnn_layers
            or len(self.cnn_pool_strides) != cnn_layers
            or len(self.cnn_dropout_rates) != cnn_layers
        ):
            raise ValueError("CNN parameters must have the same lengths")

        for feats, kernel, pool_shape, strides, drate in zip(
            self.cnn_features,
            self.cnn_kernel_sizes,
            self.cnn_pool_shapes,
            self.cnn_pool_strides,
            self.cnn_dropout_rates,
        ):
            if drate > 0.0:
                x = nn.Dropout(rate=drate, deterministic=is_eval)(x)
            x = nn.Conv(features=feats, kernel_size=kernel)(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=pool_shape, strides=strides)
        x = x.reshape(x.shape[:-3] + (-1,))

        cnn_mean = jnp.mean(x)
        cnn_var = jnp.var(x)
        self.sow("tensorboard", "cnn_mean", bobbin.ScalarSow(cnn_mean))
        self.sow("tensorboard", "cnn_var", bobbin.ScalarSow(cnn_var))

        if self.normalize_cnn_out:
            x = nn.BatchNorm(use_running_average=is_eval)(x)

        for feats, drate in zip(self.dense_features, self.dense_dropout_rates):
            if drate > 0.0:
                x = nn.Dropout(rate=drate, deterministic=is_eval)(x)
            x = nn.Dense(features=feats)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)

        logit_entropy = -jnp.sum(jax.nn.softmax(x) * jax.nn.log_softmax(x), axis=-1)
        logit_entropy = jnp.mean(logit_entropy)
        self.sow("tensorboard", "logit_entropy", bobbin.ScalarSow(logit_entropy))
        return x


def preprocess_ds(
    ds: tf.data.Dataset, *, batch_size: int, is_training: bool = False
) -> tf.data.Dataset:
    ds = ds.map(
        lambda im, lab: (tf.cast(im, tf.float32) / 255.0, lab),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    if is_training:
        ds = ds.repeat().shuffle(1024)
    return ds.batch(batch_size).prefetch(1)


@struct.dataclass
class EvalResults(bobbin.EvalResults):
    correct_count: chex.Scalar
    predict_count: chex.Scalar
    sum_logprobs: chex.Scalar

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.predict_count

    def to_log_message(self) -> str:
        return (
            f"accuracy = {self.correct_count} / {self.predict_count}"
            f" = {self.accuracy}\n"
            f"logprob = {self.sum_logprobs}"
        )

    def reduce(self, other: EvalResults) -> EvalResults:
        return EvalResults(
            correct_count=self.correct_count + other.correct_count,
            predict_count=self.predict_count + other.predict_count,
            sum_logprobs=self.sum_logprobs + other.sum_logprobs,
        )

    def is_better_than(self, other: EvalResults) -> bool:
        return self.accuracy > other.accuracy

    def write_to_tensorboard(
        self,
        current_train_state: bobbin.TrainState,
        writer: flax_tb.SummaryWriter,
    ) -> None:
        step = current_train_state.step
        writer.scalar("accuracy", self.accuracy, step=step)
        writer.scalar("logprobs", self.sum_logprobs / self.predict_count, step=step)


@struct.dataclass
class LossAuxOut:
    logits: chex.Array
    per_sample_loss: chex.Array
    predicted_labels: chex.Array


class ClassificationTask(bobbin.TrainTask):
    def __init__(self, model: nn.Module):
        super().__init__(
            model,
            example_args=(np.zeros(shape=(1, 28, 28, 1), dtype=np.float32)),
            required_rngs=("dropout", "params"),
        )

    def compute_loss(
        self,
        params: bobbin.Parameter,
        batch: bobbin.Batch,
        *,
        extra_vars: _VarCollection,
        prng_key: chex.PRNGKey,
        step: chex.Scalar,
    ) -> Tuple[chex.Scalar, Tuple[_VarCollection, LossAuxOut]]:
        inputs, labels = batch
        model_vars = extra_vars.copy()
        model_vars.update(params=params, tensorboard=dict())
        logits, updated_vars = self.model.apply(
            model_vars,
            inputs,
            rngs=self.get_rng_dict(prng_key),
            mutable=flax.core.DenyList("params"),
        )
        unused_batch_size, num_classes = logits.shape
        one_hot = jax.nn.one_hot(labels, num_classes)
        per_sample_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        loss = jnp.mean(per_sample_loss)
        return loss, (
            updated_vars,
            LossAuxOut(
                logits=logits,
                per_sample_loss=per_sample_loss,
                predicted_labels=jnp.argmax(logits, axis=-1),
            ),
        )


class EvalTask(bobbin.EvalTask):
    def __init__(self, model: nn.Module):
        self.model = model

    def create_eval_results(self, unused_dataset_name: str):
        return EvalResults(correct_count=0, predict_count=0, sum_logprobs=0.0)

    @functools.partial(
        bobbin.tpmap,
        axis_name="batch",
        argtypes=["static", "shard", "broadcast"],
        wrap_return=EvalResults.unshard_and_reduce,
    )
    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def evaluate(
        self, batch: bobbin.Batch, model_vars: bobbin.VarCollection
    ) -> EvalResults:
        inputs, labels = batch
        logits = self.model.apply(model_vars, inputs, is_eval=True)
        predicts = logits.argmax(axis=-1)
        batch_size, num_classes = logits.shape
        one_hot = jax.nn.one_hot(labels, num_classes)
        correct_count = (predicts == labels).astype(np.float32).sum()
        predict_count = len(labels)
        sum_logprobs = -jnp.sum(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        )
        return EvalResults(
            correct_count=correct_count,
            predict_count=predict_count,
            sum_logprobs=sum_logprobs,
        )


def get_datasets(
    batch_size: int = 64,
    max_train_batches: Optional[int] = None,
) -> Tuple[tf.data.Dataset, Dict[str, tf.data.Dataset]]:
    train_ds = tfds.load("mnist", split="train[:50000]", as_supervised=True)
    train_ds = preprocess_ds(train_ds, batch_size=batch_size, is_training=True)
    eval_dss = {
        "dev": tfds.load("mnist", split="train[50000:]", as_supervised=True),
        "test": tfds.load("mnist", split="test", as_supervised=True),
    }
    eval_dss = {
        k: preprocess_ds(ds, batch_size=batch_size, is_training=False)
        for k, ds in eval_dss.items()
    }

    if max_train_batches is not None:
        train_ds = train_ds.take(max_train_batches)
    return train_ds, eval_dss


def make_tx() -> optax.GradientTransformation:
    return optax.sgd(0.1, 0.9)


def make_model() -> nn.Module:
    return CNNDenseClassifier()


def main(args: argparse.Namespace):
    train_ds, eval_dss = get_datasets(max_train_batches=args.max_steps)
    all_checkpoint_path = args.log_dir_path / "all_ckpts"
    best_checkpoint_path = args.log_dir_path / "best_ckpts"
    tensorboard_path = args.log_dir_path / "tensorboard"

    model = make_model()

    task = ClassificationTask(model)
    evaler = EvalTask(model)
    train_state = task.initialize_train_state(
        jax.random.PRNGKey(0), make_tx(), checkpoint_path=all_checkpoint_path
    )
    init_train_state = train_state
    train_state = flax.jax_utils.replicate(train_state, jax.local_devices())
    eval_batch_gens = {dsname: ds.as_numpy_iterator for dsname, ds in eval_dss.items()}
    train_step_fn = bobbin.pmap_for_train_step(jax.jit(task.make_training_step_fn()))

    train_writer = flax_tb.SummaryWriter(tensorboard_path / "train")
    bobbin.publish_trainer_env_info(train_writer, init_train_state)

    warmup = 5
    crontab = bobbin.CronTab()
    crontab.schedule(
        bobbin.RunEval(
            evaler, eval_batch_gens, tensorboard_root_path=tensorboard_path
        ).and_keep_best_checkpoint(
            "dev",
            best_checkpoint_path,
        ),
        step_interval=1000,
        at_step=warmup,
    )
    crontab.schedule(
        bobbin.SaveCheckpoint(all_checkpoint_path),
        step_interval=1000,
        at_step=warmup,
    )
    crontab.schedule(
        task.make_log_writer(), time_interval=10.0, at_first_steps_of_process=warmup
    )
    crontab.schedule(bobbin.PublishTrainingProgress(train_writer), step_interval=100)

    logging.info("Main loop started.")
    prng_key = jax.random.PRNGKey(3)
    for batch in train_ds.as_numpy_iterator():
        rng, prng_key = jax.random.split(prng_key)
        train_state, step_info = train_step_fn(train_state, batch, rng)
        crontab.run(train_state, step_info=step_info)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.root.setLevel(logging.INFO)

    # Disable TF's memory preallocation if TF is built with CUDA.
    tf.config.experimental.set_visible_devices([], "GPU")

    argparser = argparse.ArgumentParser(description="MNIST training")
    argparser.add_argument("--log_dir_path", type=epath.Path, default=None)
    argparser.add_argument("--max_steps", type=int, default=None)
    args = argparser.parse_args()
    main(args)
