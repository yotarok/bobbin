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

"""MNIST example.
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


# `bobbin.VarCollection` is actually `Dict[str, chex.ArrayTree]`. This is a
# standard structure for storing model variables in Flax.  For example,
# in typical flax models, `var_collection['params']` is used for storing
# parameters, and `var_collection['batch_stats']` is used for storing
# special non-trainable variables for `flax.linen.BatchNorm`. This type-
# alias is imported as a type for such structures.
VarCollection = bobbin.VarCollection


# First, a classifier model is defined by using flax. bobbin is designed
# not to add restrictions for the definition of the model. Any flax module
# can be used.
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

        # Here, a bobbin specific data-structure is introduced.
        # `bobbin.*Summary` classes are used to wrap intermediate variable so
        # it can be retrieved and published to TensorBoard afterwards. In this
        # example, means and variances of CNN outputs are collected (even
        # though this is a bit meaningless in practice).
        self.sow("tensorboard", "cnn_mean", bobbin.ScalarSummary(cnn_mean))
        self.sow("tensorboard", "cnn_var", bobbin.ScalarSummary(cnn_var))

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

        # Same here. The reason why bobbin needed this special class is to tag
        # some sow-ed variables for TensorBoard. `MplImageSummary`, which is not
        # covered by this example, for example, includes extra padding
        # indicator variables to handle variable-size images.
        self.sow("tensorboard", "logit_entropy", bobbin.ScalarSummary(logit_entropy))
        return x


# bobbin does not support dataset inputs. Therefore, it's clients'
# responsibility to prepare datasets in a right way. Since bobbin uses datasets
# represented as iterators of batches (numpy arrays), many dataset libraries,
# including tf.data and TFDS can be integrated. (In other words, bobbin does
# not provide functionalities that relies on meta-information of datasets, and
# if you want to do something depending on meta-data, this must be done
# explicitly in the clients' code.)
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


# Here, bobbin evaluation infrastructure asks you to design evaluation results
# (metrics) as a frozen data-class (flax struct).  Evaluation task will compute
# an instance of a subclass of `bobbin.EvalResults` (here, `EvalResults`), and
# computed `EvalResults`s are reduced over batches by using `EvalResults.reduce`.
@struct.dataclass
class EvalResults(bobbin.EvalResults):
    # Here, we count the numbers of correct and processed samples, and the sum
    # of log-probabilities of the correct labels.
    correct_count: chex.Scalar
    predict_count: chex.Scalar
    sum_logprobs: chex.Scalar

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.predict_count

    # This method is required if you want a readable log message.
    def to_log_message(self) -> str:
        return (
            f"accuracy = {self.correct_count} / {self.predict_count}"
            f" = {self.accuracy}\n"
            f"logprob = {self.sum_logprobs}"
        )

    # As mentioned above, `EvalResults.reduce` is the most important method that
    # needed to be overridden. When we have two evaluation results, reduction
    # for this `EvalResults` class should be as follows:
    def reduce(self, other: EvalResults) -> EvalResults:
        return EvalResults(
            correct_count=self.correct_count + other.correct_count,
            predict_count=self.predict_count + other.predict_count,
            sum_logprobs=self.sum_logprobs + other.sum_logprobs,
        )

    # This method is required when we do early stopping. For that, comparison
    # between two metrics are important for keeping "best" parameters. Here,
    # we compare accuracies of two `EvalResults`.
    def is_better_than(self, other: EvalResults) -> bool:
        return self.accuracy > other.accuracy

    # This method defines how `EvalResults` can be published to TensorBoard.
    # here `current_train_state` is the status of training, and `writer` is
    # `flax.metrics.tensorboard.SummaryWriter` that is used to publish values
    # to TensorBoard.
    def write_to_tensorboard(
        self,
        current_train_state: bobbin.TrainState,
        writer: flax_tb.SummaryWriter,
    ) -> None:
        step = current_train_state.step
        writer.scalar("accuracy", self.accuracy, step=step)
        writer.scalar("logprobs", self.sum_logprobs / self.predict_count, step=step)

    # See API reference doc for details of `EvalResults`. Please note that
    # this class is only needed when you use evaluation functionality provided
    # by bobbin, and bobbin is designed to be a sparsely-connected set of small
    # parts which you can always choose to use, or not to use.


# How to compute `EvalResults` above is defined here as a subclass of
# `bobbin.EvalTask`.
class EvalTask(bobbin.EvalTask):
    def __init__(self, model: nn.Module):
        self.model = model

    # This `create_eval_results` function is to define "zero" for `EvalResults`.
    def create_eval_results(self, unused_dataset_name: str):
        return EvalResults(correct_count=0, predict_count=0, sum_logprobs=0.0)

    # This function `bobbin.EvalTask.evaluate` is the main part of evaluation
    # logic.  The function must be overridden to compute `EvalResults` for the
    # input batch `batch` and model variables `model_vars`.  Here, `evaluate`
    # is wrapped by `tpmap` decorator for performing multi-device parallel
    # computation. See API document for `tpmap` for details. It should be noted
    # that it runs perfectly without this decorator except for that only single
    # device will be used.
    @functools.partial(
        bobbin.tpmap,
        axis_name="batch",
        argtypes=["static", "shard", "broadcast"],
        wrap_return=EvalResults.unshard_and_reduce,
    )
    def evaluate(self, batch: bobbin.Batch, model_vars: VarCollection) -> EvalResults:
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


# A loss function interface for bobbin assumes that the loss function returns an
# auxiliary data for later use (e.g. logging).  Here, for demonstrating this
# feature, `LossAuxOut` is defined.  Any pytree (including None) can be used as
# an auxiliary output.
@struct.dataclass
class LossAuxOut:
    logits: chex.Array
    per_sample_loss: chex.Array
    predicted_labels: chex.Array


# Besides "evaluation", another main abstraction bobbin introduces is training
# step.  For providing infrastructure for training loop construction, bobbin
# asks users to implement a subclass of `bobbin.TrainTask`.
class ClassificationTask(bobbin.TrainTask):
    def __init__(self, model: nn.Module):
        # The constructor of `bobbin.TrainTask` typically requires the following
        # three arguments, the first one is `nn.Module` to be optimized. The
        # second one is example input data for `nn.Module` that will be passed
        # to `model.init` for initializing the variables. The third parameter is
        # the names of variable collections that requires random number
        # generators for training.
        super().__init__(
            model,
            example_args=(np.zeros(shape=(1, 28, 28, 1), dtype=np.float32)),
            required_rngs=("dropout",),
        )

    # `bobbin.TrainTask.compute_loss` is the central part of the training.
    # This method must be overridden to define how to compute the loss function
    # from the given parameters and the input batch.
    def compute_loss(
        self,
        params: bobbin.Parameter,
        batch: bobbin.Batch,
        *,
        extra_vars: VarCollection,
        prng_key: chex.PRNGKey,
        step: chex.Scalar,
    ) -> Tuple[chex.Scalar, Tuple[VarCollection, LossAuxOut]]:
        inputs, labels = batch

        # For some reasons, currently parameters and non-parameter variables are
        # passed separately.  So few lines needed to merge them.
        model_vars = extra_vars.copy()
        # Here, we also reset variables in "tensorboard" collection.
        # Because `nn.Module.sow` in flax constructs tuples of intermediate
        # variables when it is called multiple times. We need to reset
        # tensorboard variables for avoiding those intermediate variables to be
        # accumulated over several steps.
        model_vars.update(params=params, tensorboard=dict())

        # The main part of loss starts here.
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

        # The return value of this function must have the structure as
        # `(loss, (updated_vars, aux_output))` where `loss` is the loss scalar,
        # and `updated_vars` is the updated non-trainable variables that has the
        # same structure (or subset structure) of `extra_vars` and `aux_output`
        # is an arbitrary pytree that can be used outside of the training step
        # function. Here, we use `LossAuxOut` as `aux_output` but this can also
        # be `None`.
        return loss, (
            updated_vars,
            LossAuxOut(
                logits=logits,
                per_sample_loss=per_sample_loss,
                predicted_labels=jnp.argmax(logits, axis=-1),
            ),
        )


# In addition to the task definitions above, we need optimizer configuration.
def make_tx() -> optax.GradientTransformation:
    return optax.sgd(0.1, 0.9)


# and model configuration function.
def make_model() -> nn.Module:
    return CNNDenseClassifier()


def main(args: argparse.Namespace):
    # Unlike other frameworks "bobbin" doesn't define your main function, and
    # designed to make your main function simpler.
    # (This should also be important for Jupyter notebook experiments.)

    # First, we define some paths.
    all_checkpoint_path = args.log_dir_path / "all_ckpts"
    best_checkpoint_path = args.log_dir_path / "best_ckpts"
    tensorboard_path = args.log_dir_path / "tensorboard"

    # and we define datasets, the model, the optimizer, and the tasks defined
    # above.
    train_ds, eval_dss = get_datasets(max_train_batches=args.max_steps)
    model = make_model()
    tx = make_tx()
    task = ClassificationTask(model)
    evaler = EvalTask(model)

    # Trainer state variable is first initialized using
    # `bobbin.TrainTask.initialize_train_state`. This function takes RNG seed,
    # optimizer, and checkpoint path as arguments. Checkpoint will be loaded if
    # there's a valid checkpoint in the specified path, otherwise, it will
    # initialize the variables using the RNG seed provided.
    train_state = task.initialize_train_state(
        jax.random.PRNGKey(0), tx, checkpoint_path=all_checkpoint_path
    )

    # First, we publish meta-training information to TensorBoard using
    # `bobbin.publish_trainer_env_info`.
    train_writer = flax_tb.SummaryWriter(tensorboard_path / "train")
    bobbin.publish_trainer_env_info(train_writer, train_state)

    # Then, create and compile the training step function by calling
    # `make_training_step_fn`, and obtain multi-process version of the step
    # function using `pmap` function.
    train_step_fn = task.make_training_step_fn().pmap("batch")
    # Since we will be working on pmapped version of training function,
    # training state must be device-replicated as follows:
    train_state = flax.jax_utils.replicate(train_state, jax.local_devices())

    # bobbin's evaluation functions take a dictionary of generators (functions
    # that return iterators) as a source of test data.  Here, we convert
    # `tf.data.Dataset`s to the generators.
    eval_batch_gens = {dsname: ds.as_numpy_iterator for dsname, ds in eval_dss.items()}

    # Finally, we configure `bobbin.CronTab`.
    warmup = 5
    crontab = bobbin.CronTab()
    # This entry in crontab means that we save checkpoint for train_state for
    # each 1000 steps and after 5 (warmup) steps completed.
    crontab.schedule(
        bobbin.SaveCheckpoint(all_checkpoint_path),
        step_interval=1000,
        at_step=warmup,
    )
    # This entry in crontab means that we run evaluation for each 1000 steps and
    # after 5 (warmup) steps completed.
    # `EvalResults` will be published to `tensorboard_path`. Also, this keeps
    # the best performing parameters to a separate checkpoint path.
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
    # This entry in crontab means that the log entry will be written for each
    # approx. 10 seconds, and for first 5 (warmup) steps, log will be written
    # after every step.
    crontab.schedule(
        task.make_log_writer(), time_interval=10.0, at_first_steps_of_process=warmup
    )
    # This entry in crontab means that, for each 100 steps, training summary
    # variables will be published to TensorBoard.
    crontab.schedule(bobbin.PublishTrainingProgress(train_writer), step_interval=100)

    # The rest of this program is fairly easy.
    logging.info("Main loop started.")
    # Here, we initialize the RNG seed.
    prng_key = jax.random.PRNGKey(3)
    for batch in train_ds.as_numpy_iterator():
        # and split it for each step.
        rng, prng_key = jax.random.split(prng_key)
        # Here, `train_step_fn` is called,
        train_state, step_info = train_step_fn(train_state, batch, rng)
        # and periodical actions registered in crontab will be invoked.
        crontab.run(train_state, step_info=step_info)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.root.setLevel(logging.INFO)

    # Disable TF's memory pre-allocation if TF is built with CUDA.
    tf.config.experimental.set_visible_devices([], "GPU")

    argparser = argparse.ArgumentParser(description="MNIST training")
    argparser.add_argument("--log_dir_path", type=epath.Path, default=None)
    argparser.add_argument("--max_steps", type=int, default=None)
    args = argparser.parse_args()
    main(args)
