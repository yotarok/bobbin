#
# var_util.Copyright 2022 Google LLC
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

"""Generates feature normalizers from LibriSpeech dataset.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict, Optional, Tuple

import bobbin
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from bobbin.example_lib import asrnn
from bobbin.example_lib import asrio


_LogMelFilterBank = asrnn.LogMelFilterBank


def prepare_datasets(
    tfds_data_dir: Optional[str] = None,
    wpm_vocab_path: str = "",
    wpm_size_limit: Optional[int] = None,
) -> tuple[tf.data.Dataset, list[tf.data.Dataset]]:
    builder_kwargs = dict(config="lazy_decode")
    return tfds.load(
        "librispeech",
        split="train_clean100+train_clean360+train_other500",
        data_dir=tfds_data_dir,
        builder_kwargs=builder_kwargs,
        download=False,
    )


def compute_normalizer(dataset, feature_fn):
    def prepare(row: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        return row["speech"], tf.zeros_like(row["speech"], tf.float32)

    def tf_feature_fn(
        waveform: tf.Tensor, waveform_paddings: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.numpy_function(
            feature_fn,
            [waveform, waveform_paddings],
            Tout=(tf.float32, tf.float32),
            stateful=False,
        )

    def sum_statistics(
        features: tf.Tensor, feature_paddings: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        features = tf.cast(features, tf.float64)
        feature_paddings = tf.cast(feature_paddings, tf.float64)
        features = features * (1.0 - feature_paddings)[..., tf.newaxis]
        denom = tf.reduce_sum(1.0 - feature_paddings)
        first = tf.reduce_sum(features, axis=(0, 1))
        second = tf.reduce_sum(features * features, axis=(0, 1))
        return (denom, first, second)

    def reduce_statistics(
        xs: Tuple[tf.Tensor, ...], ys: Tuple[tf.Tensor, ...]
    ) -> Tuple[tf.Tensor, ...]:
        return tuple(x + y for x, y in zip(xs, ys))

    waves = dataset.map(
        prepare, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    waves = waves.padded_batch(32, padding_values=(tf.constant(0, tf.int16), 1.0))
    features = waves.map(
        tf_feature_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    stats = features.map(
        sum_statistics, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    feature_dims = 80
    init_stats = (
        tf.zeros([], tf.float64),
        tf.zeros([feature_dims], tf.float64),
        tf.zeros([feature_dims], tf.float64),
    )
    acc = init_stats
    logging.info("#Batches = %d", stats.cardinality())
    log_freq = stats.cardinality() // 30
    for i, x in enumerate(stats.as_numpy_iterator()):
        if i % log_freq == 0:
            logging.info("%d batches processed.", i)
        acc = reduce_statistics(acc, x)
    return tuple(x.numpy() for x in acc)


def main(args: argparse.Namespace):
    train_dataset = prepare_datasets(
        tfds_data_dir=args.tfds_data_dir,
    )

    def feature_fn(waveform, waveform_paddings):
        with jax.default_device(jax.devices("cpu")[0]):
            module = _LogMelFilterBank()
            return module.apply({}, waveform, waveform_paddings)

    start_time = time.time()
    denom, first, second = compute_normalizer(train_dataset, feature_fn)
    elapsed_time = time.time() - start_time
    mean = first / denom
    var = second / denom - mean * mean
    stddev = np.sqrt(var)
    normalizer = asrio.MeanVarNormalizer(mean=mean, stddev=stddev, n=denom)
    logging.info(
        "FPS = %d frame / %f sec = %f frame/sec",
        normalizer.n,
        elapsed_time,
        normalizer.n / elapsed_time,
    )

    json = bobbin.dump_pytree_json(normalizer)
    logging.info("Normalizer: %s", json)
    with open(args.output_path, "w") as f:
        f.write(json)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.root.setLevel(logging.INFO)

    # Disable TF's memory preallocation if TF is built with CUDA.
    tf.config.experimental.set_visible_devices([], "GPU")

    argparser = argparse.ArgumentParser(description="Compute input normalizers")
    argparser.add_argument("--tfds_data_dir", type=str, default=None)
    argparser.add_argument("--output_path", type=str, default=None)
    args = argparser.parse_args()
    main(args)
