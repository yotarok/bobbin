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
"""Test invoking the example scripts.

The tests in this module are rather integration tests than unit tests. However,
the execution time and network access are controlled so it can fit in CI.
"""

import argparse
import dataclasses
import tempfile
from typing import Dict
import unittest
import sys

from absl.testing import absltest
from absl.testing import parameterized
import chex
from etils import epath
import numpy as np
import tensorflow as tf

_EXAMPLES_DIR = epath.Path(__file__).parent.parent / "examples"


@dataclasses.dataclass
class MockedTfds:
    load_return_values: Dict[str, tf.data.Dataset]

    class ReadConfig(unittest.mock.MagicMock):
        pass

    def load(self, name, *, split, **kwargs):
        if isinstance(split, str):
            return self._load_one(name, split=split, **kwargs)
        elif hasattr(split, "__iter__"):
            return [self._load_one(name, split=s, **kwargs) for s in split]
        else:
            raise ValueError(
                "Unsupported split specifier for `MockedTfds.load` "
                " string or list of splits are expected, however "
                f"{type(split)} is given."
            )

    def _load_one(self, name, *, split, **kwargs):
        key = name + "_" + split
        if key in self.load_return_values:
            return self.load_return_values[key]
        elif "" in self.load_return_values:
            return self.load_return_values[""]

    def split_for_jax_process(self, split):
        return split


def _mocked_supervised_mnist(size: int = 128) -> tf.data.Dataset:
    def gen():
        for unused_n in range(size):
            image = np.random.randint(0, 256, size=(28, 28, 1), dtype=np.uint8)
            label = np.random.randint(0, 10, dtype=np.int64)
            yield image, label

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(28, 28, 1), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )


def _mocked_librispeech(size: int = 128) -> tf.data.Dataset:
    def gen():
        for n in range(size):
            length = np.random.randint(16000, 48000)
            speech = np.random.randint(
                -(2**15), 2**15, dtype=np.int16, size=(length,)
            )
            yield {
                "chapter_id": 0,
                "id": "mocked_speech_{n:05d}",
                "speaker_id": n % 5,
                "speech": speech,
                "text": "THIS IS {n}-th SENTENCE IN GENERATED DATASET",
            }

    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "chapter_id": tf.TensorSpec(shape=(), dtype=tf.int64),
            "id": tf.TensorSpec(shape=(), dtype=tf.string),
            "speaker_id": tf.TensorSpec(shape=(), dtype=tf.int64),
            "speech": tf.TensorSpec(shape=(None), dtype=tf.int16),
            "text": tf.TensorSpec(shape=(), dtype=tf.string),
        },
    )


class MnistExampleTest(chex.TestCase):
    def test_invoke(self):
        sys.path.append(str(_EXAMPLES_DIR))
        sys.modules["tensorflow_datasets"] = MockedTfds(
            load_return_values={"": _mocked_supervised_mnist()}
        )
        import mnist  # pytype: disable=import-error

        with tempfile.TemporaryDirectory() as logdir:
            args = argparse.Namespace()
            args.log_dir_path = epath.Path(logdir)
            args.max_steps = 5
            mnist.main(args)

            tensorboard_path = args.log_dir_path / "tensorboard"
            np.testing.assert_(tensorboard_path.is_dir())
            for split in ("train", "dev", "test"):
                np.testing.assert_((tensorboard_path / split).is_dir())

            np.testing.assert_((args.log_dir_path / "all_ckpts").is_dir())
            np.testing.assert_((args.log_dir_path / "best_ckpts").is_dir())


class LibriSpeechExampleTest(chex.TestCase):
    def test_invoke(self):
        sys.path.append(str(_EXAMPLES_DIR))
        sys.modules["tensorflow_datasets"] = MockedTfds(
            load_return_values={"": _mocked_librispeech()}
        )
        import librispeech.train  # pytype: disable=import-error

        with tempfile.TemporaryDirectory() as logdir:
            args = argparse.Namespace()
            args.log_dir_path = epath.Path(logdir)
            args.max_steps = 1

            args.tfds_data_dir = None
            args.feature_normalizer = None
            args.per_device_batch_size = 8
            args.wpm_vocab = None
            args.wpm_size_limit = 32
            args.accumulate_updates = 2
            args.multi_process = None
            args.model_type = "unittest"

            librispeech.train.main(args)

    @parameterized.product(
        model_type=("debug", "unittest", "p100m", "p100m_preln"),
        use_accumulate_updates=(False, True),
    )
    def test_configuration(self, model_type, use_accumulate_updates):
        sys.path.append(str(_EXAMPLES_DIR))
        sys.modules["tensorflow_datasets"] = MockedTfds(
            load_return_values={"": _mocked_librispeech()}
        )
        import librispeech.train  # pytype: disable=import-error

        with tempfile.TemporaryDirectory() as logdir:
            args = argparse.Namespace()
            args.log_dir_path = epath.Path(logdir)
            args.max_steps = 0

            args.tfds_data_dir = None
            args.feature_normalizer = None
            args.per_device_batch_size = 1
            args.wpm_vocab = None
            args.wpm_size_limit = 32
            args.accumulate_updates = 2 if use_accumulate_updates else None
            args.multi_process = None
            args.model_type = model_type

            librispeech.train.main(args)


if __name__ == "__main__":
    absltest.main()
