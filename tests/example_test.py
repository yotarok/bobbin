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
import sys

from absl.testing import absltest
from etils import epath
import numpy as np
import tensorflow as tf


_EXAMPLES_DIR = epath.Path(__file__).parent.parent / "examples"


@dataclasses.dataclass
class MockedTfds:
    load_return_values: Dict[str, tf.data.Dataset]

    def load(self, name, *args, split, **kwargs):
        key = name + "_" + split
        if key in self.load_return_values:
            return self.load_return_values[key]
        elif "" in self.load_return_values:
            return self.load_return_values[""]


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


class MnistExampleTest(absltest.TestCase):
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

            np.testing.assert_(args.log_dir_path / "all_ckpts")
            np.testing.assert_(args.log_dir_path / "best_ckpts")


if __name__ == "__main__":
    absltest.main()