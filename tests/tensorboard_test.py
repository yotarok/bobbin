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

"""Tests for tensorboard."""

from unittest import mock

from absl.testing import absltest
import chex
import flax
from flax.metrics import tensorboard as flax_tb
import numpy as np
import optax

from bobbin import tensorboard


class TrainIntermediatesPublisherTest(chex.TestCase):
    def test_default(self):
        variables = {
            "params": {
                "layers": (
                    {
                        "weight": np.random.normal(size=(3, 3)),
                        "not_for_tb:scalar": 3.14,
                    },
                ),
            },
            "intermediates": {
                "layers": ({"mean_output:scalar": 1.23, "var_output:scalar": 2.46},)
            },
        }
        writer = mock.create_autospec(flax_tb.SummaryWriter)
        tensorboard.publish_train_intermediates(writer, variables["intermediates"], 123)
        writer.scalar.assert_any_call("/layers/0/mean_output", 1.23, step=123)
        writer.scalar.assert_any_call("/layers/0/var_output", 2.46, step=123)


@flax.struct.dataclass
class EvalResults:
    input_norm_counts: np.ndarray
    input_norm_bins: np.ndarray
    accuracy: float

    @classmethod
    def make_random_test_data(cls):
        bins = np.arange(0.0, 1.0, 0.02)
        return cls(
            input_norm_counts=np.random.randint(0, 10, size=bins.shape),
            input_norm_bins=bins,
            accuracy=np.random.uniform(0, 1.0),
        )


def _create_mock_writer(dest_path, **kwargs):
    ret = mock.create_autospec(flax_tb.SummaryWriter)
    ret.dest_path = dest_path
    return ret


def _create_empty_state(step: int) -> flax.training.train_state.TrainState:
    state = flax.training.train_state.TrainState.create(
        apply_fn=None, params=None, tx=optax.identity()
    )
    state = state.replace(step=123)
    return state


@mock.patch("flax.metrics.tensorboard.SummaryWriter", new=_create_mock_writer)
class EvalResultsPublisherTest(chex.TestCase):
    def test_eval_results_writer(self):
        state = _create_empty_state(step=123)
        writer_fn = tensorboard.make_eval_results_writer("/dummy/root_dir")
        EvalResults.write_to_tensorboard = mock.MagicMock()
        writer_fn(
            dict(
                devset=EvalResults.make_random_test_data(),
                evalset=EvalResults.make_random_test_data(),
            ),
            state,
        )
        called_datasets = [
            call.args[1].dest_path.name
            for call in EvalResults.write_to_tensorboard.call_args_list
        ]
        self.assertSequenceEqual(sorted(called_datasets), sorted(["devset", "evalset"]))

    def test_eval_results_writer_with_filtering(self):
        state = _create_empty_state(step=123)
        writer_fn = tensorboard.make_eval_results_writer(
            "/dummy/root_dir", ["devset", "trainset"]
        )
        EvalResults.write_to_tensorboard = mock.MagicMock()
        writer_fn(
            dict(
                devset=EvalResults.make_random_test_data(),
                evalset=EvalResults.make_random_test_data(),
                trainset=EvalResults.make_random_test_data(),
            ),
            state,
        )
        called_datasets = [
            call.args[1].dest_path.name
            for call in EvalResults.write_to_tensorboard.call_args_list
        ]
        self.assertSequenceEqual(
            sorted(called_datasets), sorted(["devset", "trainset"])
        )

    def test_eval_results_writer_with_custom_writer(self):
        state = _create_empty_state(step=123)
        custom_write_to_tensorboard = mock.MagicMock()
        writer_fn = tensorboard.make_eval_results_writer(
            "/dummy/root_dir", method=custom_write_to_tensorboard
        )
        writer_fn(
            dict(
                devset=EvalResults.make_random_test_data(),
                evalset=EvalResults.make_random_test_data(),
            ),
            state,
        )
        called_datasets = [
            call.args[2].dest_path.name
            for call in custom_write_to_tensorboard.call_args_list
        ]
        self.assertSequenceEqual(sorted(called_datasets), sorted(["devset", "evalset"]))


if __name__ == "__main__":
    absltest.main()
