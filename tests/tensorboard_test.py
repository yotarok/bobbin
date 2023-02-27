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
from absl.testing import parameterized
import chex
import flax
from flax.metrics import tensorboard as flax_tb
import jax
import numpy as np
import optax

import bobbin
from bobbin import tensorboard


class TrainSummaryPublisherTest(chex.TestCase):
    def test_default(self):
        variables = {
            "params": {
                "layers": (
                    {
                        "weight": np.random.normal(size=(3, 3)),
                    },
                ),
            },
            "tensorboard": {
                "layers": (
                    {
                        "mean_output": tensorboard.ScalarSummary(1.23),
                        "var_output": tensorboard.ScalarSummary(2.46),
                    },
                )
            },
        }
        writer = mock.create_autospec(flax_tb.SummaryWriter)
        tensorboard.publish_train_intermediates(writer, variables["tensorboard"], 123)
        print(writer.scalar.call_args_list)
        writer.scalar.assert_any_call("summary/layers/0/mean_output", 1.23, step=123)
        writer.scalar.assert_any_call("summary/layers/0/var_output", 2.46, step=123)

    def test_image(self):
        image = np.random.uniform(size=(13, 13, 3))
        tb_vars = dict(x=tensorboard.ImageSummary(image=image))
        writer = mock.create_autospec(flax_tb.SummaryWriter)
        tensorboard.publish_train_intermediates(writer, tb_vars, 123)
        writer.image.assert_any_call("summary/x", image, step=123)

    def test_mpl_image(self):
        data = np.random.uniform(size=(17, 11))
        h_paddings = (np.arange(11) >= 7).astype(np.float32)
        v_paddings = (np.arange(17) >= 11).astype(np.float32)
        tb_vars = dict(
            x=tensorboard.MplImageSummary(
                image=data, h_paddings=h_paddings, v_paddings=v_paddings
            )
        )
        writer = mock.create_autospec(flax_tb.SummaryWriter)
        tensorboard.publish_train_intermediates(writer, tb_vars, 123)
        writer.image.assert_any_call("summary/x", mock.ANY, step=123)


class EvalResults(bobbin.evaluation.EvalResults):
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


_SummaryWriterOriginal = flax_tb.SummaryWriter


def _create_mock_writer(dest_path, **kwargs):
    ret = mock.create_autospec(_SummaryWriterOriginal)
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
        writer = tensorboard.MultiDirectorySummaryWriter(
            "/dummy/root_dir", keys=("devset", "evalset"), use_threaded_writer=False
        )
        writer_fn = writer.as_result_processor()
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

    def test_eval_results_writer_with_error(self):
        state = _create_empty_state(step=123)
        writer = tensorboard.MultiDirectorySummaryWriter(
            "/dummy/root_dir", keys=("devset", "evalset")
        )
        writer_fn = writer.as_result_processor()
        EvalResults.write_to_tensorboard = mock.MagicMock()

        with np.testing.assert_raises(ValueError):
            writer_fn(
                dict(
                    devset=EvalResults.make_random_test_data(),
                    evalset=EvalResults.make_random_test_data(),
                    trainset=EvalResults.make_random_test_data(),
                ),
                state,
            )

    def test_eval_results_writer_with_custom_writer(self):
        state = _create_empty_state(step=123)
        custom_write_to_tensorboard = mock.MagicMock()
        writer = tensorboard.MultiDirectorySummaryWriter(
            "/dummy/root_dir",
            keys=("devset", "evalset"),
            use_threaded_writer=False,
        )
        writer_fn = writer.as_result_processor(method=custom_write_to_tensorboard)
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

    @mock.patch("jax.process_count")
    @mock.patch("jax.process_index")
    def test_eval_results_writer_switch_in_multi_process_env(
        self, process_index_mock, process_count_mock
    ):
        state = _create_empty_state(step=123)
        random_test_data = dict(
            devset=EvalResults.make_random_test_data(),
            evalset=EvalResults.make_random_test_data(),
        )

        custom_write_to_tensorboard = mock.MagicMock()

        process_index_mock.return_value = 1
        process_count_mock.return_value = 4
        writer = tensorboard.MultiDirectorySummaryWriter(
            "/dummy/root_dir", keys=("devset", "evalset"), use_threaded_writer=False
        )
        writer_fn = writer.as_result_processor(method=custom_write_to_tensorboard)
        writer_fn(random_test_data, state)
        custom_write_to_tensorboard.assert_called()
        for call in custom_write_to_tensorboard.call_args_list:
            unused_results, unused_train_state, writer = call.args
            np.testing.assert_(isinstance(writer, tensorboard.NullSummaryWriter))

        custom_write_to_tensorboard.reset_mock()
        process_index_mock.return_value = 0
        process_count_mock.return_value = 4
        writer = tensorboard.MultiDirectorySummaryWriter(
            "/dummy/root_dir", keys=("devset", "evalset"), use_threaded_writer=False
        )
        writer_fn = writer.as_result_processor(method=custom_write_to_tensorboard)
        writer_fn(random_test_data, state)
        custom_write_to_tensorboard.assert_called()
        for call in custom_write_to_tensorboard.call_args_list:
            unused_results, unused_train_state, writer = call.args
            print(writer, process_index_mock)
            np.testing.assert_(isinstance(writer, _SummaryWriterOriginal))

        custom_write_to_tensorboard.reset_mock()
        process_index_mock.return_value = 1
        process_count_mock.return_value = 4
        writer = tensorboard.MultiDirectorySummaryWriter(
            "/dummy/root_dir",
            keys=("devset", "evalset"),
            only_from_leader_process=False,
        )
        writer_fn = writer.as_result_processor(method=custom_write_to_tensorboard)
        writer_fn(random_test_data, state)
        custom_write_to_tensorboard.assert_called()
        for call in custom_write_to_tensorboard.call_args_list:
            unused_results, unused_train_state, writer = call.args
            np.testing.assert_(isinstance(writer, _SummaryWriterOriginal))

    def test_dispatch(self):
        # This is in fact an unexported/ unused feature, but developed for
        # cleaner API.
        writer = tensorboard.MultiDirectorySummaryWriter(
            "/dummy/root_dir", keys=("sub1", "sub2"), use_threaded_writer=False
        )

        # pytype: disable=attribute-error
        def reset():
            for s in ("sub1", "sub2"):
                writer.subwriter(s).reset_mock()

        writer.close()
        writer.subwriter("sub1").close.assert_called_once()
        writer.subwriter("sub2").close.assert_called_once()
        reset()

        writer.flush()
        writer.subwriter("sub1").flush.assert_called_once()
        writer.subwriter("sub2").flush.assert_called_once()
        reset()

        hparams = dict(x=3, y=2)
        writer.hparams(hparams)
        writer.subwriter("sub1").hparams.assert_called_once_with(hparams)
        writer.subwriter("sub2").hparams.assert_called_once_with(hparams)
        reset()

        writer.scalar("sub1/some/value", 1.23, 234)
        writer.subwriter("sub1").scalar.assert_called_once_with("some/value", 1.23, 234)
        writer.subwriter("sub2").scalar.assert_not_called()
        reset()

        with np.testing.assert_raises(ValueError):
            writer.scalar("sub3/some/value", 1.23, 234)
        reset()

        image = np.random.normal(size=(3, 3))
        writer.image("sub2/nice/image", image, 234, max_outputs=1)
        writer.subwriter("sub1").image.assert_not_called()
        writer.subwriter("sub2").image.assert_called_once_with(
            "nice/image", image, 234, max_outputs=1
        )
        reset()

        audio = np.random.normal(size=(13,))
        writer.audio("sub1/good/audio", audio, 234)
        writer.subwriter("sub1").audio.assert_called_once_with(
            "good/audio", audio, 234, sample_rate=44100, max_outputs=3
        )
        writer.subwriter("sub2").audio.assert_not_called()
        reset()

        values = np.random.normal(size=(13,))
        writer.histogram("sub1/histo_gram/__", values, 456)
        writer.subwriter("sub1").histogram.assert_called_once_with(
            "histo_gram/__", values, 456, bins=None
        )
        writer.subwriter("sub2").histogram.assert_not_called()
        reset()

        writer.text("sub1/text", "hello", 234)
        writer.subwriter("sub1").text.assert_called_once_with("text", "hello", 234)
        writer.subwriter("sub2").text.assert_not_called()
        reset()

        writer.write("sub2/array", image, 234, metadata=audio)
        writer.subwriter("sub1").write.assert_not_called()
        writer.subwriter("sub2").write.assert_called_once_with(
            "array", image, 234, metadata=audio
        )
        reset()
        # pytype: enable=attribute-error


@mock.patch("logging.log")
class TrainerEnvPublisherTest(chex.TestCase):
    def test_publish(self, mock_log):
        writer = mock.create_autospec(flax_tb.SummaryWriter)
        params = dict(w=np.zeros((7, 7)), b=np.zeros((7,)))
        extra_vars = dict(
            non_trainable=dict(
                ema_w=np.zeros((7, 7)),
                ema_b=np.zeros((7,)),
            )
        )
        state = bobbin.TrainState.create(
            apply_fn=None, params=params, tx=optax.identity(), extra_vars=extra_vars
        )
        tensorboard.publish_trainer_env_info(writer, state, prefix="")

        texts = dict()
        for call in writer.text.call_args_list:
            tag, s, step = call.args
            np.testing.assert_equal(step, 0)
            texts[tag] = s

        np.testing.assert_(
            str(bobbin.total_dimensionality(params)) in texts["total_num_params"]
        )
        np.testing.assert_(bobbin.summarize_shape(params) in texts["param_shape"])

        all_log_text = ""
        for call in mock_log.call_args_list:
            unused_loglevel, s, *format_args = call.args
            s = s % tuple(format_args)
            all_log_text += s + "\n"
        np.testing.assert_(bobbin.summarize_shape(params) in all_log_text)


class ThreadedSummaryWriterTest(chex.TestCase):
    @parameterized.named_parameters(
        ("scalar", "scalar", ("tag", 1.23, np.asarray(123))),
        ("image", "image", ("tag", np.random.uniform((6, 6, 3)), 123)),
        ("audio", "audio", ("tag", np.random.uniform((123,)), 23)),
        ("histogram", "histogram", ("tag", np.random.uniform((5,)), 234)),
        ("text", "text", ("texttag", "hello", 123)),
        ("write", "write", ("any", np.array([1, 2]), 1)),
        ("hparams", "hparams", (dict(x=3, y=4),)),
    )
    def test_eventual_write(self, method_name, test_args):
        base_writer = _create_mock_writer("dest")
        writer = bobbin.ThreadedSummaryWriter(base_writer, wait_on_flush=True)

        method = getattr(writer, method_name)
        method(*test_args)
        writer.flush()

        base_writer.flush.assert_called_once()

        wrapped_method = getattr(base_writer, method_name)
        wrapped_method.assert_called_once()
        passed_args = wrapped_method.call_args.args[: len(test_args)]
        np.testing.assert_equal(passed_args, test_args)
        for arg in passed_args:
            # Make sure arguments are not a Jax Array (that can be on device).
            np.testing.assert_(not isinstance(arg, jax.Array))


if __name__ == "__main__":
    absltest.main()
