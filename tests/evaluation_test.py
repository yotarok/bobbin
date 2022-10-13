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

"""Tests for evaluation."""

from __future__ import annotations

from absl.testing import absltest
import chex
import flax
import jax
import numpy as np

from bobbin import Batch
from bobbin import EvalResults
from bobbin import evaluation


@flax.struct.dataclass
class SimpleTreeResults(evaluation.EvalResults):
    value: chex.ArrayTree

    def reduce(self, other: SimpleTreeResults) -> SimpleTreeResults:
        return jax.tree_util.tree_map(lambda x, y: x + y, self, other)

    @classmethod
    def test_sample_1(cls):
        return cls(
            value=dict(
                layers=(
                    dict(x=np.array([1.0, 2.0, 3.0])),
                    dict(x=np.array([2.0, 4.0, 6.0])),
                ),
                losses=(dict(xent=5.0, l2=2.0)),
            )
        )


class EvalResultsTest(chex.TestCase):
    def test_unshard_and_reduce(self):
        results = SimpleTreeResults.test_sample_1()
        sharded_results = jax.tree_util.tree_map(
            lambda x: np.stack([x * 0, x, x * 2, x * 3]), results
        )
        expected = jax.tree_util.tree_map(lambda x: np.asarray(x) * 6, results)
        unsharded = sharded_results.unshard_and_reduce()
        jax.tree_util.tree_map(
            lambda x, y: np.testing.assert_allclose(x, y), unsharded, expected
        )

    def test_state_dict(self):
        results = SimpleTreeResults.test_sample_1()
        serializable = results.to_state_dict()
        # clear data, maintaining the tree structure.
        loaded = jax.tree_util.tree_map(lambda x: x * 0, results)
        loaded = loaded.load_state_dict(serializable)
        jax.tree_util.tree_map(
            lambda x, y: np.testing.assert_allclose(x, y), results, loaded
        )


class SumEvalTask(evaluation.EvalTask):
    def create_eval_results(self):
        return SimpleTreeResults(value=0.0)

    def evaluate(self, batch: Batch) -> EvalResults:
        return SimpleTreeResults(
            value=jax.tree_util.tree_reduce(lambda acc, x: acc + x.sum(), batch, 0.0)
        )


class EvalTaskTest(chex.TestCase):
    def test_eval_batches(self):
        eval_task = SumEvalTask()
        batch_shape = (3, 3)
        batches = [
            np.ones(batch_shape) * 0.1,
            np.ones(batch_shape) * 0.2,
            np.ones(batch_shape) * 0.3,
            np.ones(batch_shape) * 0.4,
        ]
        results = evaluation.eval_batches(eval_task, batches)
        np.testing.assert_allclose(results.value, 1.0 * np.prod(batch_shape))

    def test_type_check(self):
        def incompatible_evaluate_method(self, batch):
            return jax.tree_util.tree_reduce(lambda acc, x: acc + x.sum(), batch, 0.0)

        SumEvalTask.evaluate = incompatible_evaluate_method
        eval_task = SumEvalTask()
        batches = [
            np.ones((3, 3)) * 0.1,
        ]
        with self.assertRaises(TypeError):
            evaluation.eval_batches(eval_task, batches)

    def test_eval_datasets(self):
        batch_shape = (3, 3)
        batches = [
            np.ones(batch_shape) * 0.1,
            np.ones(batch_shape) * 0.2,
        ]
        batch_gens = dict(
            set1=lambda: iter(batches),
            set2=lambda: iter(batches),
        )
        eval_task = SumEvalTask()
        results = evaluation.eval_datasets(eval_task, batch_gens)
        assert "set1" in results
        assert "set2" in results


if __name__ == "__main__":
    absltest.main()
