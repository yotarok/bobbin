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

import unittest

from absl.testing import absltest
import chex
import flax
import jax
import numpy as np

import bobbin
from bobbin import evaluation

Batch = bobbin.Batch


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
        serializable = flax.serialization.to_state_dict(results)
        # clear data, maintaining the tree structure.
        loaded = jax.tree_util.tree_map(lambda x: x * 0, results)
        loaded = flax.serialization.from_state_dict(loaded, serializable)
        jax.tree_util.tree_map(
            lambda x, y: np.testing.assert_allclose(x, y), results, loaded
        )


class SumEvalTask(evaluation.EvalTask):
    def create_eval_results(self, unused_dataset_name):
        return SimpleTreeResults(value=0.0)

    def evaluate(self, batch: Batch) -> SimpleTreeResults:
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
        results = evaluation.eval_batches(eval_task, "test", batches)
        assert isinstance(results, SimpleTreeResults)
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
            evaluation.eval_batches(eval_task, "test", batches)

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
        np.testing.assert_("set1" in results)
        np.testing.assert_("set2" in results)


class SampledSetTest(chex.TestCase):
    def test_size(self):
        s = evaluation.SampledSet(5)

        np.testing.assert_equal(len(s), 0)
        num_add = 30
        for n in range(num_add):
            s = s.add(n)
            np.testing.assert_equal(len(s), min(n + 1, 5))

    def test_size_of_union_with_iterables(self):
        s = evaluation.SampledSet(5)

        np.testing.assert_equal(len(s), 0)
        num_add = 4
        for n in range(num_add):
            s = s.union(range(n))

        np.testing.assert_equal(len(s), 5)

    def test_size_of_union_with_sampled_set(self):
        s = evaluation.SampledSet(5)

        np.testing.assert_equal(len(s), 0)
        num_add = 4
        for n in range(num_add):
            y = evaluation.SampledSet(10).union(range(n))
            s = s.union(y)

        np.testing.assert_equal(len(s), 5)

    @unittest.mock.patch(
        "bobbin.evaluation.SampledSet._draw_priority", new=lambda self, x: -x
    )
    def test_with_mocked_priority(self):
        s = evaluation.SampledSet(5)
        num_add = 5
        for n in range(num_add):
            print(s)
            s = s.union(range(n))

        np.testing.assert_equal(tuple(s), (3, 2, 2, 1, 1))


if __name__ == "__main__":
    absltest.main()
