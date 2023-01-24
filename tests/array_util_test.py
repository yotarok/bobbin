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

"""Tests for array_util."""

from absl.testing import absltest
import chex
import numpy as np

from bobbin import array_util


class ArrayUtilTest(chex.TestCase):
    def test_splitter(self):
        nparts = 5
        microbatch_size = 3
        dims = 7

        microbatches = []
        for i in range(nparts):
            microbatches.append(i * np.ones((microbatch_size, dims)))

        batch = np.concatenate(microbatches, axis=0)

        split_batches = array_util.split_leading_axis(nparts, batch)

        for i, split_batch in enumerate(split_batches):
            np.testing.assert_allclose(split_batch, i)


if __name__ == "__main__":
    absltest.main()
