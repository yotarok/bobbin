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

"""Tests for cron."""

import unittest

from absl.testing import absltest
import chex
import flax
import numpy as np
import optax

from bobbin import cron

_FrozenDict = flax.core.FrozenDict


class TriggerTest(chex.TestCase):
    def test_for_each_n_steps(self):
        state = flax.training.train_state.TrainState.create(
            apply_fn=None, params=None, tx=optax.identity()
        )
        trigger = cron.ForEachNSteps(10)
        for step in range(1000):
            state = state.replace(step=step)
            self.assertEqual(trigger.check(state), step % 10 == 0)

    def test_at_first_n_steps(self):
        state = flax.training.train_state.TrainState.create(
            apply_fn=None, params=None, tx=optax.identity()
        )
        trigger = cron.AtFirstNSteps(10)
        for step in range(100):
            state = state.replace(step=step)
            self.assertEqual(trigger.check(state), step <= 10)

    def test_at_nth_step(self):
        state = flax.training.train_state.TrainState.create(
            apply_fn=None, params=None, tx=optax.identity()
        )
        trigger = cron.AtNthStep(17)
        for step in range(100):
            state = state.replace(step=step)
            self.assertEqual(trigger.check(state), step == 17)

    @unittest.mock.patch("time.time")
    def test_for_each_t_seconds(self, time_mock):
        state = flax.training.train_state.TrainState.create(
            apply_fn=None, params=None, tx=optax.identity()
        )
        trigger = cron.ForEachTSeconds(7)
        time_mock.return_value = 10.0
        trigger.check(state)

        time_mock.return_value = 15.0
        self.assertFalse(trigger.check(state))

        time_mock.return_value = 19.0
        self.assertTrue(trigger.check(state))

        # This doesn't trigger even though it's 15 seconds since the first check
        # because it's still only 6 seconds from the last check.
        time_mock.return_value = 25.0
        self.assertFalse(trigger.check(state))

        time_mock.return_value = 27.0
        self.assertTrue(trigger.check(state))

    def test_or_trigger(self):
        state = flax.training.train_state.TrainState.create(
            apply_fn=None, params=None, tx=optax.identity()
        )
        trigger = cron.AtFirstNSteps(5) | cron.ForEachNSteps(3)
        for step in range(30):
            state = state.replace(step=step)
            self.assertEqual(trigger.check(state), step <= 5 or step % 3 == 0)

    def test_trigger_creation_by_schedule(self):
        state = flax.training.train_state.TrainState.create(
            apply_fn=None, params=None, tx=optax.identity()
        )
        crontab = cron.CronTab()

        def do_nothing(*unused_args):
            pass

        crontab.schedule(do_nothing, step_interval=123, at_first_steps=10)
        unused_name, trigger, unused_action = crontab._actions[0]
        np.testing.assert_(trigger.check(state.replace(step=123)))
        np.testing.assert_(not trigger.check(state.replace(step=124)))
        np.testing.assert_(trigger.check(state.replace(step=3)))
        np.testing.assert_(not trigger.check(state.replace(step=11)))


if __name__ == "__main__":
    absltest.main()
