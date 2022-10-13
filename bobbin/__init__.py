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

"""bobbin -- a tool for making training loop with flax.linen.
"""

# flake8: noqa: F401

from .cron import AtFirstNSteps
from .cron import AtNthStep
from .cron import CronTab
from .cron import ForEachNSteps
from .cron import ForEachTSeconds
from .cron import RunEval
from .cron import RunEvalKeepBest
from .cron import SaveCheckpoint
from .cron import WriteLog

from .evaluation import EvalResults
from .evaluation import EvalTask
from .evaluation import eval_batches
from .evaluation import eval_datasets

from .pmap_util import ArgType
from .pmap_util import wrapped_pmap
from .pmap_util import BroadcastArg
from .pmap_util import RngArg
from .pmap_util import ShardArg
from .pmap_util import StaticArg
from .pmap_util import ThruArg
from .pmap_util import unshard

from .pytypes import Batch
from .pytypes import BatchGen
from .pytypes import Parameter
from .pytypes import VarCollection

from .tensorboard import make_eval_results_writer

from .training import initialize_train_state
from .training import pmap_for_train_step
from .training import TrainTask
from .training import TrainState

from .var_util import prng_keygen

__version__ = "0.0.1"
