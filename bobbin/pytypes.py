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

"""Module for typing.
"""

from typing import Callable, Dict, Iterable, Union

import chex
import jaxlib

# Type-aliases for writing self-documenting code.
Batch = chex.ArrayTree
Backend = Union[str, jaxlib.xla_extension.Client, None]
Device = jaxlib.xla_extension.Device
Parameter = chex.ArrayTree
BatchGen = Callable[[], Iterable[Batch]]
VarCollection = Dict[str, chex.ArrayTree]
