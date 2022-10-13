#!/bin/sh
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


# rm -fr devenv
# python3 -m venv devenv
source devenv/bin/activate

pip3 install --upgrade pip
pip3 install tensorflow
pip3 install tfds-nightly
pip3 install "jax[cpu]"
pip3 install flax
pip3 install fiddle

pip3 install black
pip3 install flake8
pip3 install pytype
