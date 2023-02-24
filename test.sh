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

set -xeuo pipefail

readonly VENV_DIR=/tmp/bobbin-test
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip setuptools wheel
pip install --upgrade flake8 pytest-xdist pytype black nbqa
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-test.txt

pip uninstall -y bobbin || true

## code health check
black --check bobbin examples tests
flake8 --ignore=E203,W503 --max-line-length=88 bobbin examples tests

## type checking
pytype --config=pytype.toml

for pyifile in $(find .pytype -name \*.pyi) ; do
  if [[ $(cat ${pyifile} | grep "Caught error in pytype") ]] ; then
    echo "found error in $pyifile"
    exit -1
  fi
done

## tests
PYTHONPATH=. pytest -n auto tests

pip install -r requirements/requirements-docs.txt
make -C docs html

nbqa black --check docs/*.ipynb
nbqa flake8 --ignore=E203,W503 --max-line-length=88 docs/*.ipynb
nbqa pytype docs/*.ipynb

