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

"""Benchmark script for data-pipeline.
"""

from __future__ import annotations

import argparse

import tensorflow_datasets as tfds

import train


def main(args: argparse.Namespace):
    batch_size = 32
    train_ds, eval_dss = train.prepare_datasets(
        tfds_data_dir=args.tfds_data_dir,
        wpm_vocab_path=args.wpm_vocab,
        wpm_size_limit=args.wpm_size_limit,
        train_batch_size=batch_size,
    )

    ds = train_ds
    if args.split is not None:
        ds = eval_dss[args.split]

    results = tfds.benchmark(ds, batch_size=batch_size, num_iter=100)
    print(results)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="MNIST training")
    argparser.add_argument("--tfds_data_dir", type=str, default=None)
    argparser.add_argument("--wpm_vocab", type=str, default=None)
    argparser.add_argument("--wpm_size_limit", type=int, default=None)
    argparser.add_argument("--split", type=str, default=None)
    args = argparser.parse_args()
    main(args)
