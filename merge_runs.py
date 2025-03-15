#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import List

from datasets import concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from sal.utils.hub import get_dataset_files

import os
from pathlib import Path


"""Merge revisions of a dataset into a single config.

Usage:

# Merge all revisions of a dataset for a given seed
python merge_runs.py --dataset_name ./data/home/m00918254/TTC-checkpoints/tinyllama-math-code-checkpoint-100/dvts_math500/ --approach dvts --output_dir output --dataset_split train

"""


@dataclass
class Args:
    dataset_name: str
    approach: str
    output_dir: str
    dataset_split: str


def load_single_revision(args):
    dataset_name, revision, dataset_split = args
    fname = os.path.join(dataset_name, revision)
    """Load a single dataset revision."""
    samples = load_dataset("json", data_files=fname, split=dataset_split)
    # samples = load_dataset("json", data_files=fname)
    return samples

def main():
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    revisions = get_dataset_files(args.dataset_name)
    # if args.filter_strings:
    #     revisions = [
    #         revision
    #         for revision in revisions
    #         if all(filter_string in revision for filter_string in args.filter_strings)
    #     ]

    merged_config = os.path.join(args.dataset_name, "dvts_completions.jsonl")
    print(f"Merging {len(revisions)} revisions to create config `{merged_config}`")

    # Prepare arguments for multiprocessing
    pool_args = [
        (args.dataset_name, revision, args.dataset_split) for revision in revisions
    ]


    # Use multiprocessing to load datasets in parallel
    with Pool(cpu_count()) as pool:
        datasets = list(
            tqdm(
                pool.imap(load_single_revision, pool_args),
                total=len(revisions),
                desc="Loading datasets",
            )
        )

    # Concatenate datasets
    merged_dataset = concatenate_datasets(datasets)

    # Sanity check
    if "problem" in merged_dataset.column_names and len(
        merged_dataset.unique("problem")
    ) != len(merged_dataset):
        raise ValueError("Found duplicate problems")
    if "lighteval_MATH" in merged_config and len(merged_dataset) != 5000:
        raise ValueError(f"Expected 5000 samples, got {len(merged_dataset)}")
    if "MATH-500" in merged_config and len(merged_dataset) != 500:
        raise ValueError(f"Expected 500 samples, got {len(merged_dataset)}")

    # Push merged dataset to the hub
    # url = merged_dataset.push_to_hub(
    #     args.dataset_name,
    #     config_name=merged_config,
    #     split=args.dataset_split,
    #     private=args.hub_dataset_private,
    # )
    # print(f"Pushed merged dataset to {url}")
    
    if args.output_dir is None:
            args.output_dir = f"{args.dataset_name}"
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    merged_dataset.to_json(f"{args.dataset_name}/{args.output_dir}/{args.approach}_completions.jsonl", lines=True)
    print(
            f"Saved completions to {args.dataset_name}/{args.output_dir}/{args.approach}_completions.jsonl"
        )


if __name__ == "__main__":
    main()
