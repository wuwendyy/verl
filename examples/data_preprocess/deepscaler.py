# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the deepscaler dataset to parquet format
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    """Extract the solution from the solution string, removing boxed formatting if present."""
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/deepscaler", help="Local directory to save processed data")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy processed data")

    args = parser.parse_args()

    data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    print(f"Loading the dataset from {data_source}...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    original_dataset = dataset['train']
    # There is no test set in the dataset
    
    # Split dataset: 1/100 for validation, rest for training
    total_size = len(original_dataset)
    val_size = max(1, total_size // 100)  # At least 1 sample for validation
    train_size = total_size - val_size
    
    print(f"Total dataset size: {total_size}")
    print(f"Validation size: {val_size} ({val_size/total_size*100:.2f}%)")
    print(f"Training size: {train_size} ({train_size/total_size*100:.2f}%)")
    
    # Split the dataset
    split_dataset = original_dataset.train_test_split(test_size=val_size, seed=42, shuffle=True)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']  # The 'test' split from train_test_split is our validation set

    # Instruction to add to each problem
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # Process function to transform each example
    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract the problem, answer, and solution
            problem = example.pop("problem")
            question = problem + " " + instruction_following
            
            answer = example.pop("answer")
            solution = example.pop("solution")
            
            # No need to extract the answer from solution, as we already have the answer in the dataset
           
            # Create the processed data structure
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",  # Assuming this is a math-related dataset
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split, 
                    "index": idx,
                    "ground_solution": solution,
                },
            }
            
            return data

        return process_fn

    # Apply the processing function to both datasets
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)
    
    # Prepare output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    hdfs_dir = args.hdfs_dir

    print(f"Saving processed datasets to {local_dir}...", flush=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))
    
    # Save one example from each split as JSON for reference
    train_example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_example, f, indent=2)
    
    val_example = val_dataset[0]
    with open(os.path.join(local_dir, "val_example.json"), "w") as f:
        json.dump(val_example, f, indent=2)
    
    print(f"Saved {len(train_dataset)} training samples to train.parquet")
    print(f"Saved {len(val_dataset)} validation samples to val.parquet")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)