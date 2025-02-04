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

import logging

import torch
import click
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts, particle_gibbs
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "particle_gibbs": particle_gibbs,
}

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]

    dataset = get_dataset(config)

    num_gpus = 4 if "qwen2" in config.model_path.lower() else torch.cuda.device_count()

    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )
    prm = load_prm(config)
    
    print("==="*20)
    print("config:", config)
    print("==="*20)
    print("approach_fn:", approach_fn)
    print("==="*20)

    print("Length of dataset:", len(dataset))

    # splits = [dataset.select(range(i, i+1)) for i in range(len(dataset))]

    save_steps = 10
    splits = [dataset.select(range(i, i+save_steps)) for i in range(0, len(dataset), save_steps)]

    # print number of splits and length of each split
    print("Number of splits:", len(splits))
    print("Length of each split:", [len(split) for split in splits])

    for i, dataset in enumerate(splits):
        print("--------"*20)
        print("Processing batch:", i)
        dataset = dataset.map(
            approach_fn,
            batched=False,
            batch_size=1,
            fn_kwargs={"config": config, "llm": llm, "prm": prm},
            desc="Running search",
            load_from_cache_file=False,
        )
        
        dataset.to_json(f"{config.output_dir}/batch_{i}.jsonl", orient="records", lines=True)
    # save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
