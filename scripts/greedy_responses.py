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
import tokenize

import torch
import click
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)    


qeval = {
    "llama-math-pf":(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "{output}<|eot_id|>"
    ),
    "qwen-math-pf":(
        "<|im_start|>system\n"
        "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
        "<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
}




def greedy_response(sample, config, llm):
    question = sample["problem"]
    tokenizer = llm.get_tokenizer()
    system = [
        {
            "role": "system",
            "content": config.system_prompt,
        }
    ]

 
    prompt = tokenizer.apply_chat_template(
        system + [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sample['formatted_prompt'] = prompt

    return sample
    # qwen_formatted = qeval["qwen-math-pf"][0].format(input=question, answer="{answer}")

    # breakpoint()



def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    # num_gpus = 4
    num_gpus = 4 if "qwen2" in config.model_path.lower() else torch.cuda.device_count()
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )

    
    dataset = get_dataset(config)

    print("Length of dataset:", len(dataset))

    save_steps = 50
    splits = [dataset.select(range(i, i+save_steps)) for i in range(0, len(dataset), save_steps)]

    # print number of splits and length of each split
    print("Number of splits:", len(splits))
    print("Length of each split:", [len(split) for split in splits])

    for i, dataset in enumerate(splits):
        print("--------"*20)
        print("Processing batch:", i)
        dataset = dataset.map(
            greedy_response,
            batched=False,
            batch_size=20,
            fn_kwargs={"config": config, "llm": llm,},
            desc="Running search",
            load_from_cache_file=False,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=config.max_tokens,
            stop_token_ids=(
                [151645, 151643]
                if "qwen2" in config.model_path.lower()
                else None
            ),
        )

        res_lists = llm.generate(list(dataset['formatted_prompt']), sampling_params)  
        response_texts = [res.outputs[0].text for res in res_lists] 
        dataset = dataset.add_column('greedy_response', response_texts)

        dataset.to_json(f"{config.output_dir}/{config.dataset_name}_batch_{i}.jsonl", orient="records", lines=True)
    # save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()

