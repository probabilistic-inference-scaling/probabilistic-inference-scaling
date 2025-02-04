import random
import numpy as np
from vllm import SamplingParams


def softmax(x):
    """
    Compute softmax values for a vector x.

    Args:
        x (numpy.ndarray): Input array of shape (n,)

    Returns:
        numpy.ndarray: Softmax probabilities of shape (n,)
    """
    # Subtract max for numerical stability
    # This prevents overflow when computing exp
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def inverse_sigmoid(x):
    """
    Calculate the inverse sigmoid (logit) of a value x.

    Args:
        x (float): Input value between 0 and 1 (exclusive)

    Returns:
        float: The inverse sigmoid value

    Raises:
        ValueError: If x is not between 0 and 1
    """
    # Add small epsilon to prevent log(0)
    eps = np.finfo(float).eps
    x = np.clip(x, eps, 1 - eps)

    return np.log(x) - np.log(1 - x)  # More stable than np.log(x/(1-x))


def resample_responses(responses, weights):
    """
    Resample responses based on weights using multinomial sampling.
    """
    indices = random.choices(range(len(responses)), weights, k=len(responses))
    return [responses[i] for i in indices]


def improvement_step(question, response, llm, config):
    """
    Improve a response using the LLM model.
    """
    tokenizer = llm.get_tokenizer()
    messages = [
        {
            "role": "system",
            "content": config.system_prompt,
        }
    ]
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=1024,
        top_p=1.0,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in config.model_path.lower()
            else None
        ),
    )
    improvement_prompt = "Improve the response while adhering to the format required by the task (final answer should be wrapped with the \boxed latex tag). If the response is wrong, correct the answer. If the response is correct, return it as is."
    messages.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
            {"role": "user", "content": improvement_prompt},
        ]
    )

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    response = llm.generate(prompt, sampling_params=sampling_params)

    return (
        response[0]
        .outputs[0]
        .text.strip("<|start_header_id|>assistant<|end_header_id|>")
        .strip()
    )


def initial_responses(question, llm, config):
    """
    Generate initial responses to a question using the LLM model.
    """
    tokenizer = llm.get_tokenizer()
    messages = [
        {
            "role": "system",
            "content": config.system_prompt,
        }
    ]
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=1024,
        top_p=1.0,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in config.model_path.lower()
            else None
        ),
    )
    prompt = tokenizer.apply_chat_template(
        messages + [{"role": "user", "content": question}], tokenize=False
    )
    responses = llm.generate(prompt, sampling_params=sampling_params)
    return (
        responses[0]
        .outputs[0]
        .text.strip("<|start_header_id|>assistant<|end_header_id|>")
        .strip()
    )


def particle_gibbs_improve_kernel(question, responses, model, prm, config):
    # Step 1: Evaluate the responses and assign weights
    scores = [prm.score([question], [[response]], batched=False)[-1][-1][-1] for response in responses]
    scores = [inverse_sigmoid(score) for score in scores]

    # Step 2: Resample responses based on weights
    weights = softmax(scores)
    resampled_responses = resample_responses(responses, weights)

    # Step 3: Apply the improvement step
    improved_responses = [
        improvement_step(question, response, model, config)
        for response in resampled_responses
    ]

    return improved_responses, weights


def particle_gibbs(x, config, llm, prm):
    question = x["problem"]
    res = [
        initial_responses(question, llm, config),
        initial_responses(question, llm, config),
        initial_responses(question, llm, config),
    ]

    stepwise_responses = []
    stepwise_weights = []

    for i in range(config.n):
        print(f"Running iteration {i+1} of {config.n}")
        res, weights = particle_gibbs_improve_kernel(question, res, llm, prm, config)
        stepwise_responses.append(res)
        stepwise_weights.append(weights)

    x["stepwise_responses"] = stepwise_responses
    x["stepwise_weights"] = stepwise_weights
    return x
