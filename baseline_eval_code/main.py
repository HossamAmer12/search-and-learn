import argparse
import json
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams

import warnings
import contextlib

import requests
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings

# Hossam import grader and parser
from grader import *
from parser import *


@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass


def format_prompt(question, model_name):
    """
    Format the prompt using the template based on the model name.
    """
    if "llama" in model_name.lower():
        prompt = f"""
Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.

Question: {question}
"""
    elif "qwen" in model_name.lower():
        prompt = f"""
Please reason step by step, and put your final answer within \\boxed{{}}.

Question: {question}
"""
    else:
        raise ValueError("Unsupported model. Use a LLaMA or Qwen model.")
    
    return prompt

def format_prompt_2(question, model_name):
    """
    Format the prompt using the template based on the model name.
    """
    prompt = f"""
Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.

Question: {question}
"""
   
    return prompt

def format_prompt_3(question, model_name):
    """
    Format the prompt using the template based on the model name.
    """
    prompt = f"""
Please reason step by step, and put your final answer within \\boxed{{}}.

Question: {question}
"""
   
    return prompt
    

import requests

# disable ssl warning
requests.packages.urllib3.disable_warnings()

# override the methods which you use
requests.post = lambda url, **kwargs: requests.request(
    method="POST", url=url, verify=False, **kwargs
)

requests.get = lambda url, **kwargs: requests.request(
    method="GET", url=url, verify=False, **kwargs
)


#def extract_answer(response):
#    """
#    Extracts the answer from the model response using the boxed {} format.
#    """
#    # match = re.search(r'\\boxed{([^}]*)}', response)
#    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
#    match = re.search(pattern, response)
#    # match = re.search(r'\\boxed{([^{}]+|{[^{}]+})}', response)
#    # match = re.search(r'\$\\boxed{([^}]*)}\\$', response)
#    return match.group(1) if match else None

def extract_answer_2(response):
    """
    Looks for the first occurrence of:  $\\boxed{ ... }$
    and returns everything inside the braces, ignoring nesting.
    """
    start_marker = '$\\boxed{'
    end_marker = '}$'

    # 1) Find the first occurrence of '$\\boxed{'
    start_idx = response.find(start_marker)
    if start_idx == -1:
        return None

    # 2) Calculate the actual content start (right after '$\\boxed{')
    content_start = start_idx + len(start_marker)

    # 3) Find the next '}$' after that
    end_idx = response.find(end_marker, content_start)
    if end_idx == -1:
        return None

    # 4) Return the substring inside { ... }
    return response[content_start:end_idx]
    
# def extract_answer(response):
#     """
#     Extracts the answer from the model response using the \boxed{...} format,
#     allowing an extra set of braces and some nesting.
#     """
#     # This pattern:
#     #   1) Looks for '\boxed' literally.
#     #   2) Allows one or more '{' characters: \{+
#     #   3) Captures any characters that are not braces, plus any one-level nested braces:
#     #       [^{}]*(?:\{[^{}]*\}[^{}]*)*
#     #   4) Followed by one or more '}' characters: \}+
#     pattern = r'\\boxed\{+([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}+'
    
#     match = re.search(pattern, response)
#     output = match.group(1) if match else None
#     if output is None:
#         return extract_answer_2(response)
#     else:
#         return output

def compute_accuracy(predictions, ground_truths):
    """
    Computes the exact match accuracy.
    """
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return correct / len(ground_truths)

def normalize_latex(expression):
    # Remove all whitespace
    expression = re.sub(r'\s+', '', expression)
    
    # Normalize commands: remove space between command and its argument
    expression = re.sub(r'(\\[a-zA-Z]+)\s*(\{|\[)', r'\1\2', expression)
    
    # Normalize parentheses, brackets, and braces: remove space inside
    expression = re.sub(r'([\(\[\{])\s*(.*?)\s*([\)\]\}])', r'\1\2\3', expression)
    
    # Normalize commas: remove space around commas
    expression = re.sub(r'\s*,\s*', ',', expression)

    return expression

def match_expressions(predicted, ground_truth):
    normalized_predicted = normalize_latex(predicted)
    normalized_ground_truth = normalize_latex(ground_truth)
    if normalized_ground_truth in normalized_predicted and len(normalized_ground_truth)==len(normalized_predicted)-2:
        return True
    else:
        return normalized_predicted == "{"+normalized_ground_truth+"}"

def main():
    parser = argparse.ArgumentParser(description="Run math problem generation with VLLM.")
    parser.add_argument("--model", type=str, default="TinyLlama/tinyLlama-intermediate-checkpoints-after-1T-token", help="Path or name of the LLM model.")
    parser.add_argument("--dataset", type=str, default="/data00/dataset/Math-500", help="Dataset to use.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to evaluate.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.1, help="Nucleus sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Nucleus sampling parameter.")
    parser.add_argument("--frequency_penalty", type=float, default=1.0, help="Nucleus sampling parameter.")
    parser.add_argument("--presence_penalty", type=float, default=1.0, help="Nucleus sampling parameter.")
     
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum output tokens.")
    args = parser.parse_args()

    # Load dataset
    with no_ssl_verification():
        dataset = load_dataset(args.dataset, split=args.split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    # Load model
    llm = LLM(
        model=args.model,tensor_parallel_size=1,
        dtype="float"
    )

    # Prepare sampling parameters
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty,\
    frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty)

    predictions = []
    ground_truths = []

    # Generate responses
    accuracy = []
    cnt = 0
    for sample in dataset:
        question = sample["problem"]
        ground_truth = sample["answer"]

        # Hossam format prompt
        # prompt = format_prompt(question, args.model)
        prompt = format_prompt_2(question, args.model)

        # convs = [
        #     {"role": "system", "content": prompt},
        #     {"role": "user", "content": question},
        # ]

        response = llm.generate([prompt], sampling_params)[0].outputs[0].text
        # response = llm.generate(convs, sampling_params)[0].outputs[0].text
        # print(f"response:\n{response}")
        # predicted_answer = extract_answer(response)

        # Hossam predicted answer extract answer using the code
        predicted_answer = extract_answer(response, data_name = "math", use_last_number=True)
        predictions.append(predicted_answer)
        ground_truths.append(ground_truth)

        #print(f"Q: {question}")
        print(f"Predicted: {predicted_answer}")
        print(f"Ground Truth: {ground_truth}")
        is_match = False
        if predicted_answer is not None:
            # is_match = match_expressions(predicted_answer,ground_truth)

            # Hossam is match 
            is_match = math_equal(predicted_answer, ground_truth)
            print("is match", is_match)
            
            accuracy.append(is_match)
        else:
            #print("False")
            accuracy.append(is_match)
        cnt += 1
        print(f"Sample: {cnt}: Exact Match Accuracy: {sum(accuracy)/len(accuracy):.2%}")

        # if is_match:
        #     print("debug")
        #     print(accuracy)
        #     print(sum(accuracy))
        #     print(len(accuracy))
        #     print(sum(accuracy)/len(accuracy))
        #     exit(0)
        #print("-" * 50)

    # Compute accuracy
    #accuracy = compute_accuracy(predictions, ground_truths)
    print(f"Model: {args.model}")
    print(f"Exact Match Accuracy: {sum(accuracy)/len(accuracy):.4%}")
    #data = {
    #  "model_name": args.model,
    #  "Exact Match Accuracy": f"{exact_match_accuracy:.4%}"
    #}
  
    # Define file path
    #file_path = os.path.join(args.model, "exp_info.json")
  
    # Save to JSON file
    #with open(file_path, "w") as json_file:
    #    json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    main()
