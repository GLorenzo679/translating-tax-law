import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import Tuple

# add the src directory to the Python path, needed to import the custom modules
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

lib_path = str(Path(__file__).parent.parent.parent)
if lib_path not in sys.path:
    sys.path.append(lib_path)

import cupy
import evaluate
import numpy as np
import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging,
    set_seed,
)
from unsloth import FastLanguageModel

from metrics import is_valid_syntax_tree, tree_edit_distance
from metrics.codebleu_catala import codebleu_metric


def set_global_seed(seed: int = 0):
    random.seed(seed)  # Python’s built-in random module
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch seed for CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch seed for all GPU devices
    cupy.random.seed(seed)  # CuPy seed for GPU computations
    set_seed(seed)  # Transformers library seed


set_global_seed(0)
logging.set_verbosity_info()

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to login to HuggingFace Hub if token is available
try:
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)
except Exception as e:
    print(f"Warning: Could not login to HuggingFace Hub: {e}")


def format_input(tokenizer: PreTrainedTokenizer, input_text: str, metadata: str) -> str:
    """Format the input text and metadata into the model's expected prompt format.

    Args:
        input_text (str): The law paragraph to translate.
        metadata (str): Metadata including user-defined constructs.

    Returns:
        str: Formatted prompt string.
    """

    chat = [
        {
            "role": "system",
            "content": "You are an AI assistant helping a user translate a law into code using the Catala programming language. You are provided with a law paragraph and metadata, including useful user-defined constructs. Your task is to generate the code in the Catala programming language.",
        },
        {
            "role": "user",
            "content": f"###INPUT###\n{input_text}\n###METADATA###\n{metadata}\n",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )

    return prompt


def generate_output(
    input_text: str,
    metadata: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[int, str]:
    """Generate model output based on input text and metadata.

    Args:
        input_text (str): The law paragraph to translate.
        metadata (str): Metadata including user-defined constructs.
        model (AutoModelForCausalLM): The language model.
        tokenizer (AutoTokenizer): The tokenizer for the model.

    Returns:
        Tuple[torch.Tensor, str]:
            output_ids: The generated output token IDs.
            output_text: The generated code in Catala.
    """

    with torch.no_grad():
        formatted_input = format_input(tokenizer, input_text, metadata)
        tokenized_input = tokenizer(formatted_input, return_tensors="pt").to("cuda")

        output_ids = model.generate(
            input_ids=tokenized_input["input_ids"],
            attention_mask=tokenized_input["attention_mask"],
            max_new_tokens=1024,
            do_sample=False,
            cache_implementation="offloaded",
        )[0]

        # If you want to see the decoded output with special tokens uncomment the line below
        # print(tokenizer.decode(output_ids, skip_special_tokens=False))

        output_ids_wo_input = output_ids[len(tokenized_input["input_ids"][0]) :]
        output_text = tokenizer.decode(output_ids_wo_input, skip_special_tokens=True)

        # Clear the cache and collect the IPC memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return output_ids, output_text


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    metrics,
    num_samples: int = None,
):
    """Evaluate the model on the test dataset using the specified metric.

    Args:
        model (AutoModelForCausalLM): The language model to evaluate.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        metric (evaluate.Metric): The evaluation metric to use.
        num_samples (int, optional): Number of samples to evaluate. Defaults to all samples.
    """
    with open(os.path.join(ROOT_PATH, "dataset/test.json"), "r") as f:
        dataset = json.load(f)

    if num_samples:
        if num_samples > len(dataset):
            print(
                f"Warning: num_samples ({num_samples}) is greater than dataset size ({len(dataset)}). Using full dataset."
            )
        else:
            dataset = random.sample(dataset, num_samples)

    # Determine which metrics to evaluate based on input
    if isinstance(metrics, list):
        # "All" metrics case - metrics is a list of all metrics
        metric_names = ["codebleu", "bertscore", "chrf", "ted", "is_valid"]
        metrics_dict = dict(zip(metric_names, metrics))
    else:
        # Single metric case - determine which one
        if hasattr(metrics, "name"):
            metric_name = metrics.name.lower()
        elif "tree_edit_distance" in str(metrics):
            metric_name = "ted"
        elif "is_valid_syntax_tree" in str(metrics):
            metric_name = "is_valid"
        else:
            # Try to infer from the metric object
            metric_name = "unknown"
        metrics_dict = {metric_name: metrics}

    # Initialize results dict only for metrics being evaluated
    tot_results = {key: 0 for key in metrics_dict.keys()}

    for sample in dataset:
        print(f"Sample {dataset.index(sample) + 1}:")
        _, output_text = generate_output(
            sample["input"], sample["metadata"], model, tokenizer
        )

        # print(f"Output: {sample['output']}")
        # print(f"Generated: {output_text}")

        for metric_name, metric in metrics_dict.items():
            if metric_name == "bertscore":
                result = metric.compute(
                    predictions=[output_text],
                    references=[sample["output"]],
                    lang="fr",
                    model_type="microsoft/deberta-xlarge-mnli",
                )
                tot_results[metric_name] += result["f1"][0]
            elif metric_name == "is_valid":
                result = metric.compute(predictions=[output_text])
                tot_results[metric_name] += result
            elif metric_name == "ted":
                result = metric.compute(
                    predictions=[output_text], references=[sample["output"]]
                )
                tot_results[metric_name] += result
            elif metric_name == "codebleu":
                result = metric.compute(
                    predictions=[output_text], references=[sample["output"]]
                )
                tot_results[metric_name] += result["codebleu"]
            elif metric_name == "chrf":
                result = metric.compute(
                    predictions=[output_text], references=[sample["output"]]
                )
                tot_results[metric_name] += result["score"] / 100
            else:
                # Generic handling for other metrics
                result = metric.compute(
                    predictions=[output_text], references=[sample["output"]]
                )
                tot_results[metric_name] += result

            print(f"{metric_name}: {result}")

        print()

    for metric in tot_results:
        tot_results[metric] /= len(dataset)

    print(f"Average results: {tot_results}")


def load_model_and_metric(
    model_name: str,
    metric_name: str = "All",
    unsloth_finetuned: bool = True,
    dtype: str = "bfloat16",
    load_in_4bit: bool = True,
):
    """Load the model, tokenizer, and evaluation metric based on provided names.

    Args:
        model_name (str): Name of the model folder in the models directory.
        metric_name (str): Name of the metric to use. Defaults to "All".
        unsloth_finetuned (bool): Whether the model is finetuned with Unsloth. Defaults to True.
        dtype (str): Data type for model loading. Defaults to "bfloat16".
        load_in_4bit (bool): Whether to load the model in 4-bit. Defaults to True.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer, evaluate.Metric]:
            model: The loaded language model.
            tokenizer: The tokenizer corresponding to the model.
            metric: The evaluation metric.
    """

    if unsloth_finetuned:
        # Find all checkpoints and sort to get the latest
        checkpoint_pattern = os.path.join(
            ROOT_PATH, f"models/{model_name}/checkpoint-*"
        )
        checkpoints = sorted(glob.glob(checkpoint_pattern))

        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found matching pattern: {checkpoint_pattern}"
            )

        # Use the last checkpoint (highest number)
        latest_checkpoint = checkpoints[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=latest_checkpoint,
            device_map="cuda",
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        model.forward = torch.compile(
            model.forward, mode="reduce-overhead", fullgraph=True
        )
        model.eval()
    else:
        # Convert dtype string to torch dtype
        dtype_mapping = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_mapping.get(dtype, torch.bfloat16)

        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(ROOT_PATH, f"models/{model_name}"),
            device_map="cuda",
            torch_dtype=torch_dtype,
            use_safetensors=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(ROOT_PATH, f"models/{model_name}"),
        )

    metric_mapping = {
        "BLEU": evaluate.load("bleu"),
        "chrf": evaluate.load("chrf", lowercase=True),
        "TED": tree_edit_distance,
        "All": [
            codebleu_metric,
            evaluate.load("bertscore"),
            evaluate.load("chrf", lowercase=True),
            tree_edit_distance,
            is_valid_syntax_tree,
        ],
    }

    metric = metric_mapping.get(metric_name)
    if metric is None:
        raise ValueError(f"Metric {metric_name} not recognized.")

    return model, tokenizer, metric


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned language model")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model folder in the models directory",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="All",
        choices=["BLEU", "chrf", "TED", "All"],
        help="Metric to use for evaluation (default: All)",
    )
    parser.add_argument(
        "--unsloth_finetuned",
        action="store_true",
        dest="unsloth_finetuned",
        help="Whether the model is finetuned with Unsloth (default: True)",
    )
    parser.add_argument(
        "--no_unsloth_finetuned",
        action="store_false",
        dest="unsloth_finetuned",
        help="Disable Unsloth finetuned loading",
    )
    parser.set_defaults(unsloth_finetuned=True)

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for model loading (auto, float16, bfloat16)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        dest="load_in_4bit",
        help="Load model in 4-bit precision (default: True)",
    )
    parser.add_argument(
        "--no_load_in_4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit precision loading",
    )
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (defaults to all)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    return parser.parse_args()


def main():
    """Main function to select model and metric, and evaluate the model."""
    args = parse_args()

    # Check GPU memory if CUDA is available
    try:
        free_memory, total_memory = cupy.cuda.runtime.memGetInfo()
        free_memory_gb = free_memory / 1e9
        total_memory_gb = total_memory / 1e9
        print(
            f"GPU memory: {free_memory_gb:.2f} GB free / {total_memory_gb:.2f} GB total"
        )
    except Exception as e:
        print(f"Warning: Could not check GPU memory: {e}")

    # Initialize wandb if available and not disabled
    if not args.no_wandb:
        try:
            import wandb

            wandb.init(project="PRIM-Code_Generation", name=f"{args.model_name}-Eval")
        except ImportError:
            print("Wandb is not installed. No logging will be performed.")

    model, tokenizer, metric = load_model_and_metric(
        args.model_name,
        args.metric,
        args.unsloth_finetuned,
        args.dtype,
        args.load_in_4bit,
    )
    evaluate_model(model, tokenizer, metric, args.num_samples)


if __name__ == "__main__":
    main()
