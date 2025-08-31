import argparse
import os


def is_wandb_available():
    try:
        import wandb

        os.environ["WANDB_PROJECT"] = "PRIM-Code_Generation"
        return True
    except ImportError:
        return False


import huggingface_hub
from datasets import load_dataset
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# Try to login to HuggingFace Hub if token is available
try:
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        huggingface_hub.login(hf_token)
except Exception as e:
    print(f"Warning: Could not login to HuggingFace Hub: {e}")

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_PATH = os.path.join(ROOT_PATH, "dataset")
OUTPUT_PATH = os.path.join(ROOT_PATH, "models")


def load_model_and_tokenizer(
    model_name: str, max_seq_len: int, dtype: str = "auto", load_in_4bit: bool = True
):
    """Load the model and tokenizer.

    Args:
        model_name (str): Name of the model to load.
        max_seq_len (int): Maximum sequence length.
        dtype (str): Data type for the model. Defaults to "auto".
        load_in_4bit (bool): Whether to load the model in 4-bit. Defaults to True.

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Loaded model and tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # device_map="cuda",
    )
    return model, tokenizer


def prepare_model(
    model: PreTrainedModel,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    rslora: bool = True,
) -> PeftModel:
    """Prepare the model with PEFT configurations.

    Args:
        model (PreTrainedModel): The loaded model.
        lora_r (int): LoRA rank. Defaults to 8.
        lora_alpha (int): LoRA alpha parameter. Defaults to 16.
        lora_dropout (float): LoRA dropout rate. Defaults to 0.0.
        rslora (bool): Whether to use RSLoRA. Defaults to True.

    Returns:
        PreTrainedModel: Model configured for PEFT.
    """
    return FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=rslora,
        loftq_config=None,
    )


def format_input(
    tokenizer: PreTrainedTokenizer, input_text: str, metadata: str, output_text: str
) -> str:
    """Format the input data into a prompt using the tokenizer's chat template.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        input_text (str): The law paragraph to translate.
        metadata (str): Metadata including user-defined constructs.
        output_text (str): The expected output text.

    Returns:
        str: Formatted prompt string.
    """
    chat = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant helping a user translate a law into code using "
                "the Catala programming language. You are provided with a law paragraph "
                "and metadata, including useful user-defined constructs. Your task is to "
                "generate the code in the Catala programming language."
            ),
        },
        {
            "role": "user",
            "content": f"###INPUT###\n{input_text}\n###METADATA###\n{metadata}\n",
        },
        {
            "role": "assistant",
            "content": f"{output_text}",
        },
    ]
    return tokenizer.apply_chat_template(
        chat, add_generation_prompt=False, tokenize=False
    )


def preprocess_dataset(dataset, tokenizer):
    """Preprocess the dataset by formatting inputs.

    Args:
        dataset (Dataset): The dataset to preprocess.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.

    Returns:
        Dataset: The preprocessed dataset.
    """
    return dataset.map(
        lambda x: {
            "text": format_input(tokenizer, x["input"], x["metadata"], x["output"]),
        }
    )


def train(model, tokenizer, dataset, max_seq_len: int, output_folder_name: str):
    """Train the model using the SFTTrainer.

    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        dataset (Dataset): The training dataset.
        max_seq_len (int): Maximum sequence length.
        output_folder_name (str): Name of the output folder for saving the model.
    """
    if is_wandb_available():
        logging_config = {
            "report_to": "wandb",
            "run_name": f"{output_folder_name}",
            "logging_steps": 1,
        }
    else:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(ROOT_PATH, "logs", output_folder_name)
        os.makedirs(logs_dir, exist_ok=True)
        logging_config = {
            "logging_dir": logs_dir,
            "logging_steps": 5,
        }

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=30,
            num_train_epochs=3,
            learning_rate=3e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=0,
            output_dir=os.path.join(OUTPUT_PATH, output_folder_name),
            **logging_config,
            torch_compile=True,
        ),
    )
    trainer.train()


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model using Unsloth and LoRA"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to fine-tune (e.g., unsloth/Llama-3.3-70B-Instruct)",
    )
    parser.add_argument(
        "--output_folder_name",
        type=str,
        default=None,
        help="Name of the output folder (defaults to model name without prefix)",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=4096, help="Maximum sequence length"
    )
    parser.add_argument(
        "--dtype", type=str, default="auto", help="Data type (auto, float16, bfloat16)"
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

    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.0, help="LoRA dropout rate"
    )
    parser.add_argument(
        "--rslora",
        action="store_true",
        dest="rslora",
        help="Use RSLoRA (default: True)",
    )
    parser.add_argument(
        "--no_rslora", action="store_false", dest="rslora", help="Disable RSLoRA"
    )
    parser.set_defaults(rslora=True)

    return parser.parse_args()


def main():
    """Main function to execute the fine-tuning process."""
    args = parse_args()

    # If output_folder_name not provided, derive from model_name
    if args.output_folder_name is None:
        # Extract model name after last slash
        args.output_folder_name = args.model_name.split("/")[-1]

    dataset_path = os.path.join(INPUT_PATH, "train.json")
    dataset = load_dataset("json", data_files=dataset_path)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.max_seq_len, args.dtype, args.load_in_4bit
    )
    model = prepare_model(
        model, args.lora_r, args.lora_alpha, args.lora_dropout, args.rslora
    )
    model.print_trainable_parameters()
    dataset = preprocess_dataset(dataset, tokenizer)

    train(model, tokenizer, dataset, args.max_seq_len, args.output_folder_name)


if __name__ == "__main__":
    main()
