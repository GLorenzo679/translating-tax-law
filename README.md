# Translating Tax Law to Code with LLMs: A Benchmark and Evaluation Framework

This repository contains the code and data for

Gabriele Lorenzo, Aldo Pietromatera and Nils Holzenberger<br>
[_Translating Tax Law to Code with LLMs: A Benchmark and Evaluation Framework_](https://aclanthology.org/2025.nllp-1.4.pdf)<br>
Natural Legal Language Processing Workshop, 2025

# Usage

## Prerequisites

### HuggingFace Token

To access gated models on HuggingFace (e.g., Llama models), you need to set up your HuggingFace token:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

You can obtain a token from [HuggingFace Settings](https://huggingface.co/settings/tokens).

### Weights & Biases (Optional)

If you want to use [Weights & Biases](https://wandb.ai/) for experiment tracking and logging during training and evaluation, you need to:

1. Create a free account at [wandb.ai](https://wandb.ai/)
2. Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. Login to wandb:

```bash
wandb login
# Or set the API key as an environment variable
export WANDB_API_KEY="your_wandb_api_key_here"
```

**Note:** If wandb is not installed or you don't want to use it, the scripts will automatically fall back to local logging without any issues. You can also explicitly disable wandb in the evaluation script using the `--no_wandb` flag.

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 1. Generate Dataset

This repository contains the dataset as presented in the paper in the `dataset/` folder.

For reproducibility, you can regenerate the dataset from raw Catala data using:

```bash
bash scripts/prepare_data.sh
```

This script will:

1. Clone the [Catala-examples repository](https://github.com/CatalaLang/catala-examples/)
2. Extract and process `.catala_fr` files
3. Generate the dataset with metadata
4. Split into train/test sets

The final dataset will be available in:

- `dataset/train.json` - Training set
- `dataset/test.json` - Test set
- `dataset/metadata.json` - Construct definitions

## 2. Fine-tune Model

This project uses [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning with LoRA (Low-Rank Adaptation).

### Basic Fine-tuning

```bash
python src/finetune/finetune.py \
  --model_name "unsloth/Llama-3.3-70B-Instruct"
```

### Fine-tuning with Custom Parameters

```bash
python src/finetune/finetune.py \
  --model_name "unsloth/Llama-3.3-70B-Instruct" \
  --output_folder_name "Llama-3.3-70B-Catala" \
  --max_seq_len 4096 \
  --lora_r 16 \
  --lora_alpha 32 \
  --rslora
```

### Available Arguments

| Argument               | Description                         | Default    |
| ---------------------- | ----------------------------------- | ---------- |
| `--model_name`         | Model to fine-tune (required)       | -          |
| `--output_folder_name` | Output folder name                  | Model name |
| `--max_seq_len`        | Maximum sequence length             | 4096       |
| `--dtype`              | Data type (auto, float16, bfloat16) | auto       |
| `--load_in_4bit`       | Load model in 4-bit precision       | True       |
| `--no_load_in_4bit`    | Disable 4-bit loading               | -          |
| `--lora_r`             | LoRA rank                           | 8          |
| `--lora_alpha`         | LoRA alpha parameter                | 16         |
| `--lora_dropout`       | LoRA dropout rate                   | 0.0        |
| `--rslora`             | Use RSLoRA                          | True       |
| `--no_rslora`          | Disable RSLoRA                      | -          |

The fine-tuned model will be saved in `models/<output_folder_name>/`.

## 3. Evaluate Model

Evaluate a fine-tuned model using multiple metrics (CodeBLEU, BERTScore, chrF, Tree Edit Distance, and syntax validity):

### Basic Evaluation

```bash
python src/finetune/evaluate_model.py \
  --model_name "Llama-3.3-70B-Instruct"
```

### Evaluation with Custom Settings

```bash
python src/finetune/evaluate_model.py \
  --model_name "Llama-3.3-70B-Instruct" \
  --metric "All" \
  --num_samples 50 \
  --no_wandb
```

### Available Arguments

| Argument                 | Description                          | Default     |
| ------------------------ | ------------------------------------ | ----------- |
| `--model_name`           | Model folder name (required)         | -           |
| `--metric`               | Metric to use (BLEU, chrf, TED, All) | All         |
| `--unsloth_finetuned`    | Model is Unsloth fine-tuned          | True        |
| `--no_unsloth_finetuned` | Use standard HuggingFace loading     | -           |
| `--dtype`                | Data type for loading                | bfloat16    |
| `--load_in_4bit`         | Load in 4-bit precision              | True        |
| `--no_load_in_4bit`      | Disable 4-bit loading                | -           |
| `--num_samples`          | Number of samples to evaluate        | All samples |
| `--no_wandb`             | Disable Weights & Biases logging     | -           |

### Metrics

The evaluation script computes the following metrics:

- **CodeBLEU**: Code-specific variant of BLEU
- **BERTScore**: Semantic similarity using BERT embeddings
- **chrF**: Character n-gram F-score
- **TED**: Tree Edit Distance for syntax tree comparison
- **is_valid**: Syntax validity check for generated Catala code
