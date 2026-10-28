import json

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
DATA_PATH = "data/processed/sft_dataset_anonymized.csv"


MAX_TRAIN_EXAMPLES = None # pour le mode complet
# MAX_TRAIN_EXAMPLES = 900 # pour les tests sur echantillons
MAX_EVAL_EXAMPLES = 150

# Quotas cibles par source_dataset (somme = 1.0). mediqal_oeq/mediqal_mcq du
# brief fusionnes en un seul quota mediqal (0.30+0.14) : les donnees mediqal
# reelles sont 100% QCM, aucun exemple OEQ n'existe dans ce dataset.
SFT_QUOTAS = {"mediqal": 0.44, "frenchmedmcqa": 0.06, "medquad": 0.50}

OUTPUT_DIR = "outputs/sft_lora_run5_quick" if MAX_TRAIN_EXAMPLES else "outputs/sft_lora_run5"


def format_example(row: dict) -> str:
    return f"### Instruction:\n{row['instruction']}\n\n### Reponse:\n{row['response']}"


def sample_by_quota(df: pd.DataFrame, n: int, quotas: dict, seed: int) -> pd.DataFrame:
    groups = []
    for source, quota in quotas.items():
        target = round(n * quota)
        pool = df[df["source_dataset"] == source]
        groups.append(pool.sample(n=target, replace=target > len(pool), random_state=seed))
    return pd.concat(groups)


def main() -> None:
    has_gpu = torch.cuda.is_available()

    df = pd.read_csv(DATA_PATH)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    n_train = min(MAX_TRAIN_EXAMPLES, len(train_df)) if MAX_TRAIN_EXAMPLES else len(train_df)
    train_df = sample_by_quota(train_df, n_train, SFT_QUOTAS, seed=42)
    if MAX_EVAL_EXAMPLES:
        val_df = val_df.sample(n=min(MAX_EVAL_EXAMPLES, len(val_df)), random_state=42)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3 if has_gpu else 1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8 if has_gpu else 4,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100 if has_gpu else 20,
        save_strategy="steps",
        save_steps=100 if has_gpu else 20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=has_gpu,
        use_cpu=not has_gpu,
        max_length=1024 if has_gpu else 256,
        report_to="none",
        model_init_kwargs={"dtype": "bfloat16" if has_gpu else "float32"},
    )

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        formatting_func=format_example,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    trainer.processing_class.save_pretrained(OUTPUT_DIR)
    trainer.save_state()

    log_history = trainer.state.log_history
    train_losses = [e for e in log_history if "loss" in e]
    eval_losses = [e for e in log_history if "eval_loss" in e]
    with open(f"{OUTPUT_DIR}/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "data_path": DATA_PATH,
                "max_train_examples": MAX_TRAIN_EXAMPLES,
                "max_eval_examples": MAX_EVAL_EXAMPLES,
                "lora_config": lora_config.to_dict(),
                "train_args": sft_config.to_dict(),
                "train_loss_history": train_losses,
                "eval_loss_history": eval_losses,
            },
            f,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
