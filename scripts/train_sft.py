import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
OUTPUT_ROOT = Path("outputs")
SFT_EXPERIMENTS = {
    "3_sft_medical": {"data": Path("data/processed/sft_sans_priorite.csv"), "output": "sft_medical"},
    "4_sft_medical_triage": {"data": Path("data/processed/sft_avec_priorite.csv"), "output": "sft_medical_triage"},
}

RANDOM_STATE = 42
MAX_LENGTH = 1024
MAX_EVAL_EXAMPLES = 150
QUICK_TRAIN_EXAMPLES = 200


def adapter_dir(experiment: str, quick: bool) -> Path:
    return OUTPUT_ROOT / (SFT_EXPERIMENTS[experiment]["output"] + ("_quick" if quick else ""))


def select_examples(tokenizer, quick: bool) -> tuple[list, list]:
    """Choisit les id d'entrainement et de validation, communs aux deux experiences.
    """
    frame = pd.read_csv(SFT_EXPERIMENTS["4_sft_medical_triage"]["data"])
    frame = frame[frame["split"].isin(["train", "validation"])]
    lengths = np.array([len(tokenizer(p + r + tokenizer.eos_token).input_ids) for p, r in zip(frame["prompt"], frame["reponse"])])
    print(f"Exemples ecartes car > {MAX_LENGTH} tokens : {(lengths > MAX_LENGTH).sum()} / {len(frame)}")
    frame = frame[lengths <= MAX_LENGTH]

    train = frame[frame["split"] == "train"]
    if quick:
        train = train.sample(QUICK_TRAIN_EXAMPLES, random_state=RANDOM_STATE)
    validation = frame[frame["split"] == "validation"]
    validation = validation.sample(min(MAX_EVAL_EXAMPLES, len(validation)), random_state=RANDOM_STATE)
    return train["id"].tolist(), validation["id"].tolist()


def load_examples(path: Path, ids: list) -> Dataset:
    frame = pd.read_csv(path).set_index("id").loc[ids]
    return Dataset.from_dict({"prompt": frame["prompt"].tolist(), "completion": frame["reponse"].tolist()})


def train(experiment: str, train_ids: list, val_ids: list, quick: bool) -> None:
    has_gpu = torch.cuda.is_available()
    output_dir = adapter_dir(experiment, quick)
    data_path = SFT_EXPERIMENTS[experiment]["data"]
    print(f"\n=== {experiment} : {len(train_ids)} exemples d'entrainement -> {output_dir} ===")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    steps = 10 if quick else (100 if has_gpu else 20)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1 if quick or not has_gpu else 3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8 if has_gpu else 4,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=steps,
        save_strategy="steps",
        save_steps=steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=has_gpu,
        use_cpu=not has_gpu,
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        seed=RANDOM_STATE,
        report_to="none",
        model_init_kwargs={"dtype": "bfloat16" if has_gpu else "float32"},
    )

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=sft_config,
        train_dataset=load_examples(data_path, train_ids),
        eval_dataset=load_examples(data_path, val_ids),
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    trainer.processing_class.save_pretrained(str(output_dir))
    trainer.save_state()

    log_history = trainer.state.log_history
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": experiment,
                "model_name": MODEL_NAME,
                "data_path": str(data_path),
                "n_train": len(train_ids),
                "n_validation": len(val_ids),
                "lora_config": lora_config.to_dict(),
                "train_args": sft_config.to_dict(),
                "train_loss_history": [e for e in log_history if "loss" in e],
                "eval_loss_history": [e for e in log_history if "eval_loss" in e],
            },
            f,
            indent=2,
            default=str,
        )

    # Libere le GPU avant l'entrainement ou l'evaluation suivante.
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="test rapide sur 200 exemples")
    parser.add_argument("--only", choices=list(SFT_EXPERIMENTS), help="n'entrainer qu'une experience")
    parser.add_argument("--no-eval", action="store_true", help="ne pas lancer la comparaison a la fin")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ids, val_ids = select_examples(tokenizer, args.quick)
    for experiment in [args.only] if args.only else SFT_EXPERIMENTS:
        train(experiment, train_ids, val_ids, args.quick)

    if not args.no_eval:
        from scripts import evaluate_sft

        evaluate_sft.run(quick=args.quick)


if __name__ == "__main__":
    main()
