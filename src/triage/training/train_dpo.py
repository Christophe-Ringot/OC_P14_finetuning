"""Alignement DPO de l'adaptateur SFT de triage (experience 5).

Part de l'adaptateur `outputs/sft_medical_triage` (meilleur checkpoint du SFT)
et continue de l'entrainer sur les paires de data/processed/dpo_triage.csv
(src/triage/data/build_dpo_triage.py). Le modele de reference est une copie
figee de ce meme adaptateur (TRL cree un adaptateur "ref") : la DPO mesure
l'ecart au modele SFT, pas au modele de base.

Suivi MLflow : hyperparametres, courbes (loss, marges et accuracy des
recompenses, par type de paire en evaluation), adaptateur, version dans le
Model Registry. Puis evaluation du triage (meme protocole que le SFT).
"""
import argparse
import gc
import json
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from triage.data.build_dpo_triage import OUTPUT_PATH as PAIRS_PATH
from triage.training.train_sft import MODEL_NAME, RANDOM_STATE
from triage.utils import tracking

# v1 : paires qualite + securite_triage ; v2 : paires qualite seules (le triage n'est pas une cible de la DPO).
DPO_RUNS = {
    "5_sft_triage_dpo": {"output": "dpo_triage", "types": ["qualite", "securite_triage"]},
    "6_sft_triage_dpo_qualite": {"output": "dpo_triage_qualite", "types": ["qualite"]},
}
SFT_ADAPTER = Path("outputs") / "sft_medical_triage"
OUTPUT_ROOT = Path("outputs")
QUICK_PAIRS = 64
MAX_LENGTH = 768  # couvre 91 % des paires ; au-dela, debut du cas tronque (keep_end), memoire GPU 6 Go
HPARAMS = {"beta": 0.1, "learning_rate": 2e-5, "num_train_epochs": 1, "gradient_accumulation_steps": 8}


def output_dir(experiment: str, quick: bool) -> Path:
    return OUTPUT_ROOT / (DPO_RUNS[experiment]["output"] + ("_quick" if quick else ""))


def load_pairs(experiment: str, quick: bool) -> tuple[Dataset, dict]:
    pairs = pd.read_csv(PAIRS_PATH)
    train = pairs[(pairs["split"] == "train") & pairs["type_paire"].isin(DPO_RUNS[experiment]["types"])]
    validation = pairs[pairs["split"] == "validation"]
    if quick:
        train = train.sample(QUICK_PAIRS, random_state=RANDOM_STATE)
        validation = validation.groupby("type_paire").sample(8, random_state=RANDOM_STATE)
    to_ds = lambda f: Dataset.from_pandas(f[["prompt", "chosen", "rejected"]], preserve_index=False)
    # Une evaluation par type de paire : qualite du contenu et securite du triage separees.
    evals = {t: to_ds(f) for t, f in validation.groupby("type_paire")}
    return to_ds(train), evals, {"n_train": len(train), **{f"n_validation_{t}": len(f) for t, f in validation.groupby("type_paire")},
                                 **{f"n_train_{t}": int(n) for t, n in train["type_paire"].value_counts().items()}}


def train(experiment: str, quick: bool, tracked: bool, hparams: dict) -> str | None:
    out = output_dir(experiment, quick)
    train_ds, eval_ds, counts = load_pairs(experiment, quick)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(SFT_ADAPTER), is_trainable=True)

    steps = 5 if quick else 30
    config = DPOConfig(
        output_dir=str(out),
        beta=hparams["beta"],
        learning_rate=hparams["learning_rate"],
        num_train_epochs=hparams["num_train_epochs"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
        lr_scheduler_type="cosine",
        warmup_steps=0.1,  # ratio (transformers 5)
        max_length=MAX_LENGTH,
        truncation_mode="keep_end",
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1 if quick else 5,
        eval_strategy="steps",
        eval_steps=steps,
        save_strategy="steps",
        save_steps=steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_qualite_loss",
        greater_is_better=False,
        seed=RANDOM_STATE,
        report_to="mlflow" if tracked else "none",
        run_name=experiment + ("_quick" if quick else ""),
    )
    tags = {"experience": experiment, "types_paires": ",".join(DPO_RUNS[experiment]["types"]), "mode": "quick" if quick else "complet", "phase": "dpo", "type": "entrainement",
            "git_commit": tracking.git_commit()}
    with tracking.start_run(tracked, run_name=config.run_name, tags=tags) as run:
        if run is not None:
            import mlflow
            from triage.utils.tracking import file_md5
            mlflow.log_params({"model_name": MODEL_NAME, "sft_adapter": str(SFT_ADAPTER), "pairs": str(PAIRS_PATH),
                               "pairs_md5": file_md5(PAIRS_PATH), "max_length": MAX_LENGTH, **counts})
        trainer = DPOTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer)
        # Pas de reprise sur checkpoint : avec un PeftModel passe a DPOTrainer, resume_from_checkpoint
        # restaure l'optimiseur mais pas les poids de l'adaptateur (constate le 2026-11-12).
        trainer.train()
        trainer.save_model(str(out))
        tokenizer.save_pretrained(str(out))
        history = trainer.state.log_history
        (out / "training_summary.json").write_text(json.dumps({
            "experiment": experiment, "model_name": MODEL_NAME, "sft_adapter": str(SFT_ADAPTER), **counts,
            "hparams": hparams, "best_checkpoint": trainer.state.best_model_checkpoint,
            "log_history": history,
        }, indent=2, default=str), encoding="utf-8")
        run_id = None
        if run is not None:
            mlflow.log_artifact(str(out / "training_summary.json"))
            for name in ["adapter_config.json", "adapter_model.safetensors"]:
                mlflow.log_artifact(str(out / name), artifact_path="adapter")
            run_id = run.info.run_id
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help=f"test rapide sur {QUICK_PAIRS} paires")
    parser.add_argument("--beta", type=float, default=HPARAMS["beta"])
    parser.add_argument("--lr", type=float, default=HPARAMS["learning_rate"])
    parser.add_argument("--epochs", type=float, default=HPARAMS["num_train_epochs"])
    parser.add_argument("--experiment", choices=list(DPO_RUNS), default="6_sft_triage_dpo_qualite")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    tracked = tracking.setup(not args.no_mlflow)
    hparams = {**HPARAMS, "beta": args.beta, "learning_rate": args.lr, "num_train_epochs": args.epochs}
    run_id = train(args.experiment, args.quick, tracked, hparams)

    if tracked and not args.quick:
        import mlflow
        from triage.training.backfill_mlflow import register
        register(mlflow, run_id, args.experiment)
    if not args.no_eval and not args.quick:
        from triage.evaluation import evaluate_sft
        evaluate_sft.run(tracked=tracked, run_ids={args.experiment: run_id}, only=[args.experiment])


if __name__ == "__main__":
    main()
