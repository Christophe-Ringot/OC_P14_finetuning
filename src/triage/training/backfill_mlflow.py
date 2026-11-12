"""Reverse dans MLflow les experiences deja entrainees, puis enregistre les adaptateurs.

Sert quand la base MLflow est neuve (nouveau PC, volume `mlflow/` perdu) : les
entrainements de plusieurs heures ne sont pas relances, on journalise depuis
les fichiers produits (training_summary.json, scores de triage).

Pour chaque experience : hyperparametres, courbes de loss (par step), metriques
de triage validation + test (meme regle de decision que evaluate_sft), adaptateur
en artefact, et une version dans le Model Registry (`triage-qwen3-lora`).
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from triage.evaluation.evaluate_sft import LABELS, OUTPUT_DIR, decide, triage_metrics
from triage.training.train_sft import MODEL_NAME, SFT_EXPERIMENTS, adapter_dir
from triage.utils import hf_dataset, tracking

REGISTERED_MODEL = "triage-qwen3-lora"


def triage_results(experiment: str) -> dict:
    """Metriques validation/test recalculees depuis les scores sauvegardes."""
    labels = hf_dataset.load_sft()[["id", "priorite"]].set_index("id")["priorite"]
    mode = pd.read_csv(OUTPUT_DIR / "mode_decision.csv", index_col=0)
    results = {}
    for split, path in [("validation", "scores_validation.csv"), ("test", "scores.csv")]:
        scores = pd.read_csv(OUTPUT_DIR / path)
        scores = scores[scores["experience"] == experiment].reset_index(drop=True)
        if experiment == "1_baseline_naive":
            scores = pd.read_csv(OUTPUT_DIR / path).drop_duplicates("id").reset_index(drop=True)
            pred = pd.Series([LABELS[0]] * len(scores))
        else:
            pred = decide(scores, mode.loc[experiment, "mode_retenu"])
        results[split] = triage_metrics(labels.loc[scores["id"]].reset_index(drop=True), pred)
    return results


def log_triage(mlflow, results: dict) -> None:
    for split, metrics in results.items():
        mlflow.log_metrics({f"{split}_{k}": float(v) for k, v in metrics.items()})


def register(mlflow, run_id: str, experiment: str, alias: str | None = None) -> None:
    from mlflow import MlflowClient

    # Adaptateur PEFT brut (pas un "logged model" MLflow) : version creee depuis l'URI de l'artefact.
    client = MlflowClient()
    if not client.search_registered_models(f"name = '{REGISTERED_MODEL}'"):
        client.create_registered_model(REGISTERED_MODEL, description=f"Adaptateurs LoRA de triage sur {MODEL_NAME}")
    source = f"{client.get_run(run_id).info.artifact_uri}/adapter"
    version = client.create_model_version(REGISTERED_MODEL, source, run_id=run_id, tags={"experience": experiment})
    client.update_model_version(REGISTERED_MODEL, version.version,
                                description=f"Adaptateur LoRA {experiment} sur {MODEL_NAME}")
    if alias:
        client.set_registered_model_alias(REGISTERED_MODEL, alias, version.version)
    print(f"  {REGISTERED_MODEL} v{version.version} ({experiment}{', alias ' + alias if alias else ''})")


def delete_previous(mlflow, run_name: str) -> None:
    """Rejouer le script remplace les runs reverses precedemment au lieu de les dupliquer."""
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}' and tags.source LIKE 'backfill%'")
    for run_id in runs.get("run_id", []):
        mlflow.delete_run(run_id)


def backfill_sft(mlflow, experiment: str) -> None:
    output_dir = adapter_dir(experiment, quick=False)
    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    state = json.loads((output_dir / "trainer_state.json").read_text(encoding="utf-8"))
    args = summary["train_args"]
    lora = summary["lora_config"]
    with_priority = SFT_EXPERIMENTS[experiment]["with_priority"]
    tags = {"experience": experiment, "mode": "complet", "type": "entrainement", "phase": "sft",
            "source": "backfill (fichiers d'entrainement)", "git_commit": tracking.git_commit()}
    with tracking.start_run(True, run_name=experiment, tags=tags) as run:
        mlflow.log_params({
            "model_name": MODEL_NAME, "dataset": hf_dataset.dataset_name(with_priority),
            "dataset_revision": hf_dataset.DATASET_REVISION, "n_train": summary["n_train"],
            "n_validation": summary["n_validation"], "lora_r": lora["r"], "lora_alpha": lora["lora_alpha"],
            "lora_dropout": lora["lora_dropout"], "lora_target_modules": ",".join(sorted(lora["target_modules"])),
            **{k: args[k] for k in ["learning_rate", "num_train_epochs", "per_device_train_batch_size",
                                    "gradient_accumulation_steps", "max_length", "seed", "lr_scheduler_type"]},
        })
        for entry in summary["train_loss_history"]:
            mlflow.log_metrics({"train_loss": entry["loss"], "train_token_accuracy": entry["mean_token_accuracy"]}, step=entry["step"])
        for entry in summary["eval_loss_history"]:
            mlflow.log_metrics({"eval_loss": entry["eval_loss"], "eval_token_accuracy": entry["eval_mean_token_accuracy"]}, step=entry["step"])
        mlflow.log_metrics({"best_eval_loss": state["best_metric"],
                            "best_step": int(state["best_model_checkpoint"].replace("\\", "/").rsplit("-", 1)[1])})
        log_triage(mlflow, triage_results(experiment))
        mlflow.log_artifact(str(output_dir / "training_summary.json"))
        for name in ["adapter_config.json", "adapter_model.safetensors"]:
            mlflow.log_artifact(str(output_dir / name), artifact_path="adapter")
        run_id = run.info.run_id
    register(mlflow, run_id, experiment, alias="production" if experiment == "4_sft_medical_triage" else None)


def backfill_reference(mlflow, experiment: str) -> None:
    """Baseline naive et modele de base : pas d'entrainement, seulement les metriques."""
    with tracking.start_run(True, run_name=experiment, tags={"experience": experiment, "type": "reference",
                                                             "source": "backfill (scores sauvegardes)"}):
        mlflow.log_param("model_name", "aucun (toujours P1)" if experiment == "1_baseline_naive" else MODEL_NAME)
        log_triage(mlflow, triage_results(experiment))


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    tracking.setup(True)
    import mlflow

    for experiment in ["1_baseline_naive", "2_modele_base", *SFT_EXPERIMENTS]:
        delete_previous(mlflow, experiment)
    for experiment in ["1_baseline_naive", "2_modele_base"]:
        print(experiment)
        backfill_reference(mlflow, experiment)
    for experiment in SFT_EXPERIMENTS:
        print(experiment)
        backfill_sft(mlflow, experiment)


if __name__ == "__main__":
    main()
