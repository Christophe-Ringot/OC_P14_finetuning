"""Reconstruit une experience MLflow lisible a partir des fichiers produits, sans reentrainer.

Un run par modele (1 a 6) avec une vingtaine de metriques au plus :
- parametres cles (modele, LoRA ou DPO, donnees) ;
- courbes d'entrainement par pas (loss train / validation, marges DPO) ;
- triage sur la validation et le test : accuracy, F1 macro, rappel P1, sous et sur-triage ;
- modeles entraines : triage du test avec la regle de cout deployee ; 4 et 6 : QCM et jugements en aveugle ;
puis un run `synthese` qui porte les graphiques de outputs/figures/ et les tableaux comparatifs.

Le detail (matrices, par classe, balayages) reste dans les rapports en artefacts. Les adaptateurs ne sont pas
re-televerses : chaque run renvoie par un tag a sa version du Model Registry `triage-qwen3-lora`.

    python -m triage.training.backfill_mlflow             # remplace les runs reconstruits de l'experience
    python -m triage.training.backfill_mlflow --archive   # renomme d'abord l'experience actuelle (rien n'est supprime)
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from triage.constants import BASE_MODEL, BASE_MODEL_EXPERIMENT, BASELINE, FINAL_EXPERIMENT, OUTPUTS_DIR, adapter_dir
from triage.utils import hf_dataset, tracking

REGISTERED_MODEL = "triage-qwen3-lora"
SOURCE = "reconstruction"
TRIAGE_EVAL_DIR = OUTPUTS_DIR / "triage_eval"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DESCRIPTIONS = {
    BASELINE: "Reference : predit toujours P1 (rappel P1 de 100 %, aucun apprentissage).",
    BASE_MODEL_EXPERIMENT: "Qwen3-1.7B-Base sans fine-tuning : apport du fine-tuning.",
    "3_sft_medical": "SFT LoRA sur le corpus medical, reponses sans niveau : apport du domaine seul.",
    "4_sft_medical_triage": "Modele livre : SFT LoRA avec le niveau en tete des reponses, checkpoint 1300, "
    "regle de decision au cout (alpha 0,5).",
    "5_sft_triage_dpo": "DPO v1 (paires qualite + securite) : effondrement vers P1, abandonnee. "
    "Points apres le pas 75 = reprise defectueuse ; adaptateur evalue = pas 75.",
    "6_sft_triage_dpo_qualite": "DPO v2 (paires qualite seules) : triage conserve, explications non ameliorees.",
}
MCQA = {"4_sft_medical_triage": "sft_medical_triage", "6_sft_triage_dpo_qualite": "dpo_triage_qualite"}
JUDGMENTS = {
    "jugement_reponses_prefere": (OUTPUTS_DIR / "response_eval", "jugement_aveugle"),
    "jugement_explications_prefere": (OUTPUTS_DIR / "explanation_eval", "jugement_explications"),
}


def register(mlflow, run_id: str, experiment: str, alias: str | None = None) -> None:
    """Nouvelle version du Model Registry pour l'adaptateur journalise dans `run_id` (utilise par train_dpo)."""
    from mlflow import MlflowClient

    # Adaptateur PEFT brut (pas un "logged model" MLflow) : version creee depuis l'URI de l'artefact.
    client = MlflowClient()
    if not client.search_registered_models(f"name = '{REGISTERED_MODEL}'"):
        client.create_registered_model(REGISTERED_MODEL, description=f"Adaptateurs LoRA de triage sur {BASE_MODEL}")
    source = f"{client.get_run(run_id).info.artifact_uri}/adapter"
    version = client.create_model_version(REGISTERED_MODEL, source, run_id=run_id, tags={"experience": experiment})
    client.update_model_version(
        REGISTERED_MODEL, version.version, description=f"Adaptateur LoRA {experiment} sur {BASE_MODEL}"
    )
    if alias:
        client.set_registered_model_alias(REGISTERED_MODEL, alias, version.version)
    print(f"  {REGISTERED_MODEL} v{version.version} ({experiment}{', alias ' + alias if alias else ''})")


# ---------- donnees a journaliser ----------


def triage_metrics() -> dict:
    """Metriques essentielles validation + test (decision argmax), par experience."""
    frames = {
        split: pd.read_csv(
            TRIAGE_EVAL_DIR / f"comparaison_triage{'' if split == 'test' else '_validation'}.csv", index_col=0
        )
        for split in ["validation", "test"]
    }
    return {
        experiment: {
            k: v
            for split, frame in frames.items()
            for k, v in tracking.triage_summary(frame.loc[experiment], split).items()
        }
        for experiment in frames["test"].index
    }


def cost_metrics() -> dict:
    """Test avec la regle de cout deployee, pour chaque modele entraine."""
    from triage.api.config import load_decision_rule
    from triage.evaluation.compare_models import cost_rule_results

    results, _ = cost_rule_results(load_decision_rule()["alpha"])
    names = ["accuracy", "rappel_P1", "sous_triage", "sur_triage"]
    return {
        name.replace(" + cout", ""): {f"test_cout_{k}": float(row[k]) for k in names}
        for name, row in results["test"].iterrows()
    }


def judgment_wins() -> dict:
    """Nombre de cas ou chaque modele est prefere, par jugement en aveugle."""
    wins = {}
    for metric, (folder, prefix) in JUDGMENTS.items():
        notes, key = folder / f"{prefix}_notes.jsonl", folder / f"{prefix}_cle.json"
        if not notes.exists():
            continue
        key = json.loads(key.read_text(encoding="utf-8"))
        for line in notes.read_text(encoding="utf-8").splitlines():
            note = json.loads(line)
            if note["preference"] in ("A", "B"):
                experiment = key[str(note["item"])][note["preference"]]
                wins.setdefault(experiment, {}).setdefault(metric, 0)
                wins[experiment][metric] += 1
    return wins


def curves(experiment: str) -> list[tuple[int, dict]]:
    """(pas, metriques) des courbes d'entrainement, avec les memes noms que pendant l'entrainement."""
    from triage.training.callbacks import DPO_CURVES, SFT_CURVES, select_metrics

    summary = json.loads((adapter_dir(experiment) / "training_summary.json").read_text(encoding="utf-8"))
    if "log_history" in summary:  # DPO
        history = summary["log_history"]
        if experiment == "5_sft_triage_dpo":
            history = [entry for entry in history if entry.get("step", 0) <= 75]  # avant la reprise defectueuse
        return [(e["step"], select_metrics(e, DPO_CURVES)) for e in history if "step" in e]
    entries = summary["train_loss_history"] + summary["eval_loss_history"]
    return [(e["step"], select_metrics(e, SFT_CURVES)) for e in entries]


def params(experiment: str) -> dict:
    common = {"modele_base": BASE_MODEL, "dataset": hf_dataset.dataset_name()}
    if experiment == BASELINE:
        return {"modele_base": "aucun (toujours P1)"}
    if experiment == BASE_MODEL_EXPERIMENT:
        return common
    summary = json.loads((adapter_dir(experiment) / "training_summary.json").read_text(encoding="utf-8"))
    if "lora_config" in summary:
        lora, args = summary["lora_config"], summary["train_args"]
        return common | {
            "phase": "sft",
            "n_train": summary["n_train"],
            "lora_r": lora["r"],
            "lora_alpha": lora["lora_alpha"],
            "lora_modules": ",".join(sorted(lora["target_modules"])),
            "learning_rate": args["learning_rate"],
            "epochs": args["num_train_epochs"],
            "batch_effectif": args["per_device_train_batch_size"] * args["gradient_accumulation_steps"],
            "max_length": args["max_length"],
        }
    return common | {
        "phase": "dpo",
        "adaptateur_depart": "sft_medical_triage",
        "n_paires_train": summary["n_train"],
        "types_paires": "qualite + securite" if summary.get("n_train_securite_triage") else "qualite",
        **{k: summary["hparams"][k] for k in ["beta", "learning_rate", "num_train_epochs"]},
        "checkpoint_retenu": "pas 75" if experiment == "5_sft_triage_dpo" else Path(summary["best_checkpoint"]).name,
    }


def registry_versions(client) -> dict:
    """Derniere version du Model Registry par experience."""
    versions = {}
    for version in client.search_model_versions(f"name = '{REGISTERED_MODEL}'"):
        experiment = version.tags.get("experience")
        if experiment and int(version.version) > int(versions.get(experiment, 0)):
            versions[experiment] = version.version
    return versions


# ---------- runs ----------


def log_model_run(mlflow, experiment: str, data: dict, versions: dict) -> None:
    tags = {
        "experience": experiment,
        "type": "reference" if experiment in (BASELINE, BASE_MODEL_EXPERIMENT) else "modele",
        "source": SOURCE,
        "git_commit": tracking.git_commit(),
        "mlflow.note.content": DESCRIPTIONS[experiment],
    }
    if experiment in versions:
        tags["registry"] = f"{REGISTERED_MODEL} v{versions[experiment]}"
    if experiment == FINAL_EXPERIMENT:
        tags["modele_livre"] = "oui"
    with mlflow.start_run(run_name=experiment, tags=tags):
        mlflow.log_params(params(experiment))
        if experiment in DESCRIPTIONS and adapter_dir_exists(experiment):
            for step, values in curves(experiment):
                if values:
                    mlflow.log_metrics(values, step=step)
            summary = adapter_dir(experiment) / "training_summary.json"
            mlflow.log_artifact(str(summary))
            figure = "05_courbes_sft.png" if experiment.startswith(("3_", "4_")) else "06_courbes_dpo.png"
            mlflow.log_artifact(str(FIGURES_DIR / figure), artifact_path="figures")
        metrics = {**data["triage"].get(experiment, {}), **data["cost"].get(experiment, {})}
        metrics.update({k: float(v) for k, v in data["wins"].get(experiment, {}).items()})
        if experiment in MCQA:
            mcqa = pd.read_csv(OUTPUTS_DIR / "final_eval" / f"mcqa_{MCQA[experiment]}.csv")
            metrics["qcm_exact_match"] = float(mcqa["exact_match"].mean())
        mlflow.log_metrics(metrics)
        print(f"  {experiment} : {len(metrics)} metriques de resultat")


def adapter_dir_exists(experiment: str) -> bool:
    return (
        experiment not in (BASELINE, BASE_MODEL_EXPERIMENT)
        and (adapter_dir(experiment) / "training_summary.json").exists()
    )


def log_synthesis_run(mlflow) -> None:
    tags = {
        "type": "synthese",
        "source": SOURCE,
        "git_commit": tracking.git_commit(),
        "mlflow.note.content": "Graphiques comparatifs et tableaux des 6 experiences (docs/resultats_sft_dpo.md).",
    }
    with mlflow.start_run(run_name="synthese", tags=tags):
        mlflow.log_artifacts(str(FIGURES_DIR), artifact_path="figures")
        table = pd.read_csv(TRIAGE_EVAL_DIR / "comparaison_modeles.csv")
        mlflow.log_table(table, artifact_file="comparaison_modeles.json")
        for path in [
            TRIAGE_EVAL_DIR / "comparaison_modeles.md",
            TRIAGE_EVAL_DIR / "cout_sensible.md",
            TRIAGE_EVAL_DIR / "analyse_erreurs.md",
            OUTPUTS_DIR / "explanation_eval" / "exemples_explications.md",
            Path("docs") / "resultats_sft_dpo.md",
        ]:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="rapports")
    print("  synthese : graphiques et rapports")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", action="store_true", help="renommer l'experience actuelle avant de reconstruire")
    args = parser.parse_args()
    tracking.setup(True)
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    current = client.get_experiment_by_name(name)
    if args.archive and current is not None:
        archived = f"{name} (archive {date.today().isoformat()})"
        client.rename_experiment(current.experiment_id, archived)
        print(f"Experience existante renommee : '{archived}' (runs conserves)")
        mlflow.set_experiment(name)
        mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    else:
        # Rejouer le script remplace les runs reconstruits au lieu de les dupliquer.
        previous = mlflow.search_runs(filter_string=f"tags.source = '{SOURCE}'")
        for run_id in previous.get("run_id", []):
            mlflow.delete_run(run_id)
    mlflow.set_experiment_tag(
        "mlflow.note.content",
        "POC de triage medical : un run par modele (metriques essentielles et courbes), "
        "un run `synthese` avec les graphiques. Modele livre : 4_sft_medical_triage.",
    )

    data = {"triage": triage_metrics(), "cost": cost_metrics(), "wins": judgment_wins()}
    versions = registry_versions(client)
    for experiment in DESCRIPTIONS:
        log_model_run(mlflow, experiment, data, versions)
    log_synthesis_run(mlflow)


if __name__ == "__main__":
    main()
