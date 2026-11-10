import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer
from triage.utils import hf_dataset, tracking
from triage.training.train_sft import MODEL_NAME, SFT_EXPERIMENTS, adapter_dir

OUTPUT_DIR = Path("outputs") / "triage_eval"
QUICK_TEST_EXAMPLES = 30

# Ordre d'urgence : P1 (le plus urgent) -> P3.
LABELS = ["urgence_maximale", "urgence_moderee", "differee"]
SHORT = {"urgence_maximale": "P1", "urgence_moderee": "P2", "differee": "P3"}
PRIORITY_PREFIX = "[Niveau de priorité estimé :"
# Meme gabarit que les prompts du test, sans cas ni question.
NEUTRAL_PROMPT = "### Cas clinique :\n\n\n### Question :\n\n\n### Réponse :\n"
MODES = ["calibre", "brut"]


def load_model(adapter: Path | None):
    has_gpu = torch.cuda.is_available()
    # Tout sur le GPU : un manque de memoire doit lever une erreur, pas
    # decharger silencieusement des couches sur le CPU (evaluation tres lente).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16 if has_gpu else torch.float32, device_map={"": 0} if has_gpu else None
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval()


def label_logprobs(model, tokenizer, prompt: str) -> dict:
    """Log-probabilite de " <niveau>]" apres le prompt et le debut du prefixe."""
    context = tokenizer(prompt + PRIORITY_PREFIX, add_special_tokens=False)["input_ids"]
    scores = {}
    for label in LABELS:
        continuation = tokenizer(f" {label}]", add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([context + continuation], device=model.device)
        with torch.no_grad():
            # Seules les positions qui predisent la continuation sont utiles.
            logits = model(input_ids=input_ids, logits_to_keep=len(continuation) + 1).logits[0, :-1].float()
        targets = torch.tensor(continuation, device=model.device).unsqueeze(1)
        scores[label] = logits.log_softmax(-1).gather(1, targets).sum().item()
    return scores


def score_with_model(splits: dict, adapter: Path | None) -> dict:
    """Log-probabilites brutes de chaque niveau et prior sur prompt vide, par split.

    Le prior sert a la calibration contextuelle (Zhao et al., 2021) : un modele
    qui n'a jamais vu les libelles choisit d'apres leur forme et non d'apres le
    cas, on retranche donc le score obtenu sans cas clinique. Pour un modele
    entraine sur les libelles, ce prior n'est plus neutre (un cas vide evoque
    "differee") : le choix calibre / brut est donc fait sur la validation.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(adapter)
    prior = label_logprobs(model, tokenizer, NEUTRAL_PROMPT)
    results = {}
    for name, frame in splits.items():
        raw = []
        for i, prompt in enumerate(frame["prompt"], start=1):
            raw.append(label_logprobs(model, tokenizer, prompt))
            if i % 50 == 0:
                print(f"  {name} {i}/{len(frame)}")
        scores = pd.DataFrame(raw).rename(columns=lambda label: f"raw_{SHORT[label]}")
        for label in LABELS:
            scores[f"prior_{SHORT[label]}"] = prior[label]
        results[name] = scores.assign(id=frame["id"].to_numpy())
    del model
    free_gpu_memory()
    return results


def decide(scores: pd.DataFrame, mode: str) -> pd.Series:
    """Niveau retenu : argmax des scores bruts ou calibres (brut - prior)."""
    values = pd.DataFrame({
        label: scores[f"raw_{SHORT[label]}"] - (scores[f"prior_{SHORT[label]}"] if mode == "calibre" else 0)
        for label in LABELS
    })
    return values.idxmax(axis=1)


def choose_mode(validation: pd.DataFrame, scores: pd.DataFrame) -> tuple[str, dict]:
    """Mode de decision qui maximise le F1 macro sur la validation (le test n'est pas regarde)."""
    f1 = {mode: triage_metrics(validation["priorite"], decide(scores, mode))["f1_macro"] for mode in MODES}
    return max(f1, key=f1.get), f1


def free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def triage_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    rank = {label: i for i, label in enumerate(LABELS)}
    gap = y_pred.map(rank) - y_true.map(rank)  # > 0 : predit moins urgent
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    metrics = {
        "accuracy": (gap == 0).mean(),
        "f1_macro": f1.mean(),
        "rappel_P1": recall[0],
        "sous_triage": (gap > 0).mean(),
        "sur_triage": (gap < 0).mean(),
        "erreurs_1_niveau": (gap.abs() == 1).mean(),
        "erreurs_2_niveaux": (gap.abs() == 2).mean(),
        "P1_predit_P3": ((y_true == LABELS[0]) & (y_pred == LABELS[2])).sum() / max((y_true == LABELS[0]).sum(), 1),
    }
    for i, label in enumerate(LABELS):
        metrics.update({f"precision_{SHORT[label]}": precision[i], f"rappel_{SHORT[label]}": recall[i], f"f1_{SHORT[label]}": f1[i]})
    return metrics


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| experience | " + " | ".join(frame.columns) + " |\n|" + "---|" * (len(frame.columns) + 1) + "\n"
    return header + "\n".join(f"| {name} | " + " | ".join(f"{v:.1f}" for v in row) + " |" for name, row in frame.iterrows())


def confusion_table(y_true: pd.Series, y_pred: pd.Series) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    header = "| annote \\ predit | " + " | ".join(SHORT[label] for label in LABELS) + " |\n|---|---|---|---|\n"
    rows = [f"| {SHORT[label]} | " + " | ".join(str(n) for n in matrix[i]) + " |" for i, label in enumerate(LABELS)]
    return header + "\n".join(rows)


def write_report(test: pd.DataFrame, results: pd.DataFrame, decisions: pd.DataFrame, path: Path,
                 validation: pd.DataFrame | None = None) -> None:
    percent = results.astype(float) * 100
    main_columns = ["accuracy", "f1_macro", "rappel_P1", "sous_triage", "sur_triage", "erreurs_1_niveau", "erreurs_2_niveaux", "P1_predit_P3"]
    class_columns = [c for c in results.columns if c not in main_columns]
    distribution = test["priorite"].map(SHORT).value_counts().reindex(["P1", "P2", "P3"])

    lines = [
        "# Comparaison du triage - 4 experiences",
        "",
        f"Jeu de test : {len(test)} questions sur {test['groupe'].nunique()} groupes de cas cliniques "
        f"(P1 = {distribution['P1']}, P2 = {distribution['P2']}, P3 = {distribution['P3']}).",
        "Valeurs en %. Sous-triage / sur-triage : niveau predit moins / plus urgent que l'annotation, "
        "rapporte a tous les exemples. P1_predit_P3 : part des P1 classes P3.",
        "",
        "## Regle de decision par modele",
        "",
        "Pour chaque modele, le niveau predit est l'argmax des log-probabilites, soit brutes, soit calibrees "
        "(moins le score sur prompt vide). Le mode est choisi sur la validation (F1 macro), puis applique au test.",
        "",
        "| experience | mode retenu | " + " | ".join(f"F1 validation {m}" for m in MODES) + " |\n|---|---|"
        + "---|" * len(MODES) + "\n"
        + "\n".join(f"| {name} | {row['mode_retenu']} | " + " | ".join(f"{row[f'f1_validation_{m}'] * 100:.1f}" for m in MODES) + " |"
                    for name, row in decisions.iterrows()),
        "",
        "## Metriques principales",
        "",
        markdown_table(percent[main_columns]),
        "",
        *(["Meme tableau sur la validation (regle de decision choisie sur ce jeu), a citer avec le test :", "",
           markdown_table(validation.astype(float)[main_columns] * 100), ""] if validation is not None else []),
        "## Precision, rappel et F1 par classe",
        "",
        markdown_table(percent[class_columns]),
        "",
        "## Matrices de confusion",
    ]
    for experiment in results.index:
        lines += ["", f"### {experiment}", "", confusion_table(test["priorite"], test[experiment])]
    lines += [
        "",
        "## Limites",
        "",
        "- Etiquettes de reference pre-annotees par LLM, sans validation clinique (docs/triage_grid.md).",
        "- Les questions d'un meme cas partagent le meme niveau : les exemples ne sont pas independants.",
        "- La baseline naive vaut la part de P1 dans ce jeu de test, pas 33 % (classes non equilibrees).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_split(split: str, quick: bool) -> pd.DataFrame:
    frame = hf_dataset.load_sft(with_priority=True)
    frame = frame[(frame["split"] == split) & frame["priorite"].notna()].reset_index(drop=True)
    if quick:
        frame = frame.sample(min(QUICK_TEST_EXAMPLES, len(frame)), random_state=42).reset_index(drop=True)
    return frame


def log_to_mlflow(quick: bool, results: dict, decisions: pd.DataFrame, run_ids: dict, files: list) -> None:
    """Metriques dans le run d'entrainement de chaque modele, et un run d'evaluation qui regroupe tout."""
    import mlflow

    suffix = "_quick" if quick else ""
    flat = lambda split, metrics: {f"{split}_{name}": float(value) for name, value in metrics.items()}
    for experiment, run_id in (run_ids or {}).items():
        if run_id is None or experiment not in results["test"].index:
            continue
        with tracking.start_run(True, run_id=run_id):
            for split, frame in results.items():
                mlflow.log_metrics(flat(split, frame.loc[experiment]))
            mlflow.log_param("mode_decision", decisions.loc[experiment, "mode_retenu"])

    tags = {"type": "evaluation", "mode": "quick" if quick else "complet", "git_commit": tracking.git_commit()}
    with tracking.start_run(True, run_name=f"evaluation_triage{suffix}", tags=tags):
        mlflow.log_params({
            "dataset": hf_dataset.dataset_name(), "dataset_revision": hf_dataset.DATASET_REVISION,
            "dataset_md5": hf_dataset.fingerprint(hf_dataset.load_sft()),
            **{f"mode_{e}": m for e, m in decisions["mode_retenu"].items()},
            **{f"run_id_{e}": r for e, r in (run_ids or {}).items() if r},
        })
        for split, frame in results.items():
            for experiment, metrics in frame.iterrows():
                mlflow.log_metrics({f"{experiment}.{name}": float(v) for name, v in flat(split, metrics).items()})
        for path in files:
            mlflow.log_artifact(str(path))


def run(quick: bool = False, tracked: bool = False, run_ids: dict | None = None) -> None:
    splits = {"validation": load_split("validation", quick), "test": load_split("test", quick)}
    test = splits["test"]
    suffix = "_quick" if quick else ""

    test["1_baseline_naive"] = LABELS[0]
    splits["validation"]["1_baseline_naive"] = LABELS[0]
    models = {"2_modele_base": None}
    for experiment in SFT_EXPERIMENTS:
        adapter = adapter_dir(experiment, quick)
        if not (adapter / "adapter_config.json").exists():
            print(f"{experiment} ignoree : adaptateur absent ({adapter})")
            continue
        models[experiment] = adapter

    all_scores = {name: [] for name in splits}
    decisions = {}
    for experiment, adapter in models.items():
        print(f"Evaluation de {experiment} ({len(splits['validation'])} validation, {len(test)} test)")
        scores = score_with_model(splits, adapter)
        mode, f1 = choose_mode(splits["validation"], scores["validation"])
        decisions[experiment] = {"mode_retenu": mode, **{f"f1_validation_{m}": v for m, v in f1.items()}}
        print(f"  mode retenu sur la validation : {mode} " + ", ".join(f"{m}={v:.3f}" for m, v in f1.items()))
        for name in splits:
            splits[name][experiment] = decide(scores[name], mode).to_numpy()
            all_scores[name].append(scores[name].assign(experience=experiment))

    experiments = [c for c in test.columns if c[0].isdigit()]
    results = {name: pd.DataFrame({e: triage_metrics(frame["priorite"], frame[e]) for e in experiments}).T
               for name, frame in splits.items()}
    decisions = pd.DataFrame(decisions).T

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "predictions": OUTPUT_DIR / f"predictions{suffix}.csv",
        "scores": OUTPUT_DIR / f"scores{suffix}.csv",
        "scores_validation": OUTPUT_DIR / f"scores_validation{suffix}.csv",
        "decisions": OUTPUT_DIR / f"mode_decision{suffix}.csv",
        "results": OUTPUT_DIR / f"comparaison_triage{suffix}.csv",
        "results_validation": OUTPUT_DIR / f"comparaison_triage_validation{suffix}.csv",
        "report": OUTPUT_DIR / f"comparaison_triage{suffix}.md",
    }
    test[["id", "groupe", "priorite", *experiments]].to_csv(files["predictions"], index=False)
    pd.concat(all_scores["test"]).to_csv(files["scores"], index=False)
    pd.concat(all_scores["validation"]).to_csv(files["scores_validation"], index=False)
    decisions.to_csv(files["decisions"])
    results["test"].to_csv(files["results"])
    results["validation"].to_csv(files["results_validation"])
    write_report(test, results["test"], decisions, files["report"], validation=results["validation"])
    if tracked:
        log_to_mlflow(quick, results, decisions, run_ids, list(files.values()))

    summary = ["accuracy", "f1_macro", "rappel_P1", "sous_triage", "sur_triage"]
    for name in ["validation", "test"]:
        print(f"\n{name} :\n" + (results[name][summary] * 100).round(1).to_string())
    print(f"\nRapport -> {files['report']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="evalue les adaptateurs _quick sur 30 exemples")
    parser.add_argument("--no-mlflow", action="store_true", help="ne pas journaliser l'evaluation dans MLflow")
    args = parser.parse_args()
    run(quick=args.quick, tracked=tracking.setup(not args.no_mlflow))


if __name__ == "__main__":
    main()
