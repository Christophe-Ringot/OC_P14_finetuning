import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from triage.constants import (
    ADAPTERS,
    BASE_MODEL_EXPERIMENT,
    BASELINE,
    LABELS,
    SEED,
    SHORT,
    adapter_dir,
    build_prompt,
)
from triage.evaluation.metrics import triage_metrics
from triage.evaluation.saved_scores import MODES, TRIAGE_EVAL_DIR, decision_scores, scores_path
from triage.utils import hf_dataset, tracking
from triage.utils.markdown import markdown_table

QUICK_TEST_EXAMPLES = 30
# Meme gabarit que les prompts du test, sans cas ni question.
NEUTRAL_PROMPT = build_prompt("", case="")
MAIN_COLUMNS = [
    "accuracy",
    "f1_macro",
    "rappel_P1",
    "sous_triage",
    "sur_triage",
    "erreurs_1_niveau",
    "erreurs_2_niveaux",
    "P1_predit_P3",
]


def score_with_model(splits: dict, adapter: Path | None) -> dict:
    """Log-probabilites brutes de chaque niveau et prior sur prompt vide, par split.

    Le prior sert a la calibration contextuelle (Zhao et al., 2021) : un modele
    qui n'a jamais vu les libelles choisit d'apres leur forme et non d'apres le
    cas, on retranche donc le score obtenu sans cas clinique. Pour un modele
    entraine sur les libelles, ce prior n'est plus neutre (un cas vide evoque
    "differee") : le choix calibre / brut est donc fait sur la validation.
    """
    from triage.utils.modeling import encode_context, free_gpu_memory, label_logprobs, load_model, load_tokenizer

    tokenizer = load_tokenizer()
    model = load_model(adapter)
    prior = label_logprobs(model, tokenizer, encode_context(tokenizer, NEUTRAL_PROMPT))
    results = {}
    for name, frame in splits.items():
        raw = []
        for i, prompt in enumerate(frame["prompt"], start=1):
            raw.append(label_logprobs(model, tokenizer, encode_context(tokenizer, prompt)))
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
    return decision_scores(scores, mode).idxmax(axis=1)


def choose_mode(validation: pd.DataFrame, scores: pd.DataFrame) -> tuple[str, dict]:
    """Mode de decision qui maximise le F1 macro sur la validation (le test n'est pas regarde)."""
    f1 = {mode: triage_metrics(validation["priorite"], decide(scores, mode))["f1_macro"] for mode in MODES}
    return max(f1, key=f1.get), f1


def confusion_table(y_true: pd.Series, y_pred: pd.Series) -> str:
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=LABELS),
        index=pd.Index([SHORT[label] for label in LABELS], name="annote \\ predit"),
        columns=[SHORT[label] for label in LABELS],
    )
    return markdown_table(matrix)


def write_report(
    test: pd.DataFrame, results: pd.DataFrame, decisions: pd.DataFrame, path: Path, validation: pd.DataFrame
) -> None:
    percent = results.astype(float).rename_axis("experience") * 100
    class_columns = [c for c in results.columns if c not in MAIN_COLUMNS]
    distribution = test["priorite"].map(SHORT).value_counts().reindex(["P1", "P2", "P3"])
    modes = decisions.rename_axis("experience").rename(columns={"mode_retenu": "mode retenu"})
    modes = modes.rename(columns={f"f1_validation_{m}": f"F1 validation {m}" for m in MODES})
    modes[[f"F1 validation {m}" for m in MODES]] = modes[[f"F1 validation {m}" for m in MODES]].astype(float) * 100

    lines = [
        f"# Comparaison du triage - {len(results)} experiences",
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
        markdown_table(modes),
        "",
        "## Metriques principales",
        "",
        markdown_table(percent[MAIN_COLUMNS]),
        "",
        "Meme tableau sur la validation (regle de decision choisie sur ce jeu), a citer avec le test :",
        "",
        markdown_table(validation.astype(float).rename_axis("experience")[MAIN_COLUMNS] * 100),
        "",
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
        frame = frame.sample(min(QUICK_TEST_EXAMPLES, len(frame)), random_state=SEED).reset_index(drop=True)
    return frame


def available_models(quick: bool) -> dict:
    """Modele de base puis chaque adaptateur present sur le disque."""
    models = {BASE_MODEL_EXPERIMENT: None}
    for experiment in ADAPTERS:
        adapter = adapter_dir(experiment, quick)
        if (adapter / "adapter_config.json").exists():
            models[experiment] = adapter
        else:
            print(f"{experiment} ignoree : adaptateur absent ({adapter})")
    return models


def log_to_mlflow(quick: bool, results: dict, decisions: pd.DataFrame, run_ids: dict, files: list) -> None:
    """Metriques dans le run d'entrainement de chaque modele, et un run d'evaluation qui regroupe tout."""
    import mlflow

    def flat(split: str, metrics: pd.Series) -> dict:
        return {f"{split}_{name}": float(value) for name, value in metrics.items()}

    for experiment, run_id in run_ids.items():
        if run_id is None or experiment not in results["test"].index:
            continue
        with tracking.start_run(True, run_id=run_id):
            for split, frame in results.items():
                mlflow.log_metrics(flat(split, frame.loc[experiment]))
            mlflow.log_param("mode_decision", decisions.loc[experiment, "mode_retenu"])

    tags = {"type": "evaluation", "mode": "quick" if quick else "complet", "git_commit": tracking.git_commit()}
    with tracking.start_run(True, run_name=f"evaluation_triage{'_quick' if quick else ''}", tags=tags):
        mlflow.log_params(
            {
                "dataset": hf_dataset.dataset_name(),
                "dataset_revision": hf_dataset.DATASET_REVISION,
                "dataset_md5": hf_dataset.fingerprint(hf_dataset.load_sft()),
                **{f"mode_{e}": m for e, m in decisions["mode_retenu"].items()},
                **{f"run_id_{e}": r for e, r in run_ids.items() if r},
            }
        )
        for split, frame in results.items():
            for experiment, metrics in frame.iterrows():
                mlflow.log_metrics({f"{experiment}.{name}": v for name, v in flat(split, metrics).items()})
        for path in files:
            mlflow.log_artifact(str(path))


def run(quick: bool = False, tracked: bool = False, run_ids: dict | None = None, only: list | None = None) -> None:
    """Evalue tous les modeles, ou seulement `only` en reprenant les scores deja calcules des autres."""
    run_ids = run_ids or {}
    suffix = "_quick" if quick else ""
    splits = {"validation": load_split("validation", quick), "test": load_split("test", quick)}
    for frame in splits.values():
        frame[BASELINE] = LABELS[0]

    if only:
        saved_scores = {name: pd.read_csv(scores_path(name, suffix)) for name in splits}
        saved_modes = pd.read_csv(TRIAGE_EVAL_DIR / f"mode_decision{suffix}.csv", index_col=0)

    all_scores = {name: [] for name in splits}
    decisions = {}
    for experiment, adapter in available_models(quick).items():
        if only and experiment not in only and experiment in saved_modes.index:
            scores = {}
            for name in splits:
                frame = saved_scores[name][saved_scores[name]["experience"] == experiment].drop(columns="experience")
                scores[name] = frame.set_index("id").loc[splits[name]["id"]].reset_index()
            decisions[experiment] = saved_modes.loc[experiment].to_dict()
            mode = decisions[experiment]["mode_retenu"]
            print(f"{experiment} : scores repris de l'evaluation precedente (mode {mode})")
        else:
            print(f"Evaluation de {experiment} ({len(splits['validation'])} validation, {len(splits['test'])} test)")
            scores = score_with_model(splits, adapter)
            mode, f1 = choose_mode(splits["validation"], scores["validation"])
            decisions[experiment] = {"mode_retenu": mode, **{f"f1_validation_{m}": v for m, v in f1.items()}}
            print(f"  mode retenu sur la validation : {mode} " + ", ".join(f"{m}={v:.3f}" for m, v in f1.items()))
        for name in splits:
            splits[name][experiment] = decide(scores[name], mode).to_numpy()
            all_scores[name].append(scores[name].assign(experience=experiment))

    test = splits["test"]
    experiments = [c for c in test.columns if c[0].isdigit()]
    results = {
        name: pd.DataFrame({e: triage_metrics(frame["priorite"], frame[e]) for e in experiments}).T
        for name, frame in splits.items()
    }
    decisions = pd.DataFrame(decisions).T

    TRIAGE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "predictions": TRIAGE_EVAL_DIR / f"predictions{suffix}.csv",
        "scores": scores_path("test", suffix),
        "scores_validation": scores_path("validation", suffix),
        "decisions": TRIAGE_EVAL_DIR / f"mode_decision{suffix}.csv",
        "results": TRIAGE_EVAL_DIR / f"comparaison_triage{suffix}.csv",
        "results_validation": TRIAGE_EVAL_DIR / f"comparaison_triage_validation{suffix}.csv",
        "report": TRIAGE_EVAL_DIR / f"comparaison_triage{suffix}.md",
    }
    test[["id", "groupe", "priorite", *experiments]].to_csv(files["predictions"], index=False)
    pd.concat(all_scores["test"]).to_csv(files["scores"], index=False)
    pd.concat(all_scores["validation"]).to_csv(files["scores_validation"], index=False)
    decisions.to_csv(files["decisions"])
    results["test"].to_csv(files["results"])
    results["validation"].to_csv(files["results_validation"])
    write_report(test, results["test"], decisions, files["report"], results["validation"])
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
    parser.add_argument("--only", nargs="+", help="n'evaluer que ces experiences (les autres : scores deja calcules)")
    args = parser.parse_args()
    run(quick=args.quick, tracked=tracking.setup(not args.no_mlflow), only=args.only)


if __name__ == "__main__":
    main()
