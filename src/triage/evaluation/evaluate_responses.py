"""Qualite des reponses generees (pas seulement le niveau de triage).

Chaque modele genere une reponse (decodage glouton) pour un echantillon fixe du
test : 20 cas cliniques annotes par niveau (P1, P2, P3) et 20 questions
generales (FrenchMedMCQA, MedQuAD). Le meme echantillon sert avant et apres
DPO, pour mesurer l'apport de chaque phase.

Metriques automatiques, par reponse puis moyennees :
- pertinence : ROUGE-L et F1 lexical par rapport a la reponse de reference ;
- conformite : prefixe "[Niveau de priorite estime : ...]" present sur les cas
  annotes, et niveau ecrit = niveau annote ;
- securite : formulations rassurantes a risque et chiffres absents du cas et de
  la reference (safety_check), sous-triage dans le prefixe ;
- clarte : longueur, reponse coupee (limite de tokens atteinte), repetitions.

Ces metriques sont des filtres grossiers : la grille de jugement en aveugle
(docs/evaluation_reponses.md) complete l'evaluation sur les memes reponses.

Sorties : outputs/response_eval/generations_<modele>.csv et qualite_reponses.md
"""

import argparse
from pathlib import Path

import pandas as pd

from triage.constants import BASE_MODEL_EXPERIMENT, LABELS, LEVEL_RE, OUTPUTS_DIR, RANK, SEED, SHORT, adapter_dir
from triage.evaluation.metrics import repetition, rouge_l, strip_prefix, token_f1, words
from triage.evaluation.safety_check import dangerous_language, numeric_hallucination
from triage.utils import hf_dataset, tracking
from triage.utils.markdown import markdown_table

OUTPUT_DIR = OUTPUTS_DIR / "response_eval"
MODELS = {
    BASE_MODEL_EXPERIMENT: None,
    **{e: adapter_dir(e) for e in ["4_sft_medical_triage", "5_sft_triage_dpo", "6_sft_triage_dpo_qualite"]},
}
PER_LEVEL = 20
GENERAL = 20
MAX_NEW_TOKENS = 300
METRICS = [
    "rouge_l",
    "f1_lexical",
    "prefixe_present",
    "niveau_correct",
    "sous_triage_ecrit",
    "formulation_risquee",
    "chiffre_invente",
    "longueur_mots",
    "coupee",
    "repetition",
    "vide",
]


def sample() -> pd.DataFrame:
    frame = hf_dataset.load_sft(with_priority=True)
    test = frame[frame["split"] == "test"]
    annotated = test[test["priorite"].notna()]
    # Un seul exemple par cas clinique, pour ne pas peser plusieurs fois le meme cas.
    annotated = annotated.groupby("case_id").sample(1, random_state=SEED)
    parts = [annotated[annotated["priorite"] == label].sample(PER_LEVEL, random_state=SEED) for label in LABELS]
    parts.append(test[test["priorite"].isna()].sample(GENERAL, random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)


def generate(adapter: Path | None, prompts: list) -> tuple[list, list]:
    from triage.utils import modeling

    tokenizer = modeling.load_tokenizer(padding_side="left")
    model = modeling.load_model(adapter)
    texts, truncated = modeling.generate(model, tokenizer, prompts, MAX_NEW_TOKENS)
    del model
    modeling.free_gpu_memory()
    return texts, truncated


def score(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        generated, reference = row.generated, row.reponse
        level = LEVEL_RE.search(generated)
        annotated = isinstance(row.priorite, str)
        body = strip_prefix(generated)
        rows.append(
            {
                "rouge_l": rouge_l(strip_prefix(reference), body),
                "f1_lexical": token_f1(strip_prefix(reference), body),
                "prefixe_present": float(bool(level)) if annotated else float("nan"),
                "niveau_correct": float(bool(level) and level.group(1) == row.priorite) if annotated else float("nan"),
                "sous_triage_ecrit": float(
                    bool(level) and level.group(1) in RANK and RANK[level.group(1)] > RANK[row.priorite]
                )
                if annotated
                else float("nan"),
                "formulation_risquee": float(bool(dangerous_language(generated))),
                "chiffre_invente": float(bool(numeric_hallucination(row.prompt, reference, generated))),
                "longueur_mots": len(words(body)),
                "coupee": float(row.truncated),
                "repetition": repetition(body),
                "vide": float(len(words(body)) < 3),
            }
        )
    return pd.DataFrame(rows)


def summarize(results: dict) -> pd.DataFrame:
    summary = {}
    for name, frame in results.items():
        summary[name] = frame[METRICS].mean()
        for label in LABELS:
            summary[name][f"rouge_l_{SHORT[label]}"] = frame.loc[frame["priorite"] == label, "rouge_l"].mean()
        summary[name]["rouge_l_general"] = frame.loc[frame["priorite"].isna(), "rouge_l"].mean()
    return pd.DataFrame(summary).T


def write_report(summary: pd.DataFrame, n: int) -> None:
    percent = [c for c in summary.columns if c != "longueur_mots"]
    shown = summary.copy()
    shown[percent] = shown[percent] * 100
    lines = [
        "# Qualite des reponses generees",
        "",
        f"Echantillon fixe du test : {n} questions ({PER_LEVEL} cas par niveau P1/P2/P3, un par cas clinique, "
        f"et {GENERAL} questions generales). Decodage glouton, {MAX_NEW_TOKENS} tokens max. Valeurs en % "
        "(longueur en mots). `rouge_l` et `f1_lexical` comparent a la reponse de reference sans le prefixe de niveau.",
        "",
        markdown_table(shown.rename_axis("modele")),
        "",
        "Lecture :",
        "- `niveau_correct` porte sur le niveau ecrit par le modele en generation libre ; la metrique de triage de "
        "reference reste celle des log-probabilites (comparaison_triage.md), plus stable.",
        "- `formulation_risquee` et `chiffre_invente` sont des signaux a relire, pas des erreurs averees "
        "(un chiffre calcule a partir du cas est signale aussi).",
        "- ROUGE-L recompense la proximite avec une reference courte : il mesure la pertinence par rapport au "
        "corrige, pas la qualite clinique. Le jugement en aveugle (docs/evaluation_reponses.md) la complete.",
    ]
    (OUTPUT_DIR / "qualite_reponses.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    tracked = tracking.setup(not args.no_mlflow)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = sample()

    for name in args.models:
        adapter = MODELS[name]
        if adapter is not None and not (adapter / "adapter_config.json").exists():
            print(f"{name} ignore : adaptateur absent ({adapter})")
            continue
        print(f"Generation {name} ({len(questions)} questions)")
        texts, truncated = generate(adapter, questions["prompt"].tolist())
        frame = questions[["id", "case_id", "source", "priorite", "prompt", "reponse"]].assign(
            generated=texts, truncated=truncated
        )
        frame.to_csv(OUTPUT_DIR / f"generations_{name}.csv", index=False)

    # Le rapport reprend tous les modeles deja generes (avant et apres DPO).
    results = {}
    for name in MODELS:
        path = OUTPUT_DIR / f"generations_{name}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["generated"] = frame["generated"].fillna("")
            results[name] = pd.concat([frame, score(frame)], axis=1)
            results[name].to_csv(OUTPUT_DIR / f"scores_{name}.csv", index=False)
    summary = summarize(results)
    summary.to_csv(OUTPUT_DIR / "qualite_reponses.csv")
    write_report(summary, len(questions))
    print(
        summary[
            [
                "rouge_l",
                "f1_lexical",
                "niveau_correct",
                "sous_triage_ecrit",
                "formulation_risquee",
                "chiffre_invente",
                "coupee",
            ]
        ]
        .round(3)
        .to_string()
    )

    if tracked:
        import mlflow

        with tracking.start_run(
            True, run_name="evaluation_reponses", tags={"type": "evaluation", "phase": "qualite_reponses"}
        ):
            mlflow.log_params(
                {
                    "n_questions": len(questions),
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "decodage": "glouton, repetition_penalty 1.1",
                }
            )
            for name, row in summary.iterrows():
                mlflow.log_metrics({f"{name}.{k}": float(v) for k, v in row.items() if pd.notna(v)})
            mlflow.log_artifacts(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
