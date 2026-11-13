import argparse
from pathlib import Path

import pandas as pd

from triage.constants import DATA_DIR, FINAL_EXPERIMENT, LABELS, OUTPUTS_DIR, SEED
from triage.utils import hf_dataset

ANNOTATIONS = DATA_DIR / "annotation" / "annotations_priorite.csv"
POOL = DATA_DIR / "annotation" / "pool_priorite.csv"
PREDICTIONS = OUTPUTS_DIR / "triage_eval" / "predictions.csv"
EXPORT_DIR = Path("clinical_eval")
REVIEW_DIR = OUTPUTS_DIR / "annotation_review"
REVIEW_PATH = REVIEW_DIR / "relecture_annotations.csv"
N_REVIEW = 50
CASE_COLUMNS = [
    "case_id",
    "source",
    "langue",
    "specialite",
    "priorite",
    "niveau_confiance",
    "justification",
    "age_ans",
    "sexe",
    "symptomes",
    "antecedents",
    "constantes",
]


def questions() -> pd.DataFrame:
    """Les 400 questions annotees du test, avec la justification de l'annotation."""
    frame = hf_dataset.load_sft(with_priority=True)
    test = frame[(frame["split"] == "test") & frame["priorite"].notna()]
    notes = pd.read_csv(ANNOTATIONS)[["case_id", "justification"]]
    return test.merge(notes, on="case_id", how="left").drop(columns=["split", "groupe"])


def cases(rows: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par cas clinique : c'est le cas qui porte le niveau, pas la question."""
    pool = pd.read_csv(POOL)[["case_id", "contexte_patient"]]
    out = rows.groupby("case_id", as_index=False).first()[CASE_COLUMNS].merge(pool, on="case_id", how="left")
    out["n_questions"] = out["case_id"].map(rows["case_id"].value_counts())
    return out


def review_sample(case_table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Cas a relire en priorite : mal classes par le modele final, puis P2, puis le reste."""
    pred = pd.read_csv(PREDICTIONS)[["id", "priorite", FINAL_EXPERIMENT]].merge(rows[["id", "case_id"]], on="id")
    pred["erreur"] = pred[FINAL_EXPERIMENT] != pred["priorite"]
    per_case = pred.groupby("case_id").agg(
        taux_erreur_modele=("erreur", "mean"), niveau_predit_majoritaire=(FINAL_EXPERIMENT, lambda s: s.mode().iloc[0])
    )
    table = case_table.merge(per_case, on="case_id", how="left")
    table["rang"] = 2
    table.loc[table["priorite"] == LABELS[1], "rang"] = 1
    table.loc[table["taux_erreur_modele"] > 0, "rang"] = 0
    table = table.sample(frac=1, random_state=SEED).sort_values(
        ["rang", "taux_erreur_modele"], ascending=[True, False], kind="stable"
    )
    sample = table.head(N_REVIEW)
    columns = [
        "case_id",
        "specialite",
        "contexte_patient",
        "priorite",
        "justification",
        "niveau_predit_majoritaire",
        "taux_erreur_modele",
    ]
    return sample[columns].assign(accord="", niveau_relecteur="", commentaire="")


def summarize() -> None:
    """Accord entre l'annotation LLM et le relecteur, une fois la colonne niveau_relecteur remplie."""
    from sklearn.metrics import cohen_kappa_score

    review = pd.read_csv(REVIEW_PATH).dropna(subset=["niveau_relecteur"])
    if review.empty:
        print("Aucune relecture remplie.")
        return
    agree = (review["priorite"] == review["niveau_relecteur"]).mean()
    kappa = cohen_kappa_score(review["priorite"], review["niveau_relecteur"], labels=LABELS)
    print(f"{len(review)} cas relus : accord {agree:.1%}, kappa de Cohen {kappa:.2f}")
    print(
        pd.crosstab(review["priorite"], review["niveau_relecteur"], rownames=["annotation LLM"], colnames=["relecteur"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", choices=["build", "summarize"], default="build")
    if parser.parse_args().action == "summarize":
        summarize()
        return
    rows = questions()
    case_table = cases(rows)
    (EXPORT_DIR / "data").mkdir(parents=True, exist_ok=True)
    rows.to_parquet(EXPORT_DIR / "data" / "questions.parquet", index=False)
    case_table.to_parquet(EXPORT_DIR / "data" / "cases.parquet", index=False)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    # Ne jamais ecraser une relecture deja commencee.
    if REVIEW_PATH.exists():
        print(f"{REVIEW_PATH} existe deja : non regenere")
    else:
        review_sample(case_table, rows).to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
    print(f"{len(rows)} questions, {len(case_table)} cas -> {EXPORT_DIR}")
    print(case_table["priorite"].value_counts().to_string())


if __name__ == "__main__":
    main()
