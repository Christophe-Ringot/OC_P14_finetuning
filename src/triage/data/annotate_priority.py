import pandas as pd

from triage.constants import DATA_DIR, LABELS, RANK
from triage.schemas.corpus import corpus_schema

CORPUS_PATH = DATA_DIR / "interim" / "corpus_anonymized.csv"
POOL_PATH = DATA_DIR / "annotation" / "pool_priorite.csv"
ANNOTATIONS_PATH = DATA_DIR / "annotation" / "annotations_priorite.csv"
OUTPUT_PATH = DATA_DIR / "interim" / "corpus_annotated.csv"


def check_annotations(pool: pd.DataFrame, annotations: pd.DataFrame) -> None:
    duplicated = annotations["case_id"].duplicated()
    if duplicated.any():
        raise ValueError(f"Cas annotes plusieurs fois : {annotations.loc[duplicated, 'case_id'].tolist()}")
    invalid = ~annotations["priorite"].isin(LABELS)
    if invalid.any():
        raise ValueError(f"Niveaux invalides : {annotations.loc[invalid, 'priorite'].unique().tolist()}")
    missing = set(pool["case_id"]) - set(annotations["case_id"])
    unknown = set(annotations["case_id"]) - set(pool["case_id"])
    if missing or unknown:
        raise ValueError(f"{len(missing)} cas non annotes, {len(unknown)} annotations hors vivier")

    level = dict(zip(annotations["case_id"], annotations["priorite"].map(RANK)))
    children = pool[pool["parent_case_id"].notna()]
    # Une version prolongee d'un cas ne peut pas etre moins urgente que la precedente (rang 0 = P1).
    decreasing = children[children["case_id"].map(level) > children["parent_case_id"].map(level)]
    if not decreasing.empty:
        raise ValueError(f"Niveau en baisse dans une famille : {decreasing['case_id'].tolist()}")


def main() -> None:
    corpus = pd.read_csv(CORPUS_PATH)
    pool = pd.read_csv(POOL_PATH)
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    check_annotations(pool, annotations)

    priority = corpus["case_id"].map(dict(zip(annotations["case_id"], annotations["priorite"])))
    annotated = corpus.assign(
        priorite=priority,
        niveau_confiance=priority.notna().map({True: "pre_annotation_llm", False: "source_originale"}),
    )
    validated = corpus_schema.validate(annotated, lazy=True)
    validated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    cases = pool.merge(annotations, on="case_id")
    print("Repartition par cas clinique :")
    print(cases["priorite"].value_counts().reindex(LABELS).to_string())
    print("\nRepartition par question (heritee du cas) :")
    print(validated["priorite"].value_counts().reindex(LABELS).to_string())
    print("\nControle avec la regex de gravite (% des cas de chaque groupe) :")
    crosstab = pd.crosstab(cases["signal_gravite"], cases["priorite"], normalize="index").reindex(columns=LABELS)
    print((crosstab * 100).round(1).to_string())
    print(f"\nCorpus annote : {validated['priorite'].notna().sum()} / {len(validated)} lignes -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
