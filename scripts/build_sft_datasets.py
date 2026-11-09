from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS_PATH = Path("data") / "interim" / "corpus_annotated.csv"
POOL_PATH = Path("data") / "annotation" / "pool_priorite.csv"
OUTPUT_DIR = Path("data") / "processed"
WITHOUT_PATH = OUTPUT_DIR / "sft_sans_priorite.csv"
WITH_PATH = OUTPUT_DIR / "sft_avec_priorite.csv"

RANDOM_STATE = 42
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
# Longueur du debut de texte commun qui rattache deux variantes d'un meme cas.
SHARED_PREFIX_LENGTH = 100
# Similarite cosinus (TF-IDF n-grammes de caracteres) au-dela de laquelle deux
# cas sont consideres comme des reformulations d'un meme scenario. Seuil choisi
# par relecture des paires (scripts/audit_splits.py) : vers 0.45 apparaissent
# des cas sans rapport, a 0.5 il s'agit surtout du meme scenario reformule.
SIMILARITY_THRESHOLD = 0.5
OUTPUT_COLUMNS = ["id", "split", "source", "langue", "groupe", "priorite", "prompt", "reponse"]


def build_prompt(row) -> str:
    if isinstance(row.contexte_patient, str):
        return f"### Cas clinique :\n{row.contexte_patient}\n\n### Question :\n{row.instruction}\n\n### Réponse :\n"
    return f"### Question :\n{row.instruction}\n\n### Réponse :\n"


def similar_pairs(pool: pd.DataFrame) -> list:
    """Paires de cas dont les textes sont des reformulations probables l'un de l'autre."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(pool["contexte_patient"].fillna(""))
    similarity = (matrix @ matrix.T).toarray()
    rows, columns = np.where(np.triu(similarity >= SIMILARITY_THRESHOLD, k=1))
    cases = pool["case_id"].to_numpy()
    return list(zip(cases[rows], cases[columns]))


def case_groups(pool: pd.DataFrame) -> dict:
    """Union-find : relie deux cas de meme famille, de meme debut de texte ou de texte tres similaire."""
    parent = {case: case for case in pool["case_id"]}

    def find(case):
        while parent[case] != case:
            parent[case] = parent[parent[case]]
            case = parent[case]
        return case

    for key in ["famille_id", "prefixe"]:
        for members in pool.groupby(key)["case_id"]:
            first, *others = members[1].tolist()
            for other in others:
                parent[find(other)] = find(first)
    pairs = similar_pairs(pool)
    for first, other in pairs:
        parent[find(other)] = find(first)
    print(f"{len(pairs)} paires de cas similaires (cosinus >= {SIMILARITY_THRESHOLD}) fusionnees")
    return {case: "groupe_" + find(case) for case in pool["case_id"]}


def assign_splits(frame: pd.DataFrame) -> pd.Series:
    """Tire les groupes au hasard, par source, jusqu'a atteindre les ratios en lignes."""
    rng = np.random.default_rng(RANDOM_STATE)
    split = pd.Series(index=frame.index, dtype=object)
    for _, source_rows in frame.groupby("source"):
        sizes = source_rows.groupby("groupe").size()
        sizes = sizes.iloc[rng.permutation(len(sizes))]
        bounds = np.cumsum(list(SPLIT_RATIOS.values())) * sizes.sum()
        position = sizes.cumsum() - sizes / 2
        names = np.array(list(SPLIT_RATIOS))[np.searchsorted(bounds, position)]
        split.loc[source_rows.index] = source_rows["groupe"].map(dict(zip(sizes.index, names)))
    return split


def check_datasets(without: pd.DataFrame, with_priority: pd.DataFrame) -> None:
    same = ["id", "split", "source", "langue", "groupe", "priorite", "prompt"]
    if not without[same].equals(with_priority[same]):
        raise ValueError("Les deux datasets different ailleurs que dans la reponse")
    differs = without["reponse"] != with_priority["reponse"]
    if not differs.equals(with_priority["priorite"].notna()):
        raise ValueError("Le prefixe de priorite ne correspond pas aux lignes annotees")
    for key in ["groupe", "prompt"]:
        leaking = with_priority.groupby(key)["split"].nunique()
        if (leaking > 1).any():
            raise ValueError(f"{(leaking > 1).sum()} valeurs de '{key}' presentes dans plusieurs splits")


def main() -> None:
    corpus = pd.read_csv(CORPUS_PATH)
    pool = pd.read_csv(POOL_PATH)
    pool["prefixe"] = pool["contexte_patient"].str[:SHARED_PREFIX_LENGTH]

    groups = case_groups(pool)
    # Hors cas clinique, une meme question revient parfois avec des reponses
    # redigees differemment : elle forme alors un groupe a elle seule.
    question_groups = "question_" + corpus.groupby(["source", "instruction"]).ngroup().astype(str)
    corpus["groupe"] = corpus["case_id"].map(groups).fillna(question_groups)
    corpus["split"] = assign_splits(corpus)
    corpus["prompt"] = [build_prompt(row) for row in corpus.itertuples()]

    without = corpus[OUTPUT_COLUMNS].copy()
    with_priority = without.copy()
    annotated = with_priority["priorite"].notna()
    with_priority.loc[annotated, "reponse"] = (
        "[Niveau de priorité estimé : " + with_priority.loc[annotated, "priorite"] + "]\n\n"
        + with_priority.loc[annotated, "reponse"]
    )
    check_datasets(without, with_priority)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    without.to_csv(WITHOUT_PATH, index=False, encoding="utf-8")
    with_priority.to_csv(WITH_PATH, index=False, encoding="utf-8")

    cases = corpus[corpus["case_id"].notna()]
    print(f"{cases['case_id'].nunique()} cas cliniques regroupes en {cases['groupe'].nunique()} groupes sans fuite")
    print("\nLignes par split et par source :")
    print(pd.crosstab(corpus["source"], corpus["split"], margins=True).to_string())
    print("\nPriorite par split (lignes annotees) :")
    print(pd.crosstab(cases["priorite"], cases["split"], normalize="columns").mul(100).round(1).to_string())
    print(f"\n-> {WITHOUT_PATH}\n-> {WITH_PATH}")


if __name__ == "__main__":
    main()
