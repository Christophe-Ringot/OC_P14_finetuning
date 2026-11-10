import re
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data") / "interim" / "corpus_anonymized.csv"
POOL_PATH = Path("data") / "annotation" / "pool_priorite.csv"


SEVERITY_RE = re.compile(
    r"(FC\s*[:=]|FR\s*[:=]|SpO2|Sp02|SaO2|PA\s*[:=]|GCS|Glasgow|"
    r"satur\w*|tension\s+art|fréquence\s+card|fréquence\s+respi|"
    r"T°|T\s*[:=]\s*\d|détresse|choc\s|marbrures|cyanose|coma|"
    r"urgence|réanimation|intubé|hypotension|hypoxém)",
    re.IGNORECASE,
)


def find_families(pool: pd.DataFrame) -> tuple[list, list]:
    """Relie chaque cas a la version precedente dont il prolonge le texte.
    """
    texts = dict(zip(pool["case_id"], pool["contexte_patient"]))
    by_length = sorted(texts, key=lambda case: len(texts[case]))
    parents = {}
    for i, case in enumerate(by_length):
        shorter = (other for other in reversed(by_length[:i]) if len(texts[other]) < len(texts[case]))
        parents[case] = next((other for other in shorter if texts[case].startswith(texts[other])), None)

    def root(case):
        while parents[case] is not None:
            case = parents[case]
        return case

    return [parents[case] for case in pool["case_id"]], [root(case) for case in pool["case_id"]]


def main() -> None:
    corpus = pd.read_csv(INPUT_PATH)
    cases = corpus[corpus["case_id"].notna()]
    pool = (
        cases.groupby("case_id", sort=False)
        .agg(
            contexte_patient=("contexte_patient", "first"),
            specialite=("specialite", "first"),
            n_questions=("id", "size"),
        )
        .reset_index()
    )
    pool["signal_gravite"] = pool["contexte_patient"].map(lambda text: bool(SEVERITY_RE.search(text)))
    pool["parent_case_id"], pool["famille_id"] = find_families(pool)

    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(POOL_PATH, index=False, encoding="utf-8")

    print(f"Cas cliniques uniques : {len(pool)} (couvrant {pool['n_questions'].sum()} questions)")
    print(f"Avec signal de gravite (regex) : {pool['signal_gravite'].sum()} / {len(pool)}")
    print(
        f"Familles de cas (versions successives) : {pool['famille_id'].nunique()} "
        f"dont {pool['parent_case_id'].notna().sum()} cas prolongeant une version precedente"
    )
    print(f"Vivier -> {POOL_PATH}")


if __name__ == "__main__":
    main()
