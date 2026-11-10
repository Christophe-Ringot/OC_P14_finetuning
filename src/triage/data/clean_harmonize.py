import hashlib
import re
import string
from pathlib import Path
import pandas as pd
from triage.schemas.corpus import corpus_schema
from triage.data.collect_sources import load_raw

OUTPUT_PATH = Path("data") / "interim" / "corpus_clean.csv"
RANDOM_STATE = 42
# MedQuAD est echantillonne pour ne pas noyer les sources francaises.
MEDQUAD_SAMPLE_SIZE = 700

ANSWER_COLS = ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"]
LETTERS = list(string.ascii_uppercase[:5])


def clean_text(text) -> str | None:
    if text is None or pd.isna(text):
        return None
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    # Supprime les espaces en debut/fin de ligne sans toucher aux puces "- ".
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text or None


def case_id(clinical_case: str) -> str:
    return "case_" + hashlib.sha1(clinical_case.encode("utf-8")).hexdigest()[:10]


def harmonize_mediqal_oeq() -> pd.DataFrame:
    raw = load_raw("mediqal_oeq")
    context = raw["clinical_case"].map(clean_text)
    return pd.DataFrame(
        {
            "id": "mediqal_oeq_" + raw["id"].astype(str),
            "langue": "fr",
            "source": "mediqal_oeq",
            "source_id": raw["id"].astype(str),
            "case_id": context.map(lambda c: case_id(c) if isinstance(c, str) else None),
            "instruction": raw["question"].map(clean_text),
            "contexte_patient": context,
            "reponse": raw["answer"].map(clean_text),
            "specialite": raw["medical_subject"],
            "type_question": raw["question_type"].replace({"Raisonnement": "Reasoning"}),
        }
    )


def harmonize_frenchmedmcqa() -> pd.DataFrame:
    raw = load_raw("frenchmedmcqa")
    rows = []
    for row in raw.itertuples():
        # Sous-ensemble mono-reponse : correct_answers est l'indice de l'option.
        index = int(row.correct_answers)
        options = "\n".join(f"({LETTERS[i]}) {clean_text(getattr(row, col))}" for i, col in enumerate(ANSWER_COLS))
        rows.append(
            {
                "id": f"frenchmedmcqa_{row.id}",
                "langue": "fr",
                "source": "frenchmedmcqa",
                "source_id": str(row.id),
                "case_id": None,
                "instruction": f"{clean_text(row.question)}\n{options}",
                "contexte_patient": None,
                "reponse": f"La réponse correcte est ({LETTERS[index]}) : {clean_text(getattr(row, ANSWER_COLS[index]))}",
                "specialite": None,
                "type_question": "qcm",
            }
        )
    return pd.DataFrame(rows)


def harmonize_medquad() -> pd.DataFrame:
    raw = load_raw("medquad").reset_index(names="row_index")
    return pd.DataFrame(
        {
            "id": "medquad_" + raw["row_index"].astype(str),
            "langue": "en",
            "source": "medquad",
            "source_id": raw["row_index"].astype(str),
            "case_id": None,
            "instruction": raw["Question"].map(clean_text),
            "contexte_patient": None,
            "reponse": raw["Answer"].map(clean_text),
            "specialite": raw["qtype"],
            "type_question": "qa",
        }
    )


def apply_rule(frame: pd.DataFrame, mask: pd.Series, rule: str, log: list) -> pd.DataFrame:
    """Retire les lignes `mask` et consigne combien chaque source en perd."""
    removed = frame.loc[mask, "source"].value_counts()
    for source, count in removed.items():
        log.append({"source": source, "regle": rule, "lignes_retirees": int(count)})
    return frame.loc[~mask]


def main() -> None:
    corpus = pd.concat(
        [harmonize_mediqal_oeq(), harmonize_frenchmedmcqa(), harmonize_medquad()], ignore_index=True
    )
    initial = corpus["source"].value_counts()
    log = []

    corpus = apply_rule(corpus, corpus["instruction"].isna() | corpus["reponse"].isna(), "question ou reponse vide", log)
    corpus = apply_rule(
        corpus,
        corpus.duplicated(["contexte_patient", "instruction", "reponse"], keep="first"),
        "doublon exact (contexte + question + reponse)",
        log,
    )
    # MedQuAD contient la meme question avec des reponses de sources NIH
    # differentes : on n'en garde qu'une pour eviter des cibles contradictoires.
    is_medquad = corpus["source"] == "medquad"
    corpus = apply_rule(
        corpus, is_medquad & corpus.duplicated(["source", "instruction"], keep="first"), "question MedQuAD repetee", log
    )

    medquad = corpus[corpus["source"] == "medquad"]
    kept = medquad.groupby("specialite", group_keys=False).sample(
        frac=MEDQUAD_SAMPLE_SIZE / len(medquad), random_state=RANDOM_STATE
    )
    corpus = apply_rule(
        corpus, (corpus["source"] == "medquad") & ~corpus.index.isin(kept.index),
        f"echantillonnage MedQuAD (~{MEDQUAD_SAMPLE_SIZE}, stratifie par type)", log,
    )

    corpus = corpus.assign(priorite=None, niveau_confiance="source_originale", split=None)
    validated = corpus_schema.validate(corpus.reset_index(drop=True), lazy=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Lignes retirees par regle :")
    print(pd.DataFrame(log).to_string(index=False))
    summary = pd.DataFrame({"initial": initial, "final": validated["source"].value_counts()})
    print("\nBilan par source :")
    print(summary.to_string())
    mediqal = validated[validated["source"] == "mediqal_oeq"]
    print(
        f"\nMediQAl OEQ : {mediqal['case_id'].nunique()} cas cliniques, "
        f"{mediqal['case_id'].isna().sum()} questions sans cas clinique"
    )
    print(f"Corpus nettoye : {len(validated)} lignes -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
