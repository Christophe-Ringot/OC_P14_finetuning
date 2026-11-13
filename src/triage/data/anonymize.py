import re
from collections import Counter

import pandas as pd
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from spacy.lang.fr.stop_words import STOP_WORDS as FRENCH_STOP_WORDS

from triage.constants import DATA_DIR, SEED
from triage.schemas.corpus import corpus_schema

INPUT_PATH = DATA_DIR / "interim" / "corpus_clean.csv"
OUTPUT_PATH = DATA_DIR / "interim" / "corpus_anonymized.csv"
REPORT_PATH = DATA_DIR / "interim" / "anonymization_report.csv"
SAMPLE_PATH = DATA_DIR / "interim" / "anonymization_sample.csv"

TEXT_COLUMNS = ["instruction", "contexte_patient", "reponse"]
CONTACT_ENTITIES = ["EMAIL_ADDRESS"]
PLACEHOLDER = "X"
SAMPLE_SIZE = 30

CIVILITY_RE = re.compile(
    r"\b(Monsieur|Madame|Mademoiselle|Mme|Mlle|Mr|M\.)\s+"
    r"((?:(?:Le|La|De|Du)\s+)?[A-Z][A-Za-zÀ-ÿ'-]*(?:\.[A-Z][A-Za-zÀ-ÿ'-]*)*(?:\.{1,3}|…)?)"
    r"(\s+([A-Z][A-Za-zÀ-ÿ'-]*)(?:\.{1,3}|…)?)?"
)
NAME = r"[A-ZÉÈ][a-zéèêëàâîïôöûüç]+(?:-[A-ZÉÈ][a-zéèêëàâîïôöûüç]+)?"
# Prenom en debut de phrase ou apres "enfant", "petit"..., suivi d'un age.
FIRST_NAME_RE = re.compile(
    rf"(?:^|[.!?:]\s+|\b(?:enfant|petit|petite|garçon|fille|bébé|nourrisson|patient|patiente|jeune)\s+)"
    rf"({NAME})"
    r"(?=,?\s+(?:\d+\s*(?:ans|mois|semaines|jours)|âgée?\s+de|a\s+\d+\s*(?:ans|mois)|est\s+âgée?|\(\d+\s*ans))",
    re.MULTILINE,
)


def mask_civilities(text: str, counter: Counter) -> str:
    def replace(match: re.Match) -> str:
        civility, name, second, second_word = match.groups()
        # Le second mot est souvent le prenom ("Monsieur S... Gerard"), sauf
        # s'il ouvre la phrase suivante ("de Madame T. Vous vous inquietez").
        kept = second if second_word and second_word.lower() in FRENCH_STOP_WORDS else ""
        if name.rstrip(".…") == PLACEHOLDER:
            return match.group(0)
        counter[f"CIVILITE | {match.group(0)[: len(match.group(0)) - len(kept)]}"] += 1
        return f"{civility} {PLACEHOLDER}.{kept}"

    return CIVILITY_RE.sub(replace, text)


def mask_first_names(text: str, nlp, counter: Counter) -> str:
    names = set()
    for match in FIRST_NAME_RE.finditer(text):
        doc = nlp(text[max(0, match.start() - 40) : match.end() + 40])
        token = next((t for t in doc if t.text == match.group(1).split("-")[0]), None)
        if token is not None and token.pos_ == "PROPN":
            names.add(match.group(1))
    for name in names:
        text, n = re.subn(rf"\b{re.escape(name)}\b", PLACEHOLDER, text)
        counter[f"PRENOM | {name}"] += n
    return text


def build_engines():
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "fr", "model_name": "fr_core_news_md"},
                {"lang_code": "en", "model_name": "en_core_web_md"},
            ],
        }
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["fr", "en"])
    return analyzer, AnonymizerEngine(), spacy.load("fr_core_news_md")


def anonymize_text(text, language: str, engines, counter: Counter):
    if not isinstance(text, str):
        return text
    analyzer, anonymizer, nlp_fr = engines
    text = mask_civilities(text, counter)
    if language == "fr":
        text = mask_first_names(text, nlp_fr, counter)
    results = analyzer.analyze(text=text, language=language, entities=CONTACT_ENTITIES)
    for r in results:
        counter[f"COORDONNEES | {text[r.start : r.end]}"] += 1
    if not results:
        return text
    operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "[COORDONNEES]"})}
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text


def main() -> None:
    corpus = pd.read_csv(INPUT_PATH)
    engines = build_engines()
    counter = Counter()

    # Un meme cas clinique est repete pour chacune de ses questions : chaque
    # texte distinct est anonymise une seule fois, donc toujours a l'identique.
    cache = {}
    anonymized = corpus.copy()
    for column in TEXT_COLUMNS:
        values = []
        for text, language in zip(corpus[column], corpus["langue"]):
            if (text, language) not in cache:
                cache[(text, language)] = anonymize_text(text, language, engines, counter)
            values.append(cache[(text, language)])
        anonymized[column] = values

    validated = corpus_schema.validate(anonymized, lazy=True)
    validated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    report = pd.DataFrame(
        [
            {"type": key.split(" | ", 1)[0], "texte_masque": key.split(" | ", 1)[1], "occurrences": n}
            for key, n in counter.most_common()
        ],
        columns=["type", "texte_masque", "occurrences"],
    )
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8")

    changed = pd.concat(
        [
            pd.DataFrame({"id": corpus["id"], "colonne": column, "avant": corpus[column], "apres": anonymized[column]})
            for column in TEXT_COLUMNS
        ]
    )
    changed = changed[changed["avant"].fillna("") != changed["apres"].fillna("")].drop_duplicates(["avant"])
    changed.sample(min(SAMPLE_SIZE, len(changed)), random_state=SEED).to_csv(SAMPLE_PATH, index=False, encoding="utf-8")

    rows_changed = (corpus[TEXT_COLUMNS].fillna("") != anonymized[TEXT_COLUMNS].fillna("")).any(axis=1)
    print(f"Lignes modifiees : {rows_changed.sum()} / {len(corpus)} | textes distincts modifies : {len(changed)}")
    print(report.groupby("type")["occurrences"].sum().to_string())
    print(f"\nCorpus anonymise -> {OUTPUT_PATH}")
    print(f"Detail -> {REPORT_PATH} | echantillon avant/apres -> {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
