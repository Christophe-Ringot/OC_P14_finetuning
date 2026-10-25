
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from schemas.sft_schema import sft_schema

DATA_PATH = "data/processed/sft_dataset.csv"
OUTPUT_PATH = "data/processed/sft_dataset_anonymized.csv"
REPORT_PATH = "outputs/anonymization_report.csv"

MASKED_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "US_SSN", "IP_ADDRESS"]


def build_analyzer() -> AnalyzerEngine:
    config = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "fr", "model_name": "fr_core_news_md"},
            {"lang_code": "en", "model_name": "en_core_web_md"},
        ],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["fr", "en"])


def anonymize_text(
    text: str, language: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine
) -> tuple[str, list[tuple[str, str]]]:
    results = analyzer.analyze(text=text, language=language, entities=MASKED_ENTITIES)
    if not results:
        return text, []
    masked = anonymizer.anonymize(text=text, analyzer_results=results).text
    hits = [(r.entity_type, text[r.start : r.end]) for r in results]
    return masked, hits


def main() -> None:
    analyzer = build_analyzer()
    anonymizer = AnonymizerEngine()

    df = pd.read_csv(DATA_PATH)
    report_rows = []
    for col in ["instruction", "response"]:
        masked_col = []
        for row in df.itertuples():
            masked, hits = anonymize_text(getattr(row, col), row.language, analyzer, anonymizer)
            masked_col.append(masked)
            for entity_type, span in hits:
                report_rows.append({"id": row.id, "field": col, "entity_type": entity_type, "text": span})
        df[col] = masked_col

    validated = sft_schema.validate(df, lazy=True)
    validated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    pd.DataFrame(report_rows).to_csv(REPORT_PATH, index=False, encoding="utf-8")
    print(f"{len(validated)} lignes anonymisees ecrites dans {OUTPUT_PATH}")
    print(f"{len(report_rows)} entites masquees - rapport dans {REPORT_PATH}")


if __name__ == "__main__":
    main()
