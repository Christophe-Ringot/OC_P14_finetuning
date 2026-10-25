"""Schema Pandera pour le corpus SFT (paires instruction-reponse)."""

from pandera.pandas import Check, Column, DataFrameSchema

SOURCE_DATASETS = ["frenchmedmcqa", "mediqal", "medquad"]
LANGUAGES = ["fr", "en"]
SPLITS = ["train", "validation", "test"]
TASK_TYPES = ["qcm", "clinical_case_qcm", "qa"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]

sft_schema = DataFrameSchema(
    columns={
        "id": Column(str, unique=True, nullable=False),
        "source_dataset": Column(str, Check.isin(SOURCE_DATASETS)),
        "language": Column(str, Check.isin(LANGUAGES)),
        "split": Column(str, Check.isin(SPLITS)),
        "task_type": Column(str, Check.isin(TASK_TYPES)),
        "instruction": Column(str, Check.str_length(min_value=5), nullable=False),
        "response": Column(str, Check.str_length(min_value=3), nullable=False),
        "medical_subject": Column(str, nullable=True),
        "symptoms": Column(str, nullable=True),
        "antecedents": Column(str, nullable=True),
        "vital_signs": Column(str, nullable=True),
        "confidence_level": Column(str, Check.isin(CONFIDENCE_LEVELS)),
        "license": Column(str, nullable=False),
        "original_id": Column(str, nullable=True),
    },
    checks=Check(
        lambda df: df["instruction"] != df["response"],
        error="instruction et response ne doivent pas etre identiques",
    ),
    strict=True,
    coerce=True,
)
