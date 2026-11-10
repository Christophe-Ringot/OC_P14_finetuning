from pandera.pandas import Check, Column, DataFrameSchema

LANGUAGES = ["fr", "en"]
SPLITS = ["train", "validation", "test"]
LABEL_TYPES = ["easy", "hard", "length", "human"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]

dpo_schema = DataFrameSchema(
    columns={
        "id": Column(str, unique=True, nullable=False),
        "source_dataset": Column(str, nullable=False),
        "language": Column(str, Check.isin(LANGUAGES)),
        "split": Column(str, Check.isin(SPLITS)),
        "label_type": Column(str, Check.isin(LABEL_TYPES), nullable=True),
        "prompt": Column(str, Check.str_length(min_value=5), nullable=False),
        "chosen": Column(str, Check.str_length(min_value=3), nullable=False),
        "rejected": Column(str, Check.str_length(min_value=3), nullable=False),
        "chosen_score": Column(float, Check.in_range(0, 5), nullable=True),
        "rejected_score": Column(float, Check.in_range(0, 5), nullable=True),
        "confidence_level": Column(str, Check.isin(CONFIDENCE_LEVELS)),
        "license": Column(str, nullable=False),
    },
    checks=Check(
        lambda df: df["chosen"] != df["rejected"],
        error="chosen et rejected ne doivent pas etre identiques",
    ),
    strict=True,
    coerce=True,
)
