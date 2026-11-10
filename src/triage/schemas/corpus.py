from pandera.pandas import Check, Column, DataFrameSchema

SOURCES = ["mediqal_oeq", "frenchmedmcqa", "medquad"]
LANGUES = ["fr", "en"]
PRIORITES = ["urgence_maximale", "urgence_moderee", "differee"]

corpus_schema = DataFrameSchema(
    columns={
        "id": Column(str, unique=True, nullable=False),
        "langue": Column(str, Check.isin(LANGUES)),
        "source": Column(str, Check.isin(SOURCES)),
        "source_id": Column(str, nullable=False),
        "case_id": Column(str, nullable=True),
        "instruction": Column(str, Check.str_length(min_value=3), nullable=False),
        "contexte_patient": Column(str, nullable=True),
        "reponse": Column(str, Check.str_length(min_value=1), nullable=False),
        "specialite": Column(str, nullable=True),
        "type_question": Column(str, nullable=True),
        "priorite": Column(str, Check.isin(PRIORITES), nullable=True),
        "niveau_confiance": Column(str, Check.isin(["source_originale", "pre_annotation_llm"])),
        "split": Column(str, nullable=True),
    },
    checks=[
        # Un cas clinique a toujours un case_id, et inversement.
        Check(lambda df: df["contexte_patient"].notna() == df["case_id"].notna(), error="case_id incoherent"),
        Check(lambda df: df["instruction"] != df["reponse"], error="instruction identique a la reponse"),
        # Une priorite n'existe que pour un cas clinique pre-annote (etape 4).
        Check(
            lambda df: df["priorite"].notna() == (df["niveau_confiance"] == "pre_annotation_llm"),
            error="priorite incoherente avec niveau_confiance",
        ),
        Check(lambda df: df["priorite"].isna() | df["case_id"].notna(), error="priorite sans cas clinique"),
    ],
    strict=True,
    coerce=True,
)
