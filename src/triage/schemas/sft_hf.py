"""Schema Pandera du dataset SFT final publie sur Hugging Face.

Une ligne = une paire prompt / reponse de `data/processed/sft_avec_priorite.csv`
(le fichier d'entrainement, inchange), enrichie des metadonnees du cas
clinique : source et licence, specialite, priorite et niveau de confiance,
age, sexe, symptomes, antecedents et constantes vitales.
"""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from triage.constants import LABELS
from triage.schemas.corpus import LANGUES, NIVEAUX_CONFIANCE, SOURCES

LICENCES = {"mediqal_oeq": "cc-by-4.0", "frenchmedmcqa": "apache-2.0", "medquad": "cc-by-4.0"}
SEXES = ["F", "M"]
# Bornes de plausibilite des constantes (une valeur hors bornes est ecartee a l'extraction).
CONSTANTES = {
    "fc": (20, 350),  # frequence cardiaque, /min
    "pas": (30, 300),  # pression arterielle systolique, mmHg
    "pad": (10, 200),  # pression arterielle diastolique, mmHg
    "fr": (4, 120),  # frequence respiratoire, /min
    "spo2": (30, 100),  # saturation en oxygene, %
    "temperature": (30, 43),  # degres Celsius
    "glasgow": (3, 15),  # score de Glasgow
}


def _list_of_str_or_null(values: pd.Series) -> pd.Series:
    return values.map(lambda v: v is None or (isinstance(v, list) and all(isinstance(x, str) and x for x in v)))


def _constantes_valid(values: pd.Series) -> pd.Series:
    def valid(v) -> bool:
        if not isinstance(v, dict) or set(v) != set(CONSTANTES):
            return False
        return all(v[k] is None or lo <= v[k] <= hi for k, (lo, hi) in CONSTANTES.items())

    return values.map(valid)


sft_hf_schema = DataFrameSchema(
    columns={
        "id": Column(str, unique=True, nullable=False),
        "case_id": Column(str, nullable=True),
        "source": Column(str, Check.isin(SOURCES)),
        "licence": Column(str, Check.isin(sorted(set(LICENCES.values())))),
        "langue": Column(str, Check.isin(LANGUES)),
        "groupe": Column(str, nullable=False),
        "specialite": Column(str, nullable=True),
        "type_question": Column(str, nullable=False),
        "priorite": Column(str, Check.isin(LABELS), nullable=True),
        "niveau_confiance": Column(str, Check.isin(NIVEAUX_CONFIANCE)),
        "age_ans": Column(float, Check.in_range(0, 120), nullable=True),
        "sexe": Column(str, Check.isin(SEXES), nullable=True),
        "symptomes": Column(object, Check(_list_of_str_or_null, error="liste de chaines ou null"), nullable=True),
        "antecedents": Column(object, Check(_list_of_str_or_null, error="liste de chaines ou null"), nullable=True),
        "constantes": Column(object, Check(_constantes_valid, error="constantes hors bornes ou cles manquantes")),
        "prompt": Column(str, Check.str_length(min_value=10), nullable=False),
        "reponse": Column(str, Check.str_length(min_value=1), nullable=False),
    },
    checks=[
        Check(lambda df: df["licence"] == df["source"].map(LICENCES), error="licence incoherente avec la source"),
        Check(lambda df: df["priorite"].isna() | df["case_id"].notna(), error="priorite sans cas clinique"),
        Check(
            lambda df: df["priorite"].notna() == (df["niveau_confiance"] == "pre_annotation_llm"),
            error="priorite incoherente avec niveau_confiance",
        ),
        # Symptomes et antecedents n'existent que pour un cas clinique.
        Check(lambda df: df["case_id"].notna() == df["symptomes"].notna(), error="symptomes sans cas clinique"),
        Check(lambda df: df["case_id"].notna() == df["antecedents"].notna(), error="antecedents sans cas clinique"),
    ],
    strict=True,
    coerce=False,
)
