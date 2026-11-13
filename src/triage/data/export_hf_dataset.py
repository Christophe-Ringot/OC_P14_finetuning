"""Export du dataset SFT final vers Hugging Face (dossier `hf_dataset/`).

Part du fichier d'entrainement `data/processed/sft_avec_priorite.csv`, laisse
tel quel (son empreinte MD5 reste celle enregistree dans MLflow), et y ajoute
les metadonnees du cas clinique :

- depuis `data/interim/corpus_annotated.csv` : case_id, specialite,
  type_question, niveau_confiance ;
- par regles (`extract_*` ci-dessous, deterministes) : age, sexe et
  constantes vitales (FC, PA, FR, SpO2, temperature, Glasgow), premiere valeur
  citee dans le cas ;
- depuis `data/annotation/metadata_cas.jsonl` : symptomes et antecedents,
  extraits par LLM (Claude) cas par cas, sans relecture humaine.

Sortie : `hf_dataset/data/{train,validation,test}.parquet`, valides avec
`src/triage/schemas/sft_hf.py`.

    python -m triage.data.export_hf_dataset
"""

import json
import re
from pathlib import Path

import pandas as pd

from triage.constants import DATA_DIR, SPLITS
from triage.schemas.sft_hf import CONSTANTES, LICENCES, sft_hf_schema

SFT_PATH = DATA_DIR / "processed" / "sft_avec_priorite.csv"
CORPUS_PATH = DATA_DIR / "interim" / "corpus_annotated.csv"
LLM_METADATA_PATH = DATA_DIR / "annotation" / "metadata_cas.jsonl"
OUTPUT_DIR = Path("hf_dataset") / "data"
COLUMNS = list(sft_hf_schema.columns)

# Specialites MediQAl ecrites de deux facons dans la source.
SPECIALITE_ALIASES = {"Hématologie": "Hematology"}

# Entre le libelle et la valeur : quelques mots ("est mesuree a", " = ", " : "),
# sans chiffre ni fin de phrase, pour ne pas rattacher un nombre d'une autre phrase.
GAP = r"[^\d\n.;!?]{0,40}?"
NUM = r"(\d{1,3}(?:[.,]\d)?)"
PATTERNS = {
    "fc": [
        rf"\b(?:FC|fr[ée]quence cardiaque|pouls|rythme cardiaque){GAP}(\d{{2,3}})(?![\d,.]\d)(?!\s*/\s*\d)",
        rf"\b(?:tachycardie|bradycardie){GAP}(\d{{2,3}})\s*(?:/\s*mi?n|bpm|battements)",
    ],
    "pa": [rf"\b(?:PA|TA|pression art[ée]rielle|tension art[ée]rielle){GAP}(\d{{2,3}})\s*[/.]\s*(\d{{2,3}})(?!\d)"],
    "fr": [
        rf"\b(?-i:FR)\b{GAP}(\d{{1,3}})(?![\d,.]\d)",
        rf"\b(?:fr[ée]quence respiratoire|respiratoire|polypn[ée]e){GAP}(\d{{1,3}})\s*(?:/\s*mi?n|cycles)",
    ],
    "spo2": [
        rf"\b(?:SpO2|SaO2|Sp02|Sat\s*O2|satur(?:ation)?\b(?:\s+en\s+(?:O2|oxyg[èe]ne))?)(?!\s+de\s+la\s+transf){GAP}(\d{{2,3}})\s*%"
    ],
    # "T" seul (pas "TGP", "Temoin", "Thrombocytes"), et pas les "30 °C SFBC" des dosages enzymatiques.
    "temperature": [
        rf"(?:\b(?-i:T)(?!\w)\s*°?|temp[ée]rature|fi[èe]vre|f[ée]brile|apyr[ée]tique){GAP}(\d{{2}})\s*(?:[,.°]\s*(\d))?\s*°?\s*C?\b"
    ],
    "glasgow": [rf"\b(?:Glasgow|GCS|GSC){GAP}(\d{{1,2}})(?!\d)"],
}
PATTERNS = {key: [re.compile(p, re.IGNORECASE) for p in pats] for key, pats in PATTERNS.items()}

AGE_RE = re.compile(r"(?<![\d,.])(\d{1,3})\s*(ans|mois|semaines|jours)\b", re.IGNORECASE)
# "depuis 20 ans", "il y a 3 ans", "a l'age de 4 ans" : une duree, pas l'age du patient.
NOT_AGE_RE = re.compile(
    r"(depuis|il y a|pendant|durant|l'âge de|après|avant|dans|tous les|toutes les|"
    r"enceinte de|prématurée? de|terme de|née? à)\s*$",
    re.IGNORECASE,
)
# "23 semaines de grossesse", "32 SA" : un terme de grossesse, pas un age.
GESTATION_RE = re.compile(r"\s*(de grossesse|d'aménorrhée|de gestation|SA\b)", re.IGNORECASE)
NEWBORN_RE = re.compile(r"\b(nouveau[- ]n[ée]e?s?|nouveau-né)\b", re.IGNORECASE)
# Seuls les mots qui disent le sexe : "un patient", "un nouveau-ne" ne sont que du masculin grammatical.
MALE_RE = re.compile(r"\b(homme|monsieur|garçon|mr\b|m\. ?x\b)", re.IGNORECASE)
FEMALE_RE = re.compile(
    r"\b(femme|madame|mademoiselle|mme\b|mlle\b|fille|fillette|patiente|adolescente|enceinte)\b", re.IGNORECASE
)


def extract_constantes(text: str) -> dict:
    values = {key: None for key in CONSTANTES}
    for key, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                if key == "pa":
                    pas, pad = int(match.group(1)), int(match.group(2))
                    lo_s, hi_s = CONSTANTES["pas"]
                    lo_d, hi_d = CONSTANTES["pad"]
                    if lo_s <= pas <= hi_s and lo_d <= pad <= hi_d and pad < pas:
                        values["pas"], values["pad"] = pas, pad
                        break
                    continue
                if key == "temperature":
                    value = float(match.group(1)) + (int(match.group(2)) / 10 if match.group(2) else 0)
                else:
                    value = float(match.group(1).replace(",", "."))
                lo, hi = CONSTANTES[key]
                if lo <= value <= hi:
                    values[key] = value if key == "temperature" else int(value)
                    break
            if values.get(key) is not None or (key == "pa" and values["pas"] is not None):
                break
    return values


def extract_age(text: str):
    for match in AGE_RE.finditer(text):
        if NOT_AGE_RE.search(text[max(0, match.start() - 20) : match.start()]) or GESTATION_RE.match(text, match.end()):
            continue
        value, unit = int(match.group(1)), match.group(2).lower()
        age = {"ans": value, "mois": value / 12, "semaines": value / 52, "jours": value / 365}[unit]
        if age <= 120:
            return round(age, 2)
    return 0.0 if NEWBORN_RE.search(text) else None


def extract_sexe(text: str):
    male, female = MALE_RE.search(text), FEMALE_RE.search(text)
    if male and (not female or male.start() < female.start()):
        return "M"
    if female:
        return "F"
    return None


def load_llm_metadata() -> dict:
    metadata = {}
    with open(LLM_METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            metadata[row["case_id"]] = (row["symptomes"], row["antecedents"])
    return metadata


def build() -> pd.DataFrame:
    sft = pd.read_csv(SFT_PATH)
    corpus = pd.read_csv(CORPUS_PATH)[
        ["id", "case_id", "contexte_patient", "specialite", "type_question", "niveau_confiance"]
    ]
    frame = sft.merge(corpus, on="id", how="left", validate="one_to_one")
    if frame["niveau_confiance"].isna().any():
        raise ValueError("lignes SFT absentes du corpus annote")

    # MedQuAD : le champ "specialite" du corpus porte en fait le type de question (qtype).
    medquad = frame["source"] == "medquad"
    frame.loc[medquad, "type_question"] = frame.loc[medquad, "specialite"]
    frame.loc[medquad, "specialite"] = None
    frame["specialite"] = frame["specialite"].replace(SPECIALITE_ALIASES)
    frame["licence"] = frame["source"].map(LICENCES)

    texts = frame["contexte_patient"].fillna("")
    frame["constantes"] = texts.map(extract_constantes)
    frame["age_ans"] = texts.map(lambda t: extract_age(t) if t else None).astype(float)
    frame["sexe"] = texts.map(lambda t: extract_sexe(t) if t else None)

    llm = load_llm_metadata()
    cases = set(frame["case_id"].dropna())
    missing = cases - set(llm)
    if missing:
        raise ValueError(f"{len(missing)} cas sans symptomes/antecedents dans {LLM_METADATA_PATH}")
    frame["symptomes"] = frame["case_id"].map(lambda c: llm[c][0] if isinstance(c, str) else None)
    frame["antecedents"] = frame["case_id"].map(lambda c: llm[c][1] if isinstance(c, str) else None)

    for column in ["case_id", "specialite", "priorite", "sexe"]:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def main() -> None:
    frame = build()
    sft_hf_schema.validate(frame[COLUMNS])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        part = frame.loc[frame["split"] == split, COLUMNS].reset_index(drop=True)
        part.to_parquet(OUTPUT_DIR / f"{split}.parquet", index=False)
        print(f"{split}: {len(part)} lignes")

    cases = frame.dropna(subset=["case_id"]).drop_duplicates("case_id")
    constantes = pd.DataFrame(cases["constantes"].tolist())
    print(f"\nCouverture sur {len(cases)} cas cliniques uniques :")
    print(
        f"  age {cases['age_ans'].notna().mean():.0%}, sexe {cases['sexe'].notna().mean():.0%}, "
        f"symptomes non vides {cases['symptomes'].map(len).gt(0).mean():.0%}, "
        f"antecedents non vides {cases['antecedents'].map(len).gt(0).mean():.0%}"
    )
    print("  constantes : " + ", ".join(f"{k} {constantes[k].notna().mean():.0%}" for k in CONSTANTES))


if __name__ == "__main__":
    main()
