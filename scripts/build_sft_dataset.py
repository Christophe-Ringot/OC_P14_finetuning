import string
from pathlib import Path
import pandas as pd
from schemas.sft_schema import sft_schema
from scripts.text_utils import clean_text

RANDOM_STATE = 42
FR_TARGET = 2500
EN_TARGET = 2500

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "processed" / "sft_dataset.csv"

SOURCE_LICENSES = {
    "frenchmedmcqa": "Apache 2.0 - qanastek/frenchmedmcqa (sous-ensemble mono-reponse)",
    "mediqal": "CC-BY-4.0 - ANR-MALADES/MediQAl (sous-ensemble MCQM, reponses multiples)",
    "medquad": "CC-BY-4.0 - source NIH (abachaa/MedQuAD), mirroir keivalya/MedQuad-MedicalQnADataset",
}

ANSWER_COLS = ["answer_a", "answer_b", "answer_c", "answer_d", "answer_e"]
LETTERS = list(string.ascii_uppercase[:5])


def proportional_targets(sizes: dict, total: int) -> dict:
    """Repartit `total` entre les cles de `sizes` au prorata de leur poids,
    en ajustant le dernier bucket pour que la somme soit exacte."""
    grand_total = sum(sizes.values())
    targets = {k: round(v / grand_total * total) for k, v in sizes.items()}
    drift = total - sum(targets.values())
    last_key = list(sizes.keys())[-1]
    targets[last_key] += drift
    return targets


def build_frenchmedmcqa() -> pd.DataFrame:
    rows = []
    for split, filename in [
        ("train", "train-00000-of-00001.parquet"),
        ("validation", "validation-00000-of-00001.parquet"),
        ("test", "test-00000-of-00001.parquet"),
    ]:
        df = pd.read_parquet(DATA_DIR / "frenchmedmcqa" / filename)
        for _, row in df.iterrows():
            letter_idx = int(row["correct_answers"])
            letter = LETTERS[letter_idx]
            answer_text = clean_text(row[ANSWER_COLS[letter_idx]])
            options = "\n".join(
                f"{LETTERS[i]}) {clean_text(row[col])}" for i, col in enumerate(ANSWER_COLS)
            )
            instruction = (
                f"Question : {clean_text(row['question'])}\n{options}\n"
                "Quelle est la bonne reponse ?"
            )
            response = f"La reponse correcte est {letter}) {answer_text}."
            rows.append(
                {
                    "source_dataset": "frenchmedmcqa",
                    "language": "fr",
                    "split": split,
                    "task_type": "qcm",
                    "instruction": instruction,
                    "response": response,
                    "medical_subject": None,
                    "symptoms": None,
                    "antecedents": None,
                    "vital_signs": None,
                    "confidence_level": "high",
                    "license": SOURCE_LICENSES["frenchmedmcqa"],
                    "original_id": str(row["id"]),
                }
            )
    return pd.DataFrame(rows)


def build_mediqal(target_total: int) -> pd.DataFrame:
    import json

    native_frames = {}
    for split in ["train", "validation", "test"]:
        with open(DATA_DIR / "mediqal" / f"{split}.json", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        native_frames[split] = pd.DataFrame(records)

    sizes = {split: len(df) for split, df in native_frames.items()}
    targets = proportional_targets(sizes, target_total)

    rows = []
    for split, df in native_frames.items():
        n = min(targets[split], len(df))
        sample = df.sample(n=n, random_state=RANDOM_STATE)
        for _, row in sample.iterrows():
            letters = [letter.strip() for letter in row["correct_answers"].split(",")]
            col_map = {"A": "answer_a", "B": "answer_b", "C": "answer_c", "D": "answer_d", "E": "answer_e"}
            options = "\n".join(
                f"{LETTERS[i]}) {clean_text(row[col])}" for i, col in enumerate(ANSWER_COLS)
            )
            instruction = (
                f"Cas clinique : {clean_text(row['clinical_case'])}\n\n"
                f"Question : {clean_text(row['question'])}\n{options}\n"
                "Quelle(s) proposition(s) est/sont correcte(s) ?"
            )
            correct_texts = [f"{letter}) {clean_text(row[col_map[letter]])}" for letter in letters]
            if len(correct_texts) == 1:
                response = f"La reponse correcte est {correct_texts[0]}."
            else:
                response = "Les reponses correctes sont : " + " ; ".join(correct_texts) + "."
            rows.append(
                {
                    "source_dataset": "mediqal",
                    "language": "fr",
                    "split": split,
                    "task_type": "clinical_case_qcm",
                    "instruction": instruction,
                    "response": response,
                    "medical_subject": row.get("medical_subject"),
                    "symptoms": None,
                    "antecedents": None,
                    "vital_signs": None,
                    "confidence_level": "high",
                    "license": SOURCE_LICENSES["mediqal"],
                    "original_id": str(row["id"]),
                }
            )
    return pd.DataFrame(rows)


def build_medquad(target_total: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "medquad" / "medDataset_processed.csv")
    df = df.reset_index().rename(columns={"index": "original_row_index"})

    sizes = df["qtype"].value_counts().to_dict()
    targets = proportional_targets(sizes, target_total)

    sampled_parts = []
    for qtype, group in df.groupby("qtype"):
        n = min(targets.get(qtype, 0), len(group))
        if n > 0:
            sampled_parts.append(group.sample(n=n, random_state=RANDOM_STATE))
    sample = pd.concat(sampled_parts, ignore_index=True)

    sample = sample.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    n_total = len(sample)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    split_labels = ["train"] * n_train + ["validation"] * n_val + ["test"] * (n_total - n_train - n_val)
    sample["split"] = split_labels

    rows = []
    for _, row in sample.iterrows():
        rows.append(
            {
                "source_dataset": "medquad",
                "language": "en",
                "split": row["split"],
                "task_type": "qa",
                "instruction": clean_text(row["Question"]),
                "response": clean_text(row["Answer"]),
                "medical_subject": row["qtype"],
                "symptoms": None,
                "antecedents": None,
                "vital_signs": None,
                "confidence_level": "high",
                "license": SOURCE_LICENSES["medquad"],
                "original_id": str(row["original_row_index"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df_fmcqa = build_frenchmedmcqa()
    fr_remainder = max(FR_TARGET - len(df_fmcqa), 0)
    df_mediqal = build_mediqal(fr_remainder)
    df_medquad = build_medquad(EN_TARGET)

    combined = pd.concat([df_fmcqa, df_mediqal, df_medquad], ignore_index=True)
    combined.insert(
        0,
        "id",
        [f"{row.source_dataset}_{row.split}_{i:06d}" for i, row in enumerate(combined.itertuples())],
    )

    validated = sft_schema.validate(combined, lazy=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"SFT dataset : {len(validated)} lignes ecrites dans {OUTPUT_PATH}")
    print(validated.groupby(["source_dataset", "language", "split"]).size())


if __name__ == "__main__":
    main()
