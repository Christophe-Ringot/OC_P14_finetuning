import json
from pathlib import Path
import pandas as pd
from schemas.dpo_schema import dpo_schema
from scripts.text_utils import clean_text


def proportional_targets(sizes: dict, total: int) -> dict:
    """Repartit `total` entre les cles de `sizes` au prorata de leur poids,
    en ajustant le dernier bucket pour que la somme soit exacte."""
    grand_total = sum(sizes.values())
    targets = {k: round(v / grand_total * total) for k, v in sizes.items()}
    drift = total - sum(targets.values())
    last_key = list(sizes.keys())[-1]
    targets[last_key] += drift
    return targets


RANDOM_STATE = 42
TRAIN_SAMPLE_TARGET = 3000

DATA_DIR = Path("data") / "ultramedical"
OUTPUT_PATH = Path("data") / "processed" / "dpo_dataset.csv"

LICENSE = "MIT - TsinghuaC3I/UltraMedical-Preference"

SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}


def load_split(filename: str) -> pd.DataFrame:
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["source_dataset"] = df["prompt_id"].str.split(",").str[0]
    return df


def to_rows(df: pd.DataFrame, split_label: str) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        chosen_text = clean_text(row["chosen"][-1]["content"])
        rejected_text = clean_text(row["rejected"][-1]["content"])
        metadata = row.get("metadata") or {}
        chosen_score = (metadata.get("chosen") or {}).get("score")
        rejected_score = (metadata.get("rejected") or {}).get("score")
        rows.append(
            {
                "source_dataset": row["source_dataset"],
                "language": "en",
                "split": split_label,
                "label_type": row.get("label_type"),
                "prompt": clean_text(row["prompt"]),
                "chosen": chosen_text,
                "rejected": rejected_text,
                "chosen_score": float(chosen_score) if chosen_score is not None else None,
                "rejected_score": float(rejected_score) if rejected_score is not None else None,
                "confidence_level": "medium",
                "license": LICENSE,
            }
        )
    return rows


def main() -> None:
    df_train = load_split("train.json")
    df_dev = load_split("dev.json")
    df_test = load_split("test.json")

    sizes = df_train["source_dataset"].value_counts().to_dict()
    targets = proportional_targets(sizes, TRAIN_SAMPLE_TARGET)
    sampled_parts = []
    for source, group in df_train.groupby("source_dataset"):
        n = min(targets.get(source, 0), len(group))
        if n > 0:
            sampled_parts.append(group.sample(n=n, random_state=RANDOM_STATE))
    df_train_sampled = pd.concat(sampled_parts, ignore_index=True)

    rows = (
        to_rows(df_train_sampled, "train")
        + to_rows(df_dev, "validation")
        + to_rows(df_test, "test")
    )
    combined = pd.DataFrame(rows)
    combined.insert(
        0,
        "id",
        [f"ultramedical_{row.split}_{i:06d}" for i, row in enumerate(combined.itertuples())],
    )

    validated = dpo_schema.validate(combined, lazy=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"DPO dataset : {len(validated)} lignes ecrites dans {OUTPUT_PATH}")
    print(validated.groupby(["split"]).size())
    print(validated.groupby(["source_dataset"]).size())


if __name__ == "__main__":
    main()
