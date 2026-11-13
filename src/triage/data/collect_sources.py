import json

import pandas as pd

from triage.constants import DATA_DIR

MEDIQAL_OEQ_PATH = DATA_DIR / "mediqal_oeq" / "oeq.parquet"
INVENTORY_PATH = DATA_DIR / "raw_inventory.json"
SOURCES = {
    "mediqal_oeq": {"usage": "sft", "langue": "fr", "hf_id": "ANR-MALADES/MediQAl (config oeq)"},
    "frenchmedmcqa": {"usage": "sft", "langue": "fr", "hf_id": "qanastek/frenchmedmcqa"},
    "medquad": {"usage": "sft", "langue": "en", "hf_id": "keivalya/MedQuad-MedicalQnADataset"},
    "ultramedical": {"usage": "dpo", "langue": "en", "hf_id": "TsinghuaC3I/UltraMedical-Preference"},
}


def download_mediqal_oeq() -> None:
    if MEDIQAL_OEQ_PATH.exists():
        print(f"MediQAl OEQ deja present : {MEDIQAL_OEQ_PATH}")
        return
    from datasets import load_dataset

    # La config oeq n'est publiee qu'avec un split "test" : c'est le corpus
    # complet, notre propre decoupage sera fait a l'etape 5.
    dataset = load_dataset("ANR-MALADES/MediQAl", "oeq")
    frame = pd.concat([split.to_pandas() for split in dataset.values()], ignore_index=True)
    MEDIQAL_OEQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(MEDIQAL_OEQ_PATH, index=False)
    print(f"MediQAl OEQ telecharge : {len(frame)} lignes -> {MEDIQAL_OEQ_PATH}")


def load_raw(source: str) -> pd.DataFrame:
    if source == "mediqal_oeq":
        return pd.read_parquet(MEDIQAL_OEQ_PATH)
    if source == "frenchmedmcqa":
        return pd.concat(
            [pd.read_parquet(path) for path in sorted((DATA_DIR / "frenchmedmcqa").glob("*.parquet"))],
            ignore_index=True,
        )
    if source == "medquad":
        return pd.read_csv(DATA_DIR / "medquad" / "medDataset_processed.csv")
    if source == "ultramedical":
        # Tableaux JSON classiques (pas du JSON Lines), comme dans build_dpo_dataset.
        frames = []
        for path in sorted((DATA_DIR / "ultramedical").glob("*.json")):
            with open(path, encoding="utf-8") as f:
                frames.append(pd.DataFrame(json.load(f)))
        return pd.concat(frames, ignore_index=True)
    raise ValueError(source)


def main() -> None:
    download_mediqal_oeq()

    inventory = {}
    for source, meta in SOURCES.items():
        frame = load_raw(source)
        inventory[source] = {**meta, "lignes": len(frame), "colonnes": frame.columns.tolist()}
        if source == "mediqal_oeq":
            inventory[source]["cas_cliniques_uniques"] = int(frame["clinical_case"].nunique())
            inventory[source]["questions_sans_cas"] = int(frame["clinical_case"].isna().sum())

    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    for source, info in inventory.items():
        extra = ""
        if "cas_cliniques_uniques" in info:
            extra = f" | {info['cas_cliniques_uniques']} cas uniques, {info['questions_sans_cas']} questions sans cas"
        print(f"{source:<14} {info['usage']:<4} {info['langue']}  {info['lignes']:>6} lignes{extra}")
    print(f"Inventaire -> {INVENTORY_PATH}")


if __name__ == "__main__":
    main()
