"""Jugement en aveugle de deux modeles sur les memes reponses (grille : docs/evaluation_reponses.md).

    python -m triage.evaluation.blind_judgment prepare   # paires A/B anonymisees + cle
    (le juge remplit outputs/response_eval/jugement_aveugle_notes.jsonl)
    python -m triage.evaluation.blind_judgment summarize # levee de l'aveugle et synthese

Format d'une note (une ligne JSON par item) :
{"item": 3, "A": {"pertinence": 4, "clarte": 5, "securite": 5, "conformite": 4},
 "B": {...}, "preference": "A" | "B" | "egalite", "commentaire": "..."}
"""

import argparse
import json
import random

import pandas as pd

from triage.constants import OUTPUTS_DIR
from triage.utils.markdown import markdown_table

OUTPUT_DIR = OUTPUTS_DIR / "response_eval"

MODEL_A, MODEL_B = "4_sft_medical_triage", "6_sft_triage_dpo_qualite"
CRITERIA = ["pertinence", "clarte", "securite", "conformite"]
ITEMS = OUTPUT_DIR / "jugement_aveugle_items.jsonl"
KEY = OUTPUT_DIR / "jugement_aveugle_cle.json"
NOTES = OUTPUT_DIR / "jugement_aveugle_notes.jsonl"
REPORT = OUTPUT_DIR / "jugement_aveugle.md"
SEED = 7


def prepare() -> None:
    a = pd.read_csv(OUTPUT_DIR / f"generations_{MODEL_A}.csv")
    b = pd.read_csv(OUTPUT_DIR / f"generations_{MODEL_B}.csv").set_index("id")
    a = a[a["priorite"].notna()].reset_index(drop=True)  # cas cliniques annotes seulement
    rng = random.Random(SEED)
    key = {}
    with open(ITEMS, "w", encoding="utf-8") as f:
        for i, row in a.iterrows():
            pair = [(MODEL_A, str(row["generated"])), (MODEL_B, str(b.loc[row["id"], "generated"]))]
            rng.shuffle(pair)
            key[i] = {"id": row["id"], "A": pair[0][0], "B": pair[1][0]}
            f.write(
                json.dumps(
                    {"item": i, "prompt": row["prompt"], "reference": row["reponse"], "A": pair[0][1], "B": pair[1][1]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    KEY.write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(key)} items -> {ITEMS} (cle : {KEY})")


def summarize() -> None:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    rows = []
    for line in NOTES.read_text(encoding="utf-8").splitlines():
        note = json.loads(line)
        item = key[str(note["item"])]
        for side in ["A", "B"]:
            rows.append(
                {
                    "item": note["item"],
                    "modele": item[side],
                    **note[side],
                    "prefere": note["preference"] == side,
                    "egalite": note["preference"] == "egalite",
                }
            )
    frame = pd.DataFrame(rows)
    means = frame.groupby("modele")[CRITERIA].mean()
    wins = frame.groupby("modele")["prefere"].sum()
    ties = int(frame["egalite"].sum() // 2)
    n = frame["item"].nunique()
    table = means.assign(moyenne=means.mean(axis=1), prefere=wins.astype(int)).rename_axis("modele")
    lines = [
        "# Jugement en aveugle des reponses : SFT vs SFT + DPO",
        "",
        f"{n} cas cliniques annotes de l'echantillon fixe du test, reponses presentees en ordre aleatoire "
        "(A/B) sans le nom du modele. Grille : docs/evaluation_reponses.md. Notes de 1 a 5.",
        "",
        markdown_table(table, digits=2),
        "",
        f"Egalites : {ties} / {n}.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "summarize"])
    {"prepare": prepare, "summarize": summarize}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
