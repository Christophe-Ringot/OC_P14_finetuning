"""Jugement en aveugle de deux modeles sur les memes sorties (grille : docs/evaluation_reponses.md).

Deux jeux :
- `reponses` : reponses aux questions du dataset (evaluate_responses) ;
- `explications` : explications du niveau de priorite (evaluate_explanations).

    python -m triage.evaluation.blind_judgment prepare [--set explications]   # paires A/B anonymes + cle
    (le juge remplit <prefixe>_notes.jsonl)
    python -m triage.evaluation.blind_judgment summarize [--set explications] # levee de l'aveugle et synthese

Format d'une note (une ligne JSON par item) :
{"item": 3, "A": {"<critere>": 4, ...}, "B": {...}, "preference": "A" | "B" | "egalite", "commentaire": "..."}
"""

import argparse
import json
import random
from math import comb

import pandas as pd

from triage.constants import OUTPUTS_DIR, SHORT
from triage.utils.markdown import markdown_table

MODEL_A, MODEL_B = "4_sft_medical_triage", "6_sft_triage_dpo_qualite"
SEED = 7
SETS = {
    "reponses": {
        "dir": OUTPUTS_DIR / "response_eval",
        "inputs": "generations_{}.csv",
        "prefix": "jugement_aveugle",
        "criteria": ["pertinence", "clarte", "securite", "conformite"],
        "title": "Jugement en aveugle des reponses : SFT vs SFT + DPO",
        "what": "cas cliniques annotes de l'echantillon fixe du test (question du dataset)",
    },
    "explications": {
        "dir": OUTPUTS_DIR / "explanation_eval",
        "inputs": "explications_{}.csv",
        "prefix": "jugement_explications",
        "criteria": ["coherence", "ancrage", "exactitude", "securite", "clarte"],
        "title": "Jugement en aveugle des explications du niveau : SFT vs SFT + DPO",
        "what": "cas du test, question de l'API, niveau annote force",
    },
}


def paths(name: str) -> dict:
    conf = SETS[name]
    base = conf["dir"] / conf["prefix"]
    return {
        "items": base.with_name(conf["prefix"] + "_items.jsonl"),
        "key": base.with_name(conf["prefix"] + "_cle.json"),
        "notes": base.with_name(conf["prefix"] + "_notes.jsonl"),
        "report": base.with_name(conf["prefix"] + ".md"),
    }


def prepare(name: str) -> None:
    conf, files = SETS[name], paths(name)
    a = pd.read_csv(conf["dir"] / conf["inputs"].format(MODEL_A))
    b = pd.read_csv(conf["dir"] / conf["inputs"].format(MODEL_B)).set_index("id")
    a = a[a["priorite"].notna()].reset_index(drop=True)  # cas cliniques annotes seulement
    rng = random.Random(SEED)
    key = {}
    with open(files["items"], "w", encoding="utf-8") as f:
        for i, row in a.iterrows():
            pair = [(MODEL_A, str(row["generated"])), (MODEL_B, str(b.loc[row["id"], "generated"]))]
            rng.shuffle(pair)
            key[i] = {"id": row["id"], "A": pair[0][0], "B": pair[1][0]}
            item = {"item": i, "prompt": row["prompt"], "A": pair[0][1], "B": pair[1][1]}
            if "reponse" in row:
                item["reference"] = row["reponse"]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    files["key"].write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(key)} items -> {files['items']} (cle : {files['key']})")


def sign_test(wins: int, losses: int) -> float:
    """p-valeur bilaterale du test du signe (egalites exclues)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(name: str) -> None:
    conf, files = SETS[name], paths(name)
    criteria = conf["criteria"]
    key = json.loads(files["key"].read_text(encoding="utf-8"))
    levels = {}
    inputs = conf["dir"] / conf["inputs"].format(MODEL_A)
    if inputs.exists():
        levels = pd.read_csv(inputs).set_index("id")["priorite"].map(SHORT).to_dict()
    rows = []
    for line in files["notes"].read_text(encoding="utf-8").splitlines():
        note = json.loads(line)
        item = key[str(note["item"])]
        for side in ["A", "B"]:
            rows.append(
                {
                    "item": note["item"],
                    "niveau": levels.get(item["id"], "?"),
                    "modele": item[side],
                    **{c: note[side][c] for c in criteria},
                    "prefere": note["preference"] == side,
                    "egalite": note["preference"] == "egalite",
                }
            )
    frame = pd.DataFrame(rows)
    means = frame.groupby("modele")[criteria].mean()
    wins = frame.groupby("modele")["prefere"].sum().astype(int)
    ties = int(frame["egalite"].sum() // 2)
    n = frame["item"].nunique()
    table = means.assign(moyenne=means.mean(axis=1), prefere=wins).rename_axis("modele")
    by_level = frame.pivot_table(index="niveau", columns="modele", values="prefere", aggfunc="sum").astype(int)
    by_level["egalites"] = frame[frame["modele"] == MODEL_A].groupby("niveau")["egalite"].sum().astype(int)
    p_value = sign_test(int(wins.get(MODEL_A, 0)), int(wins.get(MODEL_B, 0)))
    lines = [
        f"# {conf['title']}",
        "",
        f"{n} {conf['what']}. Sorties presentees en ordre aleatoire (A/B) sans le nom du modele. "
        "Grille : docs/evaluation_reponses.md. Notes de 1 a 5.",
        "",
        markdown_table(table, digits=2),
        "",
        f"Egalites : {ties} / {n}. Test du signe sur les cas departages : p = {p_value:.3f}.",
        "",
        "Preferences par niveau annote :",
        "",
        markdown_table(by_level),
    ]
    files["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "summarize"])
    parser.add_argument("--set", dest="name", choices=list(SETS), default="reponses")
    args = parser.parse_args()
    {"prepare": prepare, "summarize": summarize}[args.action](args.name)


if __name__ == "__main__":
    main()
