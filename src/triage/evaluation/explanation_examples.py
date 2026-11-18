"""Exemples representatifs d'explications SFT vs DPO v2, avec la note du juge (sans GPU).

Pour chaque niveau annote (P1, P2, P3), un cas ou le juge prefere le SFT, un
ou il prefere la DPO, et une egalite : 9 exemples au plus, choisis par ordre
d'item (deterministe), pas pour leur qualite. Necessite les sorties de
evaluate_explanations et les notes de blind_judgment (--set explications).

    python -m triage.evaluation.explanation_examples

Sortie : outputs/explanation_eval/exemples_explications.md
"""

import json

import pandas as pd

from triage.constants import LABELS, SHORT
from triage.evaluation.blind_judgment import MODEL_A, MODEL_B, SETS, paths
from triage.evaluation.evaluate_explanations import OUTPUT_DIR

EXCERPT = 500
NAMES = {MODEL_A: "SFT + triage", MODEL_B: "DPO v2"}


def load() -> pd.DataFrame:
    files = paths("explications")
    criteria = SETS["explications"]["criteria"]
    key = json.loads(files["key"].read_text(encoding="utf-8"))
    outputs = {m: pd.read_csv(OUTPUT_DIR / f"scores_{m}.csv").set_index("id") for m in (MODEL_A, MODEL_B)}
    rows = []
    for line in files["notes"].read_text(encoding="utf-8").splitlines():
        note = json.loads(line)
        item = key[str(note["item"])]
        side = {item["A"]: "A", item["B"]: "B"}
        sft, dpo = outputs[MODEL_A].loc[item["id"]], outputs[MODEL_B].loc[item["id"]]
        winner = item[note["preference"]] if note["preference"] in ("A", "B") else "egalite"
        rows.append(
            {
                "item": note["item"],
                "id": item["id"],
                "niveau": sft["priorite"],
                "cas": sft["cas"],
                "sft": sft["generated"],
                "dpo": dpo["generated"],
                "notes_sft": {c: note[side[MODEL_A]][c] for c in criteria},
                "notes_dpo": {c: note[side[MODEL_B]][c] for c in criteria},
                "prefere": winner,
                "commentaire": note.get("commentaire", ""),
                "flags_sft": flags(sft),
                "flags_dpo": flags(dpo),
            }
        )
    return pd.DataFrame(rows)


def flags(row: pd.Series) -> str:
    found = [
        name
        for name, column in [
            ("chiffre absent du cas", "chiffre_invente"),
            ("formulation a risque", "formulation_risquee"),
            ("vocabulaire d'un autre niveau", "niveau_contredit"),
            ("coupee", "coupee"),
        ]
        if row[column] == 1
    ]
    return ", ".join(found) or "aucun"


def pick(frame: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    for level in LABELS:
        for winner in [MODEL_A, MODEL_B, "egalite"]:
            match = frame[(frame["niveau"] == level) & (frame["prefere"] == winner)].sort_values("item")
            if len(match):
                chosen.append(match.iloc[0])
    return pd.DataFrame(chosen)


def quote(text) -> str:
    return "\n".join("> " + line for line in str(text).strip().splitlines()) or "> (vide)"


def main() -> None:
    frame = load()
    examples = pick(frame)
    lines = [
        "# Exemples d'explications du niveau : SFT contre DPO v2",
        "",
        "Cas du test, question de l'API, niveau annote force en tete (les deux modeles expliquent le meme niveau). "
        "Pour chaque niveau : un cas ou le juge prefere le SFT, un ou il prefere la DPO, une egalite, pris dans "
        "l'ordre des items (pas choisis pour leur qualite). Notes de 1 a 5 (coherence / ancrage / exactitude / "
        "securite / clarte). Juge : LLM, sans clinicien.",
        "",
    ]
    for row in examples.itertuples():
        winner = NAMES.get(row.prefere, "egalite")
        cas = str(row.cas)
        lines += [
            f"## {SHORT[row.niveau]} - item {row.item} - prefere : {winner}",
            "",
            f"**Cas** ({row.id}) : {cas[:EXCERPT].replace(chr(10), ' ')}{'...' if len(cas) > EXCERPT else ''}",
            "",
            f"**SFT + triage** ({' / '.join(str(v) for v in row.notes_sft.values())} ; signaux : {row.flags_sft})",
            "",
            quote(row.sft),
            "",
            f"**DPO v2** ({' / '.join(str(v) for v in row.notes_dpo.values())} ; signaux : {row.flags_dpo})",
            "",
            quote(row.dpo),
            "",
            f"**Commentaire du juge** : {row.commentaire}",
            "",
        ]
    path = OUTPUT_DIR / "exemples_explications.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(examples)} exemples -> {path}")


if __name__ == "__main__":
    main()
