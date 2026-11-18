"""Qualite des explications du niveau de priorite (SFT vs SFT + DPO).

evaluate_responses mesure les reponses aux questions d'examen du dataset. Ici on
evalue ce que l'agent produit vraiment : l'explication d'un niveau de triage,
avec la question de l'API ("Quel est le niveau de priorité de ce patient et
pourquoi ?") et le niveau annote force en tete de la reponse. Les deux modeles
expliquent donc le meme niveau sur les memes cas : seule la qualite de
l'explication est comparee.

Echantillon : les 60 cas cliniques de l'echantillon fixe d'evaluate_responses
(20 par niveau P1 / P2 / P3, test uniquement).

Metriques automatiques (signaux a relire, pas des verdicts) :
- ancrage : part des symptomes du cas (metadonnees du dataset) cites ;
- chiffre_invente : nombre absent du cas ;
- formulation_risquee : rassurance a risque (safety_check) ;
- niveau_contredit : vocabulaire d'un autre niveau dans l'explication ;
- longueur, coupee, repetition, vide.

Puis jugement en aveugle (blind_judgment --set explications).

    python -m triage.evaluation.evaluate_explanations           # generation (GPU) + metriques
    python -m triage.evaluation.evaluate_explanations --score   # metriques seules (sans GPU)

Sorties : outputs/explanation_eval/explications_<modele>.csv, qualite_explications.md
"""

import argparse
import ast
import re

import pandas as pd

from triage.constants import OUTPUTS_DIR, SHORT, adapter_dir, build_prompt, level_prefix
from triage.evaluation.evaluate_responses import sample as response_sample
from triage.evaluation.metrics import repetition, words
from triage.evaluation.safety_check import NUMBER_RE, dangerous_language
from triage.utils import hf_dataset
from triage.utils.markdown import markdown_table

OUTPUT_DIR = OUTPUTS_DIR / "explanation_eval"
QUESTION = "Quel est le niveau de priorité de ce patient et pourquoi ?"  # celle de l'API
MODELS = {e: adapter_dir(e) for e in ["4_sft_medical_triage", "6_sft_triage_dpo_qualite"]}
MAX_NEW_TOKENS = 200  # comme l'API
CASE_RE = re.compile(r"### Cas clinique :\n(.*?)\n\n### Question :", re.DOTALL)
# Vocabulaire propre a chaque niveau : un terme d'un autre niveau signale une explication incoherente.
LEVEL_WORDS = {
    "urgence_maximale": re.compile(r"urgence (?:vitale|maximale|absolue)|pronostic vital|imm[ée]diat", re.I),
    "urgence_moderee": re.compile(r"urgence mod[ée]r[ée]e|sans d[ée]tresse vitale|dans les heures", re.I),
    "differee": re.compile(r"pas d'urgence|non urgent|diff[ée]r[ée]e|programm[ée]e?|consultation de suivi", re.I),
}
METRICS = [
    "ancrage",
    "chiffre_invente",
    "formulation_risquee",
    "niveau_contredit",
    "longueur_mots",
    "coupee",
    "repetition",
    "vide",
]


def cases() -> pd.DataFrame:
    """Un cas par ligne : cas clinique, niveau annote et symptomes (metadonnees du dataset)."""
    picked = response_sample()
    picked = picked[picked["priorite"].notna()]
    meta = hf_dataset.load_sft().set_index("id")["symptomes"]
    out = picked[["id", "case_id", "priorite"]].copy()
    out["cas"] = picked["prompt"].map(lambda p: CASE_RE.search(p).group(1))
    out["symptomes"] = out["id"].map(lambda i: list(meta.loc[i]) if meta.loc[i] is not None else [])
    out["prompt"] = [build_prompt(QUESTION, c) + level_prefix(level) for c, level in zip(out["cas"], out["priorite"])]
    return out.reset_index(drop=True)


def generate(frame: pd.DataFrame) -> None:
    from triage.evaluation.evaluate_responses import generate as run_model

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, adapter in MODELS.items():
        print(f"Generation {name} ({len(frame)} cas)")
        texts, truncated = run_model(adapter, frame["prompt"].tolist())
        frame.assign(generated=texts, truncated=truncated).to_csv(OUTPUT_DIR / f"explications_{name}.csv", index=False)


def anchoring(symptoms: list, text: str) -> float:
    """Part des symptomes du cas dont un mot significatif (> 4 lettres) apparait dans l'explication."""
    if not symptoms:
        return float("nan")
    vocabulary = set(words(text))
    cited = [any(w in vocabulary for w in words(s) if len(w) > 4) for s in symptoms]
    return sum(cited) / len(symptoms)


def contradicts(level: str, text: str) -> bool:
    return any(pattern.search(text) for other, pattern in LEVEL_WORDS.items() if other != level)


def score(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        text = row.generated if isinstance(row.generated, str) else ""
        symptoms = row.symptomes if isinstance(row.symptomes, list) else ast.literal_eval(row.symptomes)
        rows.append(
            {
                "ancrage": anchoring(symptoms, text),
                "chiffre_invente": float(bool(set(NUMBER_RE.findall(text)) - set(NUMBER_RE.findall(row.cas)))),
                "formulation_risquee": float(bool(dangerous_language(text))),
                "niveau_contredit": float(contradicts(row.priorite, text)),
                "longueur_mots": len(words(text)),
                "coupee": float(row.truncated),
                "repetition": repetition(text),
                "vide": float(len(words(text)) < 3),
            }
        )
    return pd.DataFrame(rows)


def report() -> pd.DataFrame:
    summary, by_level = {}, {}
    for name in MODELS:
        frame = pd.read_csv(OUTPUT_DIR / f"explications_{name}.csv")
        scored = pd.concat([frame, score(frame)], axis=1)
        scored.to_csv(OUTPUT_DIR / f"scores_{name}.csv", index=False)
        summary[name] = scored[METRICS].mean()
        by_level[name] = scored.groupby(scored["priorite"].map(SHORT))[
            ["ancrage", "chiffre_invente", "niveau_contredit"]
        ].mean()
    summary = pd.DataFrame(summary).T
    shown = summary.copy()
    percent = [m for m in METRICS if m != "longueur_mots"]
    shown[percent] *= 100
    levels = pd.concat(by_level, names=["modele", "niveau"]) * 100
    lines = [
        "# Qualite des explications du niveau de priorite",
        "",
        f"60 cas du test (20 par niveau), question de l'API (`{QUESTION}`), niveau annote force en tete : "
        f"les deux modeles expliquent le meme niveau. Decodage glouton, {MAX_NEW_TOKENS} tokens max (comme l'API). "
        "Valeurs en % (longueur en mots).",
        "",
        markdown_table(shown.rename_axis("modele")),
        "",
        "Par niveau annote (%) :",
        "",
        markdown_table(levels),
        "",
        "Lecture :",
        "- `ancrage` : part des symptomes du cas (metadonnees du dataset) repris dans l'explication ; une "
        "explication qui ne cite rien du cas est generique.",
        "- `chiffre_invente` : un nombre absent du cas (constante, dose, delai) ; signal a relire, un calcul juste "
        "est signale aussi.",
        "- `niveau_contredit` : vocabulaire d'un autre niveau (ex. 'pas d'urgence' sous un P1).",
        "- Le jugement en aveugle (`jugement_explications.md`) note la coherence et l'exactitude.",
    ]
    (OUTPUT_DIR / "qualite_explications.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((shown).round(1).to_string())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true", help="recalculer les metriques sans regenerer")
    args = parser.parse_args()
    if not args.score:
        generate(cases())
    report()


if __name__ == "__main__":
    main()
