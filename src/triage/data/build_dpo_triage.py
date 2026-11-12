"""Paires de preferences DPO construites sur les cas cliniques annotes.

Meme gabarit de prompt que le SFT, memes splits par groupe de cas (train ->
paires d'entrainement, validation -> paires d'evaluation ; le test n'est jamais
utilise). Deux types de paires, pour separer les deux dimensions du projet :

- `qualite` : chosen = reponse de reference ; rejected = reponse generee par le
  modele SFT lui-meme (on-policy) avec le MEME prefixe de niveau, force. Seul le
  contenu differe : la paire apprend la qualite de la reponse, pas le triage.
- `securite_triage` : meme reponse de reference, rejected = prefixe d'un niveau
  moins urgent (P1 -> P2, P2 -> P3). Seul le niveau differe : la paire penalise
  le sous-triage, jamais le sur-triage (cout asymetrique des erreurs).

Controles qualite (outputs/dpo/qualite_paires.md) : paires identiques ou
vides, paires trop proches (ROUGE-L >= 0.6, preference ambigue), biais de
longueur chosen / rejected, recouvrement de cas avec le test.

Sortie : data/processed/dpo_triage.csv (+ cache des generations).
"""
import argparse
from pathlib import Path
import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema
from triage.evaluation.evaluate_responses import LEVEL_RE, generate, rouge_l, strip_prefix, words
from triage.evaluation.evaluate_sft import LABELS, SHORT
from triage.utils import hf_dataset

SFT_ADAPTER = Path("outputs") / "sft_medical_triage"
OUTPUT_PATH = Path("data") / "processed" / "dpo_triage.csv"
GENERATIONS_PATH = Path("data") / "processed" / "dpo_generations_sft.csv"
REPORT_DIR = Path("outputs") / "dpo"
PREFIX = "[Niveau de priorité estimé : {}]\n\n"
SEED = 42
N_QUALITY = {"train": 800, "validation": 100}
SAFETY_SHARE = 0.35  # paires securite_triage, en proportion des paires qualite
MAX_SIMILARITY = 0.6

dpo_triage_schema = DataFrameSchema(
    columns={
        "id": Column(str, unique=True),
        "case_id": Column(str),
        "split": Column(str, Check.isin(["train", "validation"])),
        "type_paire": Column(str, Check.isin(["qualite", "securite_triage"])),
        "priorite": Column(str, Check.isin(LABELS)),
        "prompt": Column(str, Check.str_length(min_value=20)),
        "chosen": Column(str, Check.str_length(min_value=10)),
        "rejected": Column(str, Check.str_length(min_value=10)),
        "similarite_rouge_l": Column(float, Check.in_range(0, 1)),
    },
    checks=[Check(lambda df: df["chosen"] != df["rejected"], error="chosen == rejected")],
    strict=True,
)


def candidates() -> pd.DataFrame:
    frame = hf_dataset.load_sft(with_priority=True)
    frame = frame[frame["split"].isin(["train", "validation"]) & frame["priorite"].notna()]
    parts = [frame[frame["split"] == split].sample(n, random_state=SEED) for split, n in N_QUALITY.items()]
    return pd.concat(parts).reset_index(drop=True)


def generations(rows: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    """Reponses du modele SFT, avec le prefixe du bon niveau force dans le prompt."""
    if GENERATIONS_PATH.exists() and not refresh:
        cached = pd.read_csv(GENERATIONS_PATH)
        if set(rows["id"]) <= set(cached["id"]):
            return cached
    prompts = [p + PREFIX.format(level) for p, level in zip(rows["prompt"], rows["priorite"])]
    texts, truncated = generate(SFT_ADAPTER, prompts)
    out = rows[["id"]].assign(generated=texts, truncated=truncated)
    GENERATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GENERATIONS_PATH, index=False)
    return out


def quality_pairs(rows: pd.DataFrame, generated: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = rows.merge(generated, on="id")
    frame["generated"] = frame["generated"].fillna("").str.strip()
    frame["rejected"] = [PREFIX.format(level) + text for level, text in zip(frame["priorite"], frame["generated"])]
    frame["similarite_rouge_l"] = [rouge_l(strip_prefix(c), t) for c, t in zip(frame["reponse"], frame["generated"])]
    drops = {
        "generation vide (< 3 mots)": frame["generated"].map(lambda t: len(words(t)) < 3),
        f"trop proche de la reference (ROUGE-L >= {MAX_SIMILARITY})": frame["similarite_rouge_l"] >= MAX_SIMILARITY,
    }
    keep = ~pd.concat(drops, axis=1).any(axis=1)
    stats = {name: int(mask.sum()) for name, mask in drops.items()}
    stats["generation coupee (limite de tokens)"] = int(frame["truncated"].sum())
    frame = frame[keep]
    return frame.assign(type_paire="qualite", chosen=frame["reponse"])[
        ["id", "case_id", "split", "type_paire", "priorite", "prompt", "chosen", "rejected", "similarite_rouge_l"]], stats


def safety_pairs(n_quality: dict) -> pd.DataFrame:
    """Meme reponse, niveau un cran moins urgent dans rejected (P3 n'a pas de sous-triage)."""
    frame = hf_dataset.load_sft(with_priority=True)
    frame = frame[frame["split"].isin(["train", "validation"]) & frame["priorite"].isin(LABELS[:2])]
    parts = []
    for split, n in n_quality.items():
        part = frame[frame["split"] == split].sample(round(n * SAFETY_SHARE), random_state=SEED + 1)
        lower = part["priorite"].map({LABELS[0]: LABELS[1], LABELS[1]: LABELS[2]})
        rejected = [r.replace(PREFIX.format(p), PREFIX.format(l), 1) for r, p, l in zip(part["reponse"], part["priorite"], lower)]
        parts.append(part.assign(type_paire="securite_triage", chosen=part["reponse"], rejected=rejected, similarite_rouge_l=1.0))
    out = pd.concat(parts)
    out["id"] = out["id"] + "_securite"
    return out[["id", "case_id", "split", "type_paire", "priorite", "prompt", "chosen", "rejected", "similarite_rouge_l"]]


def report(pairs: pd.DataFrame, drop_stats: dict, n_candidates: int) -> None:
    sft = hf_dataset.load_sft()
    test_cases = set(sft.loc[sft["split"] == "test", "case_id"].dropna())
    leak = pairs["case_id"].isin(test_cases).sum()
    quality = pairs[pairs["type_paire"] == "qualite"]
    chosen_len = quality["chosen"].map(lambda t: len(words(strip_prefix(t))))
    rejected_len = quality["rejected"].map(lambda t: len(words(strip_prefix(t))))
    counts = pairs.groupby(["split", "type_paire", pairs["priorite"].map(SHORT)]).size().unstack(fill_value=0)
    lines = [
        "# Qualite des paires DPO (triage)",
        "",
        f"{len(pairs)} paires : " + ", ".join(f"{k} {v}" for k, v in pairs.groupby(['split', 'type_paire']).size().items()) + ".",
        "",
        "## Repartition",
        "",
        "| split / type | " + " | ".join(counts.columns) + " |\n|---|" + "---|" * len(counts.columns),
        *(f"| {s} / {t} | " + " | ".join(str(v) for v in row) + " |" for (s, t), row in counts.iterrows()),
        "",
        "## Controles automatiques",
        "",
        f"- Candidats qualite generes : {n_candidates} ; ecartes : "
        + "; ".join(f"{k} = {v}" for k, v in drop_stats.items() if "coupee" not in k) + ".",
        f"- Generations coupees a la limite de tokens (gardees, rejected legitime) : {drop_stats['generation coupee (limite de tokens)']}.",
        f"- Cas du test presents dans les paires : {leak} (attendu : 0).",
        f"- Paires identiques ou vides : 0 (verifie par le schema pandera).",
        f"- Similarite ROUGE-L chosen / rejected (qualite) : mediane {quality['similarite_rouge_l'].median():.2f}, "
        f"max {quality['similarite_rouge_l'].max():.2f}.",
        f"- Longueur en mots, mediane : chosen {chosen_len.median():.0f}, rejected {rejected_len.median():.0f} ; "
        f"rejected plus long que chosen dans {(rejected_len > chosen_len).mean() * 100:.0f} % des paires. "
        "Un ecart fort ferait apprendre la longueur plutot que le contenu : a surveiller dans la longueur des reponses apres DPO.",
        "- Prefixe de niveau identique dans chosen et rejected pour toutes les paires qualite : "
        f"{'oui' if (quality['chosen'].str.extract(LEVEL_RE)[0] == quality['rejected'].str.extract(LEVEL_RE)[0]).all() else 'NON'}.",
        "",
        "## Limites",
        "",
        "- chosen = reponse de reference (corrige d'examen MediQAl) : preference supposee, non validee par un clinicien. "
        "Une relecture humaine d'un echantillon reste a faire (docs/evaluation_reponses.md).",
        "- La reference peut etre plus courte que la generation sans etre meilleure sur tous les points.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "qualite_paires.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="regenerer les reponses du modele SFT")
    args = parser.parse_args()
    rows = candidates()
    generated = generations(rows, args.refresh)
    quality, stats = quality_pairs(rows, generated)
    pairs = pd.concat([quality, safety_pairs(N_QUALITY)]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    pairs = dpo_triage_schema.validate(pairs)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    report(pairs, stats, len(rows))
    print(pairs.groupby(["split", "type_paire"]).size())
    print(f"-> {OUTPUT_PATH}, rapport {REPORT_DIR / 'qualite_paires.md'}")


if __name__ == "__main__":
    main()
