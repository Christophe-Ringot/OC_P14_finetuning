"""Audit des partitions train / validation / test (lecture seule).

Controle :
1. la repartition des lignes, cas, groupes et niveaux de priorite par split ;
2. les chevauchements exacts entre splits (cas, groupe, prompt, contexte, question) ;
3. les quasi-doublons de cas cliniques entre test et train (TF-IDF caracteres) ;
4. la presence d'indices de label dans les prompts ;
5. une baseline "plus proche voisin" : si copier le niveau du cas train le plus
   proche suffit a bien classer le test, la tache se resout par similarite.

Sorties : outputs/audit/audit_report.md et outputs/audit/near_duplicates.csv
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from triage.constants import DATA_DIR, LABELS, OUTPUTS_DIR, SHORT, SPLITS
from triage.utils.markdown import markdown_table

SFT_PATH = DATA_DIR / "processed" / "sft_avec_priorite.csv"
CORPUS_PATH = DATA_DIR / "interim" / "corpus_annotated.csv"
OUTPUT_DIR = OUTPUTS_DIR / "audit"
THRESHOLDS = [0.5, 0.7, 0.8, 0.9]
TOP_PAIRS = 30
LABEL_HINT_RE = re.compile(r"niveau de priorit|urgence_maximale|urgence_moderee|differee|\bP[123]\b", re.IGNORECASE)


def table(frame: pd.DataFrame) -> str:
    return markdown_table(frame, digits=None)


def load() -> pd.DataFrame:
    sft = pd.read_csv(SFT_PATH)
    corpus = pd.read_csv(CORPUS_PATH)[["id", "case_id", "contexte_patient", "instruction", "split"]]
    frame = sft.merge(corpus.rename(columns={"split": "split_corpus"}), on="id", how="left", validate="one_to_one")
    if frame["case_id"].isna().all():
        raise ValueError("Jointure corpus/SFT vide : verifier les id")
    return frame


def distribution(frame: pd.DataFrame) -> list:
    lines = ["## 1. Repartition par split", ""]
    summary = (
        frame.groupby("split")
        .agg(lignes=("id", "size"), cas=("case_id", "nunique"), groupes=("groupe", "nunique"))
        .reindex(SPLITS)
    )
    lines += [
        table(summary),
        "",
        "Lignes par source :",
        "",
        table(pd.crosstab(frame["source"], frame["split"])[SPLITS]),
        "",
    ]

    annotated = frame[frame["priorite"].notna()]
    by_question = pd.crosstab(annotated["priorite"].map(SHORT), annotated["split"])[SPLITS]
    cases = annotated.drop_duplicates("case_id")
    by_case = pd.crosstab(cases["priorite"].map(SHORT), cases["split"])[SPLITS]

    def pct(counts: pd.DataFrame) -> pd.DataFrame:
        return (counts / counts.sum() * 100).round(1)

    lines += [
        "Priorite par question (effectifs / %) :",
        "",
        table(by_question),
        "",
        table(pct(by_question)),
        "",
        "Priorite par cas clinique (effectifs / %) :",
        "",
        table(by_case),
        "",
        table(pct(by_case)),
        "",
    ]
    q_per_case = annotated.groupby(["split", "case_id"]).size().groupby("split").describe()[["mean", "min", "max"]]
    lines += ["Questions par cas clinique :", "", table(q_per_case.reindex(SPLITS).round(1)), ""]
    q_per_group = annotated.groupby(["split", "groupe"]).size().groupby("split").describe()[["count", "mean", "max"]]
    lines += ["Questions par groupe :", "", table(q_per_group.reindex(SPLITS).round(1)), ""]

    if "split_corpus" in frame and frame["split_corpus"].notna().any():
        lines += [
            "Split d'origine (corpus_annotated) vs split SFT :",
            "",
            table(pd.crosstab(frame["split_corpus"].fillna("NA"), frame["split"])),
            "",
        ]
    return lines


def exact_overlaps(frame: pd.DataFrame) -> list:
    lines = [
        "## 2. Chevauchements exacts entre splits",
        "",
        "Nombre de valeurs distinctes presentes dans plus d'un split.",
        "",
    ]
    rows = {}
    for key in ["case_id", "groupe", "prompt", "contexte_patient", "instruction", "reponse"]:
        values = frame.dropna(subset=[key])
        n_splits = values.groupby(key)["split"].nunique()
        shared = n_splits[n_splits > 1]
        pairs = {}
        for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]:
            sa = set(values.loc[values["split"] == a, key])
            sb = set(values.loc[values["split"] == b, key])
            pairs[f"{a}/{b}"] = len(sa & sb)
        rows[key] = {"valeurs_partagees": len(shared), **pairs}
    lines += [table(pd.DataFrame(rows).T), ""]
    lines += [
        "Note : `instruction` et `reponse` partagees peuvent etre des questions generiques "
        "(ex. 'Quel est le diagnostic ?') ; seules celles des cas cliniques sont preoccupantes.",
        "",
    ]

    cases = frame[frame["case_id"].notna()]
    shared_instr = cases.groupby("instruction")["split"].nunique()
    shared_instr = shared_instr[shared_instr > 1]
    lines += [f"Questions de cas cliniques identiques dans plusieurs splits : {len(shared_instr)}", ""]
    if len(shared_instr):
        examples = shared_instr.index.to_series().str.slice(0, 120).head(10)
        lines += [f"- `{e}`" for e in examples] + [""]
    return lines


def near_duplicates(frame: pd.DataFrame) -> tuple[list, list, pd.DataFrame]:
    """Sections 3 (quasi-doublons) et 5 (baseline plus proche voisin), et les paires test/validation - train."""
    cases = (
        frame[frame["case_id"].notna()]
        .groupby("case_id")
        .agg(
            split=("split", "first"),
            groupe=("groupe", "first"),
            priorite=("priorite", "first"),
            contexte=("contexte_patient", "first"),
            n_questions=("id", "size"),
        )
        .reset_index()
    )
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(cases["contexte"].fillna(""))
    train_mask = (cases["split"] == "train").to_numpy()
    train_idx = np.where(train_mask)[0]

    lines = ["## 3. Quasi-doublons de cas cliniques (TF-IDF n-grammes de caracteres, cosinus)", ""]
    records, knn = [], []
    for target in ["test", "validation"]:
        target_idx = np.where((cases["split"] == target).to_numpy())[0]
        sims = (matrix[target_idx] @ matrix[train_idx].T).toarray()
        best = sims.argmax(axis=1)
        best_sim = sims.max(axis=1)
        nearest = cases.iloc[train_idx[best]].reset_index(drop=True)
        current = cases.iloc[target_idx].reset_index(drop=True)
        result = pd.DataFrame(
            {
                "split": target,
                "case_id": current["case_id"],
                "groupe": current["groupe"],
                "priorite": current["priorite"],
                "n_questions": current["n_questions"],
                "train_case_id": nearest["case_id"],
                "train_groupe": nearest["groupe"],
                "train_priorite": nearest["priorite"],
                "similarite": best_sim.round(3),
                "contexte": current["contexte"].str.slice(0, 200),
                "train_contexte": nearest["contexte"].str.slice(0, 200),
            }
        )
        records.append(result)

        counts = {f">= {t}": int((best_sim >= t).sum()) for t in THRESHOLDS}
        lines += [
            f"### {target} vs train ({len(target_idx)} cas)",
            "",
            f"Similarite max au train : mediane {np.median(best_sim):.3f}, "
            f"p90 {np.percentile(best_sim, 90):.3f}, max {best_sim.max():.3f}",
            "",
            table(pd.DataFrame([counts], index=["nb_cas"])),
            "",
        ]
        for t in THRESHOLDS:
            close = result[result["similarite"] >= t]
            if len(close):
                agree = (close["priorite"] == close["train_priorite"]).mean() * 100
                lines.append(
                    f"- Seuil {t} : {len(close)} cas, {close['n_questions'].sum()} questions, "
                    f"meme niveau que le voisin train : {agree:.0f} %"
                )
        lines.append("")

        if target == "test":
            knn = [
                "## 5. Baseline plus proche voisin (niveau du cas train le plus similaire)",
                "",
                "Si cette baseline approche le score du SFT triage (63,8 % / F1 55,0), le modele peut "
                "avoir appris par similarite de surface plutot que par raisonnement.",
                "",
            ]
            for level, weights in [
                ("par cas", np.ones(len(result))),
                ("par question", result["n_questions"].to_numpy()),
            ]:
                acc = np.average(result["priorite"] == result["train_priorite"], weights=weights)
                f1 = f1_score(
                    result["priorite"],
                    result["train_priorite"],
                    labels=LABELS,
                    average="macro",
                    sample_weight=weights,
                    zero_division=0,
                )
                knn.append(f"- {level} : accuracy {acc * 100:.1f} %, F1 macro {f1 * 100:.1f} %")
            knn += [
                "",
                "Matrice de confusion (par cas) :",
                "",
                table(pd.crosstab(result["priorite"].map(SHORT), result["train_priorite"].map(SHORT))),
                "",
            ]

    pairs = pd.concat(records).sort_values("similarite", ascending=False)
    top = pairs[pairs["split"] == "test"].head(TOP_PAIRS)
    lines += [
        f"### Top {TOP_PAIRS} paires test / train les plus similaires",
        "",
        table(
            top[["case_id", "priorite", "train_case_id", "train_priorite", "similarite", "n_questions"]]
            .assign(
                priorite=lambda d: d["priorite"].map(SHORT), train_priorite=lambda d: d["train_priorite"].map(SHORT)
            )
            .set_index("case_id")
        ),
        "",
    ]
    return lines, knn, pairs


def label_hints(frame: pd.DataFrame) -> list:
    lines = ["## 4. Indices de label dans les prompts", ""]
    for split in SPLITS:
        prompts = frame.loc[frame["split"] == split, "prompt"]
        hits = prompts[prompts.str.contains(LABEL_HINT_RE)]
        lines.append(f"- {split} : {len(hits)} / {len(prompts)} prompts contiennent un motif de label")
        for text in hits.head(3):
            match = LABEL_HINT_RE.search(text)
            start = max(match.start() - 60, 0)
            lines.append(f"  - `...{text[start : match.end() + 60].replace(chr(10), ' ')}...`")
    test_with_prefix = frame.loc[frame["split"] == "test", "reponse"].str.startswith("[Niveau").sum()
    lines += [
        "",
        f"Reponses test commencant par le prefixe de niveau : {test_with_prefix} "
        "(attendu : elles ne sont pas vues a l'evaluation, qui ne score que la continuation du prompt).",
        "",
    ]
    return lines


def main() -> None:
    frame = load()
    lines = [
        "# Audit des partitions train / validation / test",
        "",
        f"Source : `{SFT_PATH}` ({len(frame)} lignes), joint a `{CORPUS_PATH}` pour `case_id` et `contexte_patient`.",
        "",
    ]
    lines += distribution(frame)
    lines += exact_overlaps(frame)
    duplicate_lines, knn_lines, pairs = near_duplicates(frame)
    lines += duplicate_lines + label_hints(frame) + knn_lines

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = OUTPUT_DIR / "audit_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pairs.to_csv(OUTPUT_DIR / "near_duplicates.csv", index=False, encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRapport -> {report}\nPaires -> {OUTPUT_DIR / 'near_duplicates.csv'}")


if __name__ == "__main__":
    main()
