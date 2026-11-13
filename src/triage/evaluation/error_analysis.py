"""Analyse des erreurs de triage d'un modele (sans GPU).

Repart des scores sauvegardes par src/triage/evaluation/evaluate_sft.py et produit :
1. la matrice de confusion par cas clinique et le detail des types d'erreur ;
2. le taux d'erreur selon les caracteristiques des cas (specialite, signal de
   gravite, longueur, nombre de questions, version d'une famille de cas) ;
3. la separation des scores entre niveaux : le modele ordonne-t-il bien les
   cas (AUC entre niveaux adjacents) ou est-il seulement decale ?
4. un ajustement des seuils de decision appris sur la validation, applique au test ;
5. les cas mal classes avec la justification de l'annotateur, pour relecture.

Sorties : outputs/triage_eval/analyse_erreurs.md et erreurs_cas.csv
"""

import argparse
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from triage.constants import DATA_DIR, FINAL_EXPERIMENT, LABELS, RANK, SHORT
from triage.evaluation import metrics
from triage.evaluation.metrics import triage_metrics
from triage.evaluation.saved_scores import TRIAGE_EVAL_DIR, decision_mode, load_decision_scores
from triage.utils import hf_dataset
from triage.utils.markdown import markdown_table as table

CORPUS_PATH = DATA_DIR / "interim" / "corpus_annotated.csv"
POOL_PATH = DATA_DIR / "annotation" / "pool_priorite.csv"
ANNOTATIONS_PATH = DATA_DIR / "annotation" / "annotations_priorite.csv"
P = [SHORT[label] for label in LABELS]
SHORT_RANK = {p: i for i, p in enumerate(P)}
BIAS_GRID = np.arange(-6, 6.01, 0.25)
MIN_RECALL_P1 = 0.95
EXCERPT = 350
MAIN = ["accuracy", "f1_macro", "rappel_P1", "rappel_P2", "rappel_P3", "sous_triage", "sur_triage"]


def softmax(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(metrics.softmax(frame.to_numpy()), index=frame.index, columns=frame.columns)


def load(experiment: str) -> tuple:
    mode = decision_mode(experiment)
    labels = hf_dataset.load_sft()[["id", "split", "priorite"]]
    corpus = pd.read_csv(CORPUS_PATH, usecols=["id", "case_id", "contexte_patient"])
    pool = pd.read_csv(POOL_PATH, usecols=["case_id", "specialite", "signal_gravite", "parent_case_id"])
    annotations = pd.read_csv(ANNOTATIONS_PATH, usecols=["case_id", "justification"])

    splits = {}
    for split in ["test", "validation"]:
        scores = load_decision_scores(experiment, split, mode)
        frame = labels.set_index("id").loc[scores.index].join(corpus.set_index("id"))
        splits[split] = (frame, scores)
    cases = pool.merge(annotations, on="case_id", how="left").set_index("case_id")
    return mode, splits, cases


def case_table(frame: pd.DataFrame, scores: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par cas : vote majoritaire (egalite -> plus urgent) et probabilites moyennes."""
    questions = frame.assign(prediction=scores.idxmax(axis=1)).join(softmax(scores).add_prefix("p_"))
    vote = questions.groupby("case_id")["prediction"].agg(lambda s: min(s.mode(), key=RANK.get))
    table_ = questions.groupby("case_id").agg(
        priorite=("priorite", "first"),
        n_questions=("prediction", "size"),
        n_niveaux_predits=("prediction", "nunique"),
        contexte=("contexte_patient", "first"),
        **{f"p_{SHORT[label]}": (f"p_{label}", "mean") for label in LABELS},
    )
    table_["prediction"] = vote
    table_["ecart"] = table_["prediction"].map(RANK) - table_["priorite"].map(RANK)  # > 0 : sous-triage
    table_["longueur"] = table_["contexte"].str.len()
    table_ = table_.join(cases)
    table_["version_derivee"] = table_["parent_case_id"].notna()
    return table_


def confusion_section(case_frame: pd.DataFrame) -> list:
    matrix = pd.DataFrame(
        confusion_matrix(case_frame["priorite"], case_frame["prediction"], labels=LABELS),
        index=[f"annote {p}" for p in P],
        columns=[f"predit {p}" for p in P],
    )
    errors = case_frame[case_frame["ecart"] != 0]
    kinds = (errors["priorite"].map(SHORT) + " -> " + errors["prediction"].map(SHORT)).value_counts()
    kinds = kinds.rename("cas").to_frame()
    kinds["sens"] = [("sous-triage" if SHORT_RANK[k[-2:]] > SHORT_RANK[k[:2]] else "sur-triage") for k in kinds.index]
    return [
        "## 1. Erreurs par cas clinique (vote majoritaire)",
        "",
        f"{len(case_frame)} cas, {len(errors)} mal classes ({len(errors) / len(case_frame) * 100:.1f} %).",
        "",
        table(matrix),
        "",
        table(kinds),
        "",
    ]


def profile_section(case_frame: pd.DataFrame) -> list:
    frame = case_frame.assign(
        erreur=case_frame["ecart"] != 0,
        sous_triage=case_frame["ecart"] > 0,
        sur_triage=case_frame["ecart"] < 0,
        longueur_quartile=pd.qcut(case_frame["longueur"], 4, labels=["Q1 court", "Q2", "Q3", "Q4 long"]),
        questions=pd.cut(case_frame["n_questions"], [0, 1, 3, 100], labels=["1", "2-3", "4+"]),
        predictions_incoherentes=case_frame["n_niveaux_predits"] > 1,
    )
    lines = [
        "## 2. Taux d'erreur selon les caracteristiques des cas",
        "",
        "Effectifs faibles : ces taux orientent la relecture, ils ne prouvent rien a eux seuls.",
        "",
    ]
    for column in [
        "priorite",
        "signal_gravite",
        "version_derivee",
        "longueur_quartile",
        "questions",
        "predictions_incoherentes",
        "specialite",
    ]:
        grouped = frame.groupby(column, observed=True).agg(
            cas=("erreur", "size"),
            erreur=("erreur", "mean"),
            sous_triage=("sous_triage", "mean"),
            sur_triage=("sur_triage", "mean"),
        )
        if column == "specialite":
            grouped = grouped[grouped["cas"] >= 3].sort_values("cas", ascending=False)
        if column == "priorite":
            grouped.index = grouped.index.map(SHORT)
        grouped[["erreur", "sous_triage", "sur_triage"]] *= 100
        lines += [f"### {column}", "", table(grouped), ""]
    return lines


def separation_section(case_frame: pd.DataFrame, frame: pd.DataFrame, scores: pd.DataFrame) -> list:
    probs = softmax(scores)
    # Urgence attendue : 0 = P1 ... 2 = P3. Score continu pour mesurer l'ordre.
    expected = probs[LABELS].to_numpy() @ np.arange(len(LABELS))
    truth = frame["priorite"].map(RANK).to_numpy()
    rows = {}
    for low, high in [(0, 1), (1, 2), (0, 2)]:
        mask = np.isin(truth, [low, high])
        rows[f"{P[low]} vs {P[high]}"] = {
            "AUC": roc_auc_score(truth[mask] == high, expected[mask]),
            "questions": int(mask.sum()),
        }
    mean_probs = case_frame.groupby(case_frame["priorite"].map(SHORT))[[f"p_{p}" for p in P]].mean() * 100
    return [
        "## 3. Separation des scores entre niveaux",
        "",
        "Probabilite moyenne attribuee a chaque niveau (softmax sur les 3 libelles, moyenne par cas), "
        "selon le niveau annote :",
        "",
        table(mean_probs.reindex(P)),
        "",
        "AUC entre niveaux, sur l'urgence attendue sum(p x rang) par question. Une AUC elevee avec une "
        "accuracy moyenne signale un modele qui ordonne bien mais dont les frontieres sont mal placees "
        "(corrigeable par un seuil) ; une AUC proche de 0.5 signale une vraie confusion.",
        "",
        table(pd.DataFrame(rows).T, digits=3),
        "",
    ]


def apply_bias(scores: pd.DataFrame, bias: tuple) -> pd.Series:
    return (scores + np.array([0.0, *bias])).idxmax(axis=1)


def threshold_section(splits: dict) -> list:
    """Biais additifs sur P2 et P3 (P1 fixe), appris sur la validation uniquement."""
    val_frame, val_scores = splits["validation"]
    test_frame, test_scores = splits["test"]
    candidates = []
    for bias in product(BIAS_GRID, BIAS_GRID):
        result = triage_metrics(val_frame["priorite"], apply_bias(val_scores, bias))
        candidates.append((bias, result))
    best_f1 = max(candidates, key=lambda c: c[1]["f1_macro"])
    safe = [c for c in candidates if c[1]["rappel_P1"] >= MIN_RECALL_P1]
    best_safe = max(safe, key=lambda c: c[1]["f1_macro"]) if safe else None

    rows = {"sans ajustement": triage_metrics(test_frame["priorite"], apply_bias(test_scores, (0, 0)))}
    rows[f"F1 max (biais P2 {best_f1[0][0]:+.2f}, P3 {best_f1[0][1]:+.2f})"] = triage_metrics(
        test_frame["priorite"], apply_bias(test_scores, best_f1[0])
    )
    if best_safe:
        rows[f"rappel P1 >= {MIN_RECALL_P1:.0%} (biais P2 {best_safe[0][0]:+.2f}, P3 {best_safe[0][1]:+.2f})"] = (
            triage_metrics(test_frame["priorite"], apply_bias(test_scores, best_safe[0]))
        )
    val_rows = {
        "sans ajustement": triage_metrics(val_frame["priorite"], apply_bias(val_scores, (0, 0))),
        "F1 max": best_f1[1],
    }
    if best_safe:
        val_rows[f"rappel P1 >= {MIN_RECALL_P1:.0%}"] = best_safe[1]
    return [
        "## 4. Ajustement des seuils de decision (appris sur la validation)",
        "",
        "Un biais est ajoute aux scores de P2 et P3 (grille de -6 a +6), choisi sur la validation, "
        "puis applique tel quel au test. Deux objectifs : F1 macro maximal, ou F1 macro maximal "
        f"sous contrainte d'un rappel P1 >= {MIN_RECALL_P1:.0%} (limiter le sous-triage des urgences vitales).",
        "",
        "Validation (donnees d'apprentissage des biais) :",
        "",
        table(pd.DataFrame(val_rows).T[MAIN] * 100),
        "",
        "Test (application sans retouche) :",
        "",
        table(pd.DataFrame(rows).T[MAIN] * 100),
        "",
    ]


def examples_section(case_frame: pd.DataFrame) -> list:
    errors = case_frame[case_frame["ecart"] != 0].copy()
    errors["confiance_prediction"] = [row[f"p_{SHORT[row['prediction']]}"] for _, row in errors.iterrows()]
    errors = errors.sort_values(["ecart", "confiance_prediction"], ascending=[False, False])
    lines = [
        "## 5. Cas mal classes (sous-triage d'abord)",
        "",
        "`justification` : motif donne par le LLM annotateur. A relire : erreur du modele, ou annotation discutable ?",
        "",
    ]
    for case_id, row in errors.iterrows():
        sens = "SOUS-TRIAGE" if row["ecart"] > 0 else "sur-triage"
        lines += [
            f"### {case_id} - annote {SHORT[row['priorite']]}, predit {SHORT[row['prediction']]} ({sens})",
            "",
            f"- Probabilites moyennes : P1 {row['p_P1']:.2f} / P2 {row['p_P2']:.2f} / P3 {row['p_P3']:.2f} ; "
            f"{row['n_questions']} question(s), {row['n_niveaux_predits']} niveau(x) predit(s)",
            f"- Specialite : {row['specialite']} ; signal de gravite (regex) : {row['signal_gravite']}",
            f"- Justification annotateur : {row['justification']}",
            f"- Cas : {str(row['contexte'])[:EXCERPT].replace(chr(10), ' ')}{'...' if len(str(row['contexte'])) > EXCERPT else ''}",
            "",
        ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=FINAL_EXPERIMENT)
    experiment = parser.parse_args().experiment
    mode, splits, cases = load(experiment)
    test_frame, test_scores = splits["test"]
    case_frame = case_table(test_frame, test_scores, cases)

    lines = [
        f"# Analyse des erreurs - {experiment}",
        "",
        f"Mode de decision : {mode} (retenu sur la validation). Jeu de test : {len(test_frame)} questions, "
        f"{len(case_frame)} cas. Valeurs en %.",
        "",
    ]
    lines += confusion_section(case_frame)
    lines += profile_section(case_frame)
    lines += separation_section(case_frame, test_frame, test_scores)
    lines += threshold_section(splits)
    lines += examples_section(case_frame)

    report = TRIAGE_EVAL_DIR / "analyse_erreurs.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    case_frame.drop(columns=["contexte"]).to_csv(TRIAGE_EVAL_DIR / "erreurs_cas.csv", encoding="utf-8")
    print("\n".join(lines[: lines.index("## 5. Cas mal classes (sous-triage d'abord)")]))
    print(f"\nRapport -> {report}")


if __name__ == "__main__":
    main()
