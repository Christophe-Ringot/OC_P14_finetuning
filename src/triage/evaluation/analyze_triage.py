"""Verification de la methode d'evaluation du triage (sans GPU).

Repart des scores sauvegardes par src/triage/evaluation/evaluate_sft.py et mesure :
1. l'effet de la calibration contextuelle (avec / sans soustraction du prior) ;
2. les metriques par cas clinique (vote majoritaire, moyenne des scores) ;
3. la coherence des predictions entre questions d'un meme cas ;
4. des intervalles de confiance par bootstrap sur les groupes ;
5. la sensibilite aux cas de test proches du train (audit_splits.py).

Sortie : outputs/triage_eval/analyse_evaluation.md
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from triage.evaluation.evaluate_sft import LABELS, OUTPUT_DIR, SHORT, triage_metrics

CORPUS_PATH = Path("data") / "interim" / "corpus_annotated.csv"
NEAR_DUPLICATES_PATH = Path("outputs") / "audit" / "near_duplicates.csv"
SIMILARITY_THRESHOLD = 0.5
N_BOOTSTRAP = 2000
RANDOM_STATE = 42
MAIN = ["accuracy", "f1_macro", "rappel_P1", "rappel_P2", "rappel_P3", "sous_triage", "sur_triage"]
P = [SHORT[label] for label in LABELS]
FROM_SHORT = {v: k for k, v in SHORT.items()}


def table(frame: pd.DataFrame, percent: bool = True) -> str:
    frame = frame * 100 if percent else frame
    header = "| | " + " | ".join(frame.columns) + " |\n|" + "---|" * (len(frame.columns) + 1)
    rows = [f"| {name} | " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for name, row in frame.iterrows()]
    return "\n".join([header, *rows])


def load(suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(OUTPUT_DIR / f"predictions{suffix}.csv")
    scores = pd.read_csv(OUTPUT_DIR / f"scores{suffix}.csv")
    cases = pd.read_csv(CORPUS_PATH, usecols=["id", "case_id"])
    predictions = predictions.merge(cases, on="id", how="left", validate="one_to_one")
    for p in P:
        scores[f"cal_{p}"] = scores[f"raw_{p}"] - scores[f"prior_{p}"]
    scores = scores.merge(predictions[["id", "case_id", "groupe", "priorite"]], on="id", how="left")
    return predictions, scores


def argmax(frame: pd.DataFrame, prefix: str) -> pd.Series:
    columns = [f"{prefix}_{p}" for p in P]
    return frame[columns].idxmax(axis=1).str.replace(f"{prefix}_", "").map(FROM_SHORT)


def calibration_effect(scores: pd.DataFrame) -> list:
    rows, priors = {}, {}
    for experiment, frame in scores.groupby("experience", sort=False):
        for mode, prefix in [("calibre", "cal"), ("brut", "raw")]:
            rows[f"{experiment} ({mode})"] = triage_metrics(frame["priorite"], argmax(frame, prefix))
        priors[experiment] = {p: frame[f"prior_{p}"].iloc[0] for p in P}
    lines = ["## 1. Effet de la calibration contextuelle (par question)", "",
             "`calibre` = methode actuelle (score - score sur prompt vide) ; `brut` = argmax des log-probabilites.", "",
             table(pd.DataFrame(rows).T[MAIN]), "",
             "Prior (log-probabilite sur prompt vide) : plus il est eleve, plus la calibration penalise le niveau.", "",
             table(pd.DataFrame(priors).T, percent=False), ""]
    return lines


def case_level(predictions: pd.DataFrame, scores: pd.DataFrame, experiments: list, modes: dict) -> list:
    """Agregation par cas ; la moyenne des scores suit le mode (calibre / brut) retenu en validation."""
    truth = predictions.groupby("case_id")["priorite"].first()
    rows = {}
    for experiment in experiments:
        rows[f"{experiment} (par question)"] = triage_metrics(predictions["priorite"], predictions[experiment])
        # Vote majoritaire, egalite tranchee vers le niveau le plus urgent (prudence).
        votes = predictions.groupby("case_id")[experiment].agg(
            lambda s: min(s.mode(), key=LABELS.index))
        rows[f"{experiment} (cas, vote)"] = triage_metrics(truth, votes.reindex(truth.index))
        frame = scores[scores["experience"] == experiment]
        if len(frame):
            prefix = "cal" if modes.get(experiment, "calibre") == "calibre" else "raw"
            mean = frame.groupby("case_id")[[f"{prefix}_{p}" for p in P]].mean()
            rows[f"{experiment} (cas, moyenne scores)"] = triage_metrics(truth, argmax(mean, prefix).reindex(truth.index))
    distribution = truth.map(SHORT).value_counts().reindex(P)
    return ["## 2. Metriques par cas clinique", "",
            f"{len(truth)} cas (P1 = {distribution['P1']}, P2 = {distribution['P2']}, P3 = {distribution['P3']}) "
            f"pour {len(predictions)} questions. Chaque cas compte une fois.", "",
            table(pd.DataFrame(rows).T[MAIN]), ""]


def consistency(predictions: pd.DataFrame, experiments: list) -> list:
    multi = predictions.groupby("case_id").filter(lambda g: len(g) > 1)
    rows = {}
    for experiment in experiments:
        n_levels = multi.groupby("case_id")[experiment].nunique()
        rows[experiment] = {"cas_coherents": (n_levels == 1).mean(), "cas_2_niveaux": (n_levels == 2).mean(),
                            "cas_3_niveaux": (n_levels == 3).mean()}
    return ["## 3. Coherence des predictions au sein d'un cas", "",
            f"Sur les {multi['case_id'].nunique()} cas ayant plusieurs questions : part des cas ou toutes les "
            "questions recoivent le meme niveau. Le label etant porte par le cas, un modele qui "
            "raisonne sur le cas devrait etre coherent.", "",
            table(pd.DataFrame(rows).T), ""]


def bootstrap(predictions: pd.DataFrame, experiments: list) -> list:
    rng = np.random.default_rng(RANDOM_STATE)
    groups = predictions["groupe"].unique()
    index = predictions.groupby("groupe").indices
    samples = {e: {"accuracy": [], "f1_macro": []} for e in experiments}
    diffs = {"accuracy": [], "f1_macro": []}
    a, b = "3_sft_medical", "4_sft_medical_triage"
    for _ in range(N_BOOTSTRAP):
        rows = np.concatenate([index[g] for g in rng.choice(groups, len(groups), replace=True)])
        sample = predictions.iloc[rows]
        values = {}
        for e in experiments:
            values[e] = {
                "accuracy": (sample["priorite"] == sample[e]).mean(),
                "f1_macro": f1_score(sample["priorite"], sample[e], labels=LABELS, average="macro", zero_division=0),
            }
            for metric, value in values[e].items():
                samples[e][metric].append(value)
        if a in values and b in values:
            for metric in diffs:
                diffs[metric].append(values[b][metric] - values[a][metric])

    def interval(values) -> str:
        low, high = np.percentile(values, [2.5, 97.5]) * 100
        return f"{np.mean(values) * 100:.1f} [{low:.1f} ; {high:.1f}]"

    frame = pd.DataFrame({e: {m: interval(v) for m, v in s.items()} for e, s in samples.items()}).T
    lines = ["## 4. Intervalles de confiance a 95 % (bootstrap sur les groupes)", "",
             f"{N_BOOTSTRAP} tirages avec remise des {len(groups)} groupes : les questions d'un meme groupe "
             "sont tirees ensemble, ce qui respecte leur dependance.", "",
             table(frame, percent=False), ""]
    if diffs["accuracy"]:
        lines += [f"Ecart {b} - {a} : accuracy {interval(diffs['accuracy'])}, F1 macro {interval(diffs['f1_macro'])}.",
                  f"Part des tirages ou l'ecart d'accuracy est <= 0 : {np.mean(np.array(diffs['accuracy']) <= 0) * 100:.1f} %.", ""]
    return lines


def sensitivity(predictions: pd.DataFrame, experiments: list) -> list:
    if not NEAR_DUPLICATES_PATH.exists():
        return ["## 5. Sensibilite aux quasi-doublons", "", f"`{NEAR_DUPLICATES_PATH}` absent : lancer src/triage/data/audit_splits.py.", ""]
    pairs = pd.read_csv(NEAR_DUPLICATES_PATH)
    suspects = set(pairs.loc[(pairs["split"] == "test") & (pairs["similarite"] >= SIMILARITY_THRESHOLD), "case_id"])
    kept = predictions[~predictions["case_id"].isin(suspects)]
    rows = {}
    for e in experiments:
        rows[f"{e} (tout)"] = triage_metrics(predictions["priorite"], predictions[e])
        rows[f"{e} (sans suspects)"] = triage_metrics(kept["priorite"], kept[e])
    return ["## 5. Sensibilite aux cas de test proches du train", "",
            f"Exclusion des {len(suspects)} cas de test dont la similarite au train est >= {SIMILARITY_THRESHOLD} "
            f"({len(predictions) - len(kept)} questions retirees, {len(kept)} restantes).", "",
            table(pd.DataFrame(rows).T[MAIN]), ""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    suffix = "_quick" if parser.parse_args().quick else ""
    predictions, scores = load(suffix)
    experiments = [c for c in predictions.columns if c[0].isdigit()]

    lines = ["# Verification de la methode d'evaluation du triage", "",
             "Valeurs en %. Genere par `src/triage/evaluation/analyze_triage.py` a partir de "
             f"`predictions{suffix}.csv` et `scores{suffix}.csv`.", ""]
    lines += calibration_effect(scores)
    decision_path = OUTPUT_DIR / f"mode_decision{suffix}.csv"
    modes = pd.read_csv(decision_path, index_col=0)["mode_retenu"].to_dict() if decision_path.exists() else {}
    lines += case_level(predictions, scores, experiments, modes)
    lines += consistency(predictions, experiments[1:])
    lines += bootstrap(predictions, experiments)
    lines += sensitivity(predictions, experiments)

    report = OUTPUT_DIR / f"analyse_evaluation{suffix}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRapport -> {report}")


if __name__ == "__main__":
    main()
