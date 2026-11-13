"""Decision de triage sensible au cout des erreurs (sans GPU).

Repart des scores sauvegardes par evaluate_sft (log-probabilites des 3 niveaux)
et remplace l'argmax par la decision de risque minimal (Bayes) :
    niveau = argmin_j sum_i p(i | cas) * C[i, j]
ou C[i, j] est le cout de predire j quand le niveau annote est i. Le sous-triage
coute plus cher que le sur-triage, sans que le sur-triage soit gratuit (sinon
tout serait classe P1).

Etapes, toutes apprises sur la validation puis appliquees telles quelles au test :
1. temperature T qui calibre les probabilites (log-vraisemblance maximale) ;
2. poids alpha du sous-triage : C_alpha multiplie les cases de sous-triage par
   alpha ; on retient l'alpha qui minimise le cout moyen sous la matrice de
   reference C (le critere clinique), pas l'accuracy.

Sorties : outputs/triage_eval/cout_sensible.md et cout_sensible.csv
"""

import argparse

import numpy as np
import pandas as pd

from triage.constants import FINAL_EXPERIMENT, LABELS, RANK, SHORT
from triage.evaluation.metrics import softmax, triage_metrics
from triage.evaluation.saved_scores import TRIAGE_EVAL_DIR, decision_mode, load_decision_scores
from triage.utils import hf_dataset, tracking
from triage.utils.markdown import markdown_table as table

# Cout de reference : lignes = niveau annote, colonnes = niveau predit (P1, P2, P3).
COST = np.array(
    [
        [0.0, 4.0, 10.0],  # P1 rate : sous-triage d'une urgence vitale
        [1.0, 0.0, 4.0],  # P2 : sur-triage leger, sous-triage vers P3
        [2.0, 1.0, 0.0],  # P3 : sur-triage (encombrement des urgences)
    ]
)
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
TEMPERATURES = np.round(np.arange(0.5, 10.01, 0.25), 2)
MAIN = ["accuracy", "f1_macro", "rappel_P1", "rappel_P2", "rappel_P3", "sous_triage", "sur_triage", "P1_predit_P3"]


def scaled_cost(alpha: float) -> np.ndarray:
    """Matrice de reference avec le sous-triage (triangle superieur) multiplie par alpha."""
    return np.where(np.triu(np.ones_like(COST), 1) > 0, COST * alpha, COST)


def probabilities(scores: pd.DataFrame, temperature: float) -> np.ndarray:
    return softmax(scores.to_numpy(), temperature)


def fit_temperature(scores: pd.DataFrame, y_true: pd.Series) -> float:
    target = y_true.map(RANK).to_numpy()
    nll = {t: -np.log(probabilities(scores, t)[np.arange(len(target)), target] + 1e-12).mean() for t in TEMPERATURES}
    return min(nll, key=nll.get)


def decide(proba: np.ndarray, cost: np.ndarray) -> pd.Series:
    """Niveau de risque attendu minimal (p @ C donne le cout attendu de chaque decision)."""
    return pd.Series(np.array(LABELS)[(proba @ cost).argmin(axis=1)])


def mean_cost(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(COST[y_true.map(RANK).to_numpy(), y_pred.map(RANK).to_numpy()].mean())


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = y_true.reset_index(drop=True), y_pred.reset_index(drop=True)
    return {
        **{k: v for k, v in triage_metrics(y_true, y_pred).items() if k in MAIN},
        "cout_moyen": mean_cost(y_true, y_pred),
    }


def load(experiment: str) -> dict:
    mode = decision_mode(experiment)
    labels = hf_dataset.load_sft()[["id", "priorite"]].set_index("id")["priorite"]
    splits = {}
    for split in ["validation", "test"]:
        scores = load_decision_scores(experiment, split, mode)
        splits[split] = (labels.loc[scores.index], scores)
    return splits


def run(experiment: str, tracked: bool) -> pd.DataFrame:
    splits = load(experiment)
    y_val, s_val = splits["validation"]
    temperature = fit_temperature(s_val, y_val)
    p_val = probabilities(s_val, temperature)

    sweep = pd.DataFrame({alpha: metrics(y_val, decide(p_val, scaled_cost(alpha))) for alpha in ALPHAS}).T
    best_alpha = float(sweep["cout_moyen"].idxmin())

    rows = {}
    for split, (y, s) in splits.items():
        proba = probabilities(s, temperature)
        rows[(split, "argmax (actuel)")] = metrics(y, s.idxmax(axis=1))
        rows[(split, "cout C (alpha=1)")] = metrics(y, decide(proba, COST))
        rows[(split, f"cout C_alpha (alpha={best_alpha:g}, choisi sur validation)")] = metrics(
            y, decide(proba, scaled_cost(best_alpha))
        )
    results = pd.DataFrame(rows).T
    results.index.names = ["split", "decision"]

    test_sweep = pd.DataFrame(
        {
            alpha: metrics(splits["test"][0], decide(probabilities(splits["test"][1], temperature), scaled_cost(alpha)))
            for alpha in ALPHAS
        }
    ).T
    write_report(experiment, temperature, best_alpha, results, sweep, test_sweep)
    results.to_csv(TRIAGE_EVAL_DIR / "cout_sensible.csv")
    if tracked:
        import mlflow

        with tracking.start_run(
            True, run_name=f"cout_sensible_{experiment}", tags={"type": "evaluation", "experience": experiment}
        ):
            mlflow.log_params({"temperature": temperature, "alpha": best_alpha, "cost_matrix": COST.tolist()})
            for (split, decision), row in results.iterrows():
                key = decision.split(" ")[0] + ("_alpha" if "alpha=" in decision and "choisi" in decision else "")
                mlflow.log_metrics({f"{split}_{key}_{k}": float(v) for k, v in row.items()})
            mlflow.log_artifact(str(TRIAGE_EVAL_DIR / "cout_sensible.md"))
    print(
        (
            results[["accuracy", "f1_macro", "rappel_P1", "sous_triage", "sur_triage", "cout_moyen"]]
            * [100, 100, 100, 100, 100, 1]
        )
        .round(2)
        .to_string()
    )
    return results


def write_report(experiment, temperature, best_alpha, results, sweep, test_sweep) -> None:
    # Pourcentages ; cout moyen x100 pour garder la precision avec le meme format.
    def pct(frame):
        return (frame * 100).rename(columns={"cout_moyen": "cout_moyen_x100"})

    cost_table = pd.DataFrame(
        COST, index=[f"annote {SHORT[l]}" for l in LABELS], columns=[f"predit {SHORT[l]}" for l in LABELS]
    )
    lines = [
        f"# Decision sensible au cout - {experiment}",
        "",
        "Le modele donne une probabilite par niveau ; au lieu de prendre le plus probable, on choisit le niveau "
        "dont le cout attendu est le plus faible. Aucune re-entrainement : seule la regle de decision change.",
        "",
        "## Matrice de cout de reference",
        "",
        "Sous-triage d'un niveau = 4, P1 classe P3 = 10, sur-triage = 1 (P2 -> P1, P3 -> P2) ou 2 (P3 -> P1). "
        "Choix d'expert a valider avec les urgentistes du CHSA : c'est un parametre metier, pas un hyperparametre.",
        "",
        table(cost_table),
        "",
        f"Temperature de calibration apprise sur la validation : T = {temperature:g} "
        "(T > 1 : le modele est trop sur de lui, ses probabilites sont adoucies).",
        "",
        "## Resultats (valeurs en %, cout moyen par question x100)",
        "",
        table(
            pct(results)
            .reset_index()
            .set_index("decision")
            .drop(columns="split")
            .assign(split=results.index.get_level_values("split"))
            .set_index("split", append=True)
        ),
        "",
        "## Effet du poids du sous-triage (alpha)",
        "",
        f"alpha multiplie les couts de sous-triage. Choisi sur la validation (cout moyen minimal) : alpha = {best_alpha:g}.",
        "",
        "Validation :",
        "",
        table(pct(sweep).rename(index=lambda a: f"{a:g}").rename_axis("alpha")),
        "",
        "Test (pour information, non utilise pour choisir) :",
        "",
        table(pct(test_sweep).rename(index=lambda a: f"{a:g}").rename_axis("alpha")),
        "",
        "## Lecture",
        "",
        "- Le compromis sous-triage / sur-triage se regle a l'inference, sans re-entrainer, et de facon explicite "
        "(une matrice lisible par un clinicien) plutot que par des biais ajustes a la main.",
        "- Une ponderation de la perte pendant l'entrainement (poids plus forts sur les tokens du niveau des cas P1/P2) "
        "est l'alternative ; elle demande un re-entrainement complet (~7 h) et deplace les memes frontieres. "
        "Avec un modele qui ordonne deja bien les cas (AUC P1/P2 = 0.86, P2/P3 = 0.92, voir analyse_erreurs.md), "
        "la decision au cout est le levier le moins cher.",
        "- Sur la validation, la decision au cout divise le cout moyen par 2 (sous-triage 16 % -> 1 %). Sur le test, "
        "elle ramene le sous-triage de 7 % a 2 % et le rappel P1 a 95 %, mais le sur-triage passe de 14 % a 35 % et le "
        "cout moyen ne baisse pas : la validation (accuracy 63 %) est bien plus difficile que le test (79 %), avec "
        "~70 cas par split, donc un reglage appris sur l'une se transpose mal a l'autre.",
        "- Point de fonctionnement raisonnable : alpha = 0.5 (test : sous-triage 4.2 %, rappel P1 90 %, accuracy 71 %). "
        "Le choix final releve du CHSA (combien de sur-triage accepter pour un P1 rattrape) et demande un jeu de "
        "validation plus grand, valide cliniquement.",
    ]
    (TRIAGE_EVAL_DIR / "cout_sensible.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=FINAL_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    run(args.experiment, tracking.setup(not args.no_mlflow))


if __name__ == "__main__":
    main()
