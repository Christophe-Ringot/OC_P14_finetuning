"""Comparaison finale des modeles SFT / DPO : tableaux et graphiques (sans GPU).

Repart des sorties deja calculees (scores de triage, historiques d'entrainement,
evaluations des reponses et des explications) et produit :
- la comparaison des 6 experiences en decision argmax ET avec la regle de cout
  deployee (alpha de decision_rule.json, temperature ajustee sur la validation
  pour chaque modele) : le sous / sur-triage de chaque version est compare a
  regle egale ;
- des intervalles de confiance bootstrap (groupes de cas) sur le test ;
- les graphiques de outputs/figures/.

    python -m triage.evaluation.compare_models

Sorties : outputs/triage_eval/comparaison_modeles.md, comparaison_modeles.csv, outputs/figures/*.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from triage.api.config import load_decision_rule
from triage.constants import LABELS, OUTPUTS_DIR, SEED, SHORT, adapter_dir
from triage.evaluation import cost_sensitive
from triage.evaluation.saved_scores import TRIAGE_EVAL_DIR
from triage.utils.markdown import markdown_table

FIGURES_DIR = OUTPUTS_DIR / "figures"
N_BOOTSTRAP = 2000
# alpha deploye (decision_rule.json), affiche a la francaise dans les titres : 0.35 -> "0,35".
DEPLOYED_ALPHA = load_decision_rule()["alpha"]
ALPHA_LABEL = f"{DEPLOYED_ALPHA:g}".replace(".", ",")
NAMES = {
    "1_baseline_naive": "1 Baseline (P1)",
    "2_modele_base": "2 Modele de base",
    "3_sft_medical": "3 SFT medical",
    "4_sft_medical_triage": "4 SFT + triage",
    "5_sft_triage_dpo": "5 DPO v1",
    "6_sft_triage_dpo_qualite": "6 DPO v2",
}
COLORS = {
    "1_baseline_naive": "#B0B0B0",
    "2_modele_base": "#8A8A8A",
    "3_sft_medical": "#4A4A4A",
    "4_sft_medical_triage": "#FF6B35",
    "5_sft_triage_dpo": "#E63946",
    "6_sft_triage_dpo_qualite": "#0066FF",
}
TRAINED = ["4_sft_medical_triage", "5_sft_triage_dpo", "6_sft_triage_dpo_qualite"]
MAIN = ["accuracy", "f1_macro", "rappel_P1", "sous_triage", "sur_triage", "P1_predit_P3"]
LABELS_FR = {
    "accuracy": "Accuracy",
    "f1_macro": "F1 macro",
    "rappel_P1": "Rappel P1",
    "sous_triage": "Sous-triage",
    "sur_triage": "Sur-triage",
}


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print(f"  -> {FIGURES_DIR / name}")


# ---------- tableaux ----------


def argmax_results() -> dict:
    return {
        split: pd.read_csv(
            TRIAGE_EVAL_DIR / f"comparaison_triage{'' if split == 'test' else '_validation'}.csv", index_col=0
        )
        for split in ["validation", "test"]
    }


def cost_rule_results(alpha: float) -> tuple[dict, dict]:
    """Regle de cout deployee appliquee a chaque modele entraine : metriques et predictions."""
    results, predictions = {"validation": {}, "test": {}}, {}
    for experiment in TRAINED:
        splits = cost_sensitive.load(experiment)
        temperature = cost_sensitive.fit_temperature(*reversed(splits["validation"]))
        for split, (y, scores) in splits.items():
            pred = cost_sensitive.decide(
                cost_sensitive.probabilities(scores, temperature), cost_sensitive.scaled_cost(alpha)
            )
            pred.index = scores.index
            results[split][experiment + " + cout"] = {**cost_sensitive.metrics(y, pred), "temperature": temperature}
            if split == "test":
                predictions[experiment] = pred
    return {s: pd.DataFrame(r).T for s, r in results.items()}, predictions


def bootstrap(predictions: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """IC 95 % de l'accuracy et du sous-triage, tirage avec remise des groupes de cas."""
    rng = np.random.default_rng(SEED)
    index = predictions.groupby("groupe").indices
    groups = np.array(list(index))
    rank = {label: i for i, label in enumerate(LABELS)}
    truth = predictions["priorite"].map(rank).to_numpy()
    preds = {name: predictions[col].map(rank).to_numpy() for name, col in columns.items()}
    samples = {name: {"accuracy": [], "sous_triage": []} for name in columns}
    for _ in range(N_BOOTSTRAP):
        rows = np.concatenate([index[g] for g in rng.choice(groups, len(groups), replace=True)])
        for name, pred in preds.items():
            gap = pred[rows] - truth[rows]
            samples[name]["accuracy"].append((gap == 0).mean())
            samples[name]["sous_triage"].append((gap > 0).mean())
    rows = {}
    for name, values in samples.items():
        gap = preds[name] - truth
        row = {"accuracy": (gap == 0).mean(), "sous_triage": (gap > 0).mean()}
        for metric, draws in values.items():
            row[f"{metric}_bas"], row[f"{metric}_haut"] = np.percentile(draws, [2.5, 97.5])
        rows[name] = row
    return pd.DataFrame(rows).T


# ---------- graphiques ----------


def fig_metrics(results: dict) -> None:
    test, val = results["test"], results["validation"]
    metrics = list(LABELS_FR)
    width = 0.13
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    for i, experiment in enumerate(NAMES):
        offset = (i - 2.5) * width
        ax.bar(
            x + offset, test.loc[experiment, metrics] * 100, width, color=COLORS[experiment], label=NAMES[experiment]
        )
        ax.scatter(x + offset, val.loc[experiment, metrics] * 100, color="black", marker="_", s=120, zorder=3)
    ax.set_xticks(x, [LABELS_FR[m] for m in metrics])
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Triage : 6 experiences sur le test (barres) et la validation (traits noirs)")
    ax.legend(ncol=3, fontsize=9, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "01_metriques_triage.png")


def fig_errors(results: dict, cost: dict) -> None:
    rows = {NAMES[e]: results["test"].loc[e] for e in NAMES}
    for e in TRAINED:
        rows[NAMES[e] + " + cout"] = cost["test"].loc[e + " + cout"]
    frame = pd.DataFrame(rows).T[["accuracy", "sur_triage", "sous_triage"]] * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    left = np.zeros(len(frame))
    for column, color, label in [
        ("accuracy", "#00A86B", "correct"),
        ("sur_triage", "#0066FF", "sur-triage"),
        ("sous_triage", "#E63946", "sous-triage"),
    ]:
        ax.barh(frame.index, frame[column], left=left, color=color, label=label)
        for y, (value, start) in enumerate(zip(frame[column], left)):
            if value >= 4:
                ax.text(start + value / 2, y, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8)
        left += frame[column].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% des questions du test")
    ax.set_title(f"Erreurs de triage : argmax puis regle de cout deployee (alpha {ALPHA_LABEL})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    save(fig, "02_sous_sur_triage.png")


def fig_confusions(predictions: pd.DataFrame, cost_predictions: dict) -> None:
    panels = [
        ("4 SFT + triage", predictions["4_sft_medical_triage"]),
        ("4 SFT + triage + cout", cost_predictions["4_sft_medical_triage"].loc[predictions["id"]].to_numpy()),
        ("5 DPO v1", predictions["5_sft_triage_dpo"]),
        ("6 DPO v2", predictions["6_sft_triage_dpo_qualite"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    short = [SHORT[label] for label in LABELS]
    for ax, (title, pred) in zip(axes, panels):
        matrix = confusion_matrix(predictions["priorite"], pred, labels=LABELS)
        ax.imshow(matrix, cmap="Oranges")
        for i in range(3):
            for j in range(3):
                color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, matrix[i, j], ha="center", va="center", color=color, fontsize=11)
        ax.set_xticks(range(3), short)
        ax.set_yticks(range(3), short)
        ax.set_xlabel("predit")
        ax.set_ylabel("annote")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Matrices de confusion sur le test (400 questions) : au-dessus de la diagonale = sous-triage")
    save(fig, "03_matrices_confusion.png")


def fig_cost_tradeoff() -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    rows = []
    for experiment in ["4_sft_medical_triage", "6_sft_triage_dpo_qualite"]:
        splits = cost_sensitive.load(experiment)
        temperature = cost_sensitive.fit_temperature(*reversed(splits["validation"]))
        for ax, split in zip(axes, ["validation", "test"]):
            y, scores = splits[split]
            proba = cost_sensitive.probabilities(scores, temperature)
            sweep = pd.DataFrame(
                {
                    a: cost_sensitive.metrics(y, cost_sensitive.decide(proba, cost_sensitive.scaled_cost(a)))
                    for a in cost_sensitive.ALPHAS
                }
            ).T
            ax.plot(
                sweep["sur_triage"] * 100,
                sweep["sous_triage"] * 100,
                "o-",
                color=COLORS[experiment],
                label=NAMES[experiment],
            )
            # Seul le point deploye est entoure et annote, texte decale par modele pour rester lisible.
            deployed = (sweep.loc[DEPLOYED_ALPHA, "sur_triage"] * 100, sweep.loc[DEPLOYED_ALPHA, "sous_triage"] * 100)
            ax.scatter(*deployed, s=160, facecolors="none", edgecolors=COLORS[experiment], linewidths=2, zorder=4)
            ax.annotate(
                f"alpha {ALPHA_LABEL}",
                deployed,
                fontsize=8,
                color=COLORS[experiment],
                xytext=(8, 8) if experiment == "4_sft_medical_triage" else (8, -16),
                textcoords="offset points",
            )
            argmax = cost_sensitive.metrics(y, scores.idxmax(axis=1))
            ax.scatter(
                argmax["sur_triage"] * 100,
                argmax["sous_triage"] * 100,
                marker="*",
                s=160,
                color=COLORS[experiment],
                edgecolor="black",
                zorder=3,
            )
            ax.set_title(split)
            ax.set_xlabel("sur-triage (%)")
            ax.grid(alpha=0.3)
            rows.append({"experience": experiment, "split": split, **sweep.loc[DEPLOYED_ALPHA].to_dict()})
    axes[0].set_ylabel("sous-triage (%)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Compromis sous / sur-triage selon le poids alpha (etoile = argmax, sans regle de cout)")
    save(fig, "04_compromis_cout.png")
    return pd.DataFrame(rows)


def fig_sft_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, experiment in zip(axes, ["3_sft_medical", "4_sft_medical_triage"]):
        summary = json.loads((adapter_dir(experiment) / "training_summary.json").read_text(encoding="utf-8"))
        train = pd.DataFrame(summary["train_loss_history"])
        evals = pd.DataFrame(summary["eval_loss_history"])
        ax.plot(
            train["step"],
            train["loss"].rolling(10, min_periods=1).mean(),
            color="#8A8A8A",
            label="train (moyenne glissante)",
        )
        ax.plot(evals["step"], evals["eval_loss"], "o-", color=COLORS[experiment], label="validation")
        best = evals.loc[evals["eval_loss"].idxmin()]
        ax.axvline(best["step"], ls="--", color="black", lw=1)
        ax.annotate(
            f"meilleur : pas {int(best['step'])}\n{best['eval_loss']:.3f}",
            (best["step"], best["eval_loss"]),
            xytext=(10, 30),
            textcoords="offset points",
            fontsize=9,
        )
        ax.set_title(NAMES[experiment])
        ax.set_xlabel("pas")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("loss")
    fig.suptitle("SFT LoRA : loss d'entrainement et de validation (checkpoint retenu en pointille)")
    save(fig, "05_courbes_sft.png")


def fig_dpo_curves() -> None:
    """Loss et marge de recompense en validation, separees par type de paire."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    panels = [
        ("eval_qualite_loss", "eval_qualite_loss", "loss, paires qualite"),
        ("eval_securite_triage_loss", "eval_securite_triage_loss", "loss, paires securite"),
        ("eval_qualite_loss", "eval_rewards/margins", "marge chosen - rejected, qualite"),
        ("eval_securite_triage_loss", "eval_rewards/margins", "marge chosen - rejected, securite"),
    ]
    for experiment in ["5_sft_triage_dpo", "6_sft_triage_dpo_qualite"]:
        history = json.loads((adapter_dir(experiment) / "training_summary.json").read_text(encoding="utf-8"))
        frame = pd.DataFrame(history["log_history"])
        for ax, (row_key, column, title) in zip(axes, panels):
            # Chaque evaluation par type de paire est une entree distincte de l'historique.
            points = frame.dropna(subset=[row_key])
            ax.plot(points["step"], points[column], "o-", color=COLORS[experiment], label=NAMES[experiment])
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("pas")
            ax.grid(alpha=0.3)
    for ax in axes:
        # v1 : coupure du PC au pas 77, reprise qui a perdu les poids de l'adaptateur ; l'adaptateur evalue est celui du pas 75.
        ax.axvline(76, color=COLORS["5_sft_triage_dpo"], ls=":", lw=1.2)
        ax.legend(fontsize=9)
    axes[0].annotate(
        "v1 : reprise defectueuse\n(adaptateur evalue = pas 75)",
        (77, 0.55),
        xycoords=("data", "axes fraction"),
        fontsize=8,
        va="top",
        color=COLORS["5_sft_triage_dpo"],
    )
    fig.suptitle("DPO en validation : v1 (qualite + securite) contre v2 (qualite seule)")
    save(fig, "06_courbes_dpo.png")


def fig_responses() -> None:
    path = OUTPUTS_DIR / "response_eval" / "qualite_reponses.csv"
    frame = pd.read_csv(path, index_col=0)
    metrics = {
        "rouge_l": "ROUGE-L",
        "f1_lexical": "F1 lexical",
        "niveau_correct": "niveau ecrit correct",
        "chiffre_invente": "chiffre a relire",
        "coupee": "coupee",
        "vide": "vide",
    }
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(metrics))
    width = 0.2
    for i, experiment in enumerate(frame.index):
        ax.bar(
            x + (i - 1.5) * width,
            frame.loc[experiment, list(metrics)] * 100,
            width,
            color=COLORS[experiment],
            label=NAMES[experiment],
        )
    ax.set_xticks(x, list(metrics.values()))
    ax.set_ylabel("%")
    ax.set_title("Reponses aux questions du test (80 questions fixes) : metriques automatiques")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "07_qualite_reponses.png")


def fig_judgments() -> None:
    sets = [
        (
            "reponses",
            OUTPUTS_DIR / "response_eval",
            "jugement_aveugle",
            ["pertinence", "clarte", "securite", "conformite"],
        ),
        (
            "explications du niveau",
            OUTPUTS_DIR / "explanation_eval",
            "jugement_explications",
            ["coherence", "ancrage", "exactitude", "securite", "clarte"],
        ),
    ]
    available = [s for s in sets if (s[1] / f"{s[2]}_notes.jsonl").exists()]
    fig, axes = plt.subplots(1, len(available), figsize=(6.5 * len(available), 4.3), squeeze=False)
    for ax, (title, folder, prefix, criteria) in zip(axes[0], available):
        key = json.loads((folder / f"{prefix}_cle.json").read_text(encoding="utf-8"))
        rows, wins = [], {"4_sft_medical_triage": 0, "6_sft_triage_dpo_qualite": 0, "egalite": 0}
        for line in (folder / f"{prefix}_notes.jsonl").read_text(encoding="utf-8").splitlines():
            note = json.loads(line)
            item = key[str(note["item"])]
            for side in "AB":
                rows.append({"modele": item[side], **{c: note[side][c] for c in criteria}})
            wins[item[note["preference"]] if note["preference"] in "AB" else "egalite"] += 1
        means = pd.DataFrame(rows).groupby("modele")[criteria].mean()
        x = np.arange(len(criteria))
        for i, experiment in enumerate(means.index):
            ax.bar(
                x + (i - 0.5) * 0.38,
                means.loc[experiment],
                0.38,
                color=COLORS[experiment],
                label=f"{NAMES[experiment]} (prefere {wins[experiment]})",
            )
        ax.set_xticks(x, criteria)
        ax.set_ylim(0, 5.2)
        ax.set_title(f"{title} : notes moyennes (1-5), egalites {wins['egalite']}", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Jugement en aveugle SFT contre DPO v2 (juge : LLM, sans clinicien)")
    save(fig, "08_jugement_aveugle.png")


def fig_bootstrap(intervals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "sous_triage", "Sous-triage")]:
        y = np.arange(len(intervals))
        values = intervals[metric] * 100
        errors = [values - intervals[f"{metric}_bas"] * 100, intervals[f"{metric}_haut"] * 100 - values]
        colors = [COLORS.get(name.replace(" + cout", ""), "#000000") for name in intervals.index]
        ax.barh(y, values, color=colors, alpha=0.85)
        ax.errorbar(values, y, xerr=errors, fmt="none", ecolor="black", capsize=3)
        ax.set_yticks(y, [NAMES.get(n, NAMES.get(n.replace(" + cout", ""), n) + " + cout") for n in intervals.index])
        ax.invert_yaxis()
        ax.set_xlabel("%")
        ax.set_title(f"{title} sur le test, IC 95 % (bootstrap sur les groupes)")
        ax.grid(axis="x", alpha=0.3)
    save(fig, "09_intervalles_confiance.png")


def main() -> None:
    alpha = load_decision_rule()["alpha"]
    results = argmax_results()
    cost, cost_predictions = cost_rule_results(alpha)
    predictions = pd.read_csv(TRIAGE_EVAL_DIR / "predictions.csv")
    rank_pred = predictions.copy()
    columns = {e: e for e in NAMES}
    for experiment, pred in cost_predictions.items():
        rank_pred[experiment + " + cout"] = pred.loc[predictions["id"]].to_numpy()
        columns[experiment + " + cout"] = experiment + " + cout"
    intervals = bootstrap(rank_pred, columns)

    print("Graphiques :")
    fig_metrics(results)
    fig_errors(results, cost)
    fig_confusions(predictions, cost_predictions)
    sweep = fig_cost_tradeoff()
    fig_sft_curves()
    fig_dpo_curves()
    fig_responses()
    fig_judgments()
    fig_bootstrap(intervals)

    table = {}
    for split in ["validation", "test"]:
        table[split] = pd.concat([results[split][MAIN], cost[split][MAIN]]).loc[
            [*NAMES, *(f"{e} + cout" for e in TRAINED)]
        ]
    combined = pd.concat(table, names=["split", "experience"])
    combined.to_csv(TRIAGE_EVAL_DIR / "comparaison_modeles.csv")
    temperatures = cost["validation"]["temperature"].rename(lambda n: n.replace(" + cout", ""))
    shown_intervals = pd.DataFrame(
        {
            "accuracy": [
                f"{r.accuracy * 100:.1f} [{r.accuracy_bas * 100:.1f} ; {r.accuracy_haut * 100:.1f}]"
                for r in intervals.itertuples()
            ],
            "sous_triage": [
                f"{r.sous_triage * 100:.1f} [{r.sous_triage_bas * 100:.1f} ; {r.sous_triage_haut * 100:.1f}]"
                for r in intervals.itertuples()
            ],
        },
        index=intervals.index,
    )
    lines = [
        "# Comparaison finale des modeles",
        "",
        "Genere par `python -m triage.evaluation.compare_models` a partir des scores sauvegardes (sans GPU). "
        f"Valeurs en %. `+ cout` : regle de decision deployee (alpha = {alpha:g}, matrice de `decision_rule.json`), "
        "temperature ajustee sur la validation pour chaque modele, appliquee telle quelle au test.",
        "",
        "## Test (400 questions, 109 cas)",
        "",
        markdown_table(table["test"].rename_axis("experience") * 100),
        "",
        "## Validation (390 questions)",
        "",
        markdown_table(table["validation"].rename_axis("experience") * 100),
        "",
        "Temperatures ajustees sur la validation : " + ", ".join(f"{k} {v:g}" for k, v in temperatures.items()) + ".",
        "",
        "## Intervalles de confiance a 95 % sur le test (bootstrap sur les groupes de cas)",
        "",
        f"{N_BOOTSTRAP} tirages avec remise des {predictions['groupe'].nunique()} groupes.",
        "",
        markdown_table(shown_intervals.rename_axis("experience")),
        "",
        f"## Regle de cout a alpha {ALPHA_LABEL} : SFT contre DPO v2",
        "",
        markdown_table(
            sweep.set_index(["experience", "split"])[["accuracy", "f1_macro", "rappel_P1", "sous_triage", "sur_triage"]]
            * 100
        ),
        "",
        "Graphiques : `outputs/figures/`.",
    ]
    (TRIAGE_EVAL_DIR / "comparaison_modeles.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[5:]))


if __name__ == "__main__":
    main()
