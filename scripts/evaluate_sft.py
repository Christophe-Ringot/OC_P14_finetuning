import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.train_sft import MODEL_NAME, SFT_EXPERIMENTS, adapter_dir

TEST_PATH = SFT_EXPERIMENTS["4_sft_medical_triage"]["data"]
OUTPUT_DIR = Path("outputs") / "triage_eval"
QUICK_TEST_EXAMPLES = 30

# Ordre d'urgence : P1 (le plus urgent) -> P3.
LABELS = ["urgence_maximale", "urgence_moderee", "differee"]
SHORT = {"urgence_maximale": "P1", "urgence_moderee": "P2", "differee": "P3"}
PRIORITY_PREFIX = "[Niveau de priorité estimé :"
# Meme gabarit que les prompts du test, sans cas ni question.
NEUTRAL_PROMPT = "### Cas clinique :\n\n\n### Question :\n\n\n### Réponse :\n"


def load_model(adapter: Path | None):
    has_gpu = torch.cuda.is_available()
    # Tout sur le GPU : un manque de memoire doit lever une erreur, pas
    # decharger silencieusement des couches sur le CPU (evaluation tres lente).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16 if has_gpu else torch.float32, device_map={"": 0} if has_gpu else None
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval()


def label_logprobs(model, tokenizer, prompt: str) -> dict:
    """Log-probabilite de " <niveau>]" apres le prompt et le debut du prefixe."""
    context = tokenizer(prompt + PRIORITY_PREFIX, add_special_tokens=False)["input_ids"]
    scores = {}
    for label in LABELS:
        continuation = tokenizer(f" {label}]", add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([context + continuation], device=model.device)
        with torch.no_grad():
            # Seules les positions qui predisent la continuation sont utiles.
            logits = model(input_ids=input_ids, logits_to_keep=len(continuation) + 1).logits[0, :-1].float()
        targets = torch.tensor(continuation, device=model.device).unsqueeze(1)
        scores[label] = logits.log_softmax(-1).gather(1, targets).sum().item()
    return scores


def predict_with_model(test: pd.DataFrame, adapter: Path | None) -> list:
    """Niveau le plus probable apres calibration contextuelle (Zhao et al., 2021).

    Les libelles different en longueur et en frequence : sans correction, un
    modele qui ne les a jamais vus choisit d'apres leur forme et non d'apres le
    cas. On retranche donc a chaque score celui obtenu sur un prompt vide : seul
    reste ce que le cas clinique ajoute a la probabilite de chaque niveau.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(adapter)
    prior = label_logprobs(model, tokenizer, NEUTRAL_PROMPT)
    predictions = []
    for i, prompt in enumerate(test["prompt"], start=1):
        scores = label_logprobs(model, tokenizer, prompt)
        calibrated = {label: scores[label] - prior[label] for label in LABELS}
        predictions.append(max(calibrated, key=calibrated.get))
        if i % 50 == 0:
            print(f"  {i}/{len(test)}")
    del model
    free_gpu_memory()
    return predictions


def free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def triage_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    rank = {label: i for i, label in enumerate(LABELS)}
    gap = y_pred.map(rank) - y_true.map(rank)  # > 0 : predit moins urgent
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    metrics = {
        "accuracy": (gap == 0).mean(),
        "f1_macro": f1.mean(),
        "rappel_P1": recall[0],
        "sous_triage": (gap > 0).mean(),
        "sur_triage": (gap < 0).mean(),
        "erreurs_1_niveau": (gap.abs() == 1).mean(),
        "erreurs_2_niveaux": (gap.abs() == 2).mean(),
        "P1_predit_P3": ((y_true == LABELS[0]) & (y_pred == LABELS[2])).sum() / max((y_true == LABELS[0]).sum(), 1),
    }
    for i, label in enumerate(LABELS):
        metrics.update({f"precision_{SHORT[label]}": precision[i], f"rappel_{SHORT[label]}": recall[i], f"f1_{SHORT[label]}": f1[i]})
    return metrics


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| experience | " + " | ".join(frame.columns) + " |\n|" + "---|" * (len(frame.columns) + 1) + "\n"
    return header + "\n".join(f"| {name} | " + " | ".join(f"{v:.1f}" for v in row) + " |" for name, row in frame.iterrows())


def confusion_table(y_true: pd.Series, y_pred: pd.Series) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    header = "| annote \\ predit | " + " | ".join(SHORT[label] for label in LABELS) + " |\n|---|---|---|---|\n"
    rows = [f"| {SHORT[label]} | " + " | ".join(str(n) for n in matrix[i]) + " |" for i, label in enumerate(LABELS)]
    return header + "\n".join(rows)


def write_report(test: pd.DataFrame, results: pd.DataFrame, path: Path) -> None:
    percent = results.astype(float) * 100
    main_columns = ["accuracy", "f1_macro", "rappel_P1", "sous_triage", "sur_triage", "erreurs_1_niveau", "erreurs_2_niveaux", "P1_predit_P3"]
    class_columns = [c for c in results.columns if c not in main_columns]
    distribution = test["priorite"].map(SHORT).value_counts().reindex(["P1", "P2", "P3"])

    lines = [
        "# Comparaison du triage - 4 experiences",
        "",
        f"Jeu de test : {len(test)} questions sur {test['groupe'].nunique()} groupes de cas cliniques "
        f"(P1 = {distribution['P1']}, P2 = {distribution['P2']}, P3 = {distribution['P3']}).",
        "Valeurs en %. Sous-triage / sur-triage : niveau predit moins / plus urgent que l'annotation, "
        "rapporte a tous les exemples. P1_predit_P3 : part des P1 classes P3.",
        "",
        "## Metriques principales",
        "",
        markdown_table(percent[main_columns]),
        "",
        "## Precision, rappel et F1 par classe",
        "",
        markdown_table(percent[class_columns]),
        "",
        "## Matrices de confusion",
    ]
    for experiment in results.index:
        lines += ["", f"### {experiment}", "", confusion_table(test["priorite"], test[experiment])]
    lines += [
        "",
        "## Limites",
        "",
        "- Etiquettes de reference pre-annotees par LLM, sans validation clinique (docs/triage_grid.md).",
        "- Les questions d'un meme cas partagent le meme niveau : les exemples ne sont pas independants.",
        "- La baseline naive vaut la part de P1 dans ce jeu de test, pas 33 % (classes non equilibrees).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(quick: bool = False) -> None:
    test = pd.read_csv(TEST_PATH)
    test = test[(test["split"] == "test") & test["priorite"].notna()].reset_index(drop=True)
    if quick:
        test = test.sample(min(QUICK_TEST_EXAMPLES, len(test)), random_state=42).reset_index(drop=True)
    suffix = "_quick" if quick else ""

    test["1_baseline_naive"] = LABELS[0]
    print(f"Evaluation de 2_modele_base sur {len(test)} exemples")
    test["2_modele_base"] = predict_with_model(test, adapter=None)
    for experiment in SFT_EXPERIMENTS:
        adapter = adapter_dir(experiment, quick)
        if not (adapter / "adapter_config.json").exists():
            print(f"{experiment} ignoree : adaptateur absent ({adapter})")
            continue
        print(f"Evaluation de {experiment}")
        test[experiment] = predict_with_model(test, adapter)

    experiments = [c for c in test.columns if c[0].isdigit()]
    results = pd.DataFrame({e: triage_metrics(test["priorite"], test[e]) for e in experiments}).T

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test[["id", "groupe", "priorite", *experiments]].to_csv(OUTPUT_DIR / f"predictions{suffix}.csv", index=False)
    results.to_csv(OUTPUT_DIR / f"comparaison_triage{suffix}.csv")
    report = OUTPUT_DIR / f"comparaison_triage{suffix}.md"
    write_report(test, results, report)

    print((results[["accuracy", "f1_macro", "rappel_P1", "sous_triage", "sur_triage"]] * 100).round(1).to_string())
    print(f"\nRapport -> {report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="evalue les adaptateurs _quick sur 30 exemples")
    run(quick=parser.parse_args().quick)


if __name__ == "__main__":
    main()
