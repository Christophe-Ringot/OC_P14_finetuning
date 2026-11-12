"""Qualite des reponses generees (pas seulement le niveau de triage).

Chaque modele genere une reponse (decodage glouton) pour un echantillon fixe du
test : 20 cas cliniques annotes par niveau (P1, P2, P3) et 20 questions
generales (FrenchMedMCQA, MedQuAD). Le meme echantillon sert avant et apres
DPO, pour mesurer l'apport de chaque phase.

Metriques automatiques, par reponse puis moyennees :
- pertinence : ROUGE-L et F1 lexical par rapport a la reponse de reference ;
- conformite : prefixe "[Niveau de priorite estime : ...]" present sur les cas
  annotes, et niveau ecrit = niveau annote ;
- securite : formulations rassurantes a risque et chiffres absents du cas et de
  la reference (safety_check), sous-triage dans le prefixe ;
- clarte : longueur, reponse coupee (limite de tokens atteinte), repetitions.

Ces metriques sont des filtres grossiers : la grille de jugement en aveugle
(docs/evaluation_reponses.md) complete l'evaluation sur les memes reponses.

Sorties : outputs/response_eval/generations_<modele>.csv et qualite_reponses.md
"""
import argparse
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer
from triage.evaluation.evaluate_sft import LABELS, PRIORITY_PREFIX, SHORT, free_gpu_memory, load_model
from triage.evaluation.safety_check import dangerous_language, numeric_hallucination
from triage.training.train_sft import MODEL_NAME
from triage.utils import hf_dataset, tracking

OUTPUT_DIR = Path("outputs") / "response_eval"
MODELS = {
    "2_modele_base": None,
    "4_sft_medical_triage": Path("outputs") / "sft_medical_triage",
    "5_sft_triage_dpo": Path("outputs") / "dpo_triage",
    "6_sft_triage_dpo_qualite": Path("outputs") / "dpo_triage_qualite",
}
PER_LEVEL = 20
GENERAL = 20
MAX_NEW_TOKENS = 300
BATCH_SIZE = 8
SEED = 42
LEVEL_RE = re.compile(r"\[Niveau de priorité estimé : ([a-z_]+)\]")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def sample() -> pd.DataFrame:
    frame = hf_dataset.load_sft(with_priority=True)
    test = frame[frame["split"] == "test"]
    annotated = test[test["priorite"].notna()]
    # Un seul exemple par cas clinique, pour ne pas peser plusieurs fois le meme cas.
    annotated = annotated.groupby("case_id").sample(1, random_state=SEED)
    parts = [annotated[annotated["priorite"] == label].sample(PER_LEVEL, random_state=SEED) for label in LABELS]
    parts.append(test[test["priorite"].isna()].sample(GENERAL, random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)


def generate(adapter: Path | None, prompts: list) -> tuple[list, list]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = load_model(adapter)
    texts, truncated = [], []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = tokenizer(prompts[start:start + BATCH_SIZE], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
        new_tokens = output[:, batch["input_ids"].shape[1]:]
        for row in new_tokens:
            text = tokenizer.decode(row, skip_special_tokens=True)
            # Le modele de base enchaine parfois un nouveau "### Cas clinique" : on coupe.
            text = text.split("\n### ")[0].strip()
            texts.append(text)
            truncated.append(bool((row != tokenizer.pad_token_id).sum() >= MAX_NEW_TOKENS))
        print(f"  {min(start + BATCH_SIZE, len(prompts))}/{len(prompts)}")
    del model
    free_gpu_memory()
    return texts, truncated


def words(text: str) -> list:
    return WORD_RE.findall(text.lower())


def strip_prefix(text: str) -> str:
    return LEVEL_RE.sub("", text, count=1).strip()


def rouge_l(reference: str, candidate: str) -> float:
    ref, cand = words(reference), words(candidate)
    if not ref or not cand:
        return 0.0
    previous = [0] * (len(cand) + 1)
    for r in ref:
        current = [0]
        for j, c in enumerate(cand, start=1):
            current.append(previous[j - 1] + 1 if r == c else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def token_f1(reference: str, candidate: str) -> float:
    common = Counter(words(reference)) & Counter(words(candidate))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(words(candidate)), overlap / len(words(reference))
    return 2 * precision * recall / (precision + recall)


def repetition(text: str) -> float:
    """Part de trigrammes repetes (0 = aucun)."""
    tokens = words(text)
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    return 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0


def score(frame: pd.DataFrame) -> pd.DataFrame:
    rank = {label: i for i, label in enumerate(LABELS)}
    rows = []
    for row in frame.itertuples(index=False):
        generated, reference = row.generated, row.reponse
        level = LEVEL_RE.search(generated)
        annotated = isinstance(row.priorite, str)
        body = strip_prefix(generated)
        rows.append({
            "rouge_l": rouge_l(strip_prefix(reference), body),
            "f1_lexical": token_f1(strip_prefix(reference), body),
            "prefixe_present": float(bool(level)) if annotated else float("nan"),
            "niveau_correct": float(bool(level) and level.group(1) == row.priorite) if annotated else float("nan"),
            "sous_triage_ecrit": float(bool(level) and level.group(1) in rank and rank[level.group(1)] > rank[row.priorite])
            if annotated else float("nan"),
            "formulation_risquee": float(bool(dangerous_language(generated))),
            "chiffre_invente": float(bool(numeric_hallucination(row.prompt, reference, generated))),
            "longueur_mots": len(words(body)),
            "coupee": float(row.truncated),
            "repetition": repetition(body),
            "vide": float(len(words(body)) < 3),
        })
    return pd.DataFrame(rows)


def summarize(results: dict) -> pd.DataFrame:
    summary = {}
    for name, frame in results.items():
        metrics = frame.filter(regex="^(rouge_l|f1_lexical|prefixe_present|niveau_correct|sous_triage_ecrit|formulation_risquee|"
                                     "chiffre_invente|longueur_mots|coupee|repetition|vide)$")
        summary[name] = metrics.mean()
        for label in LABELS:
            summary[name][f"rouge_l_{SHORT[label]}"] = frame.loc[frame["priorite"] == label, "rouge_l"].mean()
        summary[name]["rouge_l_general"] = frame.loc[frame["priorite"].isna(), "rouge_l"].mean()
    return pd.DataFrame(summary).T


def write_report(summary: pd.DataFrame, n: int) -> None:
    percent = [c for c in summary.columns if c != "longueur_mots"]
    shown = summary.copy()
    shown[percent] = shown[percent] * 100
    header = "| modele | " + " | ".join(shown.columns) + " |\n|" + "---|" * (len(shown.columns) + 1)
    rows = [f"| {name} | " + " | ".join(f"{v:.1f}" for v in row) + " |" for name, row in shown.iterrows()]
    lines = [
        "# Qualite des reponses generees",
        "",
        f"Echantillon fixe du test : {n} questions ({PER_LEVEL} cas par niveau P1/P2/P3, un par cas clinique, "
        f"et {GENERAL} questions generales). Decodage glouton, {MAX_NEW_TOKENS} tokens max. Valeurs en % "
        "(longueur en mots). `rouge_l` et `f1_lexical` comparent a la reponse de reference sans le prefixe de niveau.",
        "",
        header, *rows,
        "",
        "Lecture :",
        "- `niveau_correct` porte sur le niveau ecrit par le modele en generation libre ; la metrique de triage de "
        "reference reste celle des log-probabilites (comparaison_triage.md), plus stable.",
        "- `formulation_risquee` et `chiffre_invente` sont des signaux a relire, pas des erreurs averees "
        "(un chiffre calcule a partir du cas est signale aussi).",
        "- ROUGE-L recompense la proximite avec une reference courte : il mesure la pertinence par rapport au "
        "corrige, pas la qualite clinique. Le jugement en aveugle (docs/evaluation_reponses.md) la complete.",
    ]
    (OUTPUT_DIR / "qualite_reponses.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    tracked = tracking.setup(not args.no_mlflow)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = sample()

    for name in args.models:
        adapter = MODELS[name]
        if adapter is not None and not (adapter / "adapter_config.json").exists():
            print(f"{name} ignore : adaptateur absent ({adapter})")
            continue
        print(f"Generation {name} ({len(questions)} questions)")
        texts, truncated = generate(adapter, questions["prompt"].tolist())
        frame = questions[["id", "case_id", "source", "priorite", "prompt", "reponse"]].assign(generated=texts, truncated=truncated)
        frame.to_csv(OUTPUT_DIR / f"generations_{name}.csv", index=False)

    # Le rapport reprend tous les modeles deja generes (avant et apres DPO).
    results = {}
    for name in MODELS:
        path = OUTPUT_DIR / f"generations_{name}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["generated"] = frame["generated"].fillna("")
            results[name] = pd.concat([frame, score(frame)], axis=1)
            results[name].to_csv(OUTPUT_DIR / f"scores_{name}.csv", index=False)
    summary = summarize(results)
    summary.to_csv(OUTPUT_DIR / "qualite_reponses.csv")
    write_report(summary, len(questions))
    print(summary[["rouge_l", "f1_lexical", "niveau_correct", "sous_triage_ecrit", "formulation_risquee", "chiffre_invente", "coupee"]].round(3).to_string())

    if tracked:
        import mlflow
        with tracking.start_run(True, run_name="evaluation_reponses", tags={"type": "evaluation", "phase": "qualite_reponses"}):
            mlflow.log_params({"n_questions": len(questions), "max_new_tokens": MAX_NEW_TOKENS, "decodage": "glouton, repetition_penalty 1.1"})
            for name, row in summary.iterrows():
                mlflow.log_metrics({f"{name}.{k}": float(v) for k, v in row.items() if pd.notna(v)})
            mlflow.log_artifacts(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
