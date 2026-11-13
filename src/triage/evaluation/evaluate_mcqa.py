"""Evaluation QCM (FrenchMedMCQA) du modele SFT final, sur le test du dataset Hugging Face.

Generation gloutonne a partir du prompt du dataset (meme gabarit qu'a
l'entrainement), puis comparaison des lettres "(A)".."(E)" citees dans la
premiere ligne de la reponse avec celles de la reference.

    python -m triage.evaluation.evaluate_mcqa [--adapter outputs/dpo_triage_qualite]

Sortie : outputs/final_eval/mcqa_<adaptateur>.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from triage.constants import FINAL_EXPERIMENT, OUTPUTS_DIR, adapter_dir
from triage.utils import hf_dataset

OUTPUT_DIR = OUTPUTS_DIR / "final_eval"
MAX_NEW_TOKENS = 64
LETTER_RE = re.compile(r"\(([A-E])\)")


def extract_letters(text: str) -> set[str]:
    first_line = (text or "").strip().split("\n", 1)[0]
    return set(LETTER_RE.findall(first_line))


def prf1(pred: set, ref: set) -> tuple[float, float, float]:
    if not pred and not ref:
        return 1.0, 1.0, 1.0
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    tp = len(pred & ref)
    precision, recall = tp / len(pred), tp / len(ref)
    f1 = 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def main() -> None:
    from triage.utils import modeling

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=adapter_dir(FINAL_EXPERIMENT))
    args = parser.parse_args()

    frame = hf_dataset.load_sft(with_priority=True)
    sample = frame[(frame["split"] == "test") & (frame["type_question"] == "qcm")]
    tokenizer = modeling.load_tokenizer()
    model = modeling.load_model(args.adapter)
    # Un prompt a la fois, sans penalite de repetition : les lettres citees ne doivent pas etre penalisees.
    generated, _ = modeling.generate(
        model, tokenizer, sample["prompt"].tolist(), MAX_NEW_TOKENS, batch_size=1, repetition_penalty=1.0
    )

    rows = []
    for row, text in zip(sample.itertuples(), generated):
        ref_letters, pred_letters = extract_letters(row.reponse), extract_letters(text)
        precision, recall, f1 = prf1(pred_letters, ref_letters)
        rows.append(
            {
                "id": row.id,
                "source": row.source,
                "prompt": row.prompt,
                "reference": row.reponse,
                "generated": text,
                "reference_letters": "".join(sorted(ref_letters)),
                "predicted_letters": "".join(sorted(pred_letters)),
                "exact_match": pred_letters == ref_letters,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    report = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"mcqa_{args.adapter.name}.csv"
    report.to_csv(output_path, index=False)
    print(f"Dataset : {hf_dataset.dataset_name()} - adaptateur : {args.adapter}")
    print(f"N = {len(report)} QCM du test")
    print(f"Exact match : {report['exact_match'].mean():.3f}")
    print(f"F1 moyen    : {report['f1'].mean():.3f}")
    print(f"Detail ecrit dans {output_path}")


if __name__ == "__main__":
    main()
