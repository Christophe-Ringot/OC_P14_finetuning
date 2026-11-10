"""Evaluation QCM (FrenchMedMCQA) du modele SFT final, sur le test du dataset Hugging Face.

Generation gloutonne a partir du prompt du dataset (meme gabarit qu'a
l'entrainement), puis comparaison des lettres "(A)".."(E)" citees dans la
premiere ligne de la reponse avec celles de la reference.

    python -m scripts.evaluate_mcqa [--adapter outputs/sft_medical_triage]
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts import hf_dataset
from scripts.train_sft import MODEL_NAME, adapter_dir

OUTPUT_PATH = Path("outputs") / "sft_mcqa_eval.csv"
MAX_NEW_TOKENS = 64
LETTER_RE = re.compile(r"\(([A-E])\)")


def extract_letters(text: str) -> set:
    first_line = (text or "").strip().split("\n", 1)[0]
    return set(LETTER_RE.findall(first_line))


def prf1(pred: set, ref: set) -> tuple:
    if not pred and not ref:
        return 1.0, 1.0, 1.0
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    tp = len(pred & ref)
    precision = tp / len(pred)
    recall = tp / len(ref)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=adapter_dir("4_sft_medical_triage", quick=False))
    args = parser.parse_args()

    frame = hf_dataset.load_sft(with_priority=True)
    sample = frame[(frame["split"] == "test") & (frame["type_question"] == "qcm")]

    has_gpu = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16 if has_gpu else torch.float32, device_map={"": 0} if has_gpu else None
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter)).eval()

    rows = []
    for i, row in enumerate(sample.itertuples(), start=1):
        inputs = tokenizer(row.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        ref_letters = extract_letters(row.reponse)
        pred_letters = extract_letters(generated_text)
        precision, recall, f1 = prf1(pred_letters, ref_letters)
        rows.append({
            "id": row.id,
            "source": row.source,
            "prompt": row.prompt,
            "reference": row.reponse,
            "generated": generated_text,
            "reference_letters": "".join(sorted(ref_letters)),
            "predicted_letters": "".join(sorted(pred_letters)),
            "exact_match": pred_letters == ref_letters,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}")

    report = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset : {hf_dataset.dataset_name()} - adaptateur : {args.adapter}")
    print(f"N = {len(report)} QCM du test")
    print(f"Exact match : {report['exact_match'].mean():.3f}")
    print(f"F1 moyen    : {report['f1'].mean():.3f}")
    print(f"Detail ecrit dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
