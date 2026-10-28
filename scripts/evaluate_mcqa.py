import re

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
ADAPTER_DIR = "outputs/sft_lora_run5"
DATA_PATH = "data/processed/sft_dataset_anonymized.csv"
OUTPUT_PATH = "outputs/sft_mcqa_eval.csv"

N_PER_SOURCE = 30
MAX_NEW_TOKENS = 200 if torch.cuda.is_available() else 80

LETTER_RE = re.compile(r"\b([A-E])\)")


def build_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Reponse:\n"


def extract_letters(text: str) -> set:
    return set(LETTER_RE.findall(text or ""))


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
    df = pd.read_csv(DATA_PATH)
    test_df = df[(df["split"] == "test") & (df["task_type"].isin(["qcm", "clinical_case_qcm"]))]

    sample = pd.concat(
        g.sample(min(N_PER_SOURCE, len(g)), random_state=42)
        for _, g in test_df.groupby("source_dataset")
    )

    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    device_map = "auto" if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map=device_map)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    rows = []
    for _, row in sample.iterrows():
        prompt = build_prompt(row["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        ref_letters = extract_letters(row["response"])
        pred_letters = extract_letters(generated_text)
        precision, recall, f1 = prf1(pred_letters, ref_letters)

        rows.append({
            "source_dataset": row["source_dataset"],
            "task_type": row["task_type"],
            "instruction": row["instruction"],
            "reference": row["response"],
            "generated": generated_text,
            "reference_letters": "".join(sorted(ref_letters)),
            "predicted_letters": "".join(sorted(pred_letters)),
            "exact_match": pred_letters == ref_letters,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    report_df = pd.DataFrame(rows)
    report_df.to_csv(OUTPUT_PATH, index=False)

    print(f"N = {len(report_df)} exemples QCM evalues")
    print(f"Exact match global : {report_df['exact_match'].mean():.3f}")
    print(f"F1 moyen global     : {report_df['f1'].mean():.3f}")
    print()
    summary = report_df.groupby("source_dataset")[["exact_match", "precision", "recall", "f1"]].mean()
    print(summary)
    print(f"\nDetail ecrit dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
