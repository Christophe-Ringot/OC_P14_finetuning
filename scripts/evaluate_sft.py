import math
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
ADAPTER_DIR = "outputs/sft_lora_run5"
DATA_PATH = "data/processed/sft_dataset_anonymized.csv"
QUALITATIVE_OUTPUT = "outputs/sft_qualitative_examples.csv"
N_QUALITATIVE_EXAMPLES = 10

MAX_PERPLEXITY_EXAMPLES = None if torch.cuda.is_available() else 50
MAX_NEW_TOKENS = 200 if torch.cuda.is_available() else 80


def build_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Reponse:\n"


def compute_test_perplexity(model, tokenizer, test_df: pd.DataFrame) -> float:
    if MAX_PERPLEXITY_EXAMPLES:
        test_df = test_df.sample(min(MAX_PERPLEXITY_EXAMPLES, len(test_df)), random_state=42)

    losses = []
    for _, row in test_df.iterrows():
        text = f"{build_prompt(row['instruction'])}{row['response']}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        losses.append(loss.item())

    return sum(losses) / len(losses)


def generate_qualitative_examples(model, tokenizer, test_df: pd.DataFrame) -> pd.DataFrame:
    sample = test_df.sample(N_QUALITATIVE_EXAMPLES, random_state=42)

    rows = []
    for _, row in sample.iterrows():
        prompt = build_prompt(row["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        generated_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        rows.append({
            "source_dataset": row["source_dataset"],
            "instruction": row["instruction"],
            "reference": row["response"],
            "generated": generated_text,
        })

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    test_df = df[df["split"] == "test"]

    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    device_map = "auto" if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype, device_map=device_map)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    mean_loss = compute_test_perplexity(model, tokenizer, test_df)
    print(f"Test loss (moyenne) : {mean_loss:.4f} | perplexite : {math.exp(mean_loss):.2f}")

    report_df = generate_qualitative_examples(model, tokenizer, test_df)
    report_df.to_csv(QUALITATIVE_OUTPUT, index=False)
    print(f"Exemples qualitatifs ecrits dans {QUALITATIVE_OUTPUT}")
    print("Rappel : pas de matrice de confusion sur ce run (dataset sans label de niveau d'urgence) - a construire a la prochaine iteration du dataset.")


if __name__ == "__main__":
    main()
