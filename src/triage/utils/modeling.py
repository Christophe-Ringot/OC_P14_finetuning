import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from triage.constants import BASE_MODEL, LABELS, PRIORITY_PREFIX


def load_tokenizer(base_model: str = BASE_MODEL, padding_side: str = "right"):
    return AutoTokenizer.from_pretrained(base_model, padding_side=padding_side)


def load_model(adapter: Path | str | None = None, base_model: str = BASE_MODEL):
    """Modele de base, avec l'adaptateur LoRA s'il est fourni, en mode evaluation."""
    has_gpu = torch.cuda.is_available()
    # Tout sur le GPU : un manque de memoire doit lever une erreur, pas
    # decharger silencieusement des couches sur le CPU (inference tres lente).
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16 if has_gpu else torch.float32, device_map={"": 0} if has_gpu else None
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval()


def free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_context(tokenizer, prompt: str) -> list[int]:
    """Prompt suivi du debut du prefixe de niveau, sans token special."""
    return tokenizer(prompt + PRIORITY_PREFIX, add_special_tokens=False)["input_ids"]


def label_logprobs(model, tokenizer, context: list[int]) -> dict[str, float]:
    """Log-probabilite de " <niveau>]" apres le contexte, pour chaque niveau."""
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


def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int = 8,
    repetition_penalty: float = 1.1,
    verbose: bool = True,
) -> tuple[list[str], list[bool]]:
    """Decodage glouton par lots (tokenizer a padding a gauche) ; renvoie les textes et s'ils sont coupes."""
    texts, truncated = [], []
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row in output[:, batch["input_ids"].shape[1] :]:
            # Le modele enchaine parfois un nouveau "### Cas clinique" : on coupe.
            texts.append(tokenizer.decode(row, skip_special_tokens=True).split("\n### ")[0].strip())
            truncated.append(bool((row != tokenizer.pad_token_id).sum() >= max_new_tokens))
        if verbose:
            print(f"  {min(start + batch_size, len(prompts))}/{len(prompts)}")
    return texts, truncated
