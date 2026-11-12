import math
import threading
from dataclasses import dataclass
from triage.api.config import Settings, load_decision_rule

PROMPT_TEMPLATE = "### Cas clinique :\n{cas}\n\n### Question :\n{question}\n\n### Réponse :\n"
PRIORITY_PREFIX = "[Niveau de priorité estimé :"
SHORT = {"urgence_maximale": "P1", "urgence_moderee": "P2", "differee": "P3"}


class PromptTooLong(ValueError):
    pass


@dataclass
class Decision:
    niveau: str
    probabilites: dict
    niveau_plus_probable: str
    a_revoir: bool
    motifs_revue: list
    tokens_prompt: int


def decide(logprobs: dict, rule: dict, review_threshold: float) -> tuple[str, dict, str, bool, list]:
    """Regle de decision au cout (decision_rule.json) : niveau de cout attendu minimal."""
    labels, temperature, alpha = rule["labels"], rule["temperature"], rule["alpha"]
    scaled = [logprobs[label] / temperature for label in labels]
    top = max(scaled)
    weights = [math.exp(s - top) for s in scaled]
    proba = [w / sum(weights) for w in weights]
    cost = [[c * alpha if j > i else c for j, c in enumerate(row)] for i, row in enumerate(rule["cost_matrix"])]
    expected = [sum(proba[i] * cost[i][j] for i in range(len(labels))) for j in range(len(labels))]
    level = labels[expected.index(min(expected))]
    most_likely = labels[proba.index(max(proba))]
    reasons = []
    if max(proba) < review_threshold:
        reasons.append(f"probabilite maximale {max(proba):.2f} < {review_threshold}")
    if level != most_likely:
        reasons.append(f"niveau releve par la regle de cout ({SHORT[most_likely]} le plus probable)")
    return level, dict(zip(labels, proba)), most_likely, bool(reasons), reasons


class TriageModel:
    """Modele de base + adaptateur LoRA, charge une seule fois."""

    def __init__(self, settings: Settings):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.settings = settings
        self.rule = load_decision_rule()
        self.torch = torch
        gpu = torch.cuda.is_available()
        self.tokenizer = AutoTokenizer.from_pretrained(settings.base_model, padding_side="left")
        base = AutoModelForCausalLM.from_pretrained(settings.base_model, dtype=torch.bfloat16 if gpu else torch.float32,
                                                    device_map={"": 0} if gpu else None)
        self.model = PeftModel.from_pretrained(base, settings.adapter).eval()
        self.device = "cuda" if gpu else "cpu"
        self.lock = threading.Lock()  # un seul GPU : une requete a la fois

    def _encode(self, text: str) -> list:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _label_logprobs(self, context: list) -> dict:
        """Log-probabilite de " <niveau>]" apres le prefixe, comme a l'evaluation."""
        torch = self.torch
        scores = {}
        for label in self.rule["labels"]:
            continuation = self._encode(f" {label}]")
            ids = torch.tensor([context + continuation], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, logits_to_keep=len(continuation) + 1).logits[0, :-1].float()
            targets = torch.tensor(continuation, device=self.device).unsqueeze(1)
            scores[label] = logits.log_softmax(-1).gather(1, targets).sum().item()
        return scores

    def triage(self, cas: str, question: str) -> Decision:
        prompt = PROMPT_TEMPLATE.format(cas=cas.strip(), question=question.strip())
        context = self._encode(prompt + PRIORITY_PREFIX)
        if len(context) > self.settings.max_prompt_tokens:
            raise PromptTooLong(f"{len(context)} tokens, maximum {self.settings.max_prompt_tokens}")
        with self.lock:
            logprobs = self._label_logprobs(context)
        level, proba, most_likely, review, reasons = decide(logprobs, self.rule, self.settings.review_threshold)
        return Decision(level, proba, most_likely, review, reasons, len(context))

    def explain(self, cas: str, question: str, level: str) -> str:
        """Reponse generee apres le niveau decide (force dans le prompt)."""
        torch = self.torch
        prompt = PROMPT_TEMPLATE.format(cas=cas.strip(), question=question.strip()) + f"{PRIORITY_PREFIX} {level}]\n\n"
        batch = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.lock, torch.no_grad():
            output = self.model.generate(**batch, max_new_tokens=self.settings.max_new_tokens, do_sample=False,
                                         repetition_penalty=1.1, pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        # Le modele enchaine parfois un nouveau "### Cas clinique" : on coupe.
        return text.split("\n### ")[0].strip()
