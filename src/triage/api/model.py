import math
import threading
from dataclasses import dataclass

from triage.api.config import Settings, load_decision_rule
from triage.constants import SHORT, build_prompt, level_prefix


class PromptTooLong(ValueError):
    pass


@dataclass
class Decision:
    niveau: str
    probabilites: dict[str, float]
    niveau_plus_probable: str
    a_revoir: bool
    motifs_revue: list[str]
    tokens_prompt: int


def decide(logprobs: dict, rule: dict, review_threshold: float) -> tuple[str, dict, str, bool, list]:
    """Regle de decision au cout (decision_rule.json) : niveau de cout attendu minimal."""
    labels, temperature, alpha = rule["labels"], rule["temperature"], rule["alpha"]
    scaled = [logprobs[label] / temperature for label in labels]
    top = max(scaled)
    weights = [math.exp(s - top) for s in scaled]
    proba = [w / sum(weights) for w in weights]
    # Le poids alpha s'applique aux cases de sous-triage (triangle superieur).
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
        from triage.utils import modeling

        self.modeling = modeling
        self.settings = settings
        self.rule = load_decision_rule()
        self.tokenizer = modeling.load_tokenizer(settings.base_model, padding_side="left")
        self.model = modeling.load_model(settings.adapter, settings.base_model)
        self.lock = threading.Lock()  # un seul GPU : une requete a la fois

    def triage(self, cas: str, question: str) -> Decision:
        context = self.modeling.encode_context(self.tokenizer, build_prompt(question.strip(), cas.strip()))
        if len(context) > self.settings.max_prompt_tokens:
            raise PromptTooLong(f"{len(context)} tokens, maximum {self.settings.max_prompt_tokens}")
        with self.lock:
            logprobs = self.modeling.label_logprobs(self.model, self.tokenizer, context)
        level, proba, most_likely, review, reasons = decide(logprobs, self.rule, self.settings.review_threshold)
        return Decision(level, proba, most_likely, review, reasons, len(context))

    def explain(self, cas: str, question: str, level: str) -> str:
        """Reponse generee apres le niveau decide (force dans le prompt)."""
        prompt = build_prompt(question.strip(), cas.strip()) + level_prefix(level)
        with self.lock:
            texts, _ = self.modeling.generate(
                self.model, self.tokenizer, [prompt], self.settings.max_new_tokens, batch_size=1, verbose=False
            )
        return texts[0]
