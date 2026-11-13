import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from triage.constants import LABELS, LEVEL_RE, RANK, SHORT

WORD_RE = re.compile(r"\w+", re.UNICODE)


def triage_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    gap = y_pred.map(RANK) - y_true.map(RANK)  # > 0 : predit moins urgent
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
        metrics.update(
            {
                f"precision_{SHORT[label]}": precision[i],
                f"rappel_{SHORT[label]}": recall[i],
                f"f1_{SHORT[label]}": f1[i],
            }
        )
    return metrics


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float) / temperature
    values = np.exp(values - values.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def words(text: str) -> list[str]:
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
