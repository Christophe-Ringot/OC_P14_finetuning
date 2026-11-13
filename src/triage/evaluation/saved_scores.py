import pandas as pd

from triage.constants import LABELS, OUTPUTS_DIR, SHORT

TRIAGE_EVAL_DIR = OUTPUTS_DIR / "triage_eval"
MODES = ["calibre", "brut"]


def scores_path(split: str, suffix: str = ""):
    return TRIAGE_EVAL_DIR / (f"scores{suffix}.csv" if split == "test" else f"scores_{split}{suffix}.csv")


def decision_scores(scores: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Scores de decision par niveau : log-probabilite brute, moins le prior sur prompt vide si `calibre`."""
    return pd.DataFrame(
        {
            label: scores[f"raw_{SHORT[label]}"] - (scores[f"prior_{SHORT[label]}"] if mode == "calibre" else 0)
            for label in LABELS
        }
    )


def decision_mode(experiment: str) -> str:
    """Mode (calibre / brut) retenu sur la validation par evaluate_sft."""
    return pd.read_csv(TRIAGE_EVAL_DIR / "mode_decision.csv", index_col=0).loc[experiment, "mode_retenu"]


def load_decision_scores(experiment: str, split: str, mode: str | None = None) -> pd.DataFrame:
    """Scores de decision sauvegardes d'une experience, indexes par id de question."""
    frame = pd.read_csv(scores_path(split))
    frame = frame[frame["experience"] == experiment].set_index("id")
    return decision_scores(frame, mode or decision_mode(experiment))
