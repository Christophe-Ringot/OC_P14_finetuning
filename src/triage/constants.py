import re
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
SEED = 42
SPLITS = ["train", "validation", "test"]

DATA_DIR = Path("data")
OUTPUTS_DIR = Path("outputs")

# Niveaux de triage, du plus urgent (P1) au moins urgent (P3).
LABELS = ["urgence_maximale", "urgence_moderee", "differee"]
SHORT = {"urgence_maximale": "P1", "urgence_moderee": "P2", "differee": "P3"}
RANK = {label: i for i, label in enumerate(LABELS)}

PRIORITY_PREFIX = "[Niveau de priorité estimé :"
LEVEL_RE = re.compile(r"\[Niveau de priorité estimé : ([a-z_]+)\]")

# Experiences comparees : 1 et 2 sans adaptateur, 3 a 6 avec leur dossier d'adaptateur LoRA.
BASELINE = "1_baseline_naive"
BASE_MODEL_EXPERIMENT = "2_modele_base"
ADAPTERS = {
    "3_sft_medical": "sft_medical",
    "4_sft_medical_triage": "sft_medical_triage",
    "5_sft_triage_dpo": "dpo_triage",
    "6_sft_triage_dpo_qualite": "dpo_triage_qualite",
}
FINAL_EXPERIMENT = "4_sft_medical_triage"


def adapter_dir(experiment: str, quick: bool = False) -> Path:
    return OUTPUTS_DIR / (ADAPTERS[experiment] + ("_quick" if quick else ""))


def level_prefix(level: str) -> str:
    return f"{PRIORITY_PREFIX} {level}]\n\n"


def build_prompt(question: str, case: str | None = None) -> str:
    """Gabarit commun a l'entrainement, a l'evaluation et a l'API."""
    head = f"### Cas clinique :\n{case}\n\n" if case is not None else ""
    return f"{head}### Question :\n{question}\n\n### Réponse :\n"
