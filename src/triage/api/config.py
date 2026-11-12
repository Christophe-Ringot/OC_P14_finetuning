import json
import os
from dataclasses import dataclass, field
from pathlib import Path

LOCAL_ADAPTER = Path("outputs") / "sft_medical_triage"
HUB_ADAPTER = "christophe-ringot/p14-triage-qwen3-1.7b-lora"
DECISION_RULE_PATH = Path(__file__).with_name("decision_rule.json")


def _default_adapter() -> str:
    return str(LOCAL_ADAPTER) if (LOCAL_ADAPTER / "adapter_config.json").exists() else HUB_ADAPTER


@dataclass(frozen=True)
class Settings:
    """Lue dans l'environnement : rien de secret dans le code."""
    api_key: str = field(default_factory=lambda: os.environ.get("TRIAGE_API_KEY", ""))
    base_model: str = field(default_factory=lambda: os.environ.get("BASE_MODEL", "Qwen/Qwen3-1.7B-Base"))
    adapter: str = field(default_factory=lambda: os.environ.get("ADAPTER", _default_adapter()))
    model_version: str = field(default_factory=lambda: os.environ.get("MODEL_VERSION", "sft_medical_triage-ckpt1300"))
    log_path: Path = field(default_factory=lambda: Path(os.environ.get("LOG_PATH", "logs/interactions.jsonl")))
    # Longueur d'entrainement : au-dela, le modele n'a jamais vu de prompt aussi long.
    max_prompt_tokens: int = int(os.environ.get("MAX_PROMPT_TOKENS", "960"))
    max_new_tokens: int = int(os.environ.get("MAX_NEW_TOKENS", "200"))
    # Sous ce seuil de probabilite, la decision est signalee pour revue humaine.
    review_threshold: float = float(os.environ.get("REVIEW_THRESHOLD", "0.5"))


def load_decision_rule() -> dict:
    return json.loads(DECISION_RULE_PATH.read_text(encoding="utf-8"))
