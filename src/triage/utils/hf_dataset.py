"""Chargement du dataset SFT publie sur Hugging Face.

Source unique des scripts d'entrainement et d'evaluation :
https://huggingface.co/datasets/christophe-ringot/p14-medical-triage-sft

La revision est figee sur un commit du Hub pour que l'entrainement reste
reproductible ; la changer apres chaque nouvelle publication du dataset.

Deux variantes, memes lignes et memes splits :
- avec priorite (experience 4) : la reponse des cas annotes commence par
  "[Niveau de priorité estimé : <niveau>]" ;
- sans priorite (experience 3) : ce prefixe est retire.
"""

import hashlib
import re
from functools import lru_cache

import pandas as pd
from datasets import load_dataset

from triage.constants import LEVEL_RE, SPLITS

DATASET_ID = "christophe-ringot/p14-medical-triage-sft"
DATASET_REVISION = "c2a08bd0f372a0cb47aafc918610d801d13978e1"
PRIORITY_PREFIX_RE = re.compile("^" + LEVEL_RE.pattern + r"\n\n")


@lru_cache(maxsize=1)
def _load_hub() -> pd.DataFrame:
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    return pd.concat([dataset[split].to_pandas().assign(split=split) for split in SPLITS], ignore_index=True)


def load_sft(with_priority: bool = True) -> pd.DataFrame:
    """Toutes les lignes (colonne `split` : train / validation / test), dans l'ordre du Hub."""
    frame = _load_hub().copy()
    has_prefix = frame["reponse"].str.match(PRIORITY_PREFIX_RE)
    if not (has_prefix == frame["priorite"].notna()).all():
        raise ValueError("prefixe de priorite incoherent avec la colonne priorite")
    if not with_priority:
        frame["reponse"] = frame["reponse"].str.replace(PRIORITY_PREFIX_RE, "", n=1, regex=True)
    return frame


def dataset_name(with_priority: bool = True) -> str:
    return f"{DATASET_ID}@{DATASET_REVISION[:7]}" + ("" if with_priority else " (sans priorite)")


def fingerprint(frame: pd.DataFrame) -> str:
    """Empreinte MD5 du contenu (id, split, prompt, reponse), pour MLflow."""
    digest = hashlib.md5()
    for row in frame[["id", "split", "prompt", "reponse"]].itertuples(index=False):
        digest.update("\x1f".join(row).encode("utf-8") + b"\x1e")
    return digest.hexdigest()
