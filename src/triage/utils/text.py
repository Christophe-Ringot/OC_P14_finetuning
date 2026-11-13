import re

import pandas as pd


def clean_text(text) -> str | None:
    """Normalise espaces, fins de ligne et apostrophes ; None si le texte est vide."""
    if text is None or pd.isna(text):
        return None
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    # Supprime les espaces en debut/fin de ligne sans toucher aux puces "- ".
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or None
