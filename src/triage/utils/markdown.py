import numpy as np
import pandas as pd


def markdown_table(frame: pd.DataFrame, digits: int | None = 1) -> str:
    """Tableau markdown ; l'index devient la premiere colonne, flottants arrondis a `digits` (None : tels quels)."""
    if not isinstance(frame.index, pd.MultiIndex) and frame.index.name is None:
        frame = frame.rename_axis("")
    frame = frame.reset_index()

    def fmt(value) -> str:
        if digits is not None and isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, separator, *rows])
