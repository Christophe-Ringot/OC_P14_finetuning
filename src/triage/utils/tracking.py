"""Suivi des entrainements et evaluations avec MLflow.

Serveur : `docker compose up -d mlflow` (interface sur http://localhost:5000).
Variables d'environnement :
- MLFLOW_TRACKING_URI : adresse du serveur (defaut http://localhost:5000) ;
- MLFLOW_EXPERIMENT_NAME : experience MLflow (defaut "triage-sft").

Le serveur est verifie au demarrage : un entrainement de plusieurs heures ne
doit pas tourner sans suivi parce que le serveur n'etait pas lance. Pour
travailler sans MLflow, passer --no-mlflow aux scripts.
"""

import contextlib
import hashlib
import os
import urllib.request
from pathlib import Path

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT = "triage-sft"


def setup(enabled: bool) -> bool:
    """Configure MLflow ; renvoie False si le suivi est desactive."""
    if not enabled:
        print("Suivi MLflow desactive (--no-mlflow)")
        return False
    import mlflow

    uri = os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment = os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=5)
        except OSError as error:
            raise RuntimeError(
                f"Serveur MLflow injoignable a {uri} ({error}). "
                "Lancer `docker compose up -d mlflow`, ou passer --no-mlflow."
            ) from error
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    print(f"Suivi MLflow : {uri} (experience '{experiment}')")
    return True


@contextlib.contextmanager
def start_run(enabled: bool, run_name: str | None = None, run_id: str | None = None, tags: dict | None = None):
    """Run MLflow (nouveau ou existant), ou rien si le suivi est desactive."""
    if not enabled:
        yield None
        return
    import mlflow

    system_metrics = run_id is None  # CPU / GPU / memoire pendant l'entrainement
    with mlflow.start_run(run_name=run_name, run_id=run_id, tags=tags, log_system_metrics=system_metrics) as run:
        yield run


def git_commit() -> str:
    """Commit courant, lu dans .git sans dependre du binaire git (absent du conteneur)."""
    git = Path(".git")
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(ref):
                return line.split(" ", 1)[0]
    except OSError:
        pass
    return "inconnu"


def log_adapter(output_dir: Path) -> None:
    """Resume d'entrainement et adaptateur final (sans les checkpoints intermediaires) en artefacts MLflow."""
    import mlflow

    mlflow.log_artifact(str(output_dir / "training_summary.json"))
    for name in ["adapter_config.json", "adapter_model.safetensors"]:
        mlflow.log_artifact(str(output_dir / name), artifact_path="adapter")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
