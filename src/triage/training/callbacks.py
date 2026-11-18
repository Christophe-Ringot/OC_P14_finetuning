from transformers import TrainerCallback

# Courbes journalisees pendant l'entrainement : cle des logs du Trainer -> nom dans MLflow.
SFT_CURVES = {"loss": "train_loss", "eval_loss": "eval_loss"}
DPO_CURVES = {
    "loss": "train_loss",
    "eval_qualite_loss": "eval_loss_qualite",
    "eval_securite_triage_loss": "eval_loss_securite",
    "eval_qualite_rewards/margins": "marge_qualite",
    "eval_securite_triage_rewards/margins": "marge_securite",
}


def select_metrics(logs: dict, names: dict) -> dict:
    """Garde et renomme les valeurs de `names` dans une entree de log du Trainer."""
    # TRL evalue chaque jeu de paires a part ; ses marges arrivent sans le nom du jeu.
    dataset = next(
        (
            k[len("eval_") : -len("_loss")]
            for k in logs
            if k.startswith("eval_") and k.endswith("_loss") and k != "eval_loss"
        ),
        None,
    )
    values = {}
    for key, value in logs.items():
        if dataset and key.startswith("eval_rewards/"):
            key = f"eval_{dataset}_{key[len('eval_') :]}"
        if key in names and isinstance(value, (int, float)):
            values[names[key]] = float(value)
    return values


class MetricsCallback(TrainerCallback):
    """Remplace report_to="mlflow" : quelques courbes au lieu des dizaines de metriques et ~180
    parametres que transformers et TRL journalisent par defaut."""

    def __init__(self, names: dict):
        self.names = names

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        import mlflow

        values = select_metrics(logs, self.names)
        if values:
            mlflow.log_metrics(values, step=state.global_step)
