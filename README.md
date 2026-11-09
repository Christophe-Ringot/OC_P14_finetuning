# OC_P14_finetuning : POC Agent IA de triage médical (CHSA)

## Suivi des entraînements avec MLflow

Tous les entraînements SFT et les évaluations de triage sont enregistrés dans
MLflow (expérience `triage-sft`) :

- **run d'entraînement** (`3_sft_medical`, `4_sft_medical_triage`) : hyperparamètres,
  courbes de loss train/validation, métriques système (CPU, GPU, mémoire),
  empreinte MD5 du dataset, commit git, adaptateur LoRA final, puis les
  métriques de triage sur la validation (`validation_*`) et le test (`test_*`) ;
- **run d'évaluation** (`evaluation_triage`) : métriques des 4 expériences,
  règle de décision retenue, rapports et prédictions en artefacts.

### Tout lancer en une commande (Docker)

Prérequis : Docker Desktop (backend WSL2) avec accès au GPU NVIDIA.

```powershell
docker compose --profile train up -d --build
```

Cette commande démarre le serveur MLflow, construit l'image d'entraînement,
entraîne les deux SFT puis lance l'évaluation. L'entraînement complet prend environ 14 h.

- Interface MLflow : <http://localhost:5000> (les courbes s'affichent pendant l'entraînement)
- Suivre les logs : `docker compose logs -f train`
- Test rapide (200 exemples) : `docker compose --profile train run --rm train --quick`
- Une seule expérience : `docker compose --profile train run --rm train --only 4_sft_medical_triage`
- Arrêter MLflow : `docker compose down` (les runs sont conservés dans `./mlflow/`)

### Lancer depuis le venv local

```powershell
docker compose up -d mlflow          # serveur MLflow seul
.venv\Scripts\Activate.ps1
python -m scripts.train_sft          # MLFLOW_TRACKING_URI vaut http://localhost:5000 par défaut
```

Les scripts vérifient que le serveur répond avant de démarrer, pour éviter
un entraînement de plusieurs heures sans suivi. `--no-mlflow` désactive le
suivi (`train_sft.py`, `evaluate_sft.py`).

## Pipeline triage

```
python -m scripts.build_sft_datasets   # splits regroupés par cas (anti-fuite)
python -m scripts.audit_splits         # audit des fuites -> outputs/audit/
python -m scripts.train_sft            # 2 SFT puis évaluation -> outputs/triage_eval/
python -m scripts.analyze_triage       # calibration, métriques par cas, bootstrap
python -m scripts.error_analysis       # analyse des erreurs
```

Résultats et protocole : `docs/comparaison_triage.md`. Journal : `docs/experiments.md`.
