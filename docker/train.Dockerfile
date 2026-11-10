# Image d'entrainement SFT : memes versions que le venv local (torch 2.6.0 + CUDA 12.4).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONPATH=/app/src

WORKDIR /app

COPY docker/requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Le code est aussi monte en volume par docker-compose.yml : la copie sert a
# lancer l'image seule, le volume a eviter une reconstruction a chaque modification.
COPY src ./src

ENTRYPOINT ["python", "-m", "triage.training.train_sft"]
