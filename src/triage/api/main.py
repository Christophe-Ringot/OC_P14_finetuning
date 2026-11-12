import hashlib
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Callable
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from triage.api.config import Settings
from triage.api.model import SHORT, PromptTooLong, TriageModel

DEFAULT_QUESTION = "Quel est le niveau de priorité de ce patient et pourquoi ?"
AVERTISSEMENT = ("Aide au tri pour un POC, sans validation clinique : la decision revient au soignant. "
                 "L'explication generee peut contenir des erreurs medicales.")


class TriageRequest(BaseModel):
    cas_clinique: str = Field(min_length=20, max_length=6000, description="Description du patient et du contexte.")
    question: str = Field(default=DEFAULT_QUESTION, min_length=3, max_length=500)
    explication: bool = Field(default=True, description="Generer l'explication (plus lent).")


class TriageResponse(BaseModel):
    request_id: str
    niveau: str
    code: str
    probabilites: dict[str, float]
    niveau_plus_probable: str
    a_revoir: bool
    motifs_revue: list[str]
    explication: str | None
    modele_version: str
    latence_ms: int
    avertissement: str = AVERTISSEMENT


def _log(settings: Settings, record: dict) -> None:
    """Une ligne JSON par interaction. Le texte du cas n'est pas journalise, seulement son empreinte."""
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_app(settings: Settings | None = None, model_factory: Callable[[Settings], object] = TriageModel) -> FastAPI:
    settings = settings or Settings()
    if not settings.api_key:
        raise RuntimeError("TRIAGE_API_KEY absent : definir la cle d'API dans l'environnement")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model_factory(settings)
        yield

    app = FastAPI(title="Triage CHSA (POC)", version="0.1.0", lifespan=lifespan,
                  description="Niveau de priorite P1 / P2 / P3 d'un patient aux urgences, avec explication. " + AVERTISSEMENT)
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def check_key(key: str | None = Security(api_key_header)) -> None:
        if not key or not hmac.compare_digest(key, settings.api_key):
            raise HTTPException(status_code=401, detail="Cle d'API absente ou invalide")

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"statut": "ok", "modele_charge": getattr(request.app.state, "model", None) is not None,
                "modele_version": settings.model_version}

    @app.post("/triage", response_model=TriageResponse, dependencies=[Depends(check_key)])
    def triage(body: TriageRequest, request: Request) -> TriageResponse:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        record = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modele_version": settings.model_version,
            "entree_sha256": hashlib.sha256((body.cas_clinique + "\n" + body.question).encode("utf-8")).hexdigest(),
            "entree_caracteres": len(body.cas_clinique) + len(body.question),
            "explication_demandee": body.explication,
        }
        model = request.app.state.model
        try:
            decision = model.triage(body.cas_clinique, body.question)
            explanation = model.explain(body.cas_clinique, body.question, decision.niveau) if body.explication else None
        except PromptTooLong as error:
            _log(settings, {**record, "statut": 413, "erreur": str(error)})
            raise HTTPException(status_code=413, detail=f"Cas trop long pour le modele ({error})") from error
        except Exception as error:
            _log(settings, {**record, "statut": 500, "erreur": type(error).__name__})
            raise
        latency = int((time.perf_counter() - start) * 1000)
        _log(settings, {**record, "statut": 200, "niveau": decision.niveau, "niveau_plus_probable": decision.niveau_plus_probable,
                        "probabilites": {k: round(v, 4) for k, v in decision.probabilites.items()},
                        "a_revoir": decision.a_revoir, "tokens_prompt": decision.tokens_prompt, "latence_ms": latency})
        return TriageResponse(
            request_id=request_id, niveau=decision.niveau, code=SHORT[decision.niveau],
            probabilites={k: round(v, 4) for k, v in decision.probabilites.items()},
            niveau_plus_probable=decision.niveau_plus_probable, a_revoir=decision.a_revoir,
            motifs_revue=decision.motifs_revue, explication=explanation, modele_version=settings.model_version,
            latence_ms=latency,
        )

    return app
