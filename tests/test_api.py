import json
import pytest
from fastapi.testclient import TestClient
from triage.api.config import Settings, load_decision_rule
from triage.api.main import create_app
from triage.api.model import Decision, PromptTooLong, decide

KEY = "cle-de-test"
CASE = "Homme de 58 ans, BPCO connue, dyspnee majeure, SpO2 80 %, FR 40/min, cyanose et sueurs."


class FakeModel:
    """Remplace le modele : les tests tournent sans GPU ni telechargement."""

    def __init__(self, settings):
        self.settings = settings

    def triage(self, cas, question):
        if "TROP_LONG" in cas:
            raise PromptTooLong("2000 tokens, maximum 960")
        proba = {"urgence_maximale": 0.7, "urgence_moderee": 0.2, "differee": 0.1}
        return Decision("urgence_maximale", proba, "urgence_maximale", False, [], 42)

    def explain(self, cas, question, level):
        return "Detresse respiratoire aigue."


@pytest.fixture
def client(tmp_path):
    settings = Settings(api_key=KEY, log_path=tmp_path / "interactions.jsonl")
    with TestClient(create_app(settings, FakeModel)) as test_client:
        test_client.log_path = settings.log_path
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["modele_charge"] is True


def test_triage_ok_and_logged(client):
    response = client.post("/triage", json={"cas_clinique": CASE}, headers={"X-API-Key": KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "P1" and body["explication"]
    record = json.loads(client.log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["request_id"] == body["request_id"] and record["statut"] == 200
    # Le texte du cas ne doit pas apparaitre dans les journaux.
    assert "BPCO" not in client.log_path.read_text(encoding="utf-8")


def test_without_explanation(client):
    response = client.post("/triage", json={"cas_clinique": CASE, "explication": False}, headers={"X-API-Key": KEY})
    assert response.status_code == 200 and response.json()["explication"] is None


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "mauvaise-cle"}])
def test_auth_required(client, headers):
    assert client.post("/triage", json={"cas_clinique": CASE}, headers=headers).status_code == 401


@pytest.mark.parametrize("payload", [{}, {"cas_clinique": ""}, {"cas_clinique": "court"}, {"cas_clinique": "x" * 7000},
                                     {"cas_clinique": CASE, "question": ""}])
def test_invalid_input(client, payload):
    assert client.post("/triage", json=payload, headers={"X-API-Key": KEY}).status_code == 422


def test_prompt_too_long(client):
    response = client.post("/triage", json={"cas_clinique": CASE + " TROP_LONG"}, headers={"X-API-Key": KEY})
    assert response.status_code == 413
    assert json.loads(client.log_path.read_text(encoding="utf-8").splitlines()[-1])["statut"] == 413


def test_missing_api_key_refuses_to_start(tmp_path):
    with pytest.raises(RuntimeError):
        create_app(Settings(api_key="", log_path=tmp_path / "x.jsonl"), FakeModel)


def test_decision_rule_matches_evaluation():
    rule = load_decision_rule()
    # Scores reels d'un cas du test (outputs/triage_eval/scores.csv) : P2 le plus probable, P1 retenu par la regle de cout.
    level, proba, most_likely, review, _ = decide(
        {"urgence_maximale": -16.7028, "urgence_moderee": -15.9518, "differee": -20.0130}, rule, 0.5)
    assert most_likely == "urgence_moderee" and level == "urgence_maximale" and review
    assert abs(sum(proba.values()) - 1) < 1e-9


def test_decision_rule_clear_case():
    level, _, most_likely, review, _ = decide({"urgence_maximale": -30.0, "urgence_moderee": -25.0, "differee": -2.0},
                                             load_decision_rule(), 0.5)
    assert level == most_likely == "differee" and not review
