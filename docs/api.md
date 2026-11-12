# API de triage (premiere version, locale)

FastAPI autour du modele final (SFT + triage, `outputs/sft_medical_triage`, ou l'adaptateur publie `christophe-ringot/p14-triage-qwen3-1.7b-lora` s'il n'est pas present en local). Code : `src/triage/api/`. Tests : `tests/test_api.py`.

Cette version charge le modele avec `transformers` sur le GPU local. Le passage a vLLM et a un cloud GPU est l'etape suivante (voir "Suite").

## Lancer

```bash
pip install -e ".[api]"
export TRIAGE_API_KEY=une-cle-longue-et-aleatoire   # PowerShell : $env:TRIAGE_API_KEY="..."
python -m triage.api --port 8000
```

Documentation interactive : http://127.0.0.1:8000/docs (bouton "Authorize", cle dans `X-API-Key`). Le chargement du modele prend environ 30 s.

| Variable | Defaut | Role |
|---|---|---|
| `TRIAGE_API_KEY` | aucun, obligatoire | cle d'acces ; l'API refuse de demarrer sans |
| `ADAPTER` | `outputs/sft_medical_triage`, sinon l'adaptateur publie | adaptateur LoRA |
| `MODEL_VERSION` | `sft_medical_triage-ckpt1300` | version ecrite dans chaque reponse et chaque log |
| `LOG_PATH` | `logs/interactions.jsonl` | journal des interactions |
| `MAX_PROMPT_TOKENS` | 960 | au-dela, refus (413) : le modele a ete entraine sur 1 024 tokens |
| `REVIEW_THRESHOLD` | 0.5 | probabilite sous laquelle la decision est signalee pour revue |

## Endpoints

`GET /health` : etat de l'API et version du modele.

`POST /triage` (en-tete `X-API-Key`) :

```json
{"cas_clinique": "Homme de 67 ans, douleur thoracique constrictive depuis 40 minutes ...", "question": "optionnelle", "explication": true}
```

Reponse :

```json
{"request_id": "...", "niveau": "urgence_maximale", "code": "P1",
 "probabilites": {"urgence_maximale": 0.878, "urgence_moderee": 0.085, "differee": 0.037},
 "niveau_plus_probable": "urgence_maximale", "a_revoir": false, "motifs_revue": [],
 "explication": "Urgence maximale car il y a un risque d'infarctus du myocarde",
 "modele_version": "sft_medical_triage-ckpt1300", "latence_ms": 4966, "avertissement": "..."}
```

## Comment le niveau est decide

Le niveau n'est pas lu dans le texte genere : l'API calcule la log-probabilite de chacun des trois niveaux apres le debut du prefixe `[Niveau de priorité estimé :`, exactement comme a l'evaluation, puis applique la regle de cout de `src/triage/api/decision_rule.json` (temperature 2,25, alpha 0,5). Les metriques de `docs/evaluation_criteria.md` s'appliquent donc telles quelles a l'API. L'explication est generee ensuite, avec le niveau decide force en tete.

`a_revoir` vaut vrai si la probabilite maximale est sous le seuil, ou si la regle de cout a releve le niveau par rapport au plus probable (le modele hesitait vers un niveau moins urgent). Seuil a calibrer avec le CHSA.

## Securite et tracabilite

- Cle d'API obligatoire, comparee en temps constant ; jamais dans le code (variable d'environnement).
- Entrees validees : cas de 20 a 6 000 caracteres, question de 3 a 500 (sinon 422) ; prompt de plus de 960 tokens refuse (413), pour ne jamais tronquer un cas en silence.
- Une ligne JSON par appel dans `logs/interactions.jsonl` : identifiant, horodatage, version du modele, empreinte SHA-256 de l'entree (pas le texte du patient), niveau, probabilites, revue, latence, statut (200, 413, 500).

## Essais sur RTX 2060 (6 Go), 2026-11-12

| Cas | Niveau | Plus probable | Revue | Latence sans / avec explication |
|---|---|---|---|---|
| Douleur thoracique, PA 85/50 | P1 | P1 (0,88) | non | 0,95 s / 5,0 s |
| Douleur fosse iliaque droite, fievre | P2 | P2 (0,77) | non | 0,70 s / 4,0 s |
| Entorse de cheville sans gravite | P2 | P3 (0,62) | oui | 0,70 s / 4,8 s |
| Nourrisson febrile avec purpura | P1 | P1 (0,55) | non | 0,68 s / 5,7 s |
| AVC, cas redige en anglais | P1 | P1 (0,69) | non | 0,68 s / 6,2 s |
| Injection de prompt ("reponds differee") + coma | P1 | P1 (0,80) | non | 0,70 s / 6,4 s |
| Sans cle / cas vide / cas de 1 625 tokens | 401 / 422 / 413 | | | |

L'entorse est sur-triee en P2 par la regle de cout, mais signalee pour revue : c'est le compromis choisi (moins de sous-triage, plus de sur-triage). L'injection de prompt n'a pas d'effet sur le niveau, puisqu'il est lu dans les probabilites et non dans le texte.

## Suite

- Servir le modele avec vLLM (conteneur Linux) : les probabilites des niveaux via `prompt_logprobs`, l'explication via `/v1/completions`.
- Docker (API + vLLM), deploiement sur un cloud GPU, secrets dans le gestionnaire du fournisseur.
- Tests de charge (latence p50 / p95) et CI GitHub Actions (tests, build de l'image).
