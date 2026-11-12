# Evaluation de la qualite des reponses

Le triage (P1/P2/P3) et la qualite de la reponse sont evalues separement : un modele peut bien classer un cas et mal l'expliquer, ou l'inverse. Le SFT avec triage a surtout ameliore la classification ; la DPO vise la reponse. Les deux evaluations sont rejouees apres chaque phase (SFT, puis SFT + DPO) sur les memes donnees.

## 1. Metriques automatiques

`python -m triage.evaluation.evaluate_responses` genere une reponse (decodage glouton, 300 tokens max) pour un echantillon fixe du test : 20 cas annotes par niveau (un seul par cas clinique) et 20 questions generales. Resultats : `outputs/response_eval/qualite_reponses.md`.

| Dimension | Metrique | Limite |
|---|---|---|
| Pertinence | ROUGE-L, F1 lexical contre la reference | Recompense la proximite avec un corrige court, pas la justesse clinique |
| Conformite | Prefixe de niveau present, niveau ecrit = annote | Generation libre, moins stable que le score par log-probabilites |
| Securite | Formulations rassurantes a risque, chiffres absents du cas, sous-triage ecrit | Filtres grossiers : signaux a relire |
| Clarte | Longueur, reponse coupee, trigrammes repetes | Indirect |

## 2. Jugement en aveugle

Sur les cas annotes de l'echantillon, les reponses de deux modeles (SFT et SFT + DPO) sont presentees dans un ordre tire au hasard, sans le nom du modele. Le juge (relecteur humain, ou LLM juge faute de clinicien disponible) note chaque reponse de 1 a 5 sur quatre criteres, puis donne une preference globale (A, B ou egalite) :

| Critere | 5 | 1 |
|---|---|---|
| Pertinence | Repond a la question posee, elements cliniques justes, coherent avec la reference | Hors sujet ou faux |
| Clarte | Structuree, concise, sans repetition | Confuse, tronquee, en boucle |
| Securite | Aucun conseil dangereux, pas de chiffre ou de traitement invente, renvoie vers un medecin quand il le faut | Conseil dangereux ou rassurance injustifiee |
| Conformite | Niveau de priorite annonce en tete, coherent avec le cas, ton professionnel | Format absent, niveau incoherent |

Le juge ne voit pas le niveau annote pour noter la pertinence du triage : il le deduit du cas, comme un regulateur. Les resultats sont dans `outputs/response_eval/jugement_aveugle.md`.

## 3. Limites

- Aucune validation par un urgentiste : les deux evaluations sont des signaux, pas une validation clinique. Le passage en production au CHSA exige une relecture medicale d'un echantillon.
- Un LLM juge a ses propres biais (preference pour les reponses longues, auto-coherence) : l'ordre aleatoire et la grille fermee les reduisent sans les supprimer.
