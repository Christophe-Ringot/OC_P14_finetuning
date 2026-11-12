# Qualite des paires DPO (triage)

1195 paires : ('train', 'qualite') 785, ('train', 'securite_triage') 280, ('validation', 'qualite') 95, ('validation', 'securite_triage') 35.

## Repartition

| split / type | P1 | P2 | P3 |
|---|---|---|---|
| train / qualite | 286 | 273 | 226 |
| train / securite_triage | 130 | 150 | 0 |
| validation / qualite | 31 | 34 | 30 |
| validation / securite_triage | 17 | 18 | 0 |

## Controles automatiques

- Candidats qualite generes : 900 ; ecartes : generation vide (< 3 mots) = 20; trop proche de la reference (ROUGE-L >= 0.6) = 1.
- Generations coupees a la limite de tokens (gardees, rejected legitime) : 41.
- Cas du test presents dans les paires : 0 (attendu : 0).
- Paires identiques ou vides : 0 (verifie par le schema pandera).
- Similarite ROUGE-L chosen / rejected (qualite) : mediane 0.10, max 0.57.
- Longueur en mots, mediane : chosen 32, rejected 30 ; rejected plus long que chosen dans 51 % des paires. Un ecart fort ferait apprendre la longueur plutot que le contenu : a surveiller dans la longueur des reponses apres DPO.
- Prefixe de niveau identique dans chosen et rejected pour toutes les paires qualite : oui.

## Limites

- chosen = reponse de reference (corrige d'examen MediQAl) : preference supposee, non validee par un clinicien. Une relecture humaine d'un echantillon reste a faire (docs/evaluation_reponses.md).
- La reference peut etre plus courte que la generation sans etre meilleure sur tous les points.
