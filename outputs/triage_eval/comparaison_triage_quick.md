# Comparaison du triage - 4 experiences

Jeu de test : 30 questions sur 24 groupes de cas cliniques (P1 = 12, P2 = 8, P3 = 10).
Valeurs en %. Sous-triage / sur-triage : niveau predit moins / plus urgent que l'annotation, rapporte a tous les exemples. P1_predit_P3 : part des P1 classes P3.

## Regle de decision par modele

Pour chaque modele, le niveau predit est l'argmax des log-probabilites, soit brutes, soit calibrees (moins le score sur prompt vide). Le mode est choisi sur la validation (F1 macro), puis applique au test.

| experience | mode retenu | F1 validation calibre | F1 validation brut |
|---|---|---|---|
| 2_modele_base | brut | 12.2 | 21.1 |
| 3_sft_medical | calibre | 14.1 | 11.0 |
| 4_sft_medical_triage | calibre | 18.9 | 14.0 |

## Metriques principales

| experience | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage | erreurs_1_niveau | erreurs_2_niveaux | P1_predit_P3 |
|---|---|---|---|---|---|---|---|---|
| 1_baseline_naive | 40.0 | 19.0 | 100.0 | 0.0 | 60.0 | 26.7 | 33.3 | 0.0 |
| 2_modele_base | 36.7 | 22.5 | 8.3 | 56.7 | 6.7 | 26.7 | 36.7 | 91.7 |
| 3_sft_medical | 26.7 | 21.9 | 0.0 | 50.0 | 23.3 | 60.0 | 13.3 | 33.3 |
| 4_sft_medical_triage | 40.0 | 19.0 | 100.0 | 0.0 | 60.0 | 26.7 | 33.3 | 0.0 |

Meme tableau sur la validation (regle de decision choisie sur ce jeu), a citer avec le test :

| experience | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage | erreurs_1_niveau | erreurs_2_niveaux | P1_predit_P3 |
|---|---|---|---|---|---|---|---|---|
| 1_baseline_naive | 30.0 | 15.4 | 100.0 | 0.0 | 70.0 | 43.3 | 26.7 | 0.0 |
| 2_modele_base | 30.0 | 21.1 | 11.1 | 70.0 | 0.0 | 43.3 | 26.7 | 88.9 |
| 3_sft_medical | 23.3 | 14.1 | 0.0 | 50.0 | 26.7 | 63.3 | 13.3 | 33.3 |
| 4_sft_medical_triage | 30.0 | 18.9 | 88.9 | 3.3 | 66.7 | 43.3 | 26.7 | 0.0 |

## Precision, rappel et F1 par classe

| experience | precision_P1 | f1_P1 | precision_P2 | rappel_P2 | f1_P2 | precision_P3 | rappel_P3 | f1_P3 |
|---|---|---|---|---|---|---|---|---|
| 1_baseline_naive | 40.0 | 57.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2_modele_base | 33.3 | 13.3 | 0.0 | 0.0 | 0.0 | 37.0 | 100.0 | 54.1 |
| 3_sft_medical | 0.0 | 0.0 | 25.0 | 62.5 | 35.7 | 30.0 | 30.0 | 30.0 |
| 4_sft_medical_triage | 40.0 | 57.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Matrices de confusion

### 1_baseline_naive

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 12 | 0 | 0 |
| P2 | 8 | 0 | 0 |
| P3 | 10 | 0 | 0 |

### 2_modele_base

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 1 | 0 | 11 |
| P2 | 2 | 0 | 6 |
| P3 | 0 | 0 | 10 |

### 3_sft_medical

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 0 | 8 | 4 |
| P2 | 0 | 5 | 3 |
| P3 | 0 | 7 | 3 |

### 4_sft_medical_triage

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 12 | 0 | 0 |
| P2 | 8 | 0 | 0 |
| P3 | 10 | 0 | 0 |

## Limites

- Etiquettes de reference pre-annotees par LLM, sans validation clinique (docs/triage_grid.md).
- Les questions d'un meme cas partagent le meme niveau : les exemples ne sont pas independants.
- La baseline naive vaut la part de P1 dans ce jeu de test, pas 33 % (classes non equilibrees).
