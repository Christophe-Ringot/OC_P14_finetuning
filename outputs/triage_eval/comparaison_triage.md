# Comparaison du triage - 4 experiences

Jeu de test : 400 questions sur 72 groupes de cas cliniques (P1 = 163, P2 = 124, P3 = 113).
Valeurs en %. Sous-triage / sur-triage : niveau predit moins / plus urgent que l'annotation, rapporte a tous les exemples. P1_predit_P3 : part des P1 classes P3.

## Regle de decision par modele

Pour chaque modele, le niveau predit est l'argmax des log-probabilites, soit brutes, soit calibrees (moins le score sur prompt vide). Le mode est choisi sur la validation (F1 macro), puis applique au test.

| experience | mode retenu | F1 validation calibre | F1 validation brut |
|---|---|---|---|
| 2_modele_base | brut | 19.6 | 20.5 |
| 3_sft_medical | calibre | 24.1 | 22.4 |
| 4_sft_medical_triage | brut | 37.1 | 62.9 |

## Metriques principales

| experience | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage | erreurs_1_niveau | erreurs_2_niveaux | P1_predit_P3 |
|---|---|---|---|---|---|---|---|---|
| 1_baseline_naive | 40.8 | 19.3 | 100.0 | 0.0 | 59.2 | 31.0 | 28.2 | 0.0 |
| 2_modele_base | 27.8 | 17.7 | 6.7 | 64.2 | 8.0 | 31.0 | 41.2 | 93.3 |
| 3_sft_medical | 32.2 | 25.7 | 45.4 | 38.2 | 29.5 | 31.0 | 36.8 | 54.0 |
| 4_sft_medical_triage | 78.8 | 78.7 | 86.5 | 7.0 | 14.2 | 20.8 | 0.5 | 0.0 |

## Precision, rappel et F1 par classe

| experience | precision_P1 | f1_P1 | precision_P2 | rappel_P2 | f1_P2 | precision_P3 | rappel_P3 | f1_P3 |
|---|---|---|---|---|---|---|---|---|
| 1_baseline_naive | 40.8 | 57.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2_modele_base | 25.6 | 10.7 | 0.0 | 0.0 | 0.0 | 28.0 | 88.5 | 42.6 |
| 3_sft_medical | 38.5 | 41.7 | 50.0 | 0.8 | 1.6 | 26.2 | 47.8 | 33.9 |
| 4_sft_medical_triage | 77.9 | 82.0 | 67.2 | 64.5 | 65.8 | 94.0 | 83.2 | 88.3 |

## Matrices de confusion

### 1_baseline_naive

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 163 | 0 | 0 |
| P2 | 124 | 0 | 0 |
| P3 | 113 | 0 | 0 |

### 2_modele_base

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 11 | 0 | 152 |
| P2 | 19 | 0 | 105 |
| P3 | 13 | 0 | 100 |

### 3_sft_medical

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 74 | 1 | 88 |
| P2 | 59 | 1 | 64 |
| P3 | 59 | 0 | 54 |

### 4_sft_medical_triage

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 141 | 22 | 0 |
| P2 | 38 | 80 | 6 |
| P3 | 2 | 17 | 94 |

## Limites

- Etiquettes de reference pre-annotees par LLM, sans validation clinique (docs/triage_grid.md).
- Les questions d'un meme cas partagent le meme niveau : les exemples ne sont pas independants.
- La baseline naive vaut la part de P1 dans ce jeu de test, pas 33 % (classes non equilibrees).
