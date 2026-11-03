# Comparaison du triage - 4 experiences

Jeu de test : 384 questions sur 82 groupes de cas cliniques (P1 = 148, P2 = 153, P3 = 83).
Valeurs en %. Sous-triage / sur-triage : niveau predit moins / plus urgent que l'annotation, rapporte a tous les exemples. P1_predit_P3 : part des P1 classes P3.

## Metriques principales

| experience | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage | erreurs_1_niveau | erreurs_2_niveaux | P1_predit_P3 |
|---|---|---|---|---|---|---|---|---|
| 1_baseline_naive | 38.5 | 18.5 | 100.0 | 0.0 | 61.5 | 39.8 | 21.6 | 0.0 |
| 2_modele_base | 28.9 | 23.9 | 0.0 | 64.3 | 6.8 | 50.0 | 21.1 | 54.7 |
| 3_sft_medical | 31.0 | 24.7 | 58.1 | 32.8 | 36.2 | 41.1 | 27.9 | 38.5 |
| 4_sft_medical_triage | 63.8 | 55.0 | 100.0 | 0.0 | 36.2 | 32.0 | 4.2 | 0.0 |

## Precision, rappel et F1 par classe

| experience | precision_P1 | f1_P1 | precision_P2 | rappel_P2 | f1_P2 | precision_P3 | rappel_P3 | f1_P3 |
|---|---|---|---|---|---|---|---|---|
| 1_baseline_naive | 38.5 | 55.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2_modele_base | 0.0 | 0.0 | 36.6 | 34.6 | 35.6 | 24.4 | 69.9 | 36.1 |
| 3_sft_medical | 38.2 | 46.1 | 0.0 | 0.0 | 0.0 | 21.4 | 39.8 | 27.8 |
| 4_sft_medical_triage | 63.0 | 77.3 | 61.2 | 53.6 | 57.1 | 100.0 | 18.1 | 30.6 |

## Matrices de confusion

### 1_baseline_naive

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 148 | 0 | 0 |
| P2 | 153 | 0 | 0 |
| P3 | 83 | 0 | 0 |

### 2_modele_base

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 0 | 67 | 81 |
| P2 | 1 | 53 | 99 |
| P3 | 0 | 25 | 58 |

### 3_sft_medical

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 86 | 5 | 57 |
| P2 | 89 | 0 | 64 |
| P3 | 50 | 0 | 33 |

### 4_sft_medical_triage

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 148 | 0 | 0 |
| P2 | 71 | 82 | 0 |
| P3 | 16 | 52 | 15 |

## Limites

- Etiquettes de reference pre-annotees par LLM, sans validation clinique (docs/triage_grid.md).
- Les questions d'un meme cas partagent le meme niveau : les exemples ne sont pas independants.
- La baseline naive vaut la part de P1 dans ce jeu de test, pas 33 % (classes non equilibrees).
