# Comparaison finale des modeles

Genere par `python -m triage.evaluation.compare_models` a partir des scores sauvegardes (sans GPU). Valeurs en %. `+ cout` : regle de decision deployee (alpha = 0.35, matrice de `decision_rule.json`), temperature ajustee sur la validation pour chaque modele, appliquee telle quelle au test.

## Test (400 questions, 109 cas)

| experience | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage | P1_predit_P3 |
|---|---|---|---|---|---|---|
| 1_baseline_naive | 40.8 | 19.3 | 100.0 | 0.0 | 59.2 | 0.0 |
| 2_modele_base | 27.8 | 17.7 | 6.7 | 64.2 | 8.0 | 93.3 |
| 3_sft_medical | 32.2 | 25.7 | 45.4 | 38.2 | 29.5 | 54.0 |
| 4_sft_medical_triage | 78.8 | 78.7 | 86.5 | 7.0 | 14.2 | 0.0 |
| 5_sft_triage_dpo | 47.0 | 32.1 | 98.8 | 0.5 | 52.5 | 0.0 |
| 6_sft_triage_dpo_qualite | 76.5 | 76.3 | 85.3 | 8.0 | 15.5 | 0.0 |
| 4_sft_medical_triage + cout | 76.8 | 76.6 | 88.3 | 5.5 | 17.8 | 0.0 |
| 5_sft_triage_dpo + cout | 45.8 | 29.9 | 100.0 | 0.0 | 54.2 | 0.0 |
| 6_sft_triage_dpo_qualite + cout | 77.8 | 77.7 | 88.3 | 5.2 | 17.0 | 0.0 |

## Validation (390 questions)

| experience | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage | P1_predit_P3 |
|---|---|---|---|---|---|---|
| 1_baseline_naive | 34.4 | 17.0 | 100.0 | 0.0 | 65.6 | 0.0 |
| 2_modele_base | 31.0 | 20.5 | 10.4 | 61.5 | 7.4 | 89.6 |
| 3_sft_medical | 29.5 | 24.1 | 45.5 | 38.2 | 32.3 | 53.7 |
| 4_sft_medical_triage | 62.6 | 62.9 | 72.4 | 16.2 | 21.3 | 0.0 |
| 5_sft_triage_dpo | 45.1 | 33.4 | 100.0 | 0.0 | 54.9 | 0.0 |
| 6_sft_triage_dpo_qualite | 62.8 | 62.4 | 77.6 | 16.2 | 21.0 | 0.7 |
| 4_sft_medical_triage + cout | 64.9 | 65.2 | 81.3 | 9.0 | 26.2 | 0.0 |
| 5_sft_triage_dpo + cout | 44.1 | 32.1 | 100.0 | 0.0 | 55.9 | 0.0 |
| 6_sft_triage_dpo_qualite + cout | 64.1 | 64.0 | 83.6 | 10.0 | 25.9 | 0.0 |

Temperatures ajustees sur la validation : 4_sft_medical_triage 2.25, 5_sft_triage_dpo 10, 6_sft_triage_dpo_qualite 2.

## Intervalles de confiance a 95 % sur le test (bootstrap sur les groupes de cas)

2000 tirages avec remise des 72 groupes.

| experience | accuracy | sous_triage |
|---|---|---|
| 1_baseline_naive | 40.8 [24.9 ; 55.6] | 0.0 [0.0 ; 0.0] |
| 2_modele_base | 27.8 [18.0 ; 39.2] | 64.2 [51.9 ; 75.1] |
| 3_sft_medical | 32.2 [25.1 ; 39.5] | 38.2 [30.2 ; 45.4] |
| 4_sft_medical_triage | 78.8 [68.1 ; 87.6] | 7.0 [2.4 ; 13.1] |
| 5_sft_triage_dpo | 47.0 [32.3 ; 61.3] | 0.5 [0.0 ; 1.7] |
| 6_sft_triage_dpo_qualite | 76.5 [65.0 ; 86.4] | 8.0 [3.0 ; 14.2] |
| 4_sft_medical_triage + cout | 76.8 [66.2 ; 85.9] | 5.5 [1.5 ; 10.8] |
| 5_sft_triage_dpo + cout | 45.8 [30.6 ; 60.0] | 0.0 [0.0 ; 0.0] |
| 6_sft_triage_dpo_qualite + cout | 77.8 [67.1 ; 86.8] | 5.2 [1.3 ; 10.5] |

## Regle de cout a alpha 0,35 : SFT contre DPO v2

| experience | split | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage |
|---|---|---|---|---|---|---|
| 4_sft_medical_triage | validation | 64.9 | 65.2 | 81.3 | 9.0 | 26.2 |
| 4_sft_medical_triage | test | 76.8 | 76.6 | 88.3 | 5.5 | 17.8 |
| 6_sft_triage_dpo_qualite | validation | 64.1 | 64.0 | 83.6 | 10.0 | 25.9 |
| 6_sft_triage_dpo_qualite | test | 77.8 | 77.7 | 88.3 | 5.2 | 17.0 |

Graphiques : `outputs/figures/`.
