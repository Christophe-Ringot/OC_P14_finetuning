# Verification de la methode d'evaluation du triage

Valeurs en %. Genere par `scripts/analyze_triage.py` a partir de `predictions.csv` et `scores.csv`.

## 1. Effet de la calibration contextuelle (par question)

`calibre` = methode actuelle (score - score sur prompt vide) ; `brut` = argmax des log-probabilites.

| | accuracy | f1_macro | rappel_P1 | rappel_P2 | rappel_P3 | sous_triage | sur_triage |
|---|---|---|---|---|---|---|---|
| 2_modele_base (calibre) | 29.8 | 25.8 | 1.2 | 48.4 | 50.4 | 56.0 | 14.2 |
| 2_modele_base (brut) | 27.8 | 17.7 | 6.7 | 0.0 | 88.5 | 64.2 | 8.0 |
| 3_sft_medical (calibre) | 32.2 | 25.7 | 45.4 | 0.8 | 47.8 | 38.2 | 29.5 |
| 3_sft_medical (brut) | 30.8 | 23.2 | 22.1 | 0.0 | 77.0 | 56.5 | 12.8 |
| 4_sft_medical_triage (calibre) | 47.5 | 35.3 | 96.3 | 20.2 | 7.1 | 1.5 | 51.0 |
| 4_sft_medical_triage (brut) | 78.8 | 78.7 | 86.5 | 64.5 | 83.2 | 7.0 | 14.2 |

Prior (log-probabilite sur prompt vide) : plus il est eleve, plus la calibration penalise le niveau.

| | P1 | P2 | P3 |
|---|---|---|---|
| 2_modele_base | -27.3 | -34.7 | -27.9 |
| 3_sft_medical | -29.7 | -33.7 | -28.3 |
| 4_sft_medical_triage | -13.0 | -8.2 | -1.7 |

## 2. Metriques par cas clinique

109 cas (P1 = 53, P2 = 30, P3 = 26) pour 400 questions. Chaque cas compte une fois.

| | accuracy | f1_macro | rappel_P1 | rappel_P2 | rappel_P3 | sous_triage | sur_triage |
|---|---|---|---|---|---|---|---|
| 1_baseline_naive (par question) | 40.8 | 19.3 | 100.0 | 0.0 | 0.0 | 0.0 | 59.2 |
| 1_baseline_naive (cas, vote) | 48.6 | 21.8 | 100.0 | 0.0 | 0.0 | 0.0 | 51.4 |
| 2_modele_base (par question) | 27.8 | 17.7 | 6.7 | 0.0 | 88.5 | 64.2 | 8.0 |
| 2_modele_base (cas, vote) | 22.9 | 13.5 | 1.9 | 0.0 | 92.3 | 72.5 | 4.6 |
| 2_modele_base (cas, moyenne scores) | 22.9 | 13.3 | 1.9 | 0.0 | 92.3 | 75.2 | 1.8 |
| 3_sft_medical (par question) | 32.2 | 25.7 | 45.4 | 0.8 | 47.8 | 38.2 | 29.5 |
| 3_sft_medical (cas, vote) | 37.6 | 26.7 | 60.4 | 0.0 | 34.6 | 32.1 | 30.3 |
| 3_sft_medical (cas, moyenne scores) | 29.4 | 22.0 | 41.5 | 0.0 | 38.5 | 42.2 | 28.4 |
| 4_sft_medical_triage (par question) | 78.8 | 78.7 | 86.5 | 64.5 | 83.2 | 7.0 | 14.2 |
| 4_sft_medical_triage (cas, vote) | 80.7 | 78.5 | 92.5 | 60.0 | 80.8 | 6.4 | 12.8 |
| 4_sft_medical_triage (cas, moyenne scores) | 80.7 | 78.5 | 92.5 | 60.0 | 80.8 | 6.4 | 12.8 |

## 3. Coherence des predictions au sein d'un cas

Sur les 78 cas ayant plusieurs questions : part des cas ou toutes les questions recoivent le meme niveau. Le label etant porte par le cas, un modele qui raisonne sur le cas devrait etre coherent.

| | cas_coherents | cas_2_niveaux | cas_3_niveaux |
|---|---|---|---|
| 2_modele_base | 71.8 | 28.2 | 0.0 |
| 3_sft_medical | 16.7 | 80.8 | 2.6 |
| 4_sft_medical_triage | 80.8 | 19.2 | 0.0 |

## 4. Intervalles de confiance a 95 % (bootstrap sur les groupes)

2000 tirages avec remise des 72 groupes : les questions d'un meme groupe sont tirees ensemble, ce qui respecte leur dependance.

| | accuracy | f1_macro |
|---|---|---|
| 1_baseline_naive | 40.2 [25.1 ; 55.4] | 19.0 [13.4 ; 23.8] |
| 2_modele_base | 28.0 [17.6 ; 39.2] | 17.8 [11.9 ; 23.0] |
| 3_sft_medical | 32.1 [24.8 ; 39.5] | 25.3 [20.6 ; 30.3] |
| 4_sft_medical_triage | 78.6 [68.2 ; 87.4] | 78.1 [68.8 ; 86.5] |

Ecart 4_sft_medical_triage - 3_sft_medical : accuracy 46.5 [36.0 ; 55.8], F1 macro 52.7 [42.7 ; 62.0].
Part des tirages ou l'ecart d'accuracy est <= 0 : 0.0 %.

## 5. Sensibilite aux cas de test proches du train

Exclusion des 0 cas de test dont la similarite au train est >= 0.5 (0 questions retirees, 400 restantes).

| | accuracy | f1_macro | rappel_P1 | rappel_P2 | rappel_P3 | sous_triage | sur_triage |
|---|---|---|---|---|---|---|---|
| 1_baseline_naive (tout) | 40.8 | 19.3 | 100.0 | 0.0 | 0.0 | 0.0 | 59.2 |
| 1_baseline_naive (sans suspects) | 40.8 | 19.3 | 100.0 | 0.0 | 0.0 | 0.0 | 59.2 |
| 2_modele_base (tout) | 27.8 | 17.7 | 6.7 | 0.0 | 88.5 | 64.2 | 8.0 |
| 2_modele_base (sans suspects) | 27.8 | 17.7 | 6.7 | 0.0 | 88.5 | 64.2 | 8.0 |
| 3_sft_medical (tout) | 32.2 | 25.7 | 45.4 | 0.8 | 47.8 | 38.2 | 29.5 |
| 3_sft_medical (sans suspects) | 32.2 | 25.7 | 45.4 | 0.8 | 47.8 | 38.2 | 29.5 |
| 4_sft_medical_triage (tout) | 78.8 | 78.7 | 86.5 | 64.5 | 83.2 | 7.0 | 14.2 |
| 4_sft_medical_triage (sans suspects) | 78.8 | 78.7 | 86.5 | 64.5 | 83.2 | 7.0 | 14.2 |

