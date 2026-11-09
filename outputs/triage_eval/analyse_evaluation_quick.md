# Verification de la methode d'evaluation du triage

Valeurs en %. Genere par `scripts/analyze_triage.py` a partir de `predictions_quick.csv` et `scores_quick.csv`.

## 1. Effet de la calibration contextuelle (par question)

`calibre` = methode actuelle (score - score sur prompt vide) ; `brut` = argmax des log-probabilites.

| | accuracy | f1_macro | rappel_P1 | rappel_P2 | rappel_P3 | sous_triage | sur_triage |
|---|---|---|---|---|---|---|---|
| 2_modele_base (calibre) | 23.3 | 19.5 | 0.0 | 37.5 | 40.0 | 56.7 | 20.0 |
| 2_modele_base (brut) | 36.7 | 22.5 | 8.3 | 0.0 | 100.0 | 56.7 | 6.7 |
| 3_sft_medical (calibre) | 33.3 | 27.6 | 0.0 | 50.0 | 60.0 | 53.3 | 13.3 |
| 3_sft_medical (brut) | 43.3 | 33.5 | 50.0 | 0.0 | 70.0 | 40.0 | 16.7 |
| 4_sft_medical_triage (calibre) | 36.7 | 18.3 | 91.7 | 0.0 | 0.0 | 3.3 | 60.0 |
| 4_sft_medical_triage (brut) | 36.7 | 18.3 | 91.7 | 0.0 | 0.0 | 3.3 | 60.0 |

Prior (log-probabilite sur prompt vide) : plus il est eleve, plus la calibration penalise le niveau.

| | P1 | P2 | P3 |
|---|---|---|---|
| 2_modele_base | -27.3 | -34.7 | -27.9 |
| 3_sft_medical | -26.5 | -33.2 | -27.3 |
| 4_sft_medical_triage | -10.2 | -10.3 | -11.9 |

## 2. Metriques par cas clinique

27 cas (P1 = 11, P2 = 8, P3 = 8) pour 30 questions. Chaque cas compte une fois.

| | accuracy | f1_macro | rappel_P1 | rappel_P2 | rappel_P3 | sous_triage | sur_triage |
|---|---|---|---|---|---|---|---|
| 1_baseline_naive (par question) | 40.0 | 19.0 | 100.0 | 0.0 | 0.0 | 0.0 | 60.0 |
| 1_baseline_naive (cas, vote) | 40.7 | 19.3 | 100.0 | 0.0 | 0.0 | 0.0 | 59.3 |
| 2_modele_base (par question) | 36.7 | 22.5 | 8.3 | 0.0 | 100.0 | 56.7 | 6.7 |
| 2_modele_base (cas, vote) | 33.3 | 21.4 | 9.1 | 0.0 | 100.0 | 59.3 | 7.4 |
| 2_modele_base (cas, moyenne scores) | 22.2 | 19.1 | 0.0 | 37.5 | 37.5 | 59.3 | 18.5 |
| 3_sft_medical (par question) | 33.3 | 27.6 | 0.0 | 50.0 | 60.0 | 53.3 | 13.3 |
| 3_sft_medical (cas, vote) | 33.3 | 27.9 | 0.0 | 50.0 | 62.5 | 55.6 | 11.1 |
| 3_sft_medical (cas, moyenne scores) | 33.3 | 28.1 | 0.0 | 50.0 | 62.5 | 55.6 | 11.1 |
| 4_sft_medical_triage (par question) | 36.7 | 18.3 | 91.7 | 0.0 | 0.0 | 3.3 | 60.0 |
| 4_sft_medical_triage (cas, vote) | 37.0 | 18.5 | 90.9 | 0.0 | 0.0 | 3.7 | 59.3 |
| 4_sft_medical_triage (cas, moyenne scores) | 37.0 | 18.5 | 90.9 | 0.0 | 0.0 | 3.7 | 59.3 |

## 3. Coherence des predictions au sein d'un cas

Sur les 3 cas ayant plusieurs questions : part des cas ou toutes les questions recoivent le meme niveau. Le label etant porte par le cas, un modele qui raisonne sur le cas devrait etre coherent.

| | cas_coherents | cas_2_niveaux | cas_3_niveaux |
|---|---|---|---|
| 2_modele_base | 100.0 | 0.0 | 0.0 |
| 3_sft_medical | 66.7 | 33.3 | 0.0 |
| 4_sft_medical_triage | 100.0 | 0.0 | 0.0 |

## 4. Intervalles de confiance a 95 % (bootstrap sur les groupes)

2000 tirages avec remise des 24 groupes : les questions d'un meme groupe sont tirees ensemble, ce qui respecte leur dependance.

| | accuracy | f1_macro |
|---|---|---|
| 1_baseline_naive | 39.0 [15.4 ; 62.5] | 18.3 [8.9 ; 25.6] |
| 2_modele_base | 37.1 [17.2 ; 59.3] | 22.4 [11.1 ; 36.9] |
| 3_sft_medical | 34.0 [16.1 ; 56.0] | 27.1 [13.7 ; 41.8] |
| 4_sft_medical_triage | 35.5 [12.0 ; 59.5] | 17.5 [7.7 ; 25.2] |

Ecart 4_sft_medical_triage - 3_sft_medical : accuracy 1.5 [-39.3 ; 40.0], F1 macro -9.6 [-30.9 ; 8.8].
Part des tirages ou l'ecart d'accuracy est <= 0 : 49.0 %.

## 5. Sensibilite aux cas de test proches du train

Exclusion des 0 cas de test dont la similarite au train est >= 0.5 (0 questions retirees, 30 restantes).

| | accuracy | f1_macro | rappel_P1 | rappel_P2 | rappel_P3 | sous_triage | sur_triage |
|---|---|---|---|---|---|---|---|
| 1_baseline_naive (tout) | 40.0 | 19.0 | 100.0 | 0.0 | 0.0 | 0.0 | 60.0 |
| 1_baseline_naive (sans suspects) | 40.0 | 19.0 | 100.0 | 0.0 | 0.0 | 0.0 | 60.0 |
| 2_modele_base (tout) | 36.7 | 22.5 | 8.3 | 0.0 | 100.0 | 56.7 | 6.7 |
| 2_modele_base (sans suspects) | 36.7 | 22.5 | 8.3 | 0.0 | 100.0 | 56.7 | 6.7 |
| 3_sft_medical (tout) | 33.3 | 27.6 | 0.0 | 50.0 | 60.0 | 53.3 | 13.3 |
| 3_sft_medical (sans suspects) | 33.3 | 27.6 | 0.0 | 50.0 | 60.0 | 53.3 | 13.3 |
| 4_sft_medical_triage (tout) | 36.7 | 18.3 | 91.7 | 0.0 | 0.0 | 3.3 | 60.0 |
| 4_sft_medical_triage (sans suspects) | 36.7 | 18.3 | 91.7 | 0.0 | 0.0 | 3.3 | 60.0 |

