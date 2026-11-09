# Audit des partitions train / validation / test

Source : `data\processed\sft_avec_priorite.csv` (6747 lignes), joint a `data\interim\corpus_annotated.csv` pour `case_id` et `contexte_patient`.

## 1. Repartition par split

| split | lignes | cas | groupes |
|---|---|---|---|
| train | 5398 | 804 | 2838 |
| validation | 675 | 98 | 357 |
| test | 674 | 109 | 344 |

Lignes par source :

| source | train | validation | test |
|---|---|---|---|
| frenchmedmcqa | 862 | 107 | 108 |
| mediqal_oeq | 3975 | 498 | 496 |
| medquad | 561 | 70 | 70 |

Priorite par question (effectifs / %) :

| priorite | train | validation | test |
|---|---|---|---|
| P1 | 1108 | 134 | 163 |
| P2 | 1160 | 133 | 124 |
| P3 | 896 | 123 | 113 |

| priorite | train | validation | test |
|---|---|---|---|
| P1 | 35.0 | 34.4 | 40.8 |
| P2 | 36.7 | 34.1 | 31.0 |
| P3 | 28.3 | 31.5 | 28.2 |

Priorite par cas clinique (effectifs / %) :

| priorite | train | validation | test |
|---|---|---|---|
| P1 | 302 | 43 | 53 |
| P2 | 267 | 31 | 30 |
| P3 | 235 | 24 | 26 |

| priorite | train | validation | test |
|---|---|---|---|
| P1 | 37.6 | 43.9 | 48.6 |
| P2 | 33.2 | 31.6 | 27.5 |
| P3 | 29.2 | 24.5 | 23.9 |

Questions par cas clinique :

| split | mean | min | max |
|---|---|---|---|
| train | 3.9 | 1.0 | 12.0 |
| validation | 4.0 | 1.0 | 11.0 |
| test | 3.7 | 1.0 | 10.0 |

Questions par groupe :

| split | count | mean | max |
|---|---|---|---|
| train | 613.0 | 5.2 | 26.0 |
| validation | 76.0 | 5.1 | 20.0 |
| test | 72.0 | 5.6 | 36.0 |

## 2. Chevauchements exacts entre splits

Nombre de valeurs distinctes presentes dans plus d'un split.

| index | valeurs_partagees | train/test | train/validation | validation/test |
|---|---|---|---|---|
| case_id | 0 | 0 | 0 | 0 |
| groupe | 0 | 0 | 0 | 0 |
| prompt | 0 | 0 | 0 | 0 |
| contexte_patient | 0 | 0 | 0 | 0 |
| instruction | 18 | 15 | 7 | 4 |
| reponse | 51 | 24 | 30 | 5 |

Note : `instruction` et `reponse` partagees peuvent etre des questions generiques (ex. 'Quel est le diagnostic ?') ; seules celles des cas cliniques sont preoccupantes.

Questions de cas cliniques identiques dans plusieurs splits : 17

- `Commenter le bilan biologique.`
- `Commenter les résultats de l'hémogramme.`
- `Commentez le bilan biologique.`
- `Commentez les résultats biologiques.`
- `Où se fait le shunt droite-gauche?`
- `Que faites-vous ?`
- `Que proposez-vous ?`
- `Quel diagnostic évoquez-vous ?`
- `Quel diagnostic évoquez-vous?`
- `Quel est votre diagnostic ?`

## 3. Quasi-doublons de cas cliniques (TF-IDF n-grammes de caracteres, cosinus)

### test vs train (109 cas)

Similarite max au train : mediane 0.317, p90 0.436, max 0.490

| index | >= 0.5 | >= 0.7 | >= 0.8 | >= 0.9 |
|---|---|---|---|---|
| nb_cas | 0 | 0 | 0 | 0 |


### validation vs train (98 cas)

Similarite max au train : mediane 0.306, p90 0.405, max 0.477

| index | >= 0.5 | >= 0.7 | >= 0.8 | >= 0.9 |
|---|---|---|---|---|
| nb_cas | 0 | 0 | 0 | 0 |


### Top 30 paires test / train les plus similaires

| case_id | priorite | train_case_id | train_priorite | similarite | n_questions |
|---|---|---|---|---|---|
| case_74a22241f7 | P1 | case_011e484e0a | P1 | 0.49 | 4 |
| case_db8d0b0603 | P1 | case_2df209a060 | P1 | 0.49 | 1 |
| case_fc9aecab6c | P1 | case_5ad026db0e | P3 | 0.474 | 5 |
| case_82baaf1873 | P1 | case_13d66b22a9 | P1 | 0.472 | 1 |
| case_269290207d | P1 | case_011e484e0a | P1 | 0.471 | 1 |
| case_06f6d8f711 | P2 | case_606114ca28 | P1 | 0.461 | 1 |
| case_85b084e49a | P3 | case_21d6c75484 | P2 | 0.459 | 10 |
| case_e51bdd6e35 | P1 | case_13d66b22a9 | P1 | 0.447 | 1 |
| case_ce74b56f55 | P1 | case_5d31af121f | P2 | 0.446 | 2 |
| case_d920a6fe5a | P1 | case_5d31af121f | P2 | 0.443 | 2 |
| case_4ea8c7112c | P2 | case_7a739e8634 | P1 | 0.442 | 1 |
| case_ae6103d9fa | P1 | case_23eb4a0dd3 | P1 | 0.434 | 1 |
| case_1124400cac | P1 | case_13d66b22a9 | P1 | 0.434 | 1 |
| case_02808e3643 | P1 | case_13d66b22a9 | P1 | 0.433 | 1 |
| case_3c2ad635da | P1 | case_6d459f071d | P1 | 0.433 | 2 |
| case_984e34607e | P2 | case_899afbe4fe | P3 | 0.431 | 5 |
| case_2db54866af | P1 | case_5ea91b3ba5 | P1 | 0.431 | 5 |
| case_7cb8e388fc | P2 | case_23eb4a0dd3 | P1 | 0.43 | 1 |
| case_011ade3992 | P2 | case_7a739e8634 | P1 | 0.428 | 1 |
| case_f5d38db0d0 | P2 | case_33a018c21f | P3 | 0.425 | 3 |
| case_40e1171709 | P2 | case_fbba6d7915 | P3 | 0.419 | 6 |
| case_7b85238054 | P1 | case_13d66b22a9 | P1 | 0.411 | 2 |
| case_9a35a99611 | P2 | case_70cca33646 | P1 | 0.41 | 1 |
| case_0e34b64b97 | P1 | case_5d31af121f | P2 | 0.41 | 3 |
| case_23cdf0132a | P3 | case_c703408040 | P2 | 0.406 | 4 |
| case_937626ea16 | P1 | case_f9bbeffcbb | P1 | 0.406 | 1 |
| case_4e96c12db0 | P1 | case_2d3c127a64 | P1 | 0.398 | 2 |
| case_405937a719 | P1 | case_5d31af121f | P2 | 0.394 | 2 |
| case_6c986c7a0f | P1 | case_5d31af121f | P2 | 0.392 | 2 |
| case_4613fd2c4a | P1 | case_13d66b22a9 | P1 | 0.391 | 1 |

## 4. Indices de label dans les prompts

- train : 0 / 5398 prompts contiennent un motif de label
- validation : 0 / 675 prompts contiennent un motif de label
- test : 0 / 674 prompts contiennent un motif de label

Reponses test commencant par le prefixe de niveau : 400 (attendu : elles ne sont pas vues a l'evaluation, qui ne score que la continuation du prompt).

## 5. Baseline plus proche voisin (niveau du cas train le plus similaire)

Si cette baseline approche le score du SFT triage (63,8 % / F1 55,0), le modele peut avoir appris par similarite de surface plutot que par raisonnement.

- par cas : accuracy 54.1 %, F1 macro 47.8 %
- par question : accuracy 51.5 %, F1 macro 48.7 %

Matrice de confusion (par cas) :

| priorite | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 41 | 10 | 2 |
| P2 | 16 | 7 | 7 |
| P3 | 6 | 9 | 11 |

