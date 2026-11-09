# Audit des partitions train / validation / test

Source : `data\processed\sft_avec_priorite.csv` (6747 lignes), joint a `data\interim\corpus_annotated.csv` pour `case_id` et `contexte_patient`.

## 1. Repartition par split

| split | lignes | cas | groupes |
|---|---|---|---|
| train | 5398 | 817 | 2856 |
| validation | 674 | 90 | 357 |
| test | 675 | 104 | 372 |

Lignes par source :

| source | train | validation | test |
|---|---|---|---|
| frenchmedmcqa | 862 | 107 | 108 |
| mediqal_oeq | 3975 | 497 | 497 |
| medquad | 561 | 70 | 70 |

Priorite par question (effectifs / %) :

| priorite | train | validation | test |
|---|---|---|---|
| P1 | 1134 | 123 | 148 |
| P2 | 1114 | 150 | 153 |
| P3 | 928 | 121 | 83 |

| priorite | train | validation | test |
|---|---|---|---|
| P1 | 35.7 | 31.2 | 38.5 |
| P2 | 35.1 | 38.1 | 39.8 |
| P3 | 29.2 | 30.7 | 21.6 |

Priorite par cas clinique (effectifs / %) :

| priorite | train | validation | test |
|---|---|---|---|
| P1 | 324 | 33 | 41 |
| P2 | 261 | 30 | 37 |
| P3 | 232 | 27 | 26 |

| priorite | train | validation | test |
|---|---|---|---|
| P1 | 39.7 | 36.7 | 39.4 |
| P2 | 31.9 | 33.3 | 35.6 |
| P3 | 28.4 | 30.0 | 25.0 |

Questions par cas clinique :

| split | mean | min | max |
|---|---|---|---|
| train | 3.9 | 1.0 | 12.0 |
| validation | 4.4 | 1.0 | 10.0 |
| test | 3.7 | 1.0 | 10.0 |

Questions par groupe :

| split | count | mean | max |
|---|---|---|---|
| train | 647.0 | 4.9 | 14.0 |
| validation | 78.0 | 5.1 | 10.0 |
| test | 82.0 | 4.7 | 10.0 |

## 2. Chevauchements exacts entre splits

Nombre de valeurs distinctes presentes dans plus d'un split.

| index | valeurs_partagees | train/test | train/validation | validation/test |
|---|---|---|---|---|
| case_id | 0 | 0 | 0 | 0 |
| groupe | 0 | 0 | 0 | 0 |
| prompt | 0 | 0 | 0 | 0 |
| contexte_patient | 0 | 0 | 0 | 0 |
| instruction | 22 | 14 | 11 | 5 |
| reponse | 54 | 31 | 28 | 5 |

Note : `instruction` et `reponse` partagees peuvent etre des questions generiques (ex. 'Quel est le diagnostic ?') ; seules celles des cas cliniques sont preoccupantes.

Questions de cas cliniques identiques dans plusieurs splits : 22

- `Comment rectifiez-vous le diagnostic ?`
- `Commentez le bilan biologique.`
- `Interpréter le bilan biologique.`
- `Où se fait le shunt droite-gauche?`
- `Pourquoi ?`
- `Que devez-vous proposer en terme médico-social à ce patient et selon quelles modalités ?`
- `Que faites-vous ?`
- `Que proposez-vous ?`
- `Quel diagnostic faites-vous ?`
- `Quel diagnostic suspectez-vous ?`

## 3. Quasi-doublons de cas cliniques (TF-IDF n-grammes de caracteres, cosinus)

### test vs train (104 cas)

Similarite max au train : mediane 0.341, p90 0.450, max 0.661

| index | >= 0.5 | >= 0.7 | >= 0.8 | >= 0.9 |
|---|---|---|---|---|
| nb_cas | 5 | 0 | 0 | 0 |

- Seuil 0.5 : 5 cas, 11 questions, meme niveau que le voisin train : 100 %

### validation vs train (90 cas)

Similarite max au train : mediane 0.318, p90 0.425, max 0.881

| index | >= 0.5 | >= 0.7 | >= 0.8 | >= 0.9 |
|---|---|---|---|---|
| nb_cas | 4 | 1 | 1 | 0 |

- Seuil 0.5 : 4 cas, 15 questions, meme niveau que le voisin train : 75 %
- Seuil 0.7 : 1 cas, 4 questions, meme niveau que le voisin train : 100 %
- Seuil 0.8 : 1 cas, 4 questions, meme niveau que le voisin train : 100 %

### Top 30 paires test / train les plus similaires

| case_id | priorite | train_case_id | train_priorite | similarite | n_questions |
|---|---|---|---|---|---|
| case_ef780d2639 | P1 | case_6790ccc7e8 | P1 | 0.661 | 1 |
| case_c994da7321 | P3 | case_a417bfef55 | P3 | 0.548 | 3 |
| case_12db8adf38 | P3 | case_518312b819 | P3 | 0.543 | 3 |
| case_b34cefcf4d | P3 | case_bc45c0b000 | P3 | 0.521 | 3 |
| case_269290207d | P1 | case_74a22241f7 | P1 | 0.515 | 1 |
| case_31cdfade21 | P3 | case_6eb5dedaa9 | P3 | 0.497 | 2 |
| case_2df209a060 | P1 | case_db8d0b0603 | P1 | 0.49 | 2 |
| case_bcc9952643 | P2 | case_fa3a392426 | P1 | 0.468 | 1 |
| case_bfa56ab451 | P2 | case_9612fea68e | P1 | 0.465 | 3 |
| case_220479f076 | P2 | case_9a5b9cfe57 | P2 | 0.453 | 1 |
| case_8df5c250fb | P3 | case_9a5b9cfe57 | P2 | 0.451 | 3 |
| case_6b8a9b532b | P1 | case_fc9aecab6c | P1 | 0.448 | 3 |
| case_1cdeef21a9 | P1 | case_6cf019dfba | P1 | 0.448 | 9 |
| case_513389c8af | P2 | case_34742dfa9b | P2 | 0.444 | 1 |
| case_04df8c82f3 | P1 | case_db8d0b0603 | P1 | 0.44 | 1 |
| case_4e96c12db0 | P1 | case_fa3a392426 | P1 | 0.439 | 2 |
| case_6d459f071d | P1 | case_3c2ad635da | P1 | 0.433 | 7 |
| case_66cd2057ce | P1 | case_db8d0b0603 | P1 | 0.432 | 1 |
| case_984e34607e | P2 | case_899afbe4fe | P3 | 0.431 | 5 |
| case_33a018c21f | P3 | case_f5d38db0d0 | P2 | 0.425 | 2 |
| case_ec515f5cb9 | P2 | case_fdfdcf2384 | P3 | 0.416 | 7 |
| case_b27e5600ee | P2 | case_72aa42bf81 | P2 | 0.415 | 9 |
| case_c703408040 | P2 | case_23cdf0132a | P3 | 0.406 | 5 |
| case_edf437d4df | P2 | case_583b347407 | P3 | 0.402 | 6 |
| case_2666087206 | P3 | case_0312840a16 | P2 | 0.402 | 1 |
| case_0e41a55bed | P1 | case_fa3a392426 | P1 | 0.398 | 2 |
| case_d54b32a176 | P1 | case_aaf0a703cc | P3 | 0.394 | 1 |
| case_4613fd2c4a | P1 | case_74a22241f7 | P1 | 0.394 | 1 |
| case_05d219ac9b | P1 | case_0aef82dd1f | P1 | 0.392 | 1 |
| case_c98f0ae619 | P1 | case_23cdf0132a | P3 | 0.39 | 6 |

## 4. Indices de label dans les prompts

- train : 0 / 5398 prompts contiennent un motif de label
- validation : 0 / 674 prompts contiennent un motif de label
- test : 0 / 675 prompts contiennent un motif de label

Reponses test commencant par le prefixe de niveau : 384 (attendu : elles ne sont pas vues a l'evaluation, qui ne score que la continuation du prompt).

## 5. Baseline plus proche voisin (niveau du cas train le plus similaire)

Si cette baseline approche le score du SFT triage (63,8 % / F1 55,0), le modele peut avoir appris par similarite de surface plutot que par raisonnement.

- par cas : accuracy 51.9 %, F1 macro 51.1 %
- par question : accuracy 42.2 %, F1 macro 41.3 %

Matrice de confusion (par cas) :

| priorite | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 23 | 9 | 9 |
| P2 | 8 | 19 | 10 |
| P3 | 5 | 9 | 12 |

