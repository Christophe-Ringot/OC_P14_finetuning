# Decision sensible au cout - 4_sft_medical_triage

Le modele donne une probabilite par niveau ; au lieu de prendre le plus probable, on choisit le niveau dont le cout attendu est le plus faible. Aucune re-entrainement : seule la regle de decision change.

## Matrice de cout de reference

Sous-triage d'un niveau = 4, P1 classe P3 = 10, sur-triage = 1 (P2 -> P1, P3 -> P2) ou 2 (P3 -> P1). Choix d'expert a valider avec les urgentistes du CHSA : c'est un parametre metier, pas un hyperparametre.

|  | predit P1 | predit P2 | predit P3 |
|---|---|---|---|
| annote P1 | 0.0 | 4.0 | 10.0 |
| annote P2 | 1.0 | 0.0 | 4.0 |
| annote P3 | 2.0 | 1.0 | 0.0 |

Temperature de calibration apprise sur la validation : T = 2.25 (T > 1 : le modele est trop sur de lui, ses probabilites sont adoucies).

## Resultats (valeurs en %, cout moyen par question x100)

| decision | split | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage | P1_predit_P3 | rappel_P2 | rappel_P3 | cout_moyen_x100 |
|---|---|---|---|---|---|---|---|---|---|---|
| argmax (actuel) | validation | 62.6 | 62.9 | 72.4 | 16.2 | 21.3 | 0.0 | 45.1 | 70.7 | 85.9 |
| cout C (alpha=1) | validation | 61.5 | 58.5 | 97.0 | 1.0 | 37.4 | 0.0 | 55.6 | 29.3 | 41.8 |
| cout C_alpha (alpha=1.25, choisi sur validation) | validation | 59.7 | 55.3 | 99.3 | 0.3 | 40.0 | 0.0 | 54.1 | 22.8 | 41.3 |
| argmax (actuel) | test | 78.8 | 78.7 | 86.5 | 7.0 | 14.2 | 0.0 | 64.5 | 83.2 | 42.8 |
| cout C (alpha=1) | test | 62.5 | 58.8 | 95.1 | 2.2 | 35.2 | 0.0 | 37.1 | 43.4 | 46.5 |
| cout C_alpha (alpha=1.25, choisi sur validation) | test | 61.5 | 57.3 | 95.7 | 2.0 | 36.5 | 0.0 | 35.5 | 40.7 | 47.2 |

## Effet du poids du sous-triage (alpha)

alpha multiplie les couts de sous-triage. Choisi sur la validation (cout moyen minimal) : alpha = 1.25.

Validation :

| alpha | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage | P1_predit_P3 | rappel_P2 | rappel_P3 | cout_moyen_x100 |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 61.0 | 61.7 | 70.9 | 15.6 | 23.3 | 0.0 | 48.9 | 63.4 | 85.9 |
| 0.5 | 66.9 | 66.6 | 89.6 | 4.9 | 28.2 | 0.0 | 57.1 | 52.8 | 47.7 |
| 0.75 | 65.6 | 64.6 | 92.5 | 3.1 | 31.3 | 0.0 | 58.6 | 43.9 | 43.6 |
| 1 | 61.5 | 58.5 | 97.0 | 1.0 | 37.4 | 0.0 | 55.6 | 29.3 | 41.8 |
| 1.25 | 59.7 | 55.3 | 99.3 | 0.3 | 40.0 | 0.0 | 54.1 | 22.8 | 41.3 |
| 1.5 | 58.5 | 53.1 | 100.0 | 0.0 | 41.5 | 0.0 | 53.4 | 18.7 | 41.8 |
| 2 | 55.6 | 49.1 | 100.0 | 0.0 | 44.4 | 0.0 | 48.9 | 14.6 | 47.4 |
| 3 | 50.0 | 41.8 | 100.0 | 0.0 | 50.0 | 0.0 | 36.8 | 9.8 | 56.4 |

Test (pour information, non utilise pour choisir) :

| alpha | accuracy | f1_macro | rappel_P1 | sous_triage | sur_triage | P1_predit_P3 | rappel_P2 | rappel_P3 | cout_moyen_x100 |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 78.5 | 78.6 | 85.3 | 7.2 | 14.2 | 0.0 | 66.1 | 82.3 | 43.8 |
| 0.5 | 70.8 | 69.8 | 90.2 | 4.2 | 25.0 | 0.0 | 52.4 | 62.8 | 44.0 |
| 0.75 | 66.0 | 63.3 | 93.9 | 2.8 | 31.2 | 0.0 | 46.0 | 47.8 | 44.5 |
| 1 | 62.5 | 58.8 | 95.1 | 2.2 | 35.2 | 0.0 | 37.1 | 43.4 | 46.5 |
| 1.25 | 61.5 | 57.3 | 95.7 | 2.0 | 36.5 | 0.0 | 35.5 | 40.7 | 47.2 |
| 1.5 | 59.5 | 54.7 | 95.7 | 1.8 | 38.8 | 0.0 | 32.3 | 37.2 | 48.8 |
| 2 | 56.0 | 49.6 | 95.7 | 1.8 | 42.2 | 0.0 | 29.8 | 27.4 | 52.8 |
| 3 | 50.2 | 40.7 | 96.3 | 1.5 | 48.2 | 0.0 | 21.0 | 15.9 | 58.0 |

## Lecture

- Le compromis sous-triage / sur-triage se regle a l'inference, sans re-entrainer, et de facon explicite (une matrice lisible par un clinicien) plutot que par des biais ajustes a la main.
- Une ponderation de la perte pendant l'entrainement (poids plus forts sur les tokens du niveau des cas P1/P2) est l'alternative ; elle demande un re-entrainement complet (~7 h) et deplace les memes frontieres. Avec un modele qui ordonne deja bien les cas (AUC P1/P2 = 0.86, P2/P3 = 0.92, voir analyse_erreurs.md), la decision au cout est le levier le moins cher.
- Sur la validation, la decision au cout divise le cout moyen par 2 (sous-triage 16 % -> 1 %). Sur le test, elle ramene le sous-triage de 7 % a 2 % et le rappel P1 a 95 %, mais le sur-triage passe de 14 % a 35 % et le cout moyen ne baisse pas : la validation (accuracy 63 %) est bien plus difficile que le test (79 %), avec ~70 cas par split, donc un reglage appris sur l'une se transpose mal a l'autre.
- Point de fonctionnement raisonnable : alpha = 0.5 (test : sous-triage 4.2 %, rappel P1 90 %, accuracy 71 %). Le choix final releve du CHSA (combien de sur-triage accepter pour un P1 rattrape) et demande un jeu de validation plus grand, valide cliniquement.
