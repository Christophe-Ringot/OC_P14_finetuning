# Criteres d'evaluation et choix du modele final

Ce document fixe ce qui est mesure, avec quel outil, et ce qui a decide du modele livre. Le detail des runs est dans `docs/experiments.md`.

Contexte : projet de formation (POC). La priorite est une methode rigoureuse (memes donnees, meme protocole pour chaque modele, regles choisies sur la validation, limites ecrites) plutot qu'un score maximal.

## 1. Triage (critere principal)

Outil : `src/triage/evaluation/evaluate_sft.py`. Jeu : 400 questions annotees du test (109 cas cliniques), publie a part : `christophe-ringot/p14-medical-triage-clinical-eval`. Score : log-probabilite de chaque niveau apres le prefixe `[Niveau de priorite estime :`, mode brut ou calibre choisi sur la validation.

| Metrique | Pourquoi | Modele final (test / validation) |
|---|---|---|
| Accuracy, F1 macro | Performance globale, classes equilibrees | 78,8 / 62,6 ; 78,7 / 62,9 |
| Rappel P1 | Ne pas rater une urgence vitale | 86,5 / 72,4 |
| Sous-triage | Erreur la plus grave (patient sous-estime) | 7,0 / 16,2 |
| Sur-triage | Cout pour le service (engorgement) | 14,2 / 21,3 |
| P1 predit P3 | Erreur inacceptable | 0 / 0 |

Les deux jeux sont petits (une soixantaine de cas chacun) et la validation est plus difficile que le test : le resultat se cite comme une fourchette.

### Regle de decision au cout

Outil : `src/triage/evaluation/cost_sensitive.py`. Au lieu du niveau le plus probable, le modele choisit le niveau de cout attendu minimal (matrice : sous-triage d'un niveau = 4, P1 -> P3 = 10, sur-triage = 1 ou 2), avec un poids alpha sur le sous-triage.

Regle retenue : **alpha = 0,5, temperature 2,25**, choisis sur la validation (meilleurs F1 et accuracy). Effet : sous-triage 16,2 % -> 4,9 % en validation, 7,0 % -> 4,2 % en test, rappel P1 90 %, au prix d'un sur-triage plus eleve (25 % en test) et d'une accuracy de 70,8 % en test. La matrice de cout est un choix metier a valider par les urgentistes du CHSA. Fichier livre avec le modele : `decision_rule.json`.

## 2. Qualite des reponses

Outils : `evaluate_responses.py` (metriques automatiques, 80 questions fixes du test) et `blind_judgment.py` (jugement en aveugle, grille dans `docs/evaluation_reponses.md`).

| Metrique | SFT + triage | SFT + DPO v2 |
|---|---|---|
| ROUGE-L / F1 lexical | 18,9 / 21,9 | 19,5 / 23,2 |
| Niveau ecrit correct (generation libre) | 80,0 | 78,3 |
| Chiffres absents du cas et de la reference | 18,8 | 25,0 |
| Formulations rassurantes a risque | 0 | 0 |
| Jugement en aveugle : note moyenne sur 5 | 3,00 | 2,90 |
| Jugement en aveugle : prefere (60 cas, 32 egalites) | 20 | 8 |

Le jugement en aveugle prefere le SFT (test du signe sur les 28 cas departages : p = 0,036). Les deux modeles restent faibles en pertinence (1,6 a 1,7 sur 5) : un modele de 1,7 milliard de parametres ne repond pas juste aux questions d'examen specialisees. La reponse sert d'explication au niveau de priorite, pas de conseil medical.

Limites du jugement : un seul juge (Claude, LLM), sans clinicien ; le juge voyait le niveau annote dans la reference.

## 3. Connaissances medicales (QCM)

Outil : `evaluate_mcqa.py`, 108 QCM FrenchMedMCQA du test, generation gloutonne. Sorties : `outputs/final_eval/mcqa_<adaptateur>.csv`.

| Modele | Exact match | F1 par lettre |
|---|---|---|
| SFT + triage | 50,0 | 50,0 |
| SFT + DPO v2 | 49,1 | 49,1 |

Pour 5 reponses possibles, le hasard donne environ 20 %. La DPO ne degrade pas les connaissances (ecart d'une question sur 108). Les anciens chiffres (0,417 au Run 2) portaient sur un autre echantillon (60 QCM, ancien dataset) et ne sont pas comparables.

## 4. Securite des reponses

Outil : `safety_check.py`, filtre heuristique : chiffres absents du cas et de la reference, formulations rassurantes a risque. Applique aux 80 reponses de l'echantillon fixe. Sorties : `outputs/final_eval/safety_flags_<modele>.csv`.

| Modele | Reponses signalees (sur 80) | dont formulation rassurante a risque |
|---|---|---|
| SFT + triage | 15 | 0 |
| SFT + DPO v2 | 20 | 0 |

Tous les signalements sont des chiffres absents du cas et de la reference : a relire, pas forcement faux (un chiffre peut etre calcule a partir du cas ou etre une norme biologique juste).

Le jugement en aveugle a releve des erreurs dangereuses communes aux deux modeles, que le filtre ne voit pas : refus de prostaglandines dans des cardiopathies ducto-dependantes, poursuite des anticoagulants avant une chirurgie, doses de morphine en grammes. Elles justifient que la reponse ne soit jamais montree comme une prescription et qu'un humain valide toute decision.

## 5. Choix du modele final

**Modele livre : SFT + triage (`4_sft_medical_triage`, checkpoint 1300) avec la regle de decision au cout.**

- La DPO v1 (paires qualite + securite) a casse le triage (tout en P1, accuracy 47 %).
- La DPO v2 (paires qualite) conserve le triage, mais le jugement en aveugle prefere les reponses du SFT et elle produit plus de chiffres non sources : rien ne justifie de l'ajouter.
- Le sous-triage se traite par la regle de decision, reglable par le metier sans re-entrainer.

## 6. Ce qui manque avant un usage reel

- Validation clinique des annotations : 50 cas a relire (`outputs/annotation_review/relecture_annotations.csv`, priorite aux cas mal classes et aux P2), accord mesure par `python -m triage.data.build_clinical_eval summarize`.
- Un jeu d'evaluation plus grand, issu de vrais dossiers d'urgence, et non de cas d'examen.
- Une relecture medicale des reponses et de la matrice de cout.
