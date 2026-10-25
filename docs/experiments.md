# Journal d'experimentation SFT

Suivi des runs de Supervised Fine-Tuning (LoRA) sur le POC de triage medical.
Chaque run est documente avec ses hyperparametres, ses courbes de loss
(train/validation) et une lecture rapide du risque de surapprentissage.

## Run 1 - 2026-10-14

**Contexte** : premier run SFT complet et reproductible. Le run precedent
(debut octobre) avait produit des exemples qualitatifs
(`outputs/sft_qualitative_examples.csv`) mais aucune trace des courbes de
loss (`report_to="none"`, pas de log persiste) et le checkpoint avait ete
supprime du disque. `scripts/train_sft.py` a ete modifie pour sauvegarder
systematiquement le tokenizer, l'etat du trainer (`trainer_state.json`) et un
resume JSON des hyperparametres + historique de loss
(`outputs/sft_lora/training_summary.json`).

### Configuration

| Parametre | Valeur |
|---|---|
| Modele de base | Qwen/Qwen3-1.7B-Base |
| Methode | LoRA (r=16, alpha=32, dropout=0.05, cibles q/k/v/o_proj) |
| Dataset | `data/processed/sft_dataset_anonymized.csv` |
| Exemples train / validation | 200 / 50 (echantillonnes, seed=42) |
| Epochs | 1 |
| Batch effectif | 1 x 4 (grad accumulation) = 4 (environnement CPU-only) |
| Learning rate | 2e-4 (lineaire, sans warmup) |
| Longueur max sequence | 256 tokens |
| Materiel | CPU uniquement (pas de GPU disponible dans cet environnement) |
| Duree | ~50 min (49 min 44 s d'apres `train_runtime`) |

### Courbe de loss

| Step | Epoch | Train loss | Eval loss | Eval token accuracy |
|---|---|---|---|---|
| 5  | 0.1 | 1.932 | - | - |
| 10 | 0.2 | 1.713 | - | - |
| 15 | 0.3 | 1.671 | - | - |
| 20 | 0.4 | 1.722 | 1.464 | 0.679 |
| 25 | 0.5 | 1.390 | - | - |
| 30 | 0.6 | 1.479 | - | - |
| 35 | 0.7 | 1.436 | - | - |
| 40 | 0.8 | 1.535 | 1.391 | 0.693 |
| 45 | 0.9 | 1.393 | - | - |
| 50 | 1.0 | 1.563 | 1.380 | 0.695 |

Train loss moyenne sur l'epoch : 1.584 (`train_loss` final rapporte par le
trainer).

### Lecture / surapprentissage

- La eval_loss baisse de maniere monotone (1.464 -> 1.391 -> 1.380) sur les
  3 points de mesure, sans rebond : **aucun signe de surapprentissage** sur
  cette seule epoch.
- La train_loss est bruyante (petit batch effectif = 4, 200 exemples
  seulement) mais sa tendance generale est aussi a la baisse.
- `eval_mean_token_accuracy` progresse de 0.679 a 0.695, cohérent avec un
  apprentissage qui se poursuit sans encore converger.
- Avec un seul epoch et un dataset de 200 exemples, ce run est surtout une
  **validation de pipeline** (bout-en-bout : donnees -> LoRA -> sauvegarde ->
  evaluation) plutot qu'un entrainement representatif de la performance
  finale visee.

### Evaluation sur le jeu de test (`scripts/evaluate_sft.py`)

- **Test loss (moyenne, 50 exemples)** : 1.317 | **perplexite** : 3.73
- Ce chiffre est **meilleur** que la derniere eval_loss de validation (1.380)
  et que la train_loss moyenne de l'epoch (1.584) : aucun signe de
  surapprentissage train/test, le modele generalise au moins aussi bien sur
  le test que sur le train a ce stade (coherent avec un entrainement encore
  tres court - 1 epoch, 200 exemples).
- Bug rencontre et corrige pendant cette execution : `evaluate_sft.py`
  chargeait le modele avec `device_map="auto"`, ce qui a declenche un
  offload disque par `accelerate` (RAM libre ~4 Go sur cette machine pour un
  modele de 1.7B en float32, soit ~7 Go de poids) et faisait planter le
  forward (`NotImplementedError: Cannot copy out of meta tensor`). Corrige
  en n'utilisant `device_map="auto"` que si un GPU est disponible, comme le
  fait deja `train_sft.py`.

**Relecture qualitative (10 exemples du jeu de test, greedy decoding)** :

| # | Source | Resultat |
|---|---|---|
| 1 | mediqal (QCM clinique) | Partiellement correct : trouve B et C mais ajoute A et E en trop |
| 2 | mediqal (QCM) | Partiellement correct : trouve B et D mais manque A, ajoute C |
| 3 | mediqal (QCM) | Partiellement correct : trouve A, B, D mais ajoute C en trop |
| 4 | frenchmedmcqa (QCM mono-reponse) | **Exact** |
| 5 | mediqal (QCM, 5 bonnes reponses) | Partiellement correct : trouve 4/5, manque E |
| 6 | mediqal (QCM) | Partiellement correct : trouve B et C mais manque D, ajoute A et E |
| 7 | medquad (QA ouverte, Wilson disease) | Contenu medicalement correct et bien formule, differe du libelle exact de reference |
| 8 | mediqal (QCM) | Partiellement correct : trouve B, C, D mais manque A |
| 9 | frenchmedmcqa (QCM mono-reponse) | Incorrect : repond B au lieu de C |
| 10 | medquad (QA ouverte, chiffre epidemiologique) | Structure de reponse correcte, mais invente un chiffre different de la reference |

**Lecture** : sur les QCM a reponses multiples, le modele identifie
regulierement une partie des bonnes reponses mais se trompe encore souvent
sur l'ensemble exact (faux positifs/negatifs sur les propositions) - un
schema de **sous-apprentissage** (underfitting) plutot que de
surapprentissage, coherent avec un entrainement tres court. Sur les
questions ouvertes (medquad), le modele produit des reponses fluides et
globalement justes sur le fond mais peut halluciner des chiffres precis
(ex. #10) - point de vigilance deja identifie dans
`docs/rgpd_anonymisation.md` (risque d'hallucination de donnees chiffrees
memorisees/inventees), a surveiller specifiquement pour un futur agent de
triage.

### Limites connues de ce run

- Volume d'entrainement volontairement reduit (200/50 exemples) pour rester
  executable sur CPU en un temps raisonnable - pas representatif d'un run de
  production.
- Pas de grid/random search sur les hyperparametres a cette iteration : un
  seul jeu (LR=2e-4, r=16, 1 epoch) a ete teste.
- Pas de matrice de confusion sur un niveau d'urgence : le dataset SFT actuel
  est generaliste (QCM/QA medical), sans label de triage. Cf. note dans
  `scripts/evaluate_sft.py`.

### Prochaines experimentations envisagees

- Augmenter `MAX_TRAIN_EXAMPLES` / nombre d'epochs si le temps CPU le permet,
  ou passer sur GPU pour lever la contrainte de volume.
- Tester un learning rate plus bas (ex. 1e-4) pour reduire le bruit observe
  sur la train_loss.
- Comparer `r=16` vs `r=8`/`r=32` sur l'impact train/eval loss.
