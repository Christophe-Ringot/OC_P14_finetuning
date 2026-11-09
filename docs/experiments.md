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

## Verification de preparation au DPO - 2026-10-25

Avant de lancer le DPO, verification des prerequis du projet (voir aussi
`docs/evaluation_criteria.md` pour le detail des seuils proposes).

- **Metrique quantitative QCM manquante** : la perplexite/accuracy token-level
  ne dit rien de la justesse des reponses QCM (coeur du dataset). Ajout de
  `scripts/evaluate_mcqa.py` : parse les lettres A-E generees vs reference et
  calcule exact-match + precision/recall/F1, par source_dataset. Resultats
  (60 exemples test, 30 par source, `outputs/sft_mcqa_eval.csv`) :

  | source_dataset | exact match | precision | recall | F1 |
  |---|---|---|---|---|
  | frenchmedmcqa (mono-reponse) | 0.567 | 0.567 | 0.567 | 0.567 |
  | mediqal (multi-reponse, cas clinique) | 0.067 | 0.613 | 0.757 | 0.649 |
  | **Global** | **0.317** | - | - | **0.608** |

  Lecture : sur `mediqal` (QCM a reponses multiples), le modele a un bon
  recall (0.757, il retrouve la plupart des bonnes reponses) mais une
  precision moyenne (0.613, il ajoute des options en trop) -> l'ensemble
  exact est juste seulement 6.7% du temps. C'est la confirmation
  quantitative du schema de sous-apprentissage deja vu en qualitatif
  (section Run 1). Les 3 seuils proposes dans `docs/evaluation_criteria.md`
  sont **tous les 3 non atteints** (exact match 0.317 < 0.60 ; F1 0.608 <
  0.75 ; ecart mediqal/frenchmedmcqa 0.50 >> 0.15 attendu) : confirmation
  objective que ce run SFT (smoke test, 200 exemples) n'est pas pret a
  servir de politique de depart au DPO.
- **Controle de securite absent** : ajout de `scripts/safety_check.py`
  (heuristique : nombres generes absents instruction+reference, formulations
  rassurantes a risque). Premiere passe sur les 10 exemples qualitatifs
  existants (`outputs/sft_qualitative_examples.csv`) : 2/10 signales, tous
  deux des hallucinations numeriques deja identifiees plus haut (chromosome
  13 pour Wilson - fait correct mais absent de la reference ; chiffre
  epidemiologique invente). Aucune formulation dangereuse detectee sur cet
  echantillon (trop petit pour conclure). A relancer sur un echantillon plus
  large une fois `evaluate_mcqa.py` dispo (meme format de colonnes).
- **Dataset DPO anglais vs SFT francais** : decision actee avec l'equipe de
  garder le dataset DPO (`UltraMedical-Preference`, 100% anglais) tel quel
  pour cette iteration, faute de source de preference francaise disponible.
  Limite connue a documenter sur le modele final.
- **Pas de GPU dans l'environnement courant** : prerequis du cahier des
  charges toujours non rempli (`torch.cuda.is_available()` -> False). Bloque
  un run SFT representatif et un DPO dans un temps raisonnable - a lever
  avant la suite, hors de portee de cet environnement.
- **Pas de tracking MLflow/W&B** : toujours absent (`report_to=[]`),
  recommande par le cahier des charges avant la montee en charge.

**Conclusion** : le DPO n'est pas encore lance. Points bloquants restants :
acces GPU (externe a l'environnement) et validation humaine des seuils
proposes dans `docs/evaluation_criteria.md` une fois un run SFT
representatif disponible.

## Run 2 - 2026-10-25 (GPU, run representatif)

**Contexte** : GPU local active (RTX 2060, 6.4 Go VRAM - `torch` reinstalle
en version CUDA, `torch.cuda.is_available()` passait a `False` jusqu'ici car
le paquet installe etait la roue CPU-only). Run representatif lance par
l'utilisateur avec `scripts/train_sft.py` mis a jour : tout le train set
(3366 exemples, vs 200 au Run 1), 3 epochs, `max_length=1024` (vs 256),
`load_best_model_at_end` sur `eval_loss`. Sortie dans
`outputs/sft_lora_run2` (Run 1 conserve pour comparaison).

### Courbe eval_loss (tous les points, eval_steps=100)

| Step | Epoch | Eval loss | Eval token accuracy |
|---|---|---|---|
| 100 | 0.24 | 1.2095 | 0.7219 |
| 400 | 0.95 | 1.1626 | 0.7298 |
| 800 | 1.90 | 1.1467 | 0.7321 |
| 1100 | 2.61 | 1.1471 | 0.7303 |
| 1263 | 3.00 | 1.1457 | 0.7308 |

**Test loss (moyenne, `scripts/evaluate_sft.py`, test set complet)** : 1.1555
| **perplexite** : 3.18 (vs 1.317 / 3.73 au Run 1). Coherent avec l'eval_loss
(1.146) : toujours aucun signe de surapprentissage train/test.

**Lecture** : nette amelioration vs Run 1 (eval_loss final 1.146 vs 1.380 ;
eval_token_accuracy 0.731 vs 0.696). Mais la courbe **plateaue** a partir de
l'epoch ~1.9 (1.1467 -> 1.1482 -> 1.1494 -> 1.1471 -> 1.1456 -> 1.1457,
bruit sans tendance) : le modele a converge sur ce volume de donnees/ce LR,
pas de signe de surapprentissage (`load_best_model_at_end` n'a pas eu
besoin d'arbitrer fortement) mais rajouter des epochs sur la meme
configuration n'apporterait plus grand chose.

### Evaluation quantitative QCM (`scripts/evaluate_mcqa.py`, N=60)

| source_dataset | exact match | precision | recall | F1 |
|---|---|---|---|---|
| frenchmedmcqa (mono-reponse) | 0.600 (+0.033) | 0.600 | 0.600 | 0.600 (+0.033) |
| mediqal (multi-reponse) | 0.233 (+0.167) | 0.663 | 0.717 | 0.665 (+0.015) |
| **Global** | **0.417** (+0.100) | - | - | **0.632** (+0.025) |

**Comparaison aux seuils de `docs/evaluation_criteria.md`** :

| Seuil | Cible | Run 1 | Run 2 | Statut |
|---|---|---|---|---|
| Exact match global | >= 0.60 | 0.317 | 0.417 | **Non atteint** (progression nette) |
| F1 moyen | >= 0.75 | 0.608 | 0.632 | **Non atteint** (progression faible) |
| Ecart mediqal / frenchmedmcqa | < 0.15 | 0.50 | 0.367 | **Non atteint** (desequilibre reduit mais toujours fort) |

Le recall sur mediqal reste superieur a la precision (0.717 vs 0.663) :
le modele continue a "sur-proposer" des options sur les QCM a reponses
multiples, meme avec 17x plus de donnees d'entrainement.

### Controle de securite (`scripts/safety_check.py`)

Sur les 10 exemples qualitatifs regeneres avec le nouveau modele :
**0/10 signale** (vs 2/10 au Run 1). Les deux cas d'hallucination numerique
du Run 1 ont disparu : sur Wilson disease, la reponse reste maintenant
alignee avec la reference (pas de chromosome invente) ; sur la question
epidemiologique (BOR/BO syndrome), le modele repond desormais "prevalence
inconnue" plutot que d'inventer un chiffre different de la reference -
plus prudent, meme si toujours pas un match exact avec la reference. A
confirmer sur un echantillon plus large (le seuil propose exige >= 100
exemples, on n'en a que 10 ici) avant de considerer ce point comme valide.

### Lecture qualitative (reprise des memes 10 exemples)

- 2 reponses exactes (#4, #9 - les deux frenchmedmcqa mono-reponse), contre
  1 au Run 1.
- Sur mediqal (multi-reponse), le schema de sous-apprentissage persiste :
  le modele trouve une bonne partie des reponses mais ajoute ou omet encore
  une option sur la quasi-totalite des exemples (#1, #2, #3, #5, #6, #8).
- Sur l'exemple #6, la degradation est plus nette qu'au Run 1 (perd B,
  ajoute A et E) - signe que l'amelioration n'est pas uniforme sur tous les
  cas.

### Conclusion Run 2

Progression reelle et mesurable par rapport au Run 1 sur toutes les
metriques (eval_loss, accuracy, exact match QCM, F1, controle de securite),
mais **aucun des 3 seuils quantitatifs proposes n'est encore atteint**, et
la courbe eval_loss plateaue : repartir sur plus d'epochs avec la meme
configuration ne suffira probablement pas. Pistes a tester avant de
reconsiderer le passage au DPO :
- Elargir les modules LoRA cibles (ajouter `gate_proj`/`up_proj`/`down_proj`
  en plus de `q/k/v/o_proj`) ou augmenter `r` (16 -> 32) : le plateau de
  eval_loss suggere une limite de capacite plutot qu'un manque de donnees.
- Reduire le learning rate (ex. 1e-4) comme deja envisage au Run 1.
- Investiguer specifiquement le desequilibre mediqal/frenchmedmcqa (le
  format multi-reponse semble structurellement plus dur pour le modele a
  ce stade) - eventuellement renforcer sa part dans l'echantillonnage.
- Etendre le controle de securite a >= 100 exemples avant de valider ce
  critere.

**Le DPO reste premature.**

## Run 3 (quick) - 2026-10-25 (GPU, sous-ensemble r=32 / lr=1e-4)

**Contexte** : test rapide avant d'engager un run complet de 3h - mode
sous-ensemble ajoute a `scripts/train_sft.py` (`MAX_TRAIN_EXAMPLES=900`,
echantillonnage stratifie par `source_dataset`), avec `r=32` (vs 16),
`lora_alpha=64` (ratio alpha/r conserve a 2x), `learning_rate=1e-4` (vs
2e-4). Reste identique au Run 2 : 3 epochs, `max_length=1024`,
`load_best_model_at_end`. Sortie : `outputs/sft_lora_run3_quick`.

### Resultats

| Metrique | Run 1 (200 ex) | Run 2 (3366 ex, r=16/lr=2e-4) | Run 3 quick (900 ex, r=32/lr=1e-4) |
|---|---|---|---|
| Eval loss finale | 1.380 | 1.146 | 1.194 |
| Eval token accuracy | 0.696 | 0.731 | 0.726 |
| Test loss / perplexite | 1.317 / 3.73 | 1.156 / 3.18 | 1.210 / 3.35 |
| Exact match QCM global | 0.317 | 0.417 | 0.367 |
| F1 moyen QCM | 0.608 | 0.632 | 0.631 |
| Exact match mediqal | 0.067 | 0.233 | 0.133 |
| Exact match frenchmedmcqa | 0.567 | 0.600 | 0.600 |
| Securite (n=10) | 2 signalements | 0 signalement | 0 signalement |

**Lecture - resultat non concluant, confondu par le volume de donnees** :
Run 3 quick fait moins bien que Run 2 sur l'eval_loss, le test loss et
l'exact match global/mediqal, mais a ete entraine sur 27% des donnees de
Run 2 (900 vs 3366). On ne peut donc **pas conclure que `r=32`/`lr=1e-4`
est moins bon** que la config du Run 2 - la comparaison n'est pas
controlee (deux variables changent a la fois : les hyperparametres ET le
volume de donnees). Point positif : la courbe eval_loss de Run 3 quick
converge avec une forme similaire a Run 2 (plateau atteint tres tot,
epoch ~1.8-2, 1.2254 -> 1.1999 -> 1.1948 -> 1.1943), sans divergence ni
signe d'instabilite avec le LR plus bas - le changement d'hyperparametres
n'a pas degrade l'entrainement en soi.

**Prochaine etape pour trancher proprement** : soit (a) rejouer la config
du Run 2 (`r=16`, `lr=2e-4`) sur le meme sous-ensemble de 900 exemples
pour une comparaison a volume de donnees egal, soit (b) lancer directement
le run complet (3366 exemples, `MAX_TRAIN_EXAMPLES=None`) avec
`r=32`/`lr=1e-4` puisque rien dans le test rapide n'indique un risque a
passer a l'echelle.

### Conclusion Run 3 quick

Aucun des 3 seuils quantitatifs n'est atteint (exact match 0.367 < 0.60 ;
F1 0.631 < 0.75 ; ecart mediqal/frenchmedmcqa 0.467 > 0.15, pire que le Run
2). **Le DPO reste premature.** Ce run ne permet pas de confirmer que
`r=32`/`lr=1e-4` ameliore les choses par rapport au Run 2 - il faut soit un
comparatif controle, soit le run complet, avant de rejouer l'evaluation.

## Run 3 (complet) - 2026-10-26 (GPU, tout le train set, r=32 / lr=1e-4)

**Contexte** : run complet avec `MAX_TRAIN_EXAMPLES=None` (3366 exemples,
identique au Run 2 en volume), `r=32`, `lora_alpha=64`, `lr=1e-4`, 3 epochs.
Cette fois la comparaison avec le Run 2 est **controlee** (meme donnees,
seuls les hyperparametres changent). Sortie : `outputs/sft_lora_run3`.

### Resultats vs Run 2 (meme volume de donnees)

| Metrique | Run 2 (r=16, lr=2e-4) | Run 3 (r=32, lr=1e-4) | Delta |
|---|---|---|---|
| Eval loss finale | 1.1457 | 1.1495 | quasi identique |
| Eval token accuracy | 0.7308 | 0.7315 | quasi identique |
| Test loss / perplexite | 1.1555 / 3.18 | 1.1592 / 3.19 | quasi identique |
| Exact match QCM global | 0.417 | 0.367 | **-0.050** |
| F1 moyen QCM | 0.632 | 0.598 | **-0.034** |
| Exact match mediqal | 0.233 | 0.200 | -0.033 |
| Exact match frenchmedmcqa | 0.600 | 0.533 | **-0.067** |
| Securite (n=10) | 0 signalement | 0 signalement | egal |

**Lecture - resultat net, cette fois controle** : a volume de donnees
egal, `r=32`/`lr=1e-4` **n'ameliore pas** le modele par rapport a la config
du Run 2 (`r=16`/`lr=2e-4`) - la loss/perplexite est quasiment identique,
mais les metriques QCM (celles qui comptent reellement pour la tache) sont
**legerement moins bonnes** sur toutes les lignes. Conclusion importante :
le plateau de eval_loss observe au Run 2 n'etait donc **pas un probleme de
capacite du LoRA** (augmenter `r` n'aide pas) ni de learning rate trop
eleve (le baisser n'aide pas non plus) - l'hypothese testee est invalidee
par un resultat controle, pas juste un run qui n'a pas marche.

**Le Run 2 reste le meilleur modele obtenu a ce jour** sur toutes les
metriques QCM.

### Conclusion Run 3 (complet)

Aucun des 3 seuils n'est atteint, et ce run n'ameliore rien par rapport au
Run 2 (legere regression sur l'exact match/F1 QCM). **Le DPO reste
premature.** Piste a abandonner : agir sur `r`/`lr` du LoRA. Pistes
restantes, plus prometteuses puisque le goulot n'est pas la capacite du
modele :
- Le desequilibre mediqal (multi-reponse) vs frenchmedmcqa (mono-reponse)
  persiste a l'identique sur les 3 runs - investiguer le format/la
  difficulte structurelle de la tache multi-reponse plutot que les
  hyperparametres d'entrainement.
- Elargir les modules LoRA cibles (`gate_proj`/`up_proj`/`down_proj`) reste
  a tester - c'est un changement de nature differente (capacite sur les
  couches MLP, pas seulement l'attention) de ce qui vient d'etre invalide.
- Revenir a la config du Run 2 (`r=16`, `lr=2e-4`) comme base de reference
  pour la suite des experimentations.

## Run 4 - 2026-10-26 (GPU, tout le train set, modules LoRA elargis)

**Contexte** : retour a la config du Run 2 (`r=16`, `lora_alpha=32`,
`lr=2e-4`), seule variable qui change : `target_modules` elargi a
`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
(couches MLP en plus de l'attention). 17.4M parametres LoRA entrainables
(vs un ordre de grandeur plus bas pour attention seule). Test VRAM
prealable : 4.77 Go / 6.44 Go, ~9.6s/step (vs 8.0s/step attention seule).
Tout le train set, 3 epochs, `load_best_model_at_end` sur `eval_loss`.
Sortie : `outputs/sft_lora_run4`.

### Courbe eval_loss - un signal de surapprentissage leger apparait

| Step | Epoch | Eval loss |
|---|---|---|
| 400 | 0.95 | 1.1277 |
| 700 | 1.66 | 1.1226 |
| **800** | **1.90** | **1.1118 (meilleur point, selectionne par `load_best_model_at_end`)** |
| 900 | 2.14 | 1.1318 |
| 1000 | 2.38 | 1.1391 |
| 1263 | 3.00 | 1.1337 |

Contrairement aux Run 2/3 (plateau sans rebond), l'eval_loss **remonte**
legerement apres le step 800 (epoch ~1.9) avant de se re-stabiliser - premier
signe (leger) de surapprentissage observe sur ces runs. `load_best_model_at_end`
a bien joue son role en retenant le checkpoint-800 plutot que le dernier.

### Resultats vs Run 2 (meme donnees, memes r/lr, modules LoRA differents)

| Metrique | Run 2 (attention seule) | Run 4 (+ MLP) | Delta |
|---|---|---|---|
| Eval loss (meilleur point) | 1.1457 (step 1263) | **1.1118** (step 800) | **meilleur** |
| Test loss / perplexite | 1.1555 / 3.18 | **1.1218 / 3.07** | **meilleur** |
| Exact match QCM global | 0.417 | **0.317** | **-0.100, pire** |
| F1 moyen QCM | 0.632 | **0.618** | -0.014, pire |
| Exact match mediqal | 0.233 | **0.067** | **-0.167, pire (retombe au niveau du Run 1)** |
| Exact match frenchmedmcqa | 0.600 | 0.567 | -0.033, pire |
| Securite (n=10) | 0 signalement | 0 signalement | egal |

**Lecture - divergence importante entre loss et metrique tache** : elargir
les modules LoRA aux couches MLP **ameliore la perplexite** (meilleur
resultat des 4 runs, 3.07) mais **degrade nettement l'exact match QCM**,
en particulier sur `mediqal` qui retombe au niveau du Run 1 (0.067, pire
que Run 2 et Run 3). C'est la demonstration concrete que **la perplexite
n'est pas un bon proxy pour la qualite reelle sur ce dataset** : le modele
devient globalement plus fluide/confiant (la loss moyenne sur tous les
tokens baisse, dominee par les reponses longues de `medquad`) mais perd en
precision sur la tache de selection de lettres QCM, qui ne represente
qu'une petite fraction des tokens generes et pese donc peu dans la loss
agregee.

Autre limite methodologique a noter : `load_best_model_at_end` a selectionne
le checkpoint-800 **sur la base de `eval_loss` uniquement** - rien ne dit
que c'est le meilleur checkpoint pour l'exact match QCM (on n'a pas cette
metrique a chaque checkpoint, seulement sur le modele final). Un autre
checkpoint de ce meme run pourrait faire mieux sur le QCM ; ce n'est pas
verifie ici.

### Conclusion Run 4

**Le Run 2 reste le meilleur modele sur la metrique qui compte reellement**
(exact match QCM), malgre une perplexite moins bonne. Deux pistes de
capacite LoRA testees (rank/lr au Run 3, modules MLP au Run 4) **degradent
toutes les deux** l'exact match QCM par rapport au Run 2 - la piste
"capacite du LoRA" est maintenant epuisee dans les deux dimensions
testees. **Le DPO reste premature** (seuils toujours non atteints, et
regression sur ce run precis).

Lecon methodologique a retenir pour la suite : **ne plus utiliser
`eval_loss`/perplexite comme seul critere de selection de checkpoint ou de
run** - verifier systematiquement `evaluate_mcqa.py` avant de qualifier un
changement d'amelioration. Pistes restantes : investiguer le
desequilibre structurel mediqal/frenchmedmcqa (persistant sur les 4 runs),
ou envisager une metrique de selection de checkpoint basee sur le QCM
plutot que sur l'eval_loss generique.

## Run 5 - 2026-10-26 (GPU, quotas de sur-echantillonnage mediqal)

**Contexte** : retour a la config du Run 2 (`r=16`, `lr=2e-4`, modules
attention seule) pour isoler l'effet d'un changement de composition du
train set. Quotas cibles appliques via `sample_by_quota()` :
`{mediqal: 0.44, frenchmedmcqa: 0.06, medquad: 0.50}` (le brief distinguait
`mediqal_oeq: 0.30` / `mediqal_mcq: 0.14`, fusionnes car les donnees
mediqal reelles sont 100% QCM, aucun exemple OEQ). Mediqal n'ayant que 771
exemples train disponibles pour une cible de 1481 (44% de 3366), le
sur-echantillonnage se fait **avec remise** : ~1.9x de repetition moyenne,
distribution inegale (certains exemples non tires, d'autres tires 3-4
fois). Validation/test NON reechantillonnes (distribution naturelle
conservee pour une mesure fidele). Sortie : `outputs/sft_lora_run5`.

### Resultats vs Run 2 (meme r/lr/modules, composition du train set differente)

| Metrique | Run 2 (distribution naturelle) | Run 5 (quotas mediqal=44%) | Delta |
|---|---|---|---|
| Eval loss (meilleur point) | 1.1457 (step 1263) | 1.1722 (step 800) | pire |
| Test loss / perplexite | 1.1555 / 3.18 | 1.1945 / 3.30 | pire |
| Exact match QCM global | 0.417 | **0.333** | **-0.084, pire** |
| F1 moyen QCM | 0.632 | **0.594** | **-0.038, pire (pire score des 5 runs)** |
| Exact match mediqal | 0.233 | **0.100** | **-0.133, pire malgre 2x plus de donnees mediqal** |
| Recall mediqal | 0.717 | 0.646 | pire |
| Exact match frenchmedmcqa | 0.600 | 0.567 | pire (quota reduit a 6%) |
| Securite (n=10) | 0 signalement | 0 signalement | egal |

**Lecture - resultat contre-intuitif et net** : sur-echantillonner mediqal
(23% -> 44% du train, par oversampling avec remise) **degrade** sa propre
exact match (0.233 -> 0.100) au lieu de l'ameliorer, et degrade tout le
reste. La courbe eval_loss confirme un **vrai surapprentissage** cette
fois (contrairement aux Runs 2/3/4 qui plateauent proprement) : minimum a
1.1722 au step 800, puis remontee continue jusqu'a 1.1975 au step 1263 -
coherent avec la duplication d'exemples mediqal vus plusieurs fois par
epoch. L'hypothese "mediqal manque de volume d'entrainement" est donc
**invalidee** : le modele memorise les repetitions plutot que de mieux
generaliser sur le format multi-reponse, et starver `frenchmedmcqa`
(6% au lieu de 18%) prive en plus le modele d'exemples qui l'aidaient
peut-etre a stabiliser le format de reponse general.

### Conclusion Run 5

**Troisieme piste testee et invalidee** (apres rank/lr et modules MLP) :
rebalancer la composition du train set par oversampling simple ne resout
pas le desequilibre mediqal/frenchmedmcqa, et degrade le resultat global.
**Le Run 2 (distribution naturelle, r=16, lr=2e-4, attention seule) reste
le meilleur modele obtenu sur les 5 runs**, sur toutes les metriques QCM.
Si une piste de rebalancement devait etre retentee, elle demanderait une
methode plus fine qu'un oversampling aleatoire (ex. deduplication des
repetitions a chaque epoch, ou ponderation de la loss plutot que
duplication physique des exemples) - hors scope pour l'instant.

### Recalibrage des criteres d'acceptation (contexte educatif)

L'utilisateur precise que ce projet est **a but educatif** : l'objectif
n'est pas un score "excellent" mais un resultat "acceptable", avec une
methodologie rigoureuse (ce que les 5 runs documentes ci-dessus demontrent
: comparaisons controlees, detection d'un biais perplexite/tache, detection
d'un surapprentissage, invalidation de 3 hypotheses d'amelioration).
Les seuils initiaux de `docs/evaluation_criteria.md` (exact match >= 0.60,
F1 >= 0.75) etaient calibres sans reference externe et n'ont ete atteints
par aucun des 5 runs - ils sont revus a la baisse pour refleter un objectif
"acceptable" plutot que "production". Voir `docs/evaluation_criteria.md`
pour le detail.

**Decision** : le Run 2 est retenu comme modele SFT de reference pour la
suite (DPO), sur la base de ses resultats et du fait que 3 tentatives
independantes et controlees pour l'ameliorer ont toutes echoue - ce n'est
pas un resultat accidentel mais le meilleur point trouve dans l'espace de
configurations explore.

## Audit des donnees et re-entrainement triage - 2026-11-07 / 2026-11-08

**Contexte** : la revue des premiers resultats de triage (4 experiences :
baseline naive, modele de base, SFT medical, SFT medical + triage) a releve
un rappel P1 de 100 % suspect, un sur-triage de 36 %, des difficultes sur P2
et P3 et un jeu de test de 384 questions issues de seulement 82 groupes. Plan
suivi : (1) audit des fuites, (2) verification de la methode d'evaluation,
re-entrainement, (3) analyse des erreurs, (4) documentation. Resultats de
reference : `docs/comparaison_triage.md`.

### Etape 1 - Audit des partitions (`scripts/audit_splits.py`)

| Controle | Resultat |
|---|---|
| Doublons exacts entre splits (cas, groupe, prompt, contexte) | aucun |
| Questions identiques entre splits | 22, toutes generiques ("Que faites-vous ?") |
| Indices de label dans les prompts | 0 / 6 747 |
| Quasi-doublons test / train (TF-IDF caracteres, cosinus >= 0,5) | 5 cas, 11 questions (max 0,66) |
| Quasi-doublons validation / train | 4 cas (max 0,88 : meme cas de tuberculose reecrit) |
| Baseline "plus proche voisin" (niveau du cas train le plus similaire) | 42,2 % par question, sous les 63,8 % du SFT |

**Lecture** : pas de fuite capable d'expliquer les resultats, mais le
regroupement par prefixe de 100 caracteres laisse passer des reformulations
d'un meme cas (phrases reordonnees, ages ou formulations modifies).

### Etape 2 - Methode d'evaluation (`scripts/analyze_triage.py`)

`scripts/evaluate_sft.py` a ete modifie pour sauvegarder les log-probabilites
brutes et le prior (`scores.csv`), ce qui permet d'analyser sans GPU.

| Controle (SFT triage, ancien test) | Resultat |
|---|---|
| Exclusion des 5 cas suspects | 63,8 -> 64,1 % : impact negligeable |
| Evaluation par cas (vote) | 64,4 % : proche du score par question |
| Coherence intra-cas | 83,6 % des cas recoivent un seul niveau |
| IC 95 % (bootstrap sur les groupes) | 63,8 [52,8 ; 75,0] |
| **Sans calibration contextuelle** | **69,3 % accuracy, F1 69,0, rappel P1 70,9, P3 57,8** |

**Decouverte principale** : la calibration (score - score sur prompt vide)
etait la cause du sur-triage. Pour le SFT triage, un prompt vide evoque
"differee" (log-prob P3 -2,1 contre P1 -7,8) : retrancher ce prior penalise
P3 et pousse vers P1. Cette calibration se justifie pour un modele qui n'a
jamais vu les libelles, pas pour un modele entraine a les produire.

### Corrections appliquees

- `scripts/build_sft_datasets.py` : ajout d'un regroupement par similarite
  (TF-IDF n-grammes de caracteres 3-5, cosinus >= 0,5, union-find avec les
  familles et prefixes). 392 paires fusionnees ; seuil choisi par relecture
  (vers 0,45 apparaissent des cas sans rapport). Apres regeneration : aucun
  cas de validation/test a plus de 0,49 du train.
- `scripts/evaluate_sft.py` : scoring de la validation et du test ; pour
  chaque modele, choix entre scores bruts et calibres sur le F1 macro de
  validation, puis application au test (`mode_decision.csv`).
- Sauvegardes : anciens datasets `data/processed/avant_audit/`, anciens
  adaptateurs `outputs/avant_audit/`, anciens resultats
  `outputs/triage_eval/avant_audit/`, ancien audit `outputs/audit/*_avant.*`.

### Re-entrainement des deux SFT

Configuration inchangee (LoRA r=16, alpha=32, q/k/v/o_proj, lr 2e-4,
3 epochs, batch effectif 8, max_length 1024), seuls les splits changent.
5 345 exemples d'entrainement, 150 de validation pour la loss.

| | sft_medical | sft_medical_triage |
|---|---|---|
| Duree | 6,8 h | 6,8 h |
| Meilleure eval_loss (step 1300, epoch ~1,9) | 1.5317 | 1.3844 |
| eval_loss finale (step 2007) | 1.5459 | 1.3958 |

Meme profil pour les deux : minimum vers l'epoch 1,9 puis legere remontee,
`load_best_model_at_end` retient le step 1300.

### Resultats apres audit (par question, en %)

| experience | mode | val. accuracy | val. F1 | test accuracy | test F1 | test sous-triage | test sur-triage |
|---|---|---|---|---|---|---|---|
| baseline naive | - | 34.4 | 17.0 | 40.8 | 19.3 | 0.0 | 59.2 |
| modele de base | brut | 31.0 | 20.5 | 27.8 | 17.7 | 64.2 | 8.0 |
| SFT medical | calibre | 29.5 | 24.1 | 32.2 | 25.7 | 38.2 | 29.5 |
| **SFT medical + triage** | brut | **62.6** | **62.9** | **78.8** | **78.7** | 7.0 | 14.2 |

IC 95 % test du SFT triage : accuracy [68,2 ; 87,4]. Sous-triage en
validation : 16,2 %.

### Etape 3 - Analyse des erreurs (`scripts/error_analysis.py`)

- Le modele ordonne les niveaux (AUC 0,86 P1/P2, 0,92 P2/P3) : il ne
  privilegie pas systematiquement les niveaux critiques.
- P2 concentre les erreurs (40 % des cas P2 mal classes).
- Sous-triage : surtout des cas dont la gravite est biologique (cas
  "Pharmacy" : leucemie aigue, rhabdomyolyse, insuffisance renale severe) et
  une torsion testiculaire predite P2.
- Sur-triage : cardiopathies neonatales "sans detresse" (P2) dont les
  versions aggravees sont P1 dans le train.
- Annotations coherentes entre cas quasi identiques hors famille (0 %
  de desaccord a similarite >= 0,8).
- Ajustement des seuils appris en validation : ne se transfere pas au test,
  abandonne.
- Predictions incoherentes au sein d'un cas : 47 % d'erreur contre 15 %.

### Conclusion

L'apport des exemples de triage est confirme sur des splits audites et avec
une regle de decision choisie sans regarder le test : +46 points d'accuracy
sur le SFT medical en test, +33 en validation. Le chiffre a retenir est une
fourchette (62,6 % en validation, 78,8 % en test) plutot qu'un point unique.
Le rappel P1 de 100 % precedent etait un artefact de calibration. Limites
inchangees : annotations LLM non validees cliniquement, petits jeux
d'evaluation, corpus domine par deux specialites.

### Pistes pour les prochaines experimentations

1. **Un exemple de triage par cas** plutot qu'un prefixe repete sur chaque
   question : le niveau est porte par le cas, et un cas a 12 questions pese
   aujourd'hui 12 fois plus qu'un cas a une question.
2. **Signal d'incertitude** : evaluer une regle "a revoir par un humain"
   (niveaux incoherents entre questions, ou probabilite maximale faible) en
   mesurant le compromis couverture / erreurs, en particulier sur le
   sous-triage.
3. **Cas a gravite biologique** : verifier si le sous-triage vient de valeurs
   de laboratoire mal lues, ou de l'exclusion a l'entrainement des exemples
   de plus de 1 024 tokens (cas longs, riches en tableaux biologiques).
4. **Validation clinique d'un echantillon** d'annotations, en priorite sur
   les cas P2 et les cas mal classes.
5. Reprise du DPO une fois ces points traites, avec des paires de preference
   coherentes avec les objectifs de triage (penaliser le sous-triage).
