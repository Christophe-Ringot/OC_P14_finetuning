# Comparaison du triage - 4 experiences

Document de reference des resultats de triage (P1 = urgence maximale,
P2 = urgence moderee, P3 = differee). Mis a jour le 2026-11-08 apres l'audit
des donnees et le re-entrainement des deux SFT (voir `docs/experiments.md`,
section "Audit des donnees et re-entrainement triage").

**Les resultats anterieurs (63,8 % d'accuracy, 100 % de rappel P1) sont
remplaces** : ils reposaient sur une regle de decision biaisee (voir
"Historique" en fin de document).

## Protocole

| Element | Valeur |
|---|---|
| Modele de base | Qwen/Qwen3-1.7B-Base |
| Experiences | 1 baseline naive (toujours P1), 2 modele de base, 3 SFT medical (sans label de triage), 4 SFT medical + triage (prefixe `[Niveau de priorite estime : ...]` dans les reponses) |
| Entrainement SFT | LoRA r=16, alpha=32, q/k/v/o_proj, lr 2e-4, 3 epochs, 5 345 exemples, meilleur checkpoint sur eval_loss (step 1300) |
| Prediction | log-probabilite de chacun des 3 libelles apres le debut du prefixe, argmax |
| Regle de decision | par modele, scores bruts ou calibres (moins le score sur prompt vide), **choisie sur la validation** (F1 macro) puis appliquee au test |
| Splits | regroupes par cas clinique, famille de cas, prefixe commun et similarite TF-IDF >= 0,5 ; aucun cas de validation/test a plus de 0,49 de similarite du train |
| Validation | 390 questions, 76 groupes (P1 134, P2 133, P3 123) |
| Test | 400 questions, 109 cas, 72 groupes (P1 163, P2 124, P3 113) |

Regle de decision retenue sur la validation :

| experience | mode retenu | F1 validation calibre | F1 validation brut |
|---|---|---|---|
| 2_modele_base | brut | 19.6 | 20.5 |
| 3_sft_medical | calibre | 24.1 | 22.4 |
| 4_sft_medical_triage | brut | 37.1 | 62.9 |

## Resultats principaux (par question, en %)

| experience | split | accuracy | F1 macro | rappel P1 | rappel P2 | rappel P3 | sous-triage | sur-triage |
|---|---|---|---|---|---|---|---|---|
| 1_baseline_naive | validation | 34.4 | 17.0 | 100.0 | 0.0 | 0.0 | 0.0 | 65.6 |
| | test | 40.8 | 19.3 | 100.0 | 0.0 | 0.0 | 0.0 | 59.2 |
| 2_modele_base | validation | 31.0 | 20.5 | 10.4 | 0.0 | 87.0 | 61.5 | 7.4 |
| | test | 27.8 | 17.7 | 6.7 | 0.0 | 88.5 | 64.2 | 8.0 |
| 3_sft_medical | validation | 29.5 | 24.1 | 45.5 | 0.8 | 43.1 | 38.2 | 32.3 |
| | test | 32.2 | 25.7 | 45.4 | 0.8 | 47.8 | 38.2 | 29.5 |
| **4_sft_medical_triage** | validation | **62.6** | **62.9** | 72.4 | 45.1 | 70.7 | 16.2 | 21.3 |
| | test | **78.8** | **78.7** | 86.5 | 64.5 | 83.2 | 7.0 | 14.2 |

Sous-triage / sur-triage : part de toutes les questions dont le niveau predit
est moins / plus urgent que l'annotation.

**Lecture** :
- Seul le SFT triage depasse nettement la baseline naive. Le modele de base et
  le SFT medical generaliste restent sous la baseline en accuracy : sans
  exemples de triage, le modele ne sait pas produire ces niveaux.
- **Le score du SFT triage depend fortement du jeu evalue** : 62,6 % en
  validation contre 78,8 % en test. Les deux jeux sont petits (76 et 72
  groupes) ; la performance attendue sur de nouveaux cas se situe
  vraisemblablement entre ces deux valeurs. Les deux chiffres doivent etre
  cites ensemble.
- Le sous-triage (erreur la plus dangereuse) reste present : 7,0 % en test,
  **16,2 % en validation**.

## Robustesse (test)

| Controle | SFT triage | Lecture |
|---|---|---|
| IC 95 % accuracy (bootstrap sur 72 groupes) | 78.6 [68.2 ; 87.4] | incertitude large, liee au faible nombre de groupes |
| IC 95 % F1 macro | 78.1 [68.8 ; 86.5] | |
| Ecart d'accuracy vs SFT medical | +46.5 [36.0 ; 55.8] | ecart positif dans 100 % des tirages |
| Accuracy par cas (vote majoritaire, 109 cas) | 80.7 | proche du score par question : la dependance entre questions ne fausse pas le resultat |
| Coherence intra-cas | 80.8 % des cas recoivent un seul niveau | le modele raisonne au niveau du cas |
| Cas de test proches du train (similarite >= 0,5) | 0 | fuite par reformulation eliminee |

## Detail du SFT triage (test)

| | precision | rappel | F1 |
|---|---|---|---|
| P1 | 77.9 | 86.5 | 82.0 |
| P2 | 67.2 | 64.5 | 65.8 |
| P3 | 94.0 | 83.2 | 88.3 |

| annote \ predit | P1 | P2 | P3 |
|---|---|---|---|
| P1 | 141 | 22 | 0 |
| P2 | 38 | 80 | 6 |
| P3 | 2 | 17 | 94 |

Les erreurs sont quasiment toutes d'un seul niveau (0,5 % d'erreurs de deux
niveaux ; aucun P1 predit P3). Les matrices des autres experiences sont dans
`outputs/triage_eval/comparaison_triage.md`.

## Analyse des erreurs (SFT triage, test)

Detail et relecture cas par cas : `outputs/triage_eval/analyse_erreurs.md`.

- **Le modele ordonne bien les niveaux** : AUC 0,86 (P1/P2), 0,92 (P2/P3),
  0,99 (P1/P3). Il ne se contente pas de favoriser les niveaux critiques.
- **P2 est la classe charniere** : 40 % des cas P2 sont mal classes (contre
  7,5 % des P1 et 19 % des P3) ; un cas P2 recoit en moyenne 35 % de
  probabilite P1.
- **Sous-triage de gravite biologique** : 3 des 4 cas P1 predits P2 sont des
  cas ou la gravite se lit dans les resultats de laboratoire (leucemie aigue
  avec thrombopenie, rhabdomyolyse avec insuffisance renale aigue,
  insuffisance renale severe). Le quatrieme est une **torsion testiculaire
  predite P2 avec 82 % de confiance**, l'erreur la plus grave observee.
- **Sur-triage des cardiopathies neonatales** : des nouveau-nes cyanoses
  "sans detresse" annotes P2 sont predits P1. Leurs versions aggravees
  (meme scenario, texte prolonge) sont P1 dans le train : le modele associe le
  scenario a P1 et distingue mal la version initiale.
- **Annotations coherentes entre elles** : entre cas quasi identiques de
  familles differentes (similarite >= 0,8), aucune paire n'a de niveaux
  differents. Le bruit d'annotation n'explique pas les erreurs (ce qui ne
  vaut pas validation clinique).
- **Ajuster les seuils de decision n'aide pas** : des biais appris sur la
  validation degradent le test (F1 77,6 au lieu de 78,7) ; imposer un rappel
  P1 >= 95 % ramene le sous-triage a 2,2 % mais fait chuter l'accuracy a
  67,8 % et monter le sur-triage a 30 %.
- **Signal d'incertitude** : quand les questions d'un meme cas recoivent des
  niveaux differents, le cas est mal classe dans 47 % des cas (contre 15 %
  sinon) - piste pour orienter vers une relecture humaine.

## Limites

- Etiquettes de reference pre-annotees par LLM, sans validation clinique
  (`docs/triage_grid.md`) : les metriques mesurent la capacite a reproduire
  ces annotations, pas une fiabilite medicale.
- Jeux d'evaluation petits (72 et 76 groupes) : intervalles de confiance
  larges, ecart validation / test de 16 points.
- Les questions d'un meme cas partagent le meme niveau : les observations ne
  sont pas independantes (prise en compte par le bootstrap sur les groupes et
  l'evaluation par cas).
- Corpus domine par deux specialites (cardiologie pediatrique et pharmacie,
  environ 65 % des cas) : les resultats ne se generalisent pas a toutes les
  situations d'urgence.
- Pas de jeu de test independant constitue de nouveaux cas cliniques.

## Historique : resultats avant audit (2026-11-03)

| experience | accuracy | F1 macro | rappel P1 | sous-triage | sur-triage |
|---|---|---|---|---|---|
| 4_sft_medical_triage (calibration imposee) | 63.8 | 55.0 | 100.0 | 0.0 | 36.2 |
| 4_sft_medical_triage (memes scores, sans calibration) | 69.3 | 69.0 | 70.9 | 15.1 | 15.6 |

Ces chiffres ont ete remplaces pour deux raisons :
1. **Calibration inadaptee** : la calibration contextuelle retranche le score
   obtenu sur un prompt vide. Le SFT triage a appris qu'un cas sans signe
   clinique est "differee" et attribue donc une forte probabilite a P3 sur ce
   prompt (log-prob -2,1 contre -7,8 pour P1). La soustraction penalisait P3
   et poussait les predictions vers P1 : le rappel P1 de 100 % et le
   sur-triage de 36 % venaient de l'evaluation, pas du modele.
2. **Reformulations non regroupees** : le regroupement par prefixe de 100
   caracteres laissait passer des reformulations d'un meme cas entre splits
   (5 cas de test concernes, impact mesure inferieur a 1 point). Les splits
   ont ete regeneres avec un regroupement par similarite et les deux SFT
   re-entraines.

Les jeux de test avant et apres audit sont differents : les chiffres ne se
comparent qu'a la baseline naive de leur propre jeu.

## Reproduire

```
python -m scripts.build_sft_datasets   # splits
python -m scripts.audit_splits         # audit des fuites -> outputs/audit/
python -m scripts.train_sft            # 2 SFT puis evaluation -> outputs/triage_eval/
python -m scripts.analyze_triage       # calibration, par cas, bootstrap
python -m scripts.error_analysis       # analyse des erreurs
```
