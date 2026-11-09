# Analyse des erreurs - 4_sft_medical_triage

Mode de decision : brut (retenu sur la validation). Jeu de test : 400 questions, 109 cas. Valeurs en %.

## 1. Erreurs par cas clinique (vote majoritaire)

109 cas, 21 mal classes (19.3 %).

| index | predit P1 | predit P2 | predit P3 |
|---|---|---|---|
| annote P1 | 49 | 4 | 0 |
| annote P2 | 9 | 18 | 3 |
| annote P3 | 0 | 5 | 21 |

| index | cas | sens |
|---|---|---|
| P2 -> P1 | 9 | sur-triage |
| P3 -> P2 | 5 | sur-triage |
| P1 -> P2 | 4 | sous-triage |
| P2 -> P3 | 3 | sous-triage |

## 2. Taux d'erreur selon les caracteristiques des cas

Effectifs faibles : ces taux orientent la relecture, ils ne prouvent rien a eux seuls.

### priorite

| priorite | cas | erreur | sous_triage | sur_triage |
|---|---|---|---|---|
| P3 | 26 | 19.2 | 0.0 | 19.2 |
| P1 | 53 | 7.5 | 7.5 | 0.0 |
| P2 | 30 | 40.0 | 10.0 | 30.0 |

### signal_gravite

| signal_gravite | cas | erreur | sous_triage | sur_triage |
|---|---|---|---|---|
| False | 44 | 15.9 | 6.8 | 9.1 |
| True | 65 | 21.5 | 6.2 | 15.4 |

### version_derivee

| version_derivee | cas | erreur | sous_triage | sur_triage |
|---|---|---|---|---|
| False | 79 | 21.5 | 6.3 | 15.2 |
| True | 30 | 13.3 | 6.7 | 6.7 |

### longueur_quartile

| longueur_quartile | cas | erreur | sous_triage | sur_triage |
|---|---|---|---|---|
| Q1 court | 28 | 10.7 | 3.6 | 7.1 |
| Q2 | 27 | 25.9 | 3.7 | 22.2 |
| Q3 | 27 | 14.8 | 7.4 | 7.4 |
| Q4 long | 27 | 25.9 | 11.1 | 14.8 |

### questions

| questions | cas | erreur | sous_triage | sur_triage |
|---|---|---|---|---|
| 1 | 31 | 16.1 | 6.5 | 9.7 |
| 2-3 | 29 | 13.8 | 0.0 | 13.8 |
| 4+ | 49 | 24.5 | 10.2 | 14.3 |

### predictions_incoherentes

| predictions_incoherentes | cas | erreur | sous_triage | sur_triage |
|---|---|---|---|---|
| False | 94 | 14.9 | 6.4 | 8.5 |
| True | 15 | 46.7 | 6.7 | 40.0 |

### specialite

| specialite | cas | erreur | sous_triage | sur_triage |
|---|---|---|---|---|
| Pediatric Cardiology | 53 | 17.0 | 3.8 | 13.2 |
| Pharmacy | 24 | 29.2 | 16.7 | 12.5 |
| Gynecology and Obstetrics | 6 | 0.0 | 0.0 | 0.0 |
| Emergency Medicine | 3 | 0.0 | 0.0 | 0.0 |
| Nephro-Urology | 3 | 33.3 | 33.3 | 0.0 |
| Pulmonology | 3 | 66.7 | 0.0 | 66.7 |

## 3. Separation des scores entre niveaux

Probabilite moyenne attribuee a chaque niveau (softmax sur les 3 libelles, moyenne par cas), selon le niveau annote :

| priorite | p_P1 | p_P2 | p_P3 |
|---|---|---|---|
| P1 | 88.8 | 10.8 | 0.4 |
| P2 | 35.4 | 51.8 | 12.8 |
| P3 | 3.4 | 21.2 | 75.4 |

AUC entre niveaux, sur l'urgence attendue sum(p x rang) par question. Une AUC elevee avec une accuracy moyenne signale un modele qui ordonne bien mais dont les frontieres sont mal placees (corrigeable par un seuil) ; une AUC proche de 0.5 signale une vraie confusion.

| index | AUC | questions |
|---|---|---|
| P1 vs P2 | 0.861 | 287.000 |
| P2 vs P3 | 0.918 | 237.000 |
| P1 vs P3 | 0.988 | 276.000 |

## 4. Ajustement des seuils de decision (appris sur la validation)

Un biais est ajoute aux scores de P2 et P3 (grille de -6 a +6), choisi sur la validation, puis applique tel quel au test. Deux objectifs : F1 macro maximal, ou F1 macro maximal sous contrainte d'un rappel P1 >= 95% (limiter le sous-triage des urgences vitales).

Validation (donnees d'apprentissage des biais) :

| index | accuracy | f1_macro | rappel_P1 | rappel_P2 | rappel_P3 | sous_triage | sur_triage |
|---|---|---|---|---|---|---|---|
| sans ajustement | 62.6 | 62.9 | 72.4 | 45.1 | 70.7 | 16.2 | 21.3 |
| F1 max | 67.4 | 68.4 | 61.9 | 76.7 | 63.4 | 16.9 | 15.6 |
| rappel P1 >= 95% | 67.2 | 66.2 | 96.3 | 49.6 | 54.5 | 2.6 | 30.3 |

Test (application sans retouche) :

| index | accuracy | f1_macro | rappel_P1 | rappel_P2 | rappel_P3 | sous_triage | sur_triage |
|---|---|---|---|---|---|---|---|
| sans ajustement | 78.8 | 78.7 | 86.5 | 64.5 | 83.2 | 7.0 | 14.2 |
| F1 max (biais P2 +2.50, P3 +2.00) | 76.5 | 77.6 | 66.3 | 86.3 | 80.5 | 15.0 | 8.5 |
| rappel P1 >= 95% (biais P2 -3.00, P3 -4.75) | 67.8 | 65.4 | 95.1 | 37.1 | 61.9 | 2.2 | 30.0 |

## 5. Cas mal classes (sous-triage d'abord)

`justification` : motif donne par le LLM annotateur. A relire : erreur du modele, ou annotation discutable ?

### case_e964c9ea6a - annote P2, predit P3 (SOUS-TRIAGE)

- Probabilites moyennes : P1 0.00 / P2 0.01 / P3 0.99 ; 1 question(s), 1 niveau(x) predit(s)
- Specialite : Pediatric Cardiology ; signal de gravite (regex) : False
- Justification annotateur : CIA symptomatique à l'effort
- Cas : Vous faites le diagnostic d'une CIA ostium secundum chez un enfant de 4 ans. Cet enfant est symptomatique à l'effort et en échographie, la dilatation du ventricule droit est très importante bien que la CIA ne paraissent pas très large en échographie.

### case_9f7aba23a7 - annote P1, predit P2 (SOUS-TRIAGE)

- Probabilites moyennes : P1 0.00 / P2 0.94 / P3 0.06 ; 6 question(s), 1 niveau(x) predit(s)
- Specialite : Pharmacy ; signal de gravite (regex) : False
- Justification annotateur : leucémie aiguë, plaquettes 30 G/L, syndrome hémorragique
- Cas : Un jeune garçon de 4 ans est conduit par sa mère chez le pédiatre car depuis une dizaine de jours elle trouve l'enfant pâle et asthénique et signale une fièvre à 38°C avec gorge rouge. L'examen clinique révèle une angine. L'hémogramme prescrit par le pédiatre montre les résultats suivants : Sg Erythrocytes : 3,00 T/L Sg Hématocrite : 0,27 Sg Hémogl...

### case_bbcd6d751a - annote P1, predit P2 (SOUS-TRIAGE)

- Probabilites moyennes : P1 0.14 / P2 0.85 / P3 0.01 ; 4 question(s), 1 niveau(x) predit(s)
- Specialite : Pharmacy ; signal de gravite (regex) : True
- Justification annotateur : rhabdomyolyse avec insuffisance rénale aiguë
- Cas : Monsieur X., 63 ans, présentant dans ses antécédents une hypertension artérielle, un diabète de type 1 et une hypercholestérolémie, a subi une greffe rénale. Un mois après celle-ci, il est ré-hospitalisé pour des myalgies et un déficit moteur proximal des membres inférieurs avec amyotrophie du quadriceps et une asthénie intense.  Son traitement, en...

### case_05e915687a - annote P1, predit P2 (SOUS-TRIAGE)

- Probabilites moyennes : P1 0.17 / P2 0.83 / P3 0.01 ; 4 question(s), 1 niveau(x) predit(s)
- Specialite : Pharmacy ; signal de gravite (regex) : True
- Justification annotateur : insuffisance rénale sévère, PA 200/110
- Cas : Un homme de 43 ans consulte pour asthénie intense d'apparition progressive associée à pâleur remarquée pars son entourage, et une dyspnée à l'effort. Dans ses antécédents, on a la notion d'une protéinurie (0.60 g/l) découverte lors du service national, mais non explorée. Il se plaint de céphalées, et de crampes des membres inférieurs. Son rythme ca...

### case_e4a192ea71 - annote P1, predit P2 (SOUS-TRIAGE)

- Probabilites moyennes : P1 0.17 / P2 0.82 / P3 0.00 ; 5 question(s), 1 niveau(x) predit(s)
- Specialite : Nephro-Urology ; signal de gravite (regex) : True
- Justification annotateur : douleur scrotale brutale, torsion testiculaire suspecte
- Cas : Le jeune Erwan HET., âgé de 16 ans, sans antécédent notable, est amené aux urgences par sa mère pour des douleurs scrotales droites, d'apparition brutale, évoluant depuis le matin, dans un contexte d'apyrexie. Il a présenté un épisode de vomissements.L'examen rendu difficile par l'importance des douleurs, retrouve une volumineuse bourse discrètemen...

### case_13945e7bed - annote P2, predit P3 (SOUS-TRIAGE)

- Probabilites moyennes : P1 0.00 / P2 0.37 / P3 0.63 ; 5 question(s), 2 niveau(x) predit(s)
- Specialite : Pharmacy ; signal de gravite (regex) : False
- Justification annotateur : fièvre et hépatomégalie douloureuse
- Cas : Madame X. habite la région lyonnaise et vient en consultation à la suite d'un syndrome fébrile à 38°5C et de vomissements bilieux apparus 1 semaine avant. L'examen retrouve une asthénie avec un amigrissement de 4kg. L'auscultation montre une légère hépatomégalie et un foie douloureux à la palpation. L'interrogatoire de la patiente révèle que la vei...

### case_4ea8c7112c - annote P2, predit P3 (SOUS-TRIAGE)

- Probabilites moyennes : P1 0.10 / P2 0.30 / P3 0.60 ; 1 question(s), 1 niveau(x) predit(s)
- Specialite : Pediatric Cardiology ; signal de gravite (regex) : True
- Justification annotateur : coarctation néonatale sans détresse
- Cas : Un nouveau-né de 3 jours de vie est examiné à la sortie de maternité. Le pédiatre vous demande conseil car il ne sent pas les pouls fémoraux. Il n'entend pas de souffle à l'auscultation du précordium. La pression artérielle au membre supérieur droit est à 98/50 mm Hg pour 63/45 mm Hg au membre inférieur droit. Il n'y a pas d'hépatomégalie ni de dét...

### case_f5d38db0d0 - annote P2, predit P1 (sur-triage)

- Probabilites moyennes : P1 1.00 / P2 0.00 / P3 0.00 ; 3 question(s), 1 niveau(x) predit(s)
- Specialite : Pediatric Cardiology ; signal de gravite (regex) : True
- Justification annotateur : nouveau-né cyanosé, cardiopathie à opérer
- Cas : Un nouveau-né de 3 jours est hospitalisé pour une cyanose réfractaire modérée associée à un souffle systolique. La radiographie de thorax montre une aorte à droite, un cœur de volume normal et des poumons perfusés de façon symétrique. En échographie, vous faites le diagnostic de tronc artériel commun avec une valve troncale non sténosante, non fuya...

### case_36716c7b84 - annote P2, predit P1 (sur-triage)

- Probabilites moyennes : P1 1.00 / P2 0.00 / P3 0.00 ; 4 question(s), 1 niveau(x) predit(s)
- Specialite : Pediatric Cardiology ; signal de gravite (regex) : True
- Justification annotateur : nouveau-né cyanosé, cardiopathie à opérer
- Cas : Un nouveau-né de 3 jours est hospitalisé pour une cyanose réfractaire modérée associée à un souffle systolique. La radiographie de thorax montre une aorte à droite, un cœur de volume normal et des poumons perfusés de façon symétrique. En échographie, vous faites le diagnostic de tronc artériel commun avec une valve troncale non sténosante, non fuya...

### case_bcc9952643 - annote P2, predit P1 (sur-triage)

- Probabilites moyennes : P1 0.99 / P2 0.01 / P3 0.00 ; 1 question(s), 1 niveau(x) predit(s)
- Specialite : Pediatric Cardiology ; signal de gravite (regex) : True
- Justification annotateur : nouveau-né cyanosé sans détresse
- Cas : Un nouveau-né de 3 jours de vie vous est confié pour une cyanose réfractaire à l'oxygène. Il n'a pas de détresse circulatoire ou respiratoire. Vous auscultez un souffle systolique 3/6 et un B2 fort unique. L'ECG est normal. La radiographie montre une silhouette en œuf avec un médiastin étroit et des poumons normalement perfusés. L'échographie montr...

### case_9cee0b48c0 - annote P3, predit P2 (sur-triage)

- Probabilites moyennes : P1 0.00 / P2 0.95 / P3 0.04 ; 4 question(s), 1 niveau(x) predit(s)
- Specialite : Otorhinolaryngology (ENT) ; signal de gravite (regex) : False
- Justification annotateur : sinusite chronique
- Cas : Monsieur X. 45 ans consulte pour une obstruction nasale gauche avec une rhinorrhée purulente homolatérale et des céphalées.ATCD : appendicectomie, fracture de jambe.Cette obstruction nasale existe depuis plusieurs mois, avec des épisodes de céphalées identiques.

### case_4dc45af44d - annote P3, predit P2 (sur-triage)

- Probabilites moyennes : P1 0.05 / P2 0.90 / P3 0.05 ; 3 question(s), 1 niveau(x) predit(s)
- Specialite : Pediatric Cardiology ; signal de gravite (regex) : True
- Justification annotateur : cyanose d'effort, suivi
- Cas : Vous suivez depuis la période néonatale une jeune fille qui a une malformation d'Ebstein. Elle est modérément cyanosée au repos (SaO2 88%). Elle se dit gênée à l'effort et le test d'effort montre une cyanose importante pour un effort de 60 watts (SaO2 65%). Il n'y a pas d'obstacle pulmonaire en échographie et la valve tricuspide ne fuit qu'à peine.

### case_011ade3992 - annote P2, predit P1 (sur-triage)

- Probabilites moyennes : P1 0.83 / P2 0.16 / P3 0.01 ; 1 question(s), 1 niveau(x) predit(s)
- Specialite : Pediatric Cardiology ; signal de gravite (regex) : True
- Justification annotateur : coarctation néonatale sans détresse
- Cas : Un nouveau-né de 3 jours de vie est examiné à la sortie de maternité. Le pédiatre vous demande conseil car il ne sent pas les pouls fémoraux. Il n'entend pas de souffle à l'auscultation du précordium. La pression artérielle au membre supérieur droit est à 98/50 mm Hg pour 63/45 mm Hg au membre inférieur droit. Il n'y a pas d'hépatomégalie ni de dét...

### case_216a877133 - annote P2, predit P1 (sur-triage)

- Probabilites moyennes : P1 0.81 / P2 0.19 / P3 0.00 ; 7 question(s), 1 niveau(x) predit(s)
- Specialite : Pulmonology ; signal de gravite (regex) : True
- Justification annotateur : suspicion d'embolie pulmonaire, FC 118, hémodynamique stable
- Cas : Un homme de 55 ans est adressé aux Urgences par son médecin traitant pour des crachats hémoptoïques.Depuis son retour de congés en Guadeloupe il y a deux jours, le patient souffre d'une douleur localisée de la base droite et d'une dyspnée d'effort de stade III. La douleur est majorée par l'inspiration profonde. Le patient fume un paquet de cigarett...

### case_30daf5027d - annote P2, predit P1 (sur-triage)

- Probabilites moyennes : P1 0.67 / P2 0.27 / P3 0.05 ; 9 question(s), 2 niveau(x) predit(s)
- Specialite : Forensic Medicine and Toxicology ; signal de gravite (regex) : False
- Justification annotateur : accident de la route, blessés à évaluer
- Cas : Jean et Jeanne, sortent de boite de nuit en état d'ébriété avancé, après une soirée bien arrosée. Jean a consommé des stupéfiants (il n'arrive pas à décrocher de l'héroïne). Alors qu'ils sont en route pour leur domicile, un accident survient. Le couple vous relate qu'un véhicule leur a coupé la route à vive allure projetant leur auto dans un muret ...

### case_9f22584523 - annote P2, predit P1 (sur-triage)

- Probabilites moyennes : P1 0.67 / P2 0.32 / P3 0.01 ; 5 question(s), 2 niveau(x) predit(s)
- Specialite : Pharmacy ; signal de gravite (regex) : True
- Justification annotateur : infection urinaire nosocomiale avec bactériémie
- Cas : Monsieur X., 24 ans, 80 kg, polytraumatisé est hospitalisé en urgence dans un service de réanimation. Dix jours après le début de son hospitalisation, il présente les signes d'une infection urinaire avec une fièvre à 39°C accompagnée de frissons. Un examen cytobactériologique des urines et une hémoculture sont immédiatement réalisés. Une antibiothé...

### case_146b6a6a0c - annote P3, predit P2 (sur-triage)

- Probabilites moyennes : P1 0.32 / P2 0.67 / P3 0.01 ; 1 question(s), 1 niveau(x) predit(s)
- Specialite : Pediatric Cardiology ; signal de gravite (regex) : False
- Justification annotateur : souffle, consultation
- Cas : On vous adresse un nourrisson de 8 mois pour un souffle continu précordial. En échographie, vous ne trouvez pas de canal artériel persistant.

### case_3b0b25f671 - annote P2, predit P1 (sur-triage)

- Probabilites moyennes : P1 0.63 / P2 0.37 / P3 0.00 ; 7 question(s), 2 niveau(x) predit(s)
- Specialite : Pulmonology ; signal de gravite (regex) : True
- Justification annotateur : suspicion d'embolie pulmonaire, hémodynamique stable
- Cas : Un homme de 55 ans est adressé aux Urgences par son médecin traitant pour des crachats hémoptoïques. Depuis son retour de congés en Guadeloupe il y a deux jours, le patient souffre d'une douleur localisée de la base droite et d'une dyspnée d'effort de stade III. La douleur est majorée par l'inspiration profonde. Le patient fume un paquet de cigaret...

### case_be3099e3d4 - annote P3, predit P2 (sur-triage)

- Probabilites moyennes : P1 0.44 / P2 0.56 / P3 0.00 ; 7 question(s), 2 niveau(x) predit(s)
- Specialite : Pharmacy ; signal de gravite (regex) : True
- Justification annotateur : prise en charge post-opératoire de la douleur
- Cas : Monsieur X. est admis aux urgences pour douleurs abdominales sévères avec rectorragies. La coloscopie permet de découvrir une tumeur rectale avec sténose colique à 50 cm. Le bilan d'extension montre une volumineuse lésion siégeant sur le sigmoïde, sans métastases hépatiques. Le patient subit une colostomie, puis une résection de la tumeu...

### case_20a2816769 - annote P3, predit P2 (sur-triage)

- Probabilites moyennes : P1 0.01 / P2 0.55 / P3 0.43 ; 2 question(s), 2 niveau(x) predit(s)
- Specialite : Pediatric Cardiology ; signal de gravite (regex) : False
- Justification annotateur : asymptomatique, épreuve d'effort de suivi
- Cas : Un jeune homme de 24 ans a eu une plastie aortique pour une coarctation isthmique à l'âge de 3 ans. Il n'a aucun symptôme. Ses pouls fémoraux sont bien perçus. La pression artérielle au membre supérieur droit est à 140/75 mm Hg. Il a fait une épreuve d'effort quasi maximale au cours de laquelle la pression artérielle est montée à 290/110 mm Hg.

### case_e40981689b - annote P2, predit P1 (sur-triage)

- Probabilites moyennes : P1 0.54 / P2 0.46 / P3 0.00 ; 3 question(s), 2 niveau(x) predit(s)
- Specialite : Pharmacy ; signal de gravite (regex) : True
- Justification annotateur : hématome après chute, TCA allongé chez enfant
- Cas : Guillaume R., âgé de 2 ans, est admis aux urgences pour une volumineuse bosse frontale déclenchée par une chute contre un meuble, quelques heures auparavant. La tuméfaction est douloureuse, tend la peau et évoque un hématome sous-cutané. Les premiers examens biologiques donnent les résultats suivants: Sg-- plaquettes: 290 G/L Pl TP (taux de prothro...

