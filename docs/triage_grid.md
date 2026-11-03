# Grille d'annotation du signal de priorite

Grille utilisee a l'etape 4 (`scripts/build_annotation_pool.py` puis
`scripts/annotate_priority.py`) pour attribuer un niveau de priorite a chaque
cas clinique unique de MediQAl OEQ.

Inspiree de l'echelle FRENCH (SFMU, tri 1 a 5) et du Manchester Triage
System, ramenee aux 3 niveaux du POC CHSA
(urgence maximale, moderee, differee). **Ce n'est pas un outil de triage clinique valide** : c'est
une convention d'annotation pour un POC educatif.

## Principe

Le niveau decrit l'urgence de prise en charge de **l'etat le plus grave
decrit dans l'episode aigu du cas** : la presentation initiale, ou une
aggravation survenue pendant la prise en charge (ex. somnolence et PA 90/52
apparaissant apres l'arrivee d'un traumatise cranien). Un suivi ulterieur
(evolution favorable, consultation des mois plus tard) ne fait pas baisser
le niveau.

MediQAl decline souvent un meme cas en versions successives, chacune
prolongeant le texte de la precedente pour une nouvelle question. Ces
versions forment une **famille** (`famille_id` dans le vivier) : leur niveau
ne peut que rester egal ou monter d'une version a la suivante.

L'annotation porte sur le cas clinique, pas sur la question : toutes les
questions d'un meme cas heritent du meme niveau.

## Niveaux

| Niveau | Code | Equivalent FRENCH | Delai indicatif |
|---|---|---|---|
| Urgence maximale | `urgence_maximale` | tri 1-2 | immediat / < 20 min |
| Urgence moderee | `urgence_moderee` | tri 3 | < 1-2 h |
| Prise en charge differee | `differee` | tri 4-5 | consultation, programme |

### `urgence_maximale` - detresse vitale actuelle ou imminente

Au moins un signe de detresse d'une fonction vitale, ou une pathologie dont
le pronostic depend des minutes :

- circulatoire : arret cardiaque, choc (PAS < 90 mmHg, marbrures, TRC
  allonge), hemorragie active abondante, troubles du rythme mal toleres ;
- respiratoire : SpO2 < 90 %, FR > 30/min (adulte), tirage, cyanose,
  epuisement, obstruction des voies aeriennes ;
- neurologique : coma ou Glasgow <= 8, etat de mal convulsif, deficit
  neurologique brutal (AVC dans les delais), syndrome meninge febrile ;
- situations a risque vital : SCA avec sus-decalage ST, dissection, embolie
  pulmonaire grave, purpura fulminans, sepsis grave, anaphylaxie,
  polytraumatisme, intoxication grave, acidocetose severe, hyperkaliemie
  menacante, eclampsie / pre-eclampsie severe, hematome retroplacentaire,
  GEU rompue, nouveau-ne ou nourrisson en detresse.

### `urgence_moderee` - pathologie aigue sans detresse vitale

Situation aigue qui doit etre evaluee rapidement (heures), mais constantes
conservees :

- infection aigue necessitant un traitement rapide sans signe de gravite
  (pneumopathie, pyelonephrite, fievre de l'enfant bien toleree) ;
- douleur aigue intense (colique nephretique, traumatisme, fracture) ;
- decompensation d'une maladie chronique sans detresse (insuffisance
  cardiaque, asthme modere, diabete desequilibre) ;
- effet indesirable medicamenteux ou anomalie biologique significative
  necessitant une adaptation rapide, sans menace vitale immediate ;
- grossesse avec complication a evaluer sans detresse maternelle ou foetale.

### `differee` - pas d'urgence

Pas de pathologie aigue menacante ; la prise en charge peut etre programmee :

- consultation de suivi ou de depistage, bilan d'une maladie chronique
  stable ;
- symptome ancien, peu evolutif, bien tolere ;
- conseil, renouvellement ou analyse d'ordonnance en officine sans signe
  d'alerte ;
- prevention, vaccination, contraception, education therapeutique.

## Regles de decision

1. En cas d'hesitation entre deux niveaux, on retient le **plus urgent**
   (principe de precaution du triage : un sous-triage est plus grave qu'un
   sur-triage).
2. Un terrain fragile (nourrisson, grossesse, immunodepression, grand age)
   remonte d'un niveau un tableau limite.
3. Les examens faits a l'arrivee (ECG, gaz du sang, kaliemie...) font partie
   de la presentation. En revanche, un diagnostic pose a distance (biopsie,
   genetique, suivi) ne remonte pas le niveau d'un tableau qui n'etait pas
   urgent.

## Pre-annotation

- Methode : pre-annotation par LLM (Claude), sans relecture humaine
  (decision de cadrage). Chaque cas recoit un niveau et une justification
  courte citant le ou les elements decisifs.
- Controle de coherence : une expression reguliere de signes de gravite
  (signes usuels : constantes, choc, coma,
  detresse...) est calculee pour chaque cas ; la repartition des niveaux est
  comparee entre cas avec et sans signal.
- Limite : etiquettes non validees par des professionnels de sante. Le
  `niveau_confiance` des lignes annotees est `pre_annotation_llm`.
