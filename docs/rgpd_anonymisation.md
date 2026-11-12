# Anonymisation RGPD du corpus

Perimetre : le corpus SFT publie sur Hugging Face (`christophe-ringot/p14-medical-triage-sft`, 6 747 paires) et tout ce qui en derive (paires DPO de triage, jeu d'evaluation clinique). Script : `src/triage/data/anonymize.py`, etape du pipeline entre le nettoyage (`clean_harmonize.py`) et l'annotation (`annotate_priority.py`).

## Risque de depart

Les trois sources ne sont pas des dossiers patients :

- **MediQAl** (4 969 lignes) et **FrenchMedMCQA** (1 077) : questions d'examen de medecine et de pharmacie. Les cas cliniques sont des vignettes pedagogiques, mais elles citent des patients ("Monsieur G.", "Marie, 6 ans") qu'il faut traiter comme s'ils etaient reels.
- **MedQuAD** (701) : encyclopedie sante du NIH, sans donnee individuelle.

Le risque est faible par construction ; il est controle plutot que suppose.

## Methode

Appliquee aux colonnes `instruction`, `contexte_patient` et `reponse`. Chaque texte distinct est anonymise une seule fois (cache), pour qu'un cas repete sur plusieurs questions soit masque a l'identique.

| Entite | Detection | Remplacement |
|---|---|---|
| Civilite + nom ("Monsieur G.", "Mme Le Bihan") | expression reguliere | `Monsieur X.` |
| Prenom suivi d'un age ("Leonie, 69 ans") | expression reguliere, confirmee par spaCy `fr_core_news_md` (nom propre) | `X` |
| Adresse e-mail | Presidio (`EMAIL_ADDRESS`) | `[COORDONNEES]` |

Ecartes apres essai sur le corpus complet :

- **PERSON / LOCATION / ORGANIZATION de Presidio** : quasi 100 % de faux positifs sur un texte medical (eponymes comme "Cushing", molecules, acronymes biologiques). Les masquer detruirait le contenu sans gain de confidentialite.
- **Telephone, IBAN, carte bancaire** : les seules detections etaient des taux epidemiologiques ("3/100 000") ou des numeros publics du NIH.

## Resultat

- 1 402 lignes sur 6 747 modifiees.
- 588 civilites masquees (200 formes distinctes), 22 prenoms (13 distincts), 0 adresse e-mail.
- Corpus revalide par le schema Pandera `corpus_schema` apres masquage.
- Traces (non versionnees, car elles contiennent le texte masque) : `data/interim/anonymization_report.csv` (chaque forme masquee et son nombre d'occurrences) et `anonymization_sample.csv` (30 exemples avant / apres, pour relecture).

## Limites

- Les dates, villes et etablissements cites dans les vignettes sont conserves : ils sont fictifs et utiles au raisonnement clinique ("revenu de Guadeloupe", "en janvier", "a Nice").
- Un prenom sans age a cote n'est pas detecte. La relecture de l'echantillon n'en a pas montre, mais elle n'est pas exhaustive.
- Le jeu DPO UltraMedical (anglais, 6 009 lignes) n'est pas passe dans ce pipeline : il n'est plus utilise, les paires DPO du projet sont construites sur le corpus deja anonymise (`build_dpo_triage.py`).
- Avec de vraies donnees du CHSA (journaux de triage), ce pipeline ne suffirait pas : il faudrait des pseudonymes coherents, le masquage des dates et lieux reels, une analyse d'impact (AIPD) et une relecture manuelle avant tout entrainement.
