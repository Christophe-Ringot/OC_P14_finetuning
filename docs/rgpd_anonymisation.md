# Anonymisation RGPD du corpus SFT

Perimetre : dataset SFT (`data/processed/sft_dataset.csv`, 5001 paires). Le
dataset DPO (UltraMedical-Preference) n'est pas encore traite, cf. section
"Limites et suite".

## Outillage

- `presidio-analyzer` + `presidio-anonymizer`
- Moteurs linguistiques : `fr_core_news_md` (FR), `en_core_web_md` (EN), via
  un `NlpEngineProvider` multilingue partage par les deux langues du corpus.
- Script : `scripts/anonymize_sft.py`

## Pourquoi le risque de depart est structurellement faible

Les 4 sources du corpus SFT ne sont pas des dossiers patients, mais des
corpus publics deja depersonnalises par construction :

- **FrenchMedMCQA** / **mediqal** : banques de questions d'examen de
  pharmacie/medecine. Les "cas cliniques" sont des vignettes pedagogiques
  fictives (patients denommes "Monsieur M." ou par un prenom d'emprunt), pas
  des dossiers reels.
- **MedQuAD** : encyclopedie sante publique (NIH), question/reponse sur des
  maladies, sans donnee individuelle.

Un vrai risque RGPD n'etait donc pas attendu a priori, mais un controle
systematique a ete fait plutot que suppose.

## Methodologie

1. Scan complet des 5001 lignes (instruction + reponse) avec l'analyzer
   Presidio multilingue, sur deux groupes de categories :
   - **Categories structurees a faible taux de faux positifs** (regles
     regex/checksum, peu sensibles au vocabulaire medical) :
     `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`,
     `IP_ADDRESS`.
   - **Categories dependantes du NER spaCy** : `PERSON`, `LOCATION`,
     `ORGANIZATION`.
2. Les deux groupes ont ete evalues separement avant toute decision de
   masquage automatique.

## Resultats du scan complet

| Categorie | Detections (5001 lignes) | Nature |
|---|---|---|
| EMAIL_ADDRESS / PHONE_NUMBER / CREDIT_CARD / IBAN_CODE / US_SSN / IP_ADDRESS | 20 | Toutes des coordonnees d'organismes publics (lignes d'ecoute NIH, adresse email institutionnelle `adear@nia.nih.gov`, standard NIH), issues de MedQuAD. Aucune donnee personnelle de patient. Quelques faux positifs (taux epidemiologiques du type `45/100 000`). |
| PERSON | 5 733 | Quasi-totalite de faux positifs : eponymes de maladies (`Cushing`, `Waaler Rose`), noms de bacteries (`Campylobacter jejuni`, `Clostridium`), lettres de reponses QCM isolees. |
| LOCATION | 17 209 | Quasi-totalite de faux positifs : noms de molecules (`Colchicine`, `Ciprofloxacine`), lettres de QCM isolees. |
| ORGANIZATION | 7 547 | Quasi-totalite de faux positifs : acronymes medicaux (`IgA`, `IgG`, `CCMH`, `LLC`). |

Aucune mention de patient reel identifiable n'a ete trouvee dans l'echantillon
complet.

## Decision et justification

- **Masquage systematique** des 6 categories structurees a risque reel
  (remplacement par `<ENTITY_TYPE>`), y compris les coordonnees d'organismes
  publics : une politique uniforme ("on masque tout ce qui ressemble a un
  contact structure") est plus simple a auditer qu'une politique au cas par
  cas, et evite un autre risque independant du RGPD : un modele fine-tune
  qui "recite" un numero d'urgence memorise depuis le dataset d'entrainement
  peut se tromper a l'inference (hallucination) ; ce type d'information doit
  venir d'une source verifiee au moment de la reponse, pas de la
  memorisation du corpus.
- **Pas de masquage automatique** pour PERSON / LOCATION / ORGANIZATION : le
  taux de faux positifs mesure (quasi 100%) signifie que masquer ces
  categories detruirait le contenu medical correct (ex. remplacer
  "Carbimazole" ou "maladie de Cushing" par `<PERSON>`) pour un gain de vie
  privee nul, puisqu'aucune donnee patient reelle n'a ete trouvee dans ces
  categories lors du scan complet.

## Resultat

- `data/processed/sft_dataset_anonymized.csv` : 5001 lignes, revalidees avec
  le schema Pandera `sft_schema` apres masquage.
- `outputs/anonymization_report.csv` : trace de chaque entite masquee
  (id de ligne, champ, type d'entite, texte original) — auditabilite de la
  transformation.
- Pipeline deterministe et re-executable (`python -m scripts.anonymize_sft`).

## Limites et suite

- Le dataset **DPO** (UltraMedical-Preference) n'a pas encore ete traite par
  ce pipeline — reporte a une iteration ulterieure.
- Cette conclusion ("risque structurellement faible") est propre aux 4
  sources publiques utilisees ici. Si une phase ulterieure du projet integre
  de vraies donnees hospitalieres (ex. journaux de triage du CHSA), le meme
  pipeline Presidio devra etre reapplique avec une revue manuelle de
  PERSON/LOCATION, car l'hypothese de depart (pas de patients reels) ne
  tiendra plus.
