# Qualite des explications du niveau de priorite

60 cas du test (20 par niveau), question de l'API (`Quel est le niveau de priorité de ce patient et pourquoi ?`), niveau annote force en tete : les deux modeles expliquent le meme niveau. Decodage glouton, 200 tokens max (comme l'API). Valeurs en % (longueur en mots).

| modele | ancrage | chiffre_invente | formulation_risquee | niveau_contredit | longueur_mots | coupee | repetition | vide |
|---|---|---|---|---|---|---|---|---|
| 4_sft_medical_triage | 12.3 | 5.0 | 0.0 | 1.7 | 26.3 | 0.0 | 0.5 | 0.0 |
| 6_sft_triage_dpo_qualite | 13.5 | 5.0 | 0.0 | 0.0 | 23.8 | 0.0 | 0.2 | 0.0 |

Par niveau annote (%) :

| modele | niveau | ancrage | chiffre_invente | niveau_contredit |
|---|---|---|---|---|
| 4_sft_medical_triage | P1 | 12.6 | 5.0 | 5.0 |
| 4_sft_medical_triage | P2 | 16.3 | 0.0 | 0.0 |
| 4_sft_medical_triage | P3 | 6.7 | 10.0 | 0.0 |
| 6_sft_triage_dpo_qualite | P1 | 13.5 | 5.0 | 0.0 |
| 6_sft_triage_dpo_qualite | P2 | 15.2 | 5.0 | 0.0 |
| 6_sft_triage_dpo_qualite | P3 | 11.1 | 5.0 | 0.0 |

Lecture :
- `ancrage` : part des symptomes du cas (metadonnees du dataset) repris dans l'explication ; une explication qui ne cite rien du cas est generique.
- `chiffre_invente` : un nombre absent du cas (constante, dose, delai) ; signal a relire, un calcul juste est signale aussi.
- `niveau_contredit` : vocabulaire d'un autre niveau (ex. 'pas d'urgence' sous un P1).
- Le jugement en aveugle (`jugement_explications.md`) note la coherence et l'exactitude.
