# Qualite des reponses generees

Echantillon fixe du test : 80 questions (20 cas par niveau P1/P2/P3, un par cas clinique, et 20 questions generales). Decodage glouton, 300 tokens max. Valeurs en % (longueur en mots). `rouge_l` et `f1_lexical` comparent a la reponse de reference sans le prefixe de niveau.

| modele | rouge_l | f1_lexical | prefixe_present | niveau_correct | sous_triage_ecrit | formulation_risquee | chiffre_invente | longueur_mots | coupee | repetition | vide | rouge_l_P1 | rouge_l_P2 | rouge_l_P3 | rouge_l_general |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2_modele_base | 7.8 | 11.3 | 0.0 | 0.0 | 0.0 | 0.0 | 50.0 | 139.0 | 60.0 | 2.3 | 1.2 | 3.5 | 7.6 | 9.9 | 10.4 |
| 4_sft_medical_triage | 18.9 | 21.9 | 100.0 | 80.0 | 3.3 | 0.0 | 18.8 | 36.2 | 3.8 | 0.8 | 3.8 | 7.0 | 8.4 | 13.0 | 47.3 |
| 5_sft_triage_dpo | 19.2 | 22.7 | 100.0 | 40.0 | 0.0 | 0.0 | 25.0 | 39.7 | 3.8 | 0.6 | 1.2 | 7.2 | 9.8 | 13.1 | 46.7 |
| 6_sft_triage_dpo_qualite | 19.5 | 23.2 | 100.0 | 78.3 | 3.3 | 0.0 | 25.0 | 39.6 | 1.2 | 0.4 | 1.2 | 6.6 | 11.1 | 13.3 | 47.0 |

Lecture :
- `niveau_correct` porte sur le niveau ecrit par le modele en generation libre ; la metrique de triage de reference reste celle des log-probabilites (comparaison_triage.md), plus stable.
- `formulation_risquee` et `chiffre_invente` sont des signaux a relire, pas des erreurs averees (un chiffre calcule a partir du cas est signale aussi).
- ROUGE-L recompense la proximite avec une reference courte : il mesure la pertinence par rapport au corrige, pas la qualite clinique. Le jugement en aveugle (docs/evaluation_reponses.md) la complete.
