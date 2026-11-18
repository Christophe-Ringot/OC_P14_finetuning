# Jugement en aveugle des reponses : SFT vs SFT + DPO

60 cas cliniques annotes de l'echantillon fixe du test (question du dataset). Sorties presentees en ordre aleatoire (A/B) sans le nom du modele. Grille : docs/evaluation_reponses.md. Notes de 1 a 5.

| modele | pertinence | clarte | securite | conformite | moyenne | prefere |
|---|---|---|---|---|---|---|
| 4_sft_medical_triage | 1.68 | 2.73 | 2.82 | 4.77 | 3.00 | 20 |
| 6_sft_triage_dpo_qualite | 1.55 | 2.62 | 2.70 | 4.75 | 2.90 | 8 |

Egalites : 32 / 60. Test du signe sur les cas departages : p = 0.036.

Preferences par niveau annote :

| niveau | 4_sft_medical_triage | 6_sft_triage_dpo_qualite | egalites |
|---|---|---|---|
| P1 | 2 | 4 | 14 |
| P2 | 10 | 2 | 8 |
| P3 | 8 | 2 | 10 |
