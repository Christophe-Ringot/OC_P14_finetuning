# Jugement en aveugle des explications du niveau : SFT vs SFT + DPO

60 cas du test, question de l'API, niveau annote force. Sorties presentees en ordre aleatoire (A/B) sans le nom du modele. Grille : docs/evaluation_reponses.md. Notes de 1 a 5.

| modele | coherence | ancrage | exactitude | securite | clarte | moyenne | prefere |
|---|---|---|---|---|---|---|---|
| 4_sft_medical_triage | 2.95 | 1.98 | 1.93 | 3.67 | 3.22 | 2.75 | 27 |
| 6_sft_triage_dpo_qualite | 2.92 | 1.93 | 1.88 | 3.75 | 3.08 | 2.71 | 18 |

Egalites : 15 / 60. Test du signe sur les cas departages : p = 0.233.

Preferences par niveau annote :

| niveau | 4_sft_medical_triage | 6_sft_triage_dpo_qualite | egalites |
|---|---|---|---|
| P1 | 8 | 8 | 4 |
| P2 | 9 | 7 | 4 |
| P3 | 10 | 3 | 7 |
