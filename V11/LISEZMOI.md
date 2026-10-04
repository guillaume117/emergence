# Simulations des modèles jouets — version révisée v11

Code d'accompagnement de la note *Corrections apportées aux modèles jouets (v11)*.
Python 3.10 ou plus, avec `numpy`, `scipy`, `matplotlib`.

```
python experiences.py                  # règles v11, version rapide
python experiences.py --complet        # règles v11, version complète
python experiences.py --regles v10     # règles d'origine, pour comparaison
```

Les résultats sont écrits dans `resultats_v11/` ou `resultats_v10/`.

## Corrections implémentées

| | Correction | Où dans le code |
|---|---|---|
| C1 | Dépendance par lectures et écritures sur des items fins (`("x", e)`, `("r", (e, f))`, `("inc", v)`, `("w", v)`) | `ModeleA.pas`, bloc « journal » |
| C2 | Croissance par complétion de carrés (R3′), fermeture des seuls 4-cycles (R4′), suppression (R6) avec renouvellement, H0 biparti, plancher de réouverture ε | `ModeleA._croissance_v11`, R2 dans `pas` |
| C3 | Couche B partant d'un état aligné, relaxée par échantillonnage de Gibbs local du modèle O(d) de couplages x_k (R5′) | `ModeleA._r5_v11`, `tirer_vmf` |

Les règles v10 restent disponibles (`Parametres(regles="v10")`) ; elles bénéficient de C1, qui ne modifie que le journal de dépendances, pas la dynamique.

## Nouvelles mesures

- **1c** : le test d'invariance de section n'est déclaré informatif que si le recouvrement entre coupes est inférieur à 0,5 (`test_informatif`).
- **1e** : trace de la taille au fil des événements, pour juger de la stationnarité.
- **2a** : contrôle `relaxation_C3` — sur un 4-cycle isolé, la relaxation atteint environ 2√2 si le cycle est frustré et reste au plus 2 sinon.
- **2c** : violations selon la classe du cycle : frustré, non frustré voisin d'un cycle frustré, non frustré isolé.
- **2d** : balayage en βJ, densité de frustration de A contre fraction de cycles violant l'inégalité, avec la référence à vecteurs aléatoires.

## Points de vigilance relevés pendant la mise au point

Ces observations proviennent de tests de fumée sur de très petites tailles, pas des expériences :

1. Sans renouvellement (δ_carre = 0) ni plancher de réouverture (ε = 0), la dynamique v11 atteint rapidement un état absorbant : le complexe de carrés se sature. Les valeurs par défaut (δ_carre = ε = 0,05, βJ = 0,3) donnent un régime actif ; sa stationnarité reste à vérifier (1e).
2. À β_B ≈ 4, le bruit thermique de R5′ produit à lui seul des violations sur des cycles non frustrés. La valeur par défaut est β_B = 50.
3. Dans un réseau où la frustration est dense, les torsions se propagent : des cycles non frustrés peuvent violer l'inégalité. La prédiction stricte ne vaut que pour une composante équilibrée ; la prédiction testable est la dépendance à la densité de frustration (2d).
4. Dans le régime actif par défaut, la densité de frustration est élevée. Si la contextualité reste générique, il faudra chercher un régime actif où A est ordonné (frustration rare et localisée).
