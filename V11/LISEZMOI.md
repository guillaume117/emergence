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
| C2 | Croissance par complétion de carrés (R3′), fermeture des seuls 4-cycles (R4′), suppression (R6) sans pont avec renouvellement, H0 biparti ou germe (`h0_mode`), plancher de réouverture ε | `ModeleA._croissance_v11`, R2 dans `pas` |
| C3 | Couche B partant d'un état aligné, relaxée par échantillonnage de Gibbs local du modèle O(d) de couplages x_k (R5′) | `ModeleA._r5_v11`, `tirer_vmf` |

Les règles v10 restent disponibles (`Parametres(regles="v10")`) ; elles bénéficient de C1, qui ne modifie que le journal de dépendances, pas la dynamique.

## Diagnostic

```
python diagnostic_v11.py --n0 1000 --nu 0.0           # H0 aléatoire biparti
python diagnostic_v11.py --n0 1000 --nu 0.3 --h0 germe
```

Il mesure la fragmentation (composantes, V(r) sur la composante géante) et le retard de la couche B (défauts avant et après un recuit global à issues fixées, comparés aux vecteurs aléatoires).

## Nouvelles mesures

- **1c** : le test d'invariance de section n'est déclaré informatif que si le recouvrement entre coupes est inférieur à 0,5 (`test_informatif`).
- **1e** : trace de la taille au fil des événements, pour juger de la stationnarité.
- **2a** : contrôle `relaxation_C3` — sur un 4-cycle isolé, la relaxation atteint environ 2√2 si le cycle est frustré et reste au plus 2 sinon.
- **Géométrie** mesurée sur la composante géante ; nombre de composantes et fraction géante rapportés.
- **2c** : défaut moyen et fraction au-dessus du seuil τ = 0,1, par classe : frustré, non frustré dans une composante déséquilibrée, non frustré dans une composante équilibrée (Harary, union-find avec parité).
- **2d** : balayage en βJ, densité de frustration de A contre fraction de cycles violant l'inégalité, avec la référence à vecteurs aléatoires.

## Corrections après le premier essai complet

Le premier essai (n0 = 3000, ν = 0) donnait d_H ≈ 0,4 et des violations sur 60 % des cycles non frustrés. Trois causes, corrigées :

1. R6 supprimait des ponts et fragmentait la structure en centaines de petites composantes : R6 exige désormais un chemin alternatif de longueur ≤ L, et la géométrie est mesurée sur la composante géante.
2. Le comptage δ > 0 retenait des défauts infinitésimaux : seuil τ et défaut moyen.
3. La classe « isolé d'un cycle frustré court » n'était pas pertinente : classes par équilibre de composante.

## Points de vigilance relevés pendant la mise au point

Ces observations proviennent de tests de fumée sur de très petites tailles, pas des expériences :

1. Sans renouvellement (δ_carre = 0) ni plancher de réouverture (ε = 0), la dynamique v11 atteint rapidement un état absorbant : le complexe de carrés se sature. Les valeurs par défaut (δ_carre = ε = 0,05, βJ = 0,3) donnent un régime actif ; sa stationnarité reste à vérifier (1e).
2. À β_B ≈ 4, le bruit thermique de R5′ produit à lui seul des violations sur des cycles non frustrés. La valeur par défaut est β_B = 50.
3. Dans un réseau où la frustration est dense, les torsions se propagent : des cycles non frustrés peuvent violer l'inégalité. La prédiction stricte ne vaut que pour une composante équilibrée ; la prédiction testable est la dépendance à la densité de frustration (2d).
4. Dans le régime actif par défaut, la densité de frustration est élevée. Si la contextualité reste générique, il faudra chercher un régime actif où A est ordonné (frustration rare et localisée).
