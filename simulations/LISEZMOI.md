# Simulations des modèles jouets A et B

Code d'accompagnement du programme *Auto-structuration d'espaces probabilistes relationnels* (version 10).
Python 3.10 ou plus, avec `numpy`, `scipy`, `matplotlib`.

```
python experiences.py            # version rapide (environ 1 minute)
python experiences.py --complet  # graines et tailles plus grandes (environ 15 minutes)
```

Les résultats sont écrits dans `resultats/` : `resultats.json` et trois figures.

## Fichiers

| Fichier | Contenu |
|---|---|
| `modele_A.py` | Dynamique du modèle A (blocs, Gibbs local, R1–R4, horloges de taux λ0\|Λ\|), couche vectorielle du modèle B (R5), journal des événements avec empreintes, reconstruction sur coupes admissibles |
| `modele_B.py` | Cycles du graphe des contextes, inégalités de cycle, défaut local δ°, valeurs CHSH, contrôles mathématiques |
| `observables.py` | Cyclicité, croissance volumique, dimension spectrale, lois locales enracinées |
| `experiences.py` | Les deux expériences et les figures |

## Correspondance avec le texte

- **1a — non-terminaison** (test 6) : probabilité d'atteindre U = ∅ sur une grille (βJ, μ).
- **1b — géométrie selon ν** (test 9, expérience G) : β1(B_r)/|B_r|, V(r), d_s(s).
- **1c — invariance de section** (critère 3, expérience F2) : lois locales sur trois coupes admissibles de même cardinal — préfixe en t, coupe par profondeur dans l'ordre dérivé, coupe biaisée où une moitié de H0 est très en avance. Le bruit est estimé par deux tirages de racines sur la même coupe.
- **1d — reconstruction** : rejouer tout le journal redonne exactement la configuration finale.
- **2a — contrôles** : formule de section sur les forêts ; équivalence inégalités de cycle / existence d'une section globale par programme linéaire (Barahona–Mahjoub sur un cycle) ; borne de Tsirelson.
- **2b — contextualité** : pour d = 1, 2, 3, fraction de 4-cycles violant CHSH, S_max, fraction de sommets avec δ° > 0. Deux références : l'état initial H0, et le même graphe final muni de vecteurs aléatoires (sans R5).
- **2c — co-émergence** : capacité contextuelle contre cyclicité, d'une valeur de ν à l'autre.

## Choix d'implémentation à connaître

1. **Tirage des événements.** Avec des taux λ0|Λ|, le taux total vaut λ0|U| ; le bloc réalisé est donc celui d'une hyperarête ouverte tirée uniformément. C'est exactement le processus de la section 23.9, pas une approximation.
2. **Dépendance par empreintes.** Deux événements sont reliés s'ils partagent une hyperarête de leurs empreintes (lectures et écritures). C'est une sur-approximation de ≺_H : elle peut relier des événements qui commutent, jamais l'inverse. Les coupes construites sont donc admissibles.
3. **Ordre interne.** Les tirages de R2, R3, R4 et R5 sont faits sur la configuration précédant la règle puis appliqués simultanément ; les troncatures de degré sont faites par sous-ensemble uniforme (pour R4, par acceptation dans un ordre uniformément aléatoire).
4. **δ°(o)** est le maximum des défauts sur les cycles de longueur ≤ 6 passant par o. C'est un minorant de δ_3(o) tel que défini dans le texte.
5. **Arrêt.** Les simulations s'arrêtent à une taille maximale de sommets. C'est un critère computationnel : les observables sont ensuite évaluées sur la configuration ou sur des coupes admissibles, jamais en fonction de t.

## Ce qui n'est pas encore implémenté

Persistance des motifs sur coupes (définition de Π⁻, Π⁺), test d'invariance d'implémentation parallèle (test 7), recherche d'invariants et de spectres, motif de confusion isolé.
