# Version de calcul optimisée — moteur C++ et observables GPU

Moteur C++ des règles v11 (modèle A et couche contextuelle B), piloté depuis Python, avec
parallélisation par processus et dimension spectrale sur GPU.

## Architecture

| Couche | Fichier | Rôle |
|---|---|---|
| Moteur | `moteur/moteur.cpp` | Dynamique complète v11, journal lecture/écriture, reconstruction sur coupe, observables lourdes (composantes, volume des boules, cyclicité, contextualité) |
| Liaison | `moteur_rapide.py` | Enveloppe ctypes : `MoteurRapide`, mêmes paramètres que `modele_A.Parametres` |
| GPU | `observables_gpu.py` | Dimension spectrale par marches aléatoires en bloc (CuPy), repli automatique sur SciPy |
| Expériences | `experiences_rapide.py` | Simulations en parallèle (graines × ν), figures et JSON |
| Validation | `valider_moteur.py` | Comparaison avec l'implémentation Python de référence |

Pourquoi ctypes plutôt que pybind11 : les appels sont grossiers (simuler des millions d'événements,
exporter des tableaux), donc le coût d'interface est négligeable ; ctypes n'exige rien d'autre qu'un
compilateur C++. Pourquoi pas CUDA pour le moteur : la dynamique est séquentielle, irrégulière et
modifie la structure à chaque événement ; le GPU sert là où le calcul est régulier (observables).

## Compilation

Linux ou macOS :
```
cd moteur && make
```
Windows (Visual Studio 2022 et CMake) ou toute plateforme :
```
cd moteur
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
La bibliothèque (`libmoteur.so`, `libmoteur.dylib` ou `moteur.dll`) est placée dans `moteur/`.
L'option `-march=native` optimise pour la machine de compilation : recompilez sur chaque machine.

GPU (optionnel) : `pip install cupy-cuda12x`. Sans CuPy, tout fonctionne sur CPU.

## Validation (à lancer en premier)

```
python valider_moteur.py --graines 8 --n0 300 --evenements 20000
```
Elle vérifie :
1. que la reconstruction depuis le journal complet redonne l'état final à l'identique, et que les coupes sont fermées vers le passé ;
2. que les observables C++ (classes de frustration, défauts, composantes) coïncident exactement avec `modele_B.py` et `observables.py` sur un même état ;
3. que la dynamique C++ est statistiquement équivalente à la dynamique Python (écarts réduits |z| petits) — les générateurs aléatoires diffèrent, l'équivalence ne peut être que statistique ;
4. la vitesse relative.

## Expériences

```
python experiences_rapide.py --taille 200000 --graines 4 --nus 0 0.3 0.6 --processus 8
python experiences_rapide.py --taille 1000000 --h0 germe --graines 2 --processus 2
python experiences_rapide.py --taille 100000 --graines 2 --section      # avec test de section
```

- `--taille` : nombre maximal de sommets créés ; `--processus` : nombre de simulations simultanées.
- Sans `--section`, le journal est désactivé : c'est le mode rapide et économe en mémoire.
- Le journal coûte environ deux à trois fois la vitesse et beaucoup de mémoire (lectures, écritures et
  prédécesseurs de chaque événement) : à réserver aux tailles où le test de section est souhaité.

Sorties dans `resultats_rapide/` : `resultats.json` et `synthese_rapide.png` (volume des boules,
dimension spectrale, cyclicité, stationnarité, défaut contextuel par classe, pente locale d_H).

## Performances mesurées pendant le développement

Sur un cœur de la machine de développement (pas la vôtre), règles v11 :

| Configuration | Événements/s |
|---|---|
| Python de référence | environ 400 |
| C++ sans couche B, sans journal | environ 50 000 |
| C++ avec couche B (d = 2), sans journal | environ 27 000 |
| C++ avec couches d = 1, 2, 3 et journal | environ 8 000 |

Soit une accélération de 20 à 120 selon la configuration, multipliée par le nombre de processus.
La couche B (relaxation von Mises–Fisher) et le journal sont les postes les plus coûteux.

## Pistes d'optimisation suivantes

1. Remplacer la table de hachage des couplages par des tableaux courts par arête (le degré est borné).
2. Réduire les lectures déclarées : la déclaration actuelle est volontairement conservatrice, ce qui
   élargit les cônes de dépendance (le recouvrement des coupes reste élevé aux petites tailles).
3. Moteur parallèle exact (algorithme de Lubachevsky) sur CPU multicœur, puis éventuellement sur GPU,
   qui servirait aussi de test d'invariance d'implémentation (test 7 du programme).
