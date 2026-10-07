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

## Modèle v12 — croissance par front causal

Implémente `note_front_causal_v12.tex` : hauteur causale par sommet (immuable, écrite à la création),
germe en zigzag (cycle de 2 L0 sommets, hauteurs 0,1,0,1…), complétion causale R3c (h(a) = h(c) = h(b)+1,
nouveau sommet à h(b)+2), fermeture causale R4c (diamants seulement), fluctuations R7± (insertion d'un
voisin futur pendant ; retrait d'un sommet à un seul parent, mouvement inverse).

```
python valider_v12.py --L0 40 --graines 4          # test déterministe : cylindre plat exact
python experiences_rapide.py --modele v12 --L0 100 --taille 1000000 --zeta_plus 0.02 --zeta_moins 0.02
python experiences_rapide.py --modele v12 --L0 100 --taille 200000 --zeta_plus 0.02 --zeta_moins 0.02 --section
```

Le préréglage `--modele v12` désactive R6 (δ = δ_carre = 0) et la condition de concordance de R3c ;
tout reste surchargeable par `--param`. Sorties supplémentaires : `tranches.png` (longueur des tranches
L(h) ; L ∝ h signale une géométrie plate de dimension 2, L constant un cylindre, L exponentiel une
géométrie hyperbolique), exposant de L(h) dans la région achevée, violations de l'invariant de hauteur,
classification des 4-cycles frustrés (diamants causaux ou non), coupes à hauteur constante pour le test
de section.

## Réouverture spontanée (R0)

Chaque hyperarête actualisée porte une horloge exponentielle de réouverture de taux
`lambda_reouverture` (0,01 par défaut, à comparer à λ0 = 1 par degré ouvert). Sans cette règle,
la croissance depuis un germe s'éteint en quelques événements (U = ∅) ; avec elle, l'état absorbant
n'est plus atteignable tant qu'il reste une arête. R0 est locale, homogène, et journalisée comme
un événement à part entière (lecture et écriture de l'issue de l'arête). Elle est aussi présente
dans `modele_A.py`, pour que la validation compare les mêmes dynamiques.

## Diagnostic de phase : surface ou polymère branché ?

`experiences_rapide.py` mesure, à plusieurs tailles d'une même croissance (`--paliers`) :

- la **fraction de sommets dans la plus grande composante 2-arête-connexe** et la fraction de ponts
  (Tarjan) : elle reste finie pour une surface et tend vers zéro pour un polymère branché d'amas
  de carrés — c'est le discriminant principal ;
- la distribution des degrés, la fraction de **sommets plats** (degré 4, quatre coins dans des
  carrés, motif de Z²) et la **courbure combinatoire** κ(v) = 1 − deg/2 + coins/4 ;
- la densité de cycles β1(B_r)/|B_r| jusqu'à r = 40 (attention : c'est une densité, elle ne
  distingue pas cycles locaux et cycles à grande échelle).

Figure : `diagnostic_phase.png`. Variantes de règles, pour tester ce qui fait sortir de la phase
de polymère branché :

```
python experiences_rapide.py --h0 germe --taille 1000000 --r4 strict
python experiences_rapide.py --h0 germe --taille 1000000 --r4 sans_concordance
python experiences_rapide.py --h0 germe --taille 1000000 --r4 libre --r3_sans_concordance
python spectre_depuis_sauvegarde.py --dossier germe_r4-libre_r3-libre --s_max 3000
```

Chaque variante écrit dans son propre sous-dossier de `resultats_rapide/`.

Élagage des impasses : `--param elagage=p` autorise R6 à supprimer une arête pendante (extrémité de
degré 1) avec probabilité p, sans chemin alternatif ; cela n'isole que le sommet pendant. Attention :
un élagage fort (p ≈ 0,5) dévore les branches de proche en proche et effondre la structure.

Dimension de marche : `experiences_rapide.py` mesure le déplacement quadratique moyen ⟨d²⟩ ∼ s^{2/d_w}
(marche paresseuse, distances de graphe exactes) et trace `marche.png`, qui confronte d_s mesurée à
2 d_H / d_w (relation d'Alexander–Orbach, vérifiée par les structures fractales usuelles). Les variantes ne sont
implémentées que dans le moteur C++ ; la validation contre Python porte sur les règles strictes.

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
