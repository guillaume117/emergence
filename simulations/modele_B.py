"""
Modèle jouet B — couche contextuelle sur la structure émergente (section 24 du programme v10).

Contextes : hyperarêtes de rang 2 {u, v}. Données : c_k = <w_u, w_v>, p_k(a_u, a_v) = (1 + c_k a_u a_v)/4.
Aucune loi jointe sur l'ensemble des issues n'est jamais construite, sauf dans les contrôles
où l'on vérifie précisément son existence ou son absence.

Contenu :
  * énumération des cycles courts du graphe des contextes ;
  * pour chaque cycle, valeur maximale du membre de gauche des inégalités de cycle
    (pour un 4-cycle, c'est la valeur CHSH S maximisée sur les étiquetages) et défaut
    δ = [valeur - (n - 2)]_+ ;
  * défaut local au sommet δ°(o) = max des δ sur les cycles de longueur ≤ L passant par o
    (minorant de δ_r(o) pour r = ⌊L/2⌋) ;
  * contrôles : formule de section sur les forêts, équivalence inégalités de cycle /
    existence d'une section globale (programme linéaire), borne de Tsirelson.
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
from scipy.optimize import linprog


# ----------------------------------------------------------------------------
# Graphe des contextes
# ----------------------------------------------------------------------------

def graphe_contextes(C) -> dict:
    adj = {}
    for sommets in C.aretes.values():
        if len(sommets) == 2:
            u, v = sommets
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    return adj


def cycles_courts(adj, L_max=6):
    """Énumère chaque cycle simple de longueur 3..L_max une seule fois
    (sommet minimal en tête, sens canonique)."""
    cycles = []
    for s in adj:
        pile = [(s, [s])]
        while pile:
            u, chemin = pile.pop()
            for v in adj[u]:
                if v == s and len(chemin) >= 3:
                    if chemin[1] < chemin[-1]:          # un seul des deux sens
                        cycles.append(tuple(chemin))
                elif v > s and v not in chemin and len(chemin) < L_max:
                    pile.append((v, chemin + [v]))
    return cycles


# ----------------------------------------------------------------------------
# Inégalités de cycle
# ----------------------------------------------------------------------------

def valeur_cycle(c: np.ndarray) -> float:
    """max sur F impair de  Σ_{C\\F} c - Σ_F c.
    On met dans F les corrélations négatives ; si leur nombre est pair, on bascule
    l'arête de plus petite |c|, ce qui coûte 2 min|c|."""
    neg = int(np.sum(c < 0))
    v = float(np.sum(np.abs(c)))
    if neg % 2 == 0:
        v -= 2 * float(np.min(np.abs(c)))
    return v


def correlations(cycle, w) -> np.ndarray:
    n = len(cycle)
    return np.array([float(np.dot(w[cycle[i]], w[cycle[(i + 1) % n]])) for i in range(n)])


def vecteurs_aleatoires(C, d: int, graine=0) -> dict:
    """Référence : vecteurs tirés uniformément sur S^{d-1}, indépendamment de la dynamique."""
    rng = np.random.default_rng(graine)
    w = {}
    for v in C.inc:
        x = rng.normal(size=d)
        w[v] = x / np.linalg.norm(x)
    return w


def analyse_contextuelle(C, d: int, L_max=6, w=None) -> dict:
    """Statistiques de contextualité locale pour la couche de dimension d
    (ou pour un champ de vecteurs w fourni, par exemple aléatoire)."""
    adj = graphe_contextes(C)
    w = C.w[d] if w is None else w
    cycles = cycles_courts(adj, L_max)
    par_longueur = {}
    defaut_sommet = {v: 0.0 for v in adj}
    S4 = []
    for cy in cycles:
        c = correlations(cy, w)
        val = valeur_cycle(c)
        n = len(cy)
        dlt = max(0.0, val - (n - 2))
        st = par_longueur.setdefault(n, {"nombre": 0, "violations": 0, "defaut_max": 0.0,
                                         "valeur_max": -1e9})
        st["nombre"] += 1
        st["violations"] += dlt > 1e-9
        st["defaut_max"] = max(st["defaut_max"], dlt)
        st["valeur_max"] = max(st["valeur_max"], val)
        if n == 4:
            S4.append(val)
        for v in cy:
            defaut_sommet[v] = max(defaut_sommet[v], dlt)
    dv = np.array(list(defaut_sommet.values())) if defaut_sommet else np.zeros(1)
    S4 = np.array(S4)
    return {
        "d": d,
        "sommets_contextes": len(adj),
        "cycles_par_longueur": par_longueur,
        "fraction_sommets_defaut_positif": float(np.mean(dv > 1e-9)),
        "defaut_moyen": float(dv.mean()),
        "nb_4cycles": int(len(S4)),
        "fraction_CHSH_viole": float(np.mean(S4 > 2 + 1e-9)) if len(S4) else float("nan"),
        "S_max": float(S4.max()) if len(S4) else float("nan"),
        "S_moyen": float(S4.mean()) if len(S4) else float("nan"),
        "S_echantillon": S4[:3000].tolist(),
        "tsirelson_respectee": bool((S4 <= 2 * math.sqrt(2) + 1e-9).all()) if len(S4) else True,
    }


# ----------------------------------------------------------------------------
# Contrôles mathématiques (indépendants de la dynamique)
# ----------------------------------------------------------------------------

def existe_section_globale(c: np.ndarray) -> bool:
    """Existe-t-il une loi sur {±1}^n, n = longueur du cycle, à marges uniformes,
    dont les corrélations d'arêtes valent c ? Programme linéaire sur les 2^n affectations."""
    n = len(c)
    confs = np.array(list(itertools.product((-1, 1), repeat=n)))
    A_eq, b_eq = [np.ones(len(confs))], [1.0]
    for i in range(n):
        A_eq.append(confs[:, i] * confs[:, (i + 1) % n])
        b_eq.append(c[i])
        A_eq.append(confs[:, i])
        b_eq.append(0.0)
    res = linprog(np.zeros(len(confs)), A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=(0, None), method="highs")
    return res.status == 0


def controle_inegalites_de_cycle(n_essais=400, graine=1) -> dict:
    """Barahona–Mahjoub sur un cycle (pas de mineur K5) : section globale  <=>  valeur ≤ n - 2."""
    rng = np.random.default_rng(graine)
    desaccords = 0
    for _ in range(n_essais):
        n = int(rng.integers(3, 7))
        c = rng.uniform(-1, 1, size=n)
        if (valeur_cycle(c) <= n - 2 + 1e-9) != existe_section_globale(c):
            desaccords += 1
    return {"essais": n_essais, "desaccords": desaccords}


def controle_foret(n_sommets=9, graine=2) -> dict:
    """Vérifie que P_A(a) = 2^{-|X|} Π (1 + c_uv a_u a_v) est une loi de marges p_k sur un arbre."""
    rng = random.Random(graine)
    aretes = [(rng.randrange(i), i) for i in range(1, n_sommets)]   # arbre aléatoire
    c = {e: rng.uniform(-1, 1) for e in aretes}
    confs = list(itertools.product((-1, 1), repeat=n_sommets))
    P = np.array([np.prod([1 + c[(u, v)] * a[u] * a[v] for (u, v) in aretes]) for a in confs])
    P /= 2 ** n_sommets
    err = abs(P.sum() - 1)
    for (u, v) in aretes:
        for au, av in itertools.product((-1, 1), repeat=2):
            m = sum(P[k] for k, a in enumerate(confs) if a[u] == au and a[v] == av)
            err = max(err, abs(m - (1 + c[(u, v)] * au * av) / 4))
    return {"positivite": bool((P >= -1e-15).all()), "erreur_max": float(err)}


def controle_tsirelson(n_essais=20000, graine=3) -> dict:
    """Pour des vecteurs unitaires de dimension 2 à 4, la valeur CHSH d'un 4-cycle ne dépasse pas 2√2,
    et la valeur 2√2 est atteinte en dimension 2."""
    rng = np.random.default_rng(graine)
    vmax = 0.0
    for _ in range(n_essais):
        d = int(rng.integers(2, 5))
        W = rng.normal(size=(4, d))
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        c = np.array([W[i] @ W[(i + 1) % 4] for i in range(4)])
        vmax = max(vmax, valeur_cycle(c))
    W = np.stack([np.cos([0, np.pi / 4, np.pi / 2, -np.pi / 4]), np.sin([0, np.pi / 4, np.pi / 2, -np.pi / 4])], 1)
    c_opt = np.array([W[i] @ W[(i + 1) % 4] for i in range(4)])
    return {"valeur_max_aleatoire": vmax, "borne": 2 * math.sqrt(2),
            "valeur_configuration_optimale": valeur_cycle(c_opt)}
