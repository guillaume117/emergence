"""
Observables géométriques et structurelles sur une configuration (graphe primal de H).

Toutes les observables sont des fonctions de la configuration seule : elles ne lisent ni le
paramètre auxiliaire t, ni l'ordre du journal. Elles sont donc évaluables sur n'importe quelle
coupe admissible.
"""

from __future__ import annotations

import math
from collections import Counter, deque

import numpy as np
import scipy.sparse as sp


def graphe_primal(C) -> dict:
    """Liste d'adjacence du graphe primal : u ~ v si une hyperarête contient u et v."""
    adj = {v: set() for v, s in C.inc.items() if s}
    for sommets in C.aretes.values():
        for i, u in enumerate(sommets):
            for v in sommets[i + 1:]:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def boule(adj, o, r) -> dict:
    """Distances depuis o, jusqu'au rayon r (BFS)."""
    dist = {o: 0}
    file = deque([o])
    while file:
        u = file.popleft()
        if dist[u] == r:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                file.append(v)
    return dist


def stats_structure(C) -> dict:
    """Taille, fraction ouverte, taille du plus grand bloc rapportée à |U|."""
    ouvertes = [e for e, x in C.x.items() if x == 0]
    vus, plus_grand = set(), 0
    for e0 in ouvertes:
        if e0 in vus:
            continue
        b = C.bloc(e0)
        vus |= b
        plus_grand = max(plus_grand, len(b))
    return {
        "sommets": sum(1 for s in C.inc.values() if s),
        "hyperaretes": len(C.aretes),
        "fraction_ouverte": len(ouvertes) / max(1, len(C.aretes)),
        "plus_grand_bloc_sur_U": plus_grand / max(1, len(ouvertes)),
        "couplage_moyen": float(np.mean(list(C.rho.values()))) if C.rho else 0.0,
    }


def cyclicite(adj, rng, n_racines=300, rayons=(2, 3, 4)) -> dict:
    """Nombre cyclomatique par sommet des boules B_r(o) : β1 = |E| - |V| + 1 (boule connexe)."""
    sommets = list(adj)
    racines = [sommets[rng.randrange(len(sommets))] for _ in range(n_racines)]
    res = {}
    for r in rayons:
        vals = []
        for o in racines:
            d = boule(adj, o, r)
            nv = len(d)
            ne = sum(1 for u in d for v in adj[u] if v in d) // 2
            vals.append((ne - nv + 1) / nv)
        res[f"beta1_par_sommet_r{r}"] = float(np.mean(vals))
    # proportion de sommets portant un cycle court (longueur ≤ 6) : β1(B_3) > 0 au voisinage
    res["degre_moyen"] = float(np.mean([len(adj[v]) for v in sommets]))
    return res


def croissance_volumique(adj, rng, n_racines=200, r_max=12) -> dict:
    """V(r) moyen et deux ajustements : loi de puissance (dimension d_H) et exponentielle."""
    sommets = list(adj)
    V = np.zeros(r_max + 1)
    for _ in range(n_racines):
        o = sommets[rng.randrange(len(sommets))]
        d = boule(adj, o, r_max)
        c = Counter(d.values())
        V += np.cumsum([c.get(r, 0) for r in range(r_max + 1)])
    V /= n_racines
    r = np.arange(r_max + 1)
    sel = (r >= 3) & (V < 0.3 * len(sommets))      # fenêtre avant saturation de taille finie
    out = {"V": V.tolist()}
    if sel.sum() >= 3:
        out["d_H_puissance"] = float(np.polyfit(np.log(r[sel]), np.log(V[sel]), 1)[0])
        out["taux_exponentiel"] = float(np.polyfit(r[sel], np.log(V[sel]), 1)[0])
    return out


def dimension_spectrale(adj, rng, n_racines=20, s_max=120) -> dict:
    """Probabilité de retour d'une marche paresseuse et d_s(s) = -2 dlogP/dlogs.
    Le paramètre s est un paramètre de diffusion, non un temps physique."""
    sommets = list(adj)
    idx = {v: i for i, v in enumerate(sommets)}
    lignes, cols = [], []
    for u in sommets:
        for v in adj[u]:
            lignes.append(idx[v])
            cols.append(idx[u])
    n = len(sommets)
    A = sp.csr_matrix((np.ones(len(lignes)), (lignes, cols)), shape=(n, n))
    deg = np.asarray(A.sum(axis=0)).ravel()
    deg[deg == 0] = 1
    T = 0.5 * (sp.identity(n) + A @ sp.diags(1 / deg))   # marche paresseuse
    P = np.zeros(s_max + 1)
    for _ in range(n_racines):
        o = idx[sommets[rng.randrange(n)]]
        p = np.zeros(n)
        p[o] = 1
        for s in range(s_max + 1):
            P[s] += p[o]
            p = T @ p
    P /= n_racines
    s = np.arange(1, s_max + 1)
    ds = -2 * np.gradient(np.log(P[1:]), np.log(s))
    return {"P_retour": P.tolist(), "d_s_10_40": float(np.mean(ds[9:40])),
            "d_s_40_120": float(np.mean(ds[39:]))}


def distance_variation_totale(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    cles = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in cles)


def lois_locales(adj, rng, n_racines=400, r=2) -> dict:
    """Lois empiriques de voisinages enracinés simples : degré de la racine et |B_r(o)|."""
    sommets = list(adj)
    deg, taille = Counter(), Counter()
    for _ in range(n_racines):
        o = sommets[rng.randrange(len(sommets))]
        deg[len(adj[o])] += 1
        taille[len(boule(adj, o, r))] += 1
    return {"degre": deg, f"taille_B{r}": taille}
