"""
Dimension spectrale par marches aléatoires paresseuses, en bloc : on fait évoluer simultanément
R distributions initialement concentrées sur R racines, X <- T X avec T = (I + A D^-1)/2 creuse.
C'est un produit matrice creuse × matrice dense, exactement le type de calcul où un GPU excelle.

Sur GPU, nécessite CuPy (pip install cupy-cuda12x pour une RTX 4090 avec CUDA 12).
Sans CuPy, le même calcul est fait sur CPU avec SciPy.

    from observables_gpu import dimension_spectrale
    res = dimension_spectrale(moteur, n_racines=64, s_max=2000)
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    GPU = True
except Exception:          # CuPy absent ou pas de GPU utilisable
    GPU = False
import scipy.sparse as sp


def matrice_marche(ab: np.ndarray, vivante: np.ndarray, n_sommets: int):
    """Matrice de transition de la marche paresseuse sur le graphe primal (CSR, colonnes stochastiques)."""
    e = ab[vivante]
    lignes = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(len(lignes), np.float32), (lignes, cols)), shape=(n_sommets, n_sommets))
    deg = np.asarray(A.sum(axis=0)).ravel()
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0).astype(np.float32)
    T = 0.5 * (sp.identity(n_sommets, dtype=np.float32, format="csr") + A @ sp.diags(inv))
    return T.tocsr()


def composante_geante(ab: np.ndarray, vivante: np.ndarray, n_sommets: int) -> np.ndarray:
    from scipy.sparse.csgraph import connected_components
    e = ab[vivante]
    A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_sommets, n_sommets))
    _, lab = connected_components(A, directed=False)
    deg = np.bincount(e.ravel(), minlength=n_sommets)
    lab = np.where(deg > 0, lab, -1)
    g = np.bincount(lab[lab >= 0]).argmax()
    return np.nonzero(lab == g)[0]


def dimension_spectrale_tableaux(ab, vivante, n_sommets, n_racines=64, s_max=1000, graine=0,
                                 utiliser_gpu=None) -> dict:
    """P_retour(s) moyennée sur des racines de la composante géante, et d_s(s) = -2 dlogP/dlogs."""
    candidats = composante_geante(ab, vivante, n_sommets)
    rng = np.random.default_rng(graine)
    rac = rng.choice(candidats, size=min(n_racines, len(candidats)), replace=False)
    T = matrice_marche(ab, vivante, n_sommets)
    n = n_sommets
    gpu = GPU if utiliser_gpu is None else (utiliser_gpu and GPU)
    if gpu:
        Tg = csp.csr_matrix(T)
        X = cp.zeros((n, len(rac)), dtype=cp.float32)
        idx_r, idx_c = cp.asarray(rac), cp.arange(len(rac))
        X[idx_r, idx_c] = 1.0
        P = cp.zeros(s_max + 1, dtype=cp.float64)
        for s in range(s_max + 1):
            P[s] = X[idx_r, idx_c].mean()
            X = Tg @ X
        P = cp.asnumpy(P)
    else:
        X = np.zeros((n, len(rac)), dtype=np.float32)
        X[rac, np.arange(len(rac))] = 1.0
        P = np.zeros(s_max + 1)
        for s in range(s_max + 1):
            P[s] = X[rac, np.arange(len(rac))].mean()
            X = T @ X
    s = np.arange(1, s_max + 1)
    ds = -2 * np.gradient(np.log(np.maximum(P[1:], 1e-300)), np.log(s))
    return {"P_retour": P, "d_s": ds, "gpu": gpu, "n_racines": len(rac)}


def dimension_spectrale(moteur, **kw) -> dict:
    """Même calcul à partir d'un MoteurRapide ou d'un état reconstruit."""
    ab, _, viv = moteur.aretes()
    return dimension_spectrale_tableaux(ab, viv, len(moteur.composantes()[1]), **kw)
