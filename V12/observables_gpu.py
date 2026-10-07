"""
Dimension spectrale par marches aléatoires paresseuses, en bloc : on fait évoluer simultanément
R distributions initialement concentrées sur R racines, X <- T X avec T = (I + A D^-1)/2 creuse.
C'est un produit matrice creuse × matrice dense, exactement le type de calcul où un GPU excelle.

Sur GPU, nécessite CuPy et les bibliothèques CUDA correspondantes (cuBLAS, cuSPARSE) ; voir
LISEZMOI.md. Si CuPy est absent ou si une bibliothèque CUDA ne se charge pas, le module le détecte
à l'import (GPU = False, raison dans RAISON_SANS_GPU) et le même calcul est fait sur CPU avec SciPy.

Diagnostic rapide :  python observables_gpu.py

    from observables_gpu import dimension_spectrale
    res = dimension_spectrale(moteur, n_racines=64, s_max=2000)
"""

from __future__ import annotations

import numpy as np

import warnings

GPU = False
RAISON_SANS_GPU = ""
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    # Test réel : importer CuPy ne suffit pas, il faut que cuSPARSE et cuBLAS se chargent.
    # Un produit creux × dense en ordre Fortran sollicite les deux bibliothèques.
    _T = csp.csr_matrix(cp.eye(4, dtype=cp.float32))
    _X = cp.ones((4, 2), dtype=cp.float32)
    float((_T @ _X).sum())
    GPU = True
except Exception as exc:                       # CuPy absent, pilote ou bibliothèques CUDA manquants
    RAISON_SANS_GPU = f"{type(exc).__name__}: {exc}"
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
    P = None
    if gpu:
        try:
            Tg = csp.csr_matrix(T)
            X = cp.zeros((n, len(rac)), dtype=cp.float32, order="F")
            idx_r, idx_c = cp.asarray(rac), cp.arange(len(rac))
            X[idx_r, idx_c] = 1.0
            Pg = cp.zeros(s_max + 1, dtype=cp.float64)
            for s in range(s_max + 1):
                Pg[s] = X[idx_r, idx_c].mean()
                X = Tg @ X
            P = cp.asnumpy(Pg)
        except Exception as exc:               # repli sur CPU plutôt que d'interrompre l'expérience
            warnings.warn(f"échec du calcul GPU, repli sur CPU ({type(exc).__name__}: {exc})")
            gpu = False
    if P is None:
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


if __name__ == "__main__":
    if GPU:
        print("GPU utilisable :", cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
              "| CUDA runtime", cp.cuda.runtime.runtimeGetVersion(),
              "| pilote", cp.cuda.runtime.driverGetVersion())
    else:
        print("GPU non utilisable, calcul sur CPU. Raison :", RAISON_SANS_GPU or "CuPy non installé")
