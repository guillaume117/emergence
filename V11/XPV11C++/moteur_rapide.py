"""
Enveloppe Python du moteur C++ (moteur/moteur.cpp), par ctypes.

Les appels sont grossiers (simuler des millions d'événements, exporter des tableaux NumPy) :
le coût de l'interface est négligeable, et ctypes évite toute dépendance de compilation côté
Python (pas de pybind11 à installer). L'interface reproduit celle de modele_A.ModeleA pour
les usages courants, et peut produire un objet modele_A.Configuration afin de réutiliser
observables.py et modele_B.py sur des tailles modérées.

Exemple :
    from moteur_rapide import MoteurRapide, Parametres
    m = MoteurRapide(Parametres(n0=3000, nu=0.3, graine=0))
    issue = m.simuler(10_000_000, max_sommets=1_000_000)
    print(m.volume(r_max=60), m.contexte(d_index=1))
"""

from __future__ import annotations

import ctypes as ct
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ICI = Path(__file__).parent / "moteur"
_NOMS = {"Windows": ["moteur.dll", "Release/moteur.dll"], "Darwin": ["libmoteur.dylib"]}


def _charger():
    for nom in _NOMS.get(platform.system(), ["libmoteur.so"]):
        chemin = _ICI / nom
        if chemin.exists():
            return ct.CDLL(str(chemin))
    raise OSError(f"Bibliothèque du moteur introuvable dans {_ICI} : compilez-la d'abord "
                  "(make, ou cmake ; voir LISEZMOI.md).")


_lib = _charger()


class _Params(ct.Structure):
    _fields_ = [(n, ct.c_int32) for n in (
        "n0", "k0", "h0_germe", "k_max", "taille_exacte", "balayages_gibbs", "iterations_B", "rayon_B",
        "L_alternatif", "trace_tous_les", "journal", "ndims")] + \
        [("dims", ct.c_int32 * 4)] + \
        [(n, ct.c_double) for n in (
            "p", "rho0", "beta_J", "eta", "lambda0", "nu", "mu_carre", "mu_pendant", "delta",
            "delta_carre", "epsilon", "beta_B", "bruit_initial")] + \
        [("graine", ct.c_uint64)]


@dataclass
class Parametres:
    """Mêmes noms et mêmes valeurs par défaut que modele_A.Parametres (règles v11)."""
    n0: int = 1000
    k0: int = 3
    h0_mode: str = "configuration"      # ou "germe"
    p: float = 0.25
    rho0: float = 0.5
    beta_J: float = 0.3
    eta: float = 0.15
    k_max: int = 4
    lambda0: float = 1.0
    nu: float = 0.3
    mu_carre: float = 0.3
    mu_pendant: float = 0.0
    delta: float = 0.1
    delta_carre: float = 0.05
    L_alternatif: int = 5
    epsilon: float = 0.05
    dims: tuple = (1, 2, 3)
    beta_B: float = 50.0
    iterations_B: int = 5
    rayon_B: int = 1
    bruit_initial: float = 0.05
    taille_exacte: int = 12
    balayages_gibbs: int = 300
    trace_tous_les: int = 500
    journal: bool = True                 # désactiver pour les très grandes simulations
    graine: int = 0

    def vers_c(self) -> _Params:
        assert 0 <= len(self.dims) <= 4 and all(1 <= d <= 16 for d in self.dims)
        q = _Params()
        for nom in ("n0", "k0", "k_max", "taille_exacte", "balayages_gibbs", "iterations_B", "rayon_B",
                    "L_alternatif", "trace_tous_les"):
            setattr(q, nom, int(getattr(self, nom)))
        q.h0_germe = int(self.h0_mode == "germe")
        q.journal = int(self.journal)
        q.ndims = len(self.dims)
        for i, d in enumerate(self.dims):
            q.dims[i] = int(d)
        for nom in ("p", "rho0", "beta_J", "eta", "lambda0", "nu", "mu_carre", "mu_pendant", "delta",
                    "delta_carre", "epsilon", "beta_B", "bruit_initial"):
            setattr(q, nom, float(getattr(self, nom)))
        q.graine = int(self.graine)
        return q


# signatures
_v, _i32, _i64, _u64, _d = ct.c_void_p, ct.c_int32, ct.c_int64, ct.c_uint64, ct.c_double
_P = np.ctypeslib.ndpointer
for nom, res, args in [
    ("moteur_creer", _v, [ct.POINTER(_Params)]),
    ("moteur_detruire", None, [_v]),
    ("moteur_simuler", ct.c_int, [_v, _i64, _i64]),
    ("moteur_nb_sommets", _i64, [_v]), ("moteur_nb_aretes_total", _i64, [_v]),
    ("moteur_nb_couplages", _i64, [_v]), ("moteur_nb_evenements", _i64, [_v]),
    ("moteur_taille_trace", _i64, [_v]), ("moteur_nb_preds", _i64, [_v]),
    ("moteur_taille_ouverte", _i64, [_v]),
    ("moteur_exporter_aretes", None, [_v, _P(np.int32), _P(np.int8), _P(np.uint8)]),
    ("moteur_exporter_couplages", None, [_v, _P(np.int32), _P(np.float64)]),
    ("moteur_exporter_vecteurs", None, [_v, ct.c_int, _P(np.float64)]),
    ("moteur_exporter_trace", None, [_v, _P(np.int64)]),
    ("moteur_exporter_journal", None, [_v, _P(np.float64), _P(np.int32), _P(np.int32), _P(np.int32),
                                       _P(np.int64), _P(np.int32)]),
    ("moteur_fermeture", None, [_v, _P(np.int32), _i64, _P(np.uint8)]),
    ("moteur_reconstruire", _v, [_v, _P(np.int32), _i64]),
    ("moteur_etats_egaux", ct.c_int, [_v, _v]),
    ("obs_composantes", _i32, [_v, _P(np.int32), ct.POINTER(_i32)]),
    ("obs_volume", None, [_v, ct.c_int, ct.c_int, _u64, ct.c_int, _P(np.float64)]),
    ("obs_cyclicite", _d, [_v, ct.c_int, ct.c_int, _u64, ct.c_int]),
    ("obs_contexte", None, [_v, ct.c_int, ct.c_int, _d, ct.c_int, _u64, _P(np.float64)]),
]:
    f = getattr(_lib, nom)
    f.restype, f.argtypes = res, args

ISSUES = {0: "budget", 1: "absorbant", 2: "taille"}
CLASSES = ("frustres", "non_frustres_desequilibres", "non_frustres_equilibres", "indetermines")


class _EtatC:
    """Accès commun aux exports et aux observables, pour un moteur ou un état reconstruit."""

    def __init__(self, h, dims):
        self._h, self.dims = h, tuple(dims)

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.moteur_detruire(self._h)
            self._h = None

    # --- exports
    def aretes(self):
        n = _lib.moteur_nb_aretes_total(self._h)
        ab = np.empty(2 * n, np.int32)
        x = np.empty(n, np.int8)
        viv = np.empty(n, np.uint8)
        _lib.moteur_exporter_aretes(self._h, ab, x, viv)
        return ab.reshape(-1, 2), x, viv.astype(bool)

    def couplages(self):
        n = _lib.moteur_nb_couplages(self._h)
        ef = np.empty(2 * n, np.int32)
        val = np.empty(n, np.float64)
        _lib.moteur_exporter_couplages(self._h, ef, val)
        return ef.reshape(-1, 2), val

    def vecteurs(self, d_index: int) -> np.ndarray:
        d = self.dims[d_index]
        out = np.empty(_lib.moteur_nb_sommets(self._h) * d, np.float64)
        _lib.moteur_exporter_vecteurs(self._h, d_index, out)
        return out.reshape(-1, d)

    def configuration(self):
        """Objet modele_A.Configuration (dictionnaires) pour réutiliser observables.py et modele_B.py.
        À réserver aux tailles modérées."""
        from modele_A import Configuration
        ab, x, viv = self.aretes()
        C = Configuration()
        nV = _lib.moteur_nb_sommets(self._h)
        C.inc = {v: set() for v in range(nV)}
        for e in np.nonzero(viv)[0]:
            a, b = int(ab[e, 0]), int(ab[e, 1])
            C.aretes[int(e)] = (a, b)
            C.x[int(e)] = int(x[e])
            C.inc[a].add(int(e))
            C.inc[b].add(int(e))
        ef, val = self.couplages()
        C.rho = {(int(min(e, f)), int(max(e, f))): float(v) for (e, f), v in zip(ef, val)}
        for k, d in enumerate(self.dims):
            W = self.vecteurs(k)
            C.w[d] = {v: W[v].copy() for v in range(nV)}
        return C

    # --- observables C++
    def composantes(self):
        lab = np.empty(_lib.moteur_nb_sommets(self._h), np.int32)
        g = _i32()
        n = _lib.obs_composantes(self._h, lab, ct.byref(g))
        return n, lab, g.value

    def volume(self, n_racines=400, r_max=40, graine=0, geante=True) -> np.ndarray:
        V = np.empty(r_max + 1, np.float64)
        _lib.obs_volume(self._h, n_racines, r_max, graine, int(geante), V)
        return V

    def cyclicite(self, r=3, n_racines=400, graine=0, geante=True) -> float:
        return _lib.obs_cyclicite(self._h, n_racines, r, graine, int(geante))

    def contexte(self, d_index=1, L_max=6, tau=0.1, aleatoire=False, graine=0) -> dict:
        out = np.empty(18, np.float64)
        _lib.obs_contexte(self._h, d_index, L_max, tau, int(aleatoire), graine, out)
        res = {"d": self.dims[d_index], "tau": tau, "frustration": {}}
        for i, nom in enumerate(CLASSES):
            res["frustration"][nom] = {"cycles": int(out[3 * i]), "violations": int(out[3 * i + 1]),
                                       "fraction_violee": (out[3 * i + 1] / out[3 * i]) if out[3 * i] else float("nan"),
                                       "defaut_moyen": float(out[3 * i + 2])}
        res.update(densite_frustration=float(out[12]), fraction_violee_determines=float(out[13]),
                   nb_4cycles=int(out[14]), fraction_CHSH_viole=float(out[15]), S_max=float(out[16]),
                   fraction_sommets_defaut_positif=float(out[17]))
        return res

    def egal(self, autre: "_EtatC") -> bool:
        return bool(_lib.moteur_etats_egaux(self._h, autre._h))


class MoteurRapide(_EtatC):

    def __init__(self, P: Parametres):
        self.P = P
        self._cp = P.vers_c()
        super().__init__(_lib.moteur_creer(ct.byref(self._cp)), P.dims)
        self._journal = None

    def simuler(self, max_evenements: int, max_sommets: int | None = None) -> str:
        self._journal = None
        return ISSUES[_lib.moteur_simuler(self._h, int(max_evenements), int(max_sommets or 0))]

    @property
    def n_evenements(self) -> int:
        return _lib.moteur_nb_evenements(self._h)

    @property
    def taille_U(self) -> int:
        return _lib.moteur_taille_ouverte(self._h)

    def trace(self) -> np.ndarray:
        T = np.empty(_lib.moteur_taille_trace(self._h), np.int64)
        _lib.moteur_exporter_trace(self._h, T)
        return T.reshape(-1, 4)      # (événement, sommets actifs, arêtes, |U|)

    # --- journal et coupes admissibles
    def journal(self) -> dict:
        if not self.P.journal:
            raise RuntimeError("journal désactivé (Parametres.journal = False)")
        if self._journal is None:
            n = self.n_evenements
            J = {"t": np.empty(n), "profondeur": np.empty(n, np.int32), "graine_bloc": np.empty(n, np.int32),
                 "taille_bloc": np.empty(n, np.int32), "pred_off": np.empty(n + 1, np.int64),
                 "preds": np.empty(_lib.moteur_nb_preds(self._h), np.int32)}
            _lib.moteur_exporter_journal(self._h, J["t"], J["profondeur"], J["graine_bloc"],
                                         J["taille_bloc"], J["pred_off"], J["preds"])
            self._journal = J
        return self._journal

    def fermeture_passe(self, indices) -> np.ndarray:
        ids = np.ascontiguousarray(indices, np.int32)
        m = np.empty(self.n_evenements, np.uint8)
        _lib.moteur_fermeture(self._h, ids, len(ids), m)
        return np.nonzero(m)[0].astype(np.int32)

    def est_fermee(self, indices) -> bool:
        return len(self.fermeture_passe(indices)) == len(np.unique(indices))

    def coupe_par_temps(self, n: int) -> np.ndarray:
        return np.arange(n, dtype=np.int32)

    def coupe_par_profondeur(self, n_cible: int) -> np.ndarray:
        prof = self.journal()["profondeur"]
        D = np.sort(prof)[min(n_cible, len(prof)) - 1]
        return np.nonzero(prof <= D)[0].astype(np.int32)

    def coupe_biaisee(self, region_aretes, frac_region=0.9, frac_reste=0.25) -> np.ndarray:
        """Comme modele_A.coupe_biaisee, l'événement étant localisé par l'hyperarête qui a
        engendré son bloc (graine_bloc)."""
        J = self.journal()
        n = len(J["t"])
        i = np.arange(n)
        dans = np.isin(J["graine_bloc"], np.fromiter(region_aretes, np.int32))
        graines = np.nonzero((i < frac_reste * n) | ((i < frac_region * n) & dans))[0]
        return self.fermeture_passe(graines)

    def reconstruire(self, indices) -> _EtatC:
        ids = np.ascontiguousarray(indices, np.int32)
        return _EtatC(_lib.moteur_reconstruire(self._h, ids, len(ids)), self.dims)
