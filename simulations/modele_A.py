"""
Modèle jouet A — réécriture probabiliste sur hypergraphe (section 23 du programme v10).

Implémente fidèlement :
  * blocs de couplage probabiliste Λ (composantes des hyperarêtes ouvertes reliées par ρ > 0) ;
  * spécification de Gibbs locale γ_Λ, normalisée sur le bloc et conditionnée par ∂_C Λ ;
  * règles R1 (renforcement), R2 (réouverture), R3 (croissance clairsemée),
    R4 (fermeture de cycles) appliquées dans cet ordre à l'intérieur d'un événement composite a_Λ ;
  * horloges locales sans mémoire de taux λ0 |Λ| (taux additifs) ;
  * couche contextuelle du modèle B : vecteurs unitaires w_x ∈ S^{d-1} et règle R5,
    pour plusieurs dimensions d simultanément (d = 1 sert de contrôle classique).

Chaque événement est journalisé avec son empreinte et la liste des écritures qu'il effectue.
Cela permet de reconstruire la configuration sur n'importe quelle coupe admissible
(ensemble d'événements fermé vers le passé pour la relation de dépendance par empreintes),
et donc de tester l'invariance de section (critère 3, expérience F2).

Toutes les étapes internes d'un événement sont écrites de façon à ne dépendre d'aucun ordre
de parcours : les tirages sont faits sur la configuration avant la règle, puis appliqués
simultanément ; les troncatures de degré sont faites par sous-ensemble uniforme.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------------

@dataclass
class Parametres:
    # état initial
    n0: int = 1000            # nombre de sommets de H0
    k0: int = 3               # degré des sommets de H0 (modèle de configuration)
    rang0: int = 2            # rang des hyperarêtes de H0
    p: float = 0.25           # probabilité de couplage initial (sous-critique si p (K-1) < 1)
    rho0: float = 0.5         # couplage initial et couplage des hyperarêtes créées
    # dynamique
    beta_J: float = 1.0       # βJ de la spécification de Gibbs (et de R2)
    eta: float = 0.15         # pas de renforcement R1
    mu: float = 0.15          # probabilité de croissance R3
    nu: float = 0.3           # probabilité de fermeture de cycle R4
    k_max: int = 4            # degré maximal des sommets
    lambda0: float = 1.0      # taux par degré de possibilité (horloges)
    # couche contextuelle (modèle B)
    dims: tuple = (1, 2, 3)   # dimensions d des couches de vecteurs (0 : aucune)
    kappa: float = 0.2        # pas d'alignement R5 (0 < κ < 1)
    # numérique
    taille_exacte: int = 12   # énumération exacte de γ_Λ jusqu'à cette taille de bloc
    balayages_gibbs: int = 300
    graine: int = 0


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

def cle(e: int, f: int) -> tuple:
    return (e, f) if e < f else (f, e)


class Configuration:
    """Configuration de calcul C = (H, U, x, ρ) et couches de vecteurs w^(d)."""

    def __init__(self):
        self.aretes: dict[int, tuple] = {}      # eid -> sommets
        self.inc: dict[int, set] = {}           # sommet -> eids
        self.x: dict[int, int] = {}             # eid -> -1, +1, ou 0 (ouverte)
        self.rho: dict[tuple, float] = {}       # (e, f) incidentes -> couplage
        self.w: dict[int, dict[int, np.ndarray]] = {}  # d -> (sommet -> vecteur)

    def copie(self) -> "Configuration":
        c = Configuration()
        c.aretes = dict(self.aretes)
        c.inc = {v: set(s) for v, s in self.inc.items()}
        c.x = dict(self.x)
        c.rho = dict(self.rho)
        c.w = {d: {v: vec.copy() for v, vec in lay.items()} for d, lay in self.w.items()}
        return c

    # --- accès élémentaires
    def voisines(self, e: int) -> set:
        """N(e) : hyperarêtes partageant au moins un sommet avec e."""
        s = set()
        for v in self.aretes[e]:
            s |= self.inc[v]
        s.discard(e)
        return s

    def r(self, e: int, f: int) -> float:
        return self.rho.get(cle(e, f), 0.0)

    def degre(self, v: int) -> int:
        return len(self.inc[v])

    def ouvertes(self) -> list:
        return [e for e, xe in self.x.items() if xe == 0]

    def ajouter_arete(self, eid: int, sommets: tuple):
        self.aretes[eid] = sommets
        self.x[eid] = 0
        for v in sommets:
            self.inc.setdefault(v, set()).add(eid)

    def bloc(self, e0: int) -> set:
        """Composante de e0 dans le graphe de couplage des hyperarêtes ouvertes."""
        lam, pile = {e0}, [e0]
        while pile:
            e = pile.pop()
            for f in self.voisines(e):
                if f not in lam and self.x[f] == 0 and self.r(e, f) > 0:
                    lam.add(f)
                    pile.append(f)
        return lam


# ----------------------------------------------------------------------------
# Journal d'événements
# ----------------------------------------------------------------------------

@dataclass
class Evenement:
    indice: int
    t: float                         # paramètre auxiliaire du processus (non physique)
    taille_bloc: int
    empreinte: frozenset             # hyperarêtes lues ou écrites
    predecesseurs: tuple             # événements immédiatement antérieurs dans ≺ (par empreinte)
    profondeur: int                  # longueur de la plus longue chaîne de dépendances finissant ici
    ecritures: list = field(default_factory=list)  # ("x"|"rho"|"arete"|"w", clé, valeur)


# ----------------------------------------------------------------------------
# Dynamique
# ----------------------------------------------------------------------------

class ModeleA:

    def __init__(self, P: Parametres):
        self.P = P
        self.rng = random.Random(P.graine)
        self.nprng = np.random.default_rng(P.graine)
        self.C = Configuration()
        self.prochain_sommet = 0
        self.prochaine_arete = 0
        self.t = 0.0
        self.journal: list[Evenement] = []
        self.dernier_contact: dict[int, int] = {}   # eid -> dernier événement l'ayant touchée
        self._etat_initial()
        self.C0 = self.C.copie()                    # pour la reconstruction sur coupes
        # index des hyperarêtes ouvertes (tirage uniforme en O(1))
        self.U: list[int] = self.C.ouvertes()
        self.posU = {e: i for i, e in enumerate(self.U)}

    # --- état initial homogène (section 23.10)
    def _vecteur_aleatoire(self, d: int) -> np.ndarray:
        if d == 1:
            return np.array([self.rng.choice((-1.0, 1.0))])
        v = self.nprng.normal(size=d)
        return v / np.linalg.norm(v)

    def _nouveau_sommet(self) -> int:
        v = self.prochain_sommet
        self.prochain_sommet += 1
        self.C.inc[v] = set()
        for d in self.P.dims:
            self.C.w.setdefault(d, {})[v] = self._vecteur_aleatoire(d)
        return v

    def _etat_initial(self):
        P = self.P
        for d in P.dims:
            self.C.w[d] = {}
        sommets = [self._nouveau_sommet() for _ in range(P.n0)]
        # modèle de configuration : demi-arêtes appariées par paquets de rang0,
        # en rejetant les hyperarêtes dégénérées (sommet répété) ou dupliquées
        demi = [v for v in sommets for _ in range(P.k0)]
        self.rng.shuffle(demi)
        vues = set()
        for i in range(0, len(demi) - P.rang0 + 1, P.rang0):
            groupe = tuple(sorted(demi[i:i + P.rang0]))
            if len(set(groupe)) < P.rang0 or groupe in vues:
                continue
            vues.add(groupe)
            self.C.ajouter_arete(self.prochaine_arete, groupe)
            self.prochaine_arete += 1
        # couplages initiaux de Bernoulli entre hyperarêtes incidentes
        for e in list(self.C.aretes):
            for f in self.C.voisines(e):
                if e < f and self.rng.random() < P.p:
                    self.C.rho[(e, f)] = P.rho0

    # --- index U
    def _ouvrir(self, e):
        if e not in self.posU:
            self.posU[e] = len(self.U)
            self.U.append(e)

    def _fermer(self, e):
        i = self.posU.pop(e, None)
        if i is None:
            return
        dernier = self.U.pop()
        if dernier != e:
            self.U[i] = dernier
            self.posU[dernier] = i

    # --- spécification de Gibbs locale γ_Λ (sections 23.3–23.5)
    def _tirer_bloc(self, lam: list, bord: list) -> dict:
        C, bJ = self.C, self.P.beta_J
        n = len(lam)
        idx = {e: i for i, e in enumerate(lam)}
        J = np.zeros((n, n))
        h = np.zeros(n)
        for i, e in enumerate(lam):
            for f in C.voisines(e):
                if f in idx:
                    J[i, idx[f]] = C.r(e, f)
                elif f in bord:
                    h[i] += C.r(e, f) * C.x[f]
        if n <= self.P.taille_exacte:
            confs = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=float)
            energie = -0.5 * np.einsum("ki,ij,kj->k", confs, J, confs) - confs @ h
            poids = np.exp(-bJ * (energie - energie.min()))
            k = self.nprng.choice(len(confs), p=poids / poids.sum())
            sigma = confs[k]
        else:
            # échantillonneur de Gibbs : l'ordre des mises à jour est purement computationnel
            sigma = self.nprng.choice((-1.0, 1.0), size=n)
            for _ in range(self.P.balayages_gibbs):
                for i in self.nprng.permutation(n):
                    champ = J[i] @ sigma + h[i]
                    sigma[i] = 1.0 if self.nprng.random() < 1 / (1 + math.exp(-2 * bJ * champ)) else -1.0
        return {e: int(sigma[i]) for i, e in enumerate(lam)}

    # --- un événement composite a_Λ
    def pas(self) -> bool:
        """Réalise un événement. Renvoie False si l'état est absorbant (U vide)."""
        C, P, rng = self.C, self.P, self.rng
        if not self.U:
            return False
        # horloges exponentielles de taux λ0|Λ| : le taux total vaut λ0|U|, et le bloc
        # réalisé est celui d'une hyperarête ouverte tirée uniformément
        self.t += rng.expovariate(P.lambda0 * len(self.U))
        e0 = self.U[rng.randrange(len(self.U))]

        lam = C.bloc(e0)
        N_lam = set().union(*(C.voisines(e) for e in lam)) - lam
        bord = {f for f in N_lam if C.x[f] != 0 and any(C.r(e, f) > 0 for e in lam)}
        empreinte = set(lam) | N_lam
        for f in bord:
            empreinte |= C.voisines(f)
        ecr = []

        def ecrire_x(e, val):
            C.x[e] = val
            ecr.append(("x", e, val))
            (self._ouvrir if val == 0 else self._fermer)(e)

        def ecrire_rho(k, val):
            if val > 0:
                C.rho[k] = val
            else:
                C.rho.pop(k, None)
            ecr.append(("rho", k, val))

        # tirage conjoint du bloc
        lam_l = sorted(lam)
        sigma = self._tirer_bloc(lam_l, bord)
        for e, s in sigma.items():
            ecrire_x(e, s)

        # R1 — renforcement / affaiblissement (paires bloc–bloc et bloc–voisinage actualisé)
        paires = set()
        for e in lam:
            for f in C.voisines(e):
                if C.x[f] != 0:
                    paires.add(cle(e, f))
        for (e, f) in paires:
            nv = min(1.0, max(0.0, C.r(e, f) + P.eta * C.x[e] * C.x[f]))
            if nv != C.r(e, f):
                ecrire_rho((e, f), nv)

        # R2 — réouverture par instabilité locale (tirages simultanés)
        a_rouvrir = []
        for e in lam | bord:
            he = sum(C.r(e, f) * C.x[f] for f in C.voisines(e) if C.x[f] != 0)
            re = 1.0 / (1.0 + math.exp(2 * P.beta_J * C.x[e] * he))
            if rng.random() < re:
                a_rouvrir.append(e)
        for e in a_rouvrir:
            ecrire_x(e, 0)

        nouvelles = []

        def creer(sommets, couples_a):
            eid = self.prochaine_arete
            self.prochaine_arete += 1
            C.ajouter_arete(eid, sommets)
            self._ouvrir(eid)
            ecr.append(("arete", eid, sommets))
            for g in couples_a:
                ecrire_rho(cle(eid, g), P.rho0)
            nouvelles.append(eid)
            return eid

        # R3 — croissance clairsemée (tirages sur la configuration avant R3, troncature uniforme)
        candidats = {}
        for e in lam:
            if C.x[e] != 0:
                for v in C.aretes[e]:
                    if C.degre(v) < P.k_max and rng.random() < P.mu:
                        candidats.setdefault(v, []).append(e)
        for v, origines in candidats.items():
            place = P.k_max - C.degre(v)
            if len(origines) > place:
                origines = rng.sample(origines, place)
            for e in origines:
                vs = self._nouveau_sommet()
                ecr.append(("sommet", vs, {d: C.w[d][vs].copy() for d in P.dims}))
                creer(tuple(sorted((v, vs))), [e])

        # R4 — fermeture relationnelle de cycles
        generateurs = {}
        for e in lam:
            if C.x[e] == 0:
                continue
            fg = [f for f in bord & C.voisines(e) if C.x[f] == C.x[e]]
            for f, g in itertools.combinations(fg, 2):
                for u in set(C.aretes[f]) - set(C.aretes[e]):
                    for w in set(C.aretes[g]) - set(C.aretes[e]):
                        if u != w and not (C.inc[u] & C.inc[w]):
                            generateurs.setdefault(tuple(sorted((u, w))), set()).update((f, g))
        retenues = [pr for pr in generateurs if rng.random() < P.nu]
        rng.shuffle(retenues)  # ordre uniforme : troncature de degré par sous-ensemble uniforme
        for (u, w) in retenues:
            if C.degre(u) < P.k_max and C.degre(w) < P.k_max and not (C.inc[u] & C.inc[w]):
                g_couples = [g for g in generateurs[(u, w)] if u in C.aretes[g] or w in C.aretes[g]]
                creer((u, w), g_couples)

        # R5 — alignement contextuel (modèle B), simultané, équivariant sous O(d)
        for d in P.dims:
            lay = C.w[d]
            contributions = {}
            for k in lam:
                if C.x[k] != 0 and len(C.aretes[k]) == 2:
                    u, v = C.aretes[k]
                    contributions.setdefault(u, []).append(C.x[k] * lay[v])
                    contributions.setdefault(v, []).append(C.x[k] * lay[u])
            nouveaux = {}
            for u, termes in contributions.items():
                vec = lay[u] + P.kappa * np.sum(termes, axis=0)
                nrm = np.linalg.norm(vec)
                if nrm > 1e-12:
                    nouveaux[u] = vec / nrm
            for u, vec in nouveaux.items():
                lay[u] = vec
                ecr.append(("w", (d, u), vec.copy()))

        # journal : dépendance par empreintes (lectures et écritures)
        empreinte |= set(nouvelles)
        preds = {self.dernier_contact[e] for e in empreinte if e in self.dernier_contact}
        prof = 1 + max((self.journal[i].profondeur for i in preds), default=0)
        i = len(self.journal)
        for e in empreinte:
            self.dernier_contact[e] = i
        self.journal.append(Evenement(i, self.t, len(lam), frozenset(empreinte),
                                      tuple(sorted(preds)), prof, ecr))
        return True

    # --- boucle
    def simuler(self, max_evenements: int, max_sommets: int | None = None) -> str:
        for _ in range(max_evenements):
            if max_sommets and self.prochain_sommet >= max_sommets:
                return "taille"
            if not self.pas():
                return "absorbant"
        return "budget"

    # --- reconstruction de la configuration sur une coupe admissible
    def configuration_sur(self, indices) -> Configuration:
        """Applique, dans l'ordre d'origine, les écritures d'un ensemble d'événements
        fermé vers le passé. L'ordre d'origine restreint à cet ensemble est une extension
        linéaire de l'ordre de dépendance ; le résultat ne dépend donc que de l'ensemble."""
        C = self.C0.copie()
        for i in sorted(indices):
            for (typ, k, val) in self.journal[i].ecritures:
                if typ == "x":
                    C.x[k] = val
                elif typ == "rho":
                    if val > 0:
                        C.rho[k] = val
                    else:
                        C.rho.pop(k, None)
                elif typ == "sommet":
                    C.inc.setdefault(k, set())
                    for d, vec in val.items():
                        C.w[d][k] = vec.copy()
                elif typ == "arete":
                    C.ajouter_arete(k, val)
                elif typ == "w":
                    d, u = k
                    C.w[d][u] = val.copy()
        return C

    def coupe_par_temps(self, n_evenements: int) -> list:
        """Coupe définie par le paramètre auxiliaire t : préfixe du journal (feuilletage computationnel)."""
        return list(range(n_evenements))

    def coupe_par_profondeur(self, n_cible: int) -> list:
        """Coupe définie par l'ordre dérivé : tous les événements de profondeur ≤ D, D choisi
        pour que la coupe contienne environ n_cible événements. Fermée vers le passé car
        tout prédécesseur a une profondeur strictement inférieure."""
        profs = sorted(ev.profondeur for ev in self.journal)
        D = profs[min(n_cible, len(profs)) - 1]
        return [ev.indice for ev in self.journal if ev.profondeur <= D]

    def fermeture_passe(self, indices) -> set:
        """Plus petit ensemble fermé vers le passé contenant les indices donnés."""
        s, pile = set(indices), list(indices)
        while pile:
            i = pile.pop()
            for p in self.journal[i].predecesseurs:
                if p not in s:
                    s.add(p)
                    pile.append(p)
        return s

    def coupe_biaisee(self, region: set, frac_region=0.9, frac_reste=0.25) -> list:
        """Coupe admissible fortement non simultanée : la région (ensemble d'hyperarêtes de H0)
        est avancée jusqu'à frac_region du journal, le reste seulement jusqu'à frac_reste.
        Elle est définie comme la fermeture vers le passé des événements retenus."""
        n = len(self.journal)
        graines = [ev.indice for ev in self.journal
                   if ev.indice < frac_reste * n
                   or (ev.indice < frac_region * n and ev.empreinte & region)]
        return sorted(self.fermeture_passe(graines))

    def est_fermee(self, indices) -> bool:
        s = set(indices)
        return all(p in s for i in s for p in self.journal[i].predecesseurs)
