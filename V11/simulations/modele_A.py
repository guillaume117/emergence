"""
Modèle jouet A — version révisée v11 (voir corrections_modeles_v11.tex).

Trois corrections par rapport à la v10 :

  C1. Dépendance par lectures et écritures. Chaque événement déclare explicitement ce qu'il lit
      (R_e) et ce qu'il écrit (W_e), sur des items fins : ("x", e) statut et issue d'une hyperarête,
      ("r", (e, f)) couplage, ("inc", v) incidence d'un sommet, ("w", v) vecteurs de la couche B.
      Deux événements ne sont dépendants que si l'un écrit ce que l'autre lit ou écrit ;
      deux lectures communes ne créent plus de dépendance.

  C2. Croissance par complétion de carrés (R3'), fermeture limitée aux 4-cycles (R4') et
      suppression des arêtes hors de tout carré (R6). La croissance pendante n'est plus qu'un
      événement rare (mu_pendant). Avec un H0 biparti, la structure reste bipartie.

  C3. Couche contextuelle relaxée (R5'). Les vecteurs partent d'un état presque aligné (classique)
      et évoluent par échantillonnage de Gibbs local du modèle O(d)
          H_B = - sum_k x_k <w_u, w_v>,
      dont les couplages sont les issues du modèle A. La contextualité ne peut alors apparaître que
      sur les cycles frustrés de A (nombre impair de relations discordantes).

Les règles v10 (R3, R4, R5) restent disponibles avec Parametres(regles="v10") pour comparaison.
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
    regles: str = "v11"        # "v11" (corrigé) ou "v10" (version d'origine)
    # état initial
    n0: int = 1000             # nombre de sommets de H0
    k0: int = 3                # degré des sommets de H0
    rang0: int = 2             # rang des hyperarêtes de H0 (v11 : 2 obligatoirement)
    h0_biparti: bool = True    # v11 : H0 biparti (deux classes de n0/2 sommets)
    h0_mode: str = "configuration"  # v11 : "configuration" (H0 aléatoire biparti) ou "germe"
                                    #       (un seul 4-cycle, croissance depuis un germe connexe)
    p: float = 0.25            # couplage initial (sous-critique si p (K-1) < 1)
    rho0: float = 0.5
    # dynamique de A
    beta_J: float = 0.3
    eta: float = 0.15
    k_max: int = 4             # v11 : degré cible du complexe de carrés
    lambda0: float = 1.0
    # v10
    mu: float = 0.5            # croissance pendante R3 (v10)
    nu: float = 0.3            # fermeture R4 (v10) ou R4' (v11)
    # v11
    mu_carre: float = 0.3      # complétion de carrés R3'
    mu_pendant: float = 0.0    # croissance pendante résiduelle (v11)
    delta: float = 0.1         # suppression R6 des arêtes hors de tout carré
    delta_carre: float = 0.05  # suppression R6 résiduelle des arêtes prises dans un carré (renouvellement)
    L_alternatif: int = 5      # R6 : une arête n'est supprimée que si ses extrémités restent reliées
                               #      par un autre chemin de longueur ≤ L (jamais de pont : pas de
                               #      fragmentation)
    epsilon: float = 0.05      # plancher de réouverture R2 : r_e' = ε + (1 - ε) r_e (v11)
    # couche contextuelle (modèle B)
    dims: tuple = (1, 2, 3)
    kappa: float = 0.2         # v10 : pas de la montée de gradient R5
    beta_B: float = 50.0       # v11 : inverse de température de R5' (élevé : proche de l'état fondamental ;
                               #       à β_B ≈ 4 le bruit thermique seul produit des violations)
    iterations_B: int = 5      # v11 : balayages de Gibbs par événement
    rayon_B: int = 1           # v11 : rayon de relaxation autour des sommets du bloc (0 : sommets du bloc)
    bruit_initial: float = 0.05  # v11 : écart à l'alignement initial
    # numérique
    taille_exacte: int = 12
    balayages_gibbs: int = 300
    trace_tous_les: int = 500  # fréquence d'enregistrement de la taille (stationnarité)
    graine: int = 0


def cle(e: int, f: int) -> tuple:
    return (e, f) if e < f else (f, e)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

class Configuration:
    """Configuration de calcul C = (H, U, x, ρ) et couches de vecteurs w^(d)."""

    def __init__(self):
        self.aretes: dict[int, tuple] = {}
        self.inc: dict[int, set] = {}
        self.x: dict[int, int] = {}
        self.rho: dict[tuple, float] = {}
        self.w: dict[int, dict[int, np.ndarray]] = {}

    def copie(self) -> "Configuration":
        c = Configuration()
        c.aretes = dict(self.aretes)
        c.inc = {v: set(s) for v, s in self.inc.items()}
        c.x = dict(self.x)
        c.rho = dict(self.rho)
        c.w = {d: {v: vec.copy() for v, vec in lay.items()} for d, lay in self.w.items()}
        return c

    def voisines(self, e: int) -> set:
        s = set()
        for v in self.aretes[e]:
            s |= self.inc[v]
        s.discard(e)
        return s

    def voisins(self, v: int) -> set:
        """Voisins de v dans le graphe primal."""
        s = set()
        for e in self.inc[v]:
            s.update(self.aretes[e])
        s.discard(v)
        return s

    def adjacents(self, u: int, v: int) -> bool:
        return bool(self.inc[u] & self.inc[v])

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

    def supprimer_arete(self, eid: int):
        for f in self.voisines(eid):
            self.rho.pop(cle(eid, f), None)
        for v in self.aretes[eid]:
            self.inc[v].discard(eid)
        del self.aretes[eid]
        del self.x[eid]

    def bloc(self, e0: int) -> set:
        lam, pile = {e0}, [e0]
        while pile:
            e = pile.pop()
            for f in self.voisines(e):
                if f not in lam and self.x[f] == 0 and self.r(e, f) > 0:
                    lam.add(f)
                    pile.append(f)
        return lam

    def dans_un_carre(self, e: int) -> bool:
        """L'arête de rang 2 e = {a, b} appartient-elle à un 4-cycle a-b-d-c ?"""
        a, b = self.aretes[e]
        Nb = self.voisins(b) - {a}
        for c in self.voisins(a) - {b}:
            if self.voisins(c) & Nb:
                return True
        return False


# ----------------------------------------------------------------------------
# Journal
# ----------------------------------------------------------------------------

@dataclass
class Evenement:
    indice: int
    t: float
    taille_bloc: int
    lectures: frozenset
    ecritures_items: frozenset
    predecesseurs: tuple
    profondeur: int
    ecritures: list = field(default_factory=list)

    @property
    def empreinte(self) -> frozenset:
        """Hyperarêtes touchées (compatibilité avec la coupe biaisée par région)."""
        return frozenset(k[1] for k in self.lectures | self.ecritures_items if k[0] == "x")


# ----------------------------------------------------------------------------
# Échantillonnage de von Mises–Fisher (relaxation R5')
# ----------------------------------------------------------------------------

def tirer_vmf(mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Tire un vecteur unitaire de densité ∝ exp(kappa <w, mu>) sur S^{d-1} (algorithme de Wood)."""
    d = len(mu)
    if d == 1:
        return np.array([1.0 if rng.random() < 1 / (1 + math.exp(-2 * kappa * mu[0])) else -1.0])
    if kappa < 1e-9:
        v = rng.normal(size=d)
        return v / np.linalg.norm(v)
    b = (-2 * kappa + math.sqrt(4 * kappa ** 2 + (d - 1) ** 2)) / (d - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * math.log(1 - x0 ** 2)
    while True:
        z = rng.beta((d - 1) / 2, (d - 1) / 2)
        W = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        if kappa * W + (d - 1) * math.log(1 - x0 * W) - c >= math.log(rng.random()):
            break
    v = rng.normal(size=d)
    v -= v.dot(mu) * mu
    v /= np.linalg.norm(v)
    return W * mu + math.sqrt(max(0.0, 1 - W ** 2)) * v


# ----------------------------------------------------------------------------
# Dynamique
# ----------------------------------------------------------------------------

class ModeleA:

    def __init__(self, P: Parametres):
        if P.regles == "v11":
            assert P.rang0 == 2, "la version v11 travaille sur des contextes de rang 2"
        self.P = P
        self.rng = random.Random(P.graine)
        self.nprng = np.random.default_rng(P.graine)
        self.C = Configuration()
        self.prochain_sommet = 0
        self.prochaine_arete = 0
        self.t = 0.0
        self.journal: list[Evenement] = []
        self.dernier_ecrivain: dict = {}     # item -> dernier événement l'ayant écrit
        self.lecteurs: dict = {}             # item -> événements l'ayant lu depuis cette écriture
        self.trace: list = []                # (événements, sommets actifs, hyperarêtes, |U|)
        self._etat_initial()
        self.C0 = self.C.copie()
        self.U: list[int] = self.C.ouvertes()
        self.posU = {e: i for i, e in enumerate(self.U)}

    # --- vecteurs de la couche B
    def _vecteur_initial(self, d: int) -> np.ndarray:
        if self.P.regles == "v10":            # v10 : vecteurs aléatoires uniformes
            if d == 1:
                return np.array([self.rng.choice((-1.0, 1.0))])
            v = self.nprng.normal(size=d)
            return v / np.linalg.norm(v)
        # v11 : état presque aligné, donc classique
        if d == 1:
            return np.array([1.0])
        v = np.zeros(d)
        v[0] = 1.0
        v += self.P.bruit_initial * self.nprng.normal(size=d)
        return v / np.linalg.norm(v)

    def _nouveau_sommet(self, parents=()) -> int:
        v = self.prochain_sommet
        self.prochain_sommet += 1
        self.C.inc[v] = set()
        for d in self.P.dims:
            lay = self.C.w.setdefault(d, {})
            if self.P.regles == "v11" and parents:
                # v11 : le nouveau sommet hérite de la moyenne de ses parents, légèrement bruitée
                m = np.mean([lay[u] for u in parents], axis=0)
                if d > 1:
                    m = m + self.P.bruit_initial * self.nprng.normal(size=d)
                n = np.linalg.norm(m)
                lay[v] = m / n if n > 1e-12 else self._vecteur_initial(d)
                if d == 1:
                    lay[v] = np.array([1.0 if m[0] >= 0 else -1.0])
            else:
                lay[v] = self._vecteur_initial(d)
        return v

    # --- état initial
    def _etat_initial(self):
        P = self.P
        for d in P.dims:
            self.C.w[d] = {}
        sommets = [self._nouveau_sommet() for _ in range(P.n0)]
        vues = set()
        if P.regles == "v11" and P.h0_mode == "germe":
            # germe connexe minimal : un 4-cycle sur les quatre premiers sommets (les autres
            # sommets créés restent isolés et sont ignorés par les observables)
            a, b, c, d = sommets[:4]
            paires = [(a, b), (b, c), (c, d), (d, a)]
        elif P.regles == "v11" and P.h0_biparti:
            gauche, droite = sommets[: P.n0 // 2], sommets[P.n0 // 2:]
            dg = [v for v in gauche for _ in range(P.k0)]
            dd = [v for v in droite for _ in range(P.k0)]
            self.rng.shuffle(dg)
            self.rng.shuffle(dd)
            paires = zip(dg, dd)
        else:
            demi = [v for v in sommets for _ in range(P.k0)]
            self.rng.shuffle(demi)
            paires = (tuple(demi[i:i + P.rang0]) for i in range(0, len(demi) - P.rang0 + 1, P.rang0))
        for groupe in paires:
            groupe = tuple(sorted(groupe))
            if len(set(groupe)) < len(groupe) or groupe in vues:
                continue
            vues.add(groupe)
            self.C.ajouter_arete(self.prochaine_arete, groupe)
            self.prochaine_arete += 1
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

    # --- spécification de Gibbs locale
    def _tirer_bloc(self, lam: list, bord: set) -> dict:
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
            sigma = confs[self.nprng.choice(len(confs), p=poids / poids.sum())]
        else:
            sigma = self.nprng.choice((-1.0, 1.0), size=n)
            for _ in range(self.P.balayages_gibbs):
                for i in self.nprng.permutation(n):
                    champ = J[i] @ sigma + h[i]
                    sigma[i] = 1.0 if self.nprng.random() < 1 / (1 + math.exp(-2 * bJ * champ)) else -1.0
        return {e: int(sigma[i]) for i, e in enumerate(lam)}

    # ------------------------------------------------------------------------
    # Un événement composite a_Λ
    # ------------------------------------------------------------------------
    def pas(self) -> bool:
        C, P, rng = self.C, self.P, self.rng
        if not self.U:
            return False
        self.t += rng.expovariate(P.lambda0 * len(self.U))
        e0 = self.U[rng.randrange(len(self.U))]

        R, W = set(), set()          # items lus et écrits (C1)
        ecr = []                     # écritures journalisées pour la reconstruction

        def lire_arete(e):
            R.add(("x", e))
            for v in C.aretes[e]:
                R.add(("inc", v))

        def lire_voisinage(e):
            lire_arete(e)
            for f in C.voisines(e):
                R.add(("x", f))
                R.add(("r", cle(e, f)))

        def ecrire_x(e, val):
            C.x[e] = val
            W.add(("x", e))
            ecr.append(("x", e, val))
            (self._ouvrir if val == 0 else self._fermer)(e)

        def ecrire_rho(k, val):
            if val > 0:
                C.rho[k] = val
            else:
                C.rho.pop(k, None)
            W.add(("r", k))
            ecr.append(("rho", k, val))

        # bloc, frontière
        lam = C.bloc(e0)
        for e in lam:
            lire_voisinage(e)
        N_lam = set().union(*(C.voisines(e) for e in lam)) - lam
        bord = {f for f in N_lam if C.x[f] != 0 and any(C.r(e, f) > 0 for e in lam)}

        # tirage conjoint
        sigma = self._tirer_bloc(sorted(lam), bord)
        for e, s in sigma.items():
            ecrire_x(e, s)

        # R1
        paires = {cle(e, f) for e in lam for f in C.voisines(e) if C.x[f] != 0}
        for (e, f) in paires:
            nv = min(1.0, max(0.0, C.r(e, f) + P.eta * C.x[e] * C.x[f]))
            if nv != C.r(e, f):
                ecrire_rho((e, f), nv)

        # R2
        a_rouvrir = []
        for e in lam | bord:
            lire_voisinage(e)
            he = sum(C.r(e, f) * C.x[f] for f in C.voisines(e) if C.x[f] != 0)
            re = 1.0 / (1.0 + math.exp(2 * P.beta_J * C.x[e] * he))
            if P.regles == "v11":
                re = P.epsilon + (1 - P.epsilon) * re
            if rng.random() < re:
                a_rouvrir.append(e)
        for e in a_rouvrir:
            ecrire_x(e, 0)

        nouvelles = []

        def creer_arete(sommets, couples_a):
            eid = self.prochaine_arete
            self.prochaine_arete += 1
            C.ajouter_arete(eid, sommets)
            self._ouvrir(eid)
            ecr.append(("arete", eid, sommets))
            W.add(("x", eid))
            for v in sommets:
                W.add(("inc", v))
            for g in couples_a:
                if g in C.aretes:
                    ecrire_rho(cle(eid, g), P.rho0)
            nouvelles.append(eid)
            return eid

        def creer_sommet(parents):
            vs = self._nouveau_sommet(parents)
            W.add(("inc", vs))
            for d in P.dims:
                W.add(("w", vs))
            ecr.append(("sommet", vs, {d: C.w[d][vs].copy() for d in P.dims}))
            return vs

        if P.regles == "v10":
            self._croissance_v10(lam, bord, creer_arete, creer_sommet, R)
        else:
            self._croissance_v11(lam, bord, creer_arete, creer_sommet, R, W, ecr)

        # couche contextuelle
        if P.regles == "v10":
            self._r5_v10(lam, R, W, ecr)
        else:
            self._r5_v11(lam, R, W, ecr)

        # journal : dépendance lecture/écriture (C1)
        preds = set()
        for it in R | W:
            if it in self.dernier_ecrivain:
                preds.add(self.dernier_ecrivain[it])
        for it in W:
            preds |= self.lecteurs.get(it, set())
        i = len(self.journal)
        preds.discard(i)
        for it in W:
            self.dernier_ecrivain[it] = i
            self.lecteurs[it] = set()
        for it in R - W:
            self.lecteurs.setdefault(it, set()).add(i)
        prof = 1 + max((self.journal[j].profondeur for j in preds), default=0)
        self.journal.append(Evenement(i, self.t, len(lam), frozenset(R), frozenset(W),
                                      tuple(sorted(preds)), prof, ecr))
        if i % P.trace_tous_les == 0:
            actifs = sum(1 for s in C.inc.values() if s)
            self.trace.append((i, actifs, len(C.aretes), len(self.U)))
        return True

    # --- croissance v10 (R3 pendante, R4 triangles et carrés)
    def _croissance_v10(self, lam, bord, creer_arete, creer_sommet, R):
        C, P, rng = self.C, self.P, self.rng
        for f in bord:
            for g in C.voisines(f):
                R.add(("x", g))
                for v in C.aretes[g]:
                    R.add(("inc", v))
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
                vs = creer_sommet((v,))
                creer_arete(tuple(sorted((v, vs))), [e])
        generateurs = {}
        for e in lam:
            if C.x[e] == 0:
                continue
            fg = [f for f in bord & C.voisines(e) if C.x[f] == C.x[e]]
            for f, g in itertools.combinations(fg, 2):
                for u in set(C.aretes[f]) - set(C.aretes[e]):
                    for w in set(C.aretes[g]) - set(C.aretes[e]):
                        if u != w and not C.adjacents(u, w):
                            generateurs.setdefault(tuple(sorted((u, w))), set()).update((f, g))
        retenues = [pr for pr in generateurs if rng.random() < P.nu]
        rng.shuffle(retenues)
        for (u, w) in retenues:
            if C.degre(u) < P.k_max and C.degre(w) < P.k_max and not C.adjacents(u, w):
                creer_arete((u, w), [g for g in generateurs[(u, w)]
                                     if u in C.aretes[g] or w in C.aretes[g]])

    # --- croissance v11 (C2) : R3' carrés, R4' carrés, R6 suppression
    def _croissance_v11(self, lam, bord, creer_arete, creer_sommet, R, W, ecr):
        C, P, rng = self.C, self.P, self.rng

        def lire_sommet(v):
            """Lire N(v) dans le graphe primal : incidence de v (les sommets d'une arête sont immuables)."""
            R.add(("inc", v))

        actualisees = [e for e in lam if C.x[e] != 0]

        # R3' — complétion de carrés : coin a-b-c avec e = {a,b} ∈ Λ, g = {b,c} concordante,
        # a et c sans autre voisin commun que b ; création de v* relié à a et à c.
        coins = {}
        for e in actualisees:
            for a, b in (C.aretes[e], C.aretes[e][::-1]):
                lire_sommet(b)
                for g in C.inc[b]:
                    if g == e:
                        continue
                    R.add(("x", g))
                    if C.x[g] != C.x[e]:
                        continue
                    c = C.aretes[g][0] if C.aretes[g][1] == b else C.aretes[g][1]
                    lire_sommet(a)
                    lire_sommet(c)
                    for z in C.voisins(a) | C.voisins(c):
                        lire_sommet(z)
                    if C.voisins(a) & C.voisins(c) == {b}:
                        coins.setdefault(tuple(sorted((a, c))), (e, g, b))
        retenus = [k for k in coins if rng.random() < P.mu_carre]
        rng.shuffle(retenus)
        for (a, c) in retenus:
            e, g, b = coins[(a, c)]
            if (C.degre(a) < P.k_max and C.degre(c) < P.k_max
                    and C.voisins(a) & C.voisins(c) == {b}):
                vs = creer_sommet((a, c))
                n1 = creer_arete(tuple(sorted((a, vs))), [e])
                n2 = creer_arete(tuple(sorted((c, vs))), [g])
                if cle(n1, n2) not in C.rho:
                    C.rho[cle(n1, n2)] = P.rho0
                    W.add(("r", cle(n1, n2)))
                    ecr.append(("rho", cle(n1, n2), P.rho0))

        # croissance pendante résiduelle
        if P.mu_pendant > 0:
            for e in actualisees:
                for v in C.aretes[e]:
                    lire_sommet(v)
                    if C.degre(v) < P.k_max and rng.random() < P.mu_pendant:
                        vs = creer_sommet((v,))
                        creer_arete(tuple(sorted((v, vs))), [e])

        # R4' — fermeture de 4-cycles : e = {a,b} ∈ Λ, f = {a,u}, g = {b,w} concordantes de ∂Λ,
        # u, w non adjacents, et les coins u-a-b et a-b-w ne sont pas déjà dans un carré.
        generateurs = {}
        for e in actualisees:
            a, b = C.aretes[e]
            for f in bord & C.inc[a]:
                if f == e or C.x[f] != C.x[e]:
                    continue
                u = C.aretes[f][0] if C.aretes[f][1] == a else C.aretes[f][1]
                for g in bord & C.inc[b]:
                    if g == e or C.x[g] != C.x[e]:
                        continue
                    w = C.aretes[g][0] if C.aretes[g][1] == b else C.aretes[g][1]
                    for v in (a, b, u, w):
                        lire_sommet(v)
                    for z in C.voisins(u) | C.voisins(w):
                        lire_sommet(z)
                    if (u != w and not C.adjacents(u, w)
                            and C.voisins(u) & C.voisins(b) == {a}
                            and C.voisins(a) & C.voisins(w) == {b}):
                        generateurs.setdefault(tuple(sorted((u, w))), (f, g, a, b))
        retenues = [k for k in generateurs if rng.random() < P.nu]
        rng.shuffle(retenues)
        for (u, w) in retenues:
            f, g, a, b = generateurs[(u, w)]
            if (C.degre(u) < P.k_max and C.degre(w) < P.k_max and not C.adjacents(u, w)
                    and C.voisins(u) & C.voisins(b) == {a} and C.voisins(a) & C.voisins(w) == {b}):
                creer_arete(tuple(sorted((u, w))), [f, g])

        # R6 — suppression des arêtes actualisées de Λ ∪ ∂Λ : probabilité δ hors de tout carré,
        # δ_carre (petite) sinon. Une arête n'est supprimée que si ses extrémités restent reliées
        # par un autre chemin de longueur ≤ L_alternatif : R6 ne supprime jamais de pont et ne
        # peut donc pas fragmenter la structure. Les tirages sont simultanés ; les suppressions
        # sont acceptées dans un ordre uniforme avec revérification du chemin alternatif.
        def chemin_alternatif(e):
            a, b = C.aretes[e]
            vus, front = {a}, [a]
            for _ in range(P.L_alternatif):
                suivant = []
                for u in front:
                    lire_sommet(u)
                    for k in C.inc[u]:
                        if k == e:
                            continue
                        for v in C.aretes[k]:
                            if v == b:
                                return True
                            if v not in vus:
                                vus.add(v)
                                suivant.append(v)
                front = suivant
            return False

        candidats_r6 = []
        for e in (lam | bord):
            if e not in C.aretes or C.x[e] == 0:
                continue
            a, b = C.aretes[e]
            lire_sommet(a)
            lire_sommet(b)
            for z in C.voisins(a) | C.voisins(b):
                lire_sommet(z)
            proba = P.delta_carre if C.dans_un_carre(e) else P.delta
            if rng.random() < proba:
                candidats_r6.append(e)
        rng.shuffle(candidats_r6)
        for e in candidats_r6:
            if not chemin_alternatif(e):
                continue
            for f in C.voisines(e):
                W.add(("r", cle(e, f)))
            for v in C.aretes[e]:
                W.add(("inc", v))
            W.add(("x", e))
            self._fermer(e)
            C.supprimer_arete(e)
            ecr.append(("suppr", e, None))

    # --- R5 v10 : montée de gradient à partir de vecteurs aléatoires
    def _r5_v10(self, lam, R, W, ecr):
        C, P = self.C, self.P
        for d in P.dims:
            lay = C.w[d]
            contributions = {}
            for k in lam:
                if k in C.aretes and C.x[k] != 0 and len(C.aretes[k]) == 2:
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
                W.add(("w", u))
                ecr.append(("w", (d, u), vec.copy()))
            for u in contributions:
                for v in C.voisins(u):
                    R.add(("w", v))

    # --- R5' v11 (C3) : relaxation de Gibbs locale du modèle O(d) de couplages x_k
    def _r5_v11(self, lam, R, W, ecr):
        C, P = self.C, self.P
        sites = {v for k in lam if k in C.aretes for v in C.aretes[k]}
        for _ in range(P.rayon_B):
            sites |= {z for u in list(sites) for z in C.voisins(u)}
        sites = sorted(sites)
        if not sites:
            return
        for u in sites:
            R.add(("inc", u))
            for k in C.inc[u]:
                R.add(("x", k))
                R.add(("w", C.aretes[k][0]))
                R.add(("w", C.aretes[k][1]))
        for d in P.dims:
            lay = C.w[d]
            modifies = set()
            for _ in range(P.iterations_B):
                # l'ordre des mises à jour est purement computationnel
                for j in self.nprng.permutation(len(sites)):
                    u = sites[j]
                    h = np.zeros(d)
                    for k in C.inc[u]:
                        if C.x[k] != 0:
                            a, b = C.aretes[k]
                            h += C.x[k] * lay[b if a == u else a]
                    nh = np.linalg.norm(h)
                    if nh < 1e-12:
                        continue
                    lay[u] = tirer_vmf(h / nh, P.beta_B * nh, self.nprng)
                    modifies.add(u)
            for u in modifies:
                W.add(("w", u))
                ecr.append(("w", (d, u), lay[u].copy()))

    # ------------------------------------------------------------------------
    def simuler(self, max_evenements: int, max_sommets: int | None = None) -> str:
        for _ in range(max_evenements):
            if max_sommets and self.prochain_sommet >= max_sommets:
                return "taille"
            if not self.pas():
                return "absorbant"
        return "budget"

    def configuration_sur(self, indices) -> Configuration:
        """Reconstruit la configuration sur un ensemble d'événements fermé vers le passé."""
        C = self.C0.copie()
        for i in sorted(indices):
            for (typ, k, val) in self.journal[i].ecritures:
                if typ == "x":
                    if k in C.aretes:
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
                elif typ == "suppr":
                    C.supprimer_arete(k)
                elif typ == "w":
                    d, u = k
                    C.w[d][u] = val.copy()
        return C

    def coupe_par_temps(self, n_evenements: int) -> list:
        return list(range(n_evenements))

    def coupe_par_profondeur(self, n_cible: int) -> list:
        profs = sorted(ev.profondeur for ev in self.journal)
        D = profs[min(n_cible, len(profs)) - 1]
        return [ev.indice for ev in self.journal if ev.profondeur <= D]

    def fermeture_passe(self, indices) -> set:
        s, pile = set(indices), list(indices)
        while pile:
            i = pile.pop()
            for p in self.journal[i].predecesseurs:
                if p not in s:
                    s.add(p)
                    pile.append(p)
        return s

    def coupe_biaisee(self, region: set, frac_region=0.9, frac_reste=0.25) -> list:
        n = len(self.journal)
        graines = [ev.indice for ev in self.journal
                   if ev.indice < frac_reste * n
                   or (ev.indice < frac_region * n and ev.empreinte & region)]
        return sorted(self.fermeture_passe(graines))

    def est_fermee(self, indices) -> bool:
        s = set(indices)
        return all(p in s for i in s for p in self.journal[i].predecesseurs)
