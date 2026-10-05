"""
Diagnostic des premiers résultats v11 (ν = 0) : deux hypothèses à départager.

  H-geo  La structure se fragmente en composantes de petite taille (effet de R6). Les boules
         saturent alors à la taille des composantes, ce qui écrase V(r), d_H et d_s.
         Test : distribution des tailles de composantes, et V(r) mesuré sur la composante géante.

  H-ctx  La couche B est en retard sur le modèle A. Quand une issue x_k change, les vecteurs ne sont
         relaxés que localement (quelques balayages sur les extrémités du bloc) et gardent la mémoire
         de l'ancienne frustration : un cycle redevenu non frustré peut rester tordu et violer
         l'inégalité. Test : relaxation globale recuite de la couche B sur les issues finales,
         puis nouvelle mesure des classes. Si les violations des cycles isolés s'effondrent,
         le défaut vient du retard, non de la correction C3 elle-même.

Usage :  python diagnostic_v11.py --n0 1000 --nu 0.0 --graine 0
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, deque

import numpy as np

from modele_A import ModeleA, Parametres, tirer_vmf
import modele_B as mb
import observables as obs


def composantes(adj) -> list:
    vus, tailles, geante = set(), [], set()
    for s in adj:
        if s in vus:
            continue
        comp, file = {s}, deque([s])
        while file:
            u = file.popleft()
            for v in adj[u]:
                if v not in comp:
                    comp.add(v)
                    file.append(v)
        vus |= comp
        tailles.append(len(comp))
        if len(comp) > len(geante):
            geante = comp
    return sorted(tailles, reverse=True), geante


def volume_sur(adj, sommets, rng, n_racines=300, r_max=40) -> np.ndarray:
    sommets = list(sommets)
    V = np.zeros(r_max + 1)
    for _ in range(n_racines):
        d = obs.boule(adj, sommets[rng.randrange(len(sommets))], r_max)
        c = Counter(d.values())
        V += np.cumsum([c.get(r, 0) for r in range(r_max + 1)])
    return V / n_racines


def relaxation_globale(C, d, balayages=200, beta_min=5.0, beta_max=500.0, graine=0):
    """Recuit de la couche B à issues x fixées : β croît géométriquement de beta_min à beta_max."""
    rng = np.random.default_rng(graine)
    lay = {v: w.copy() for v, w in C.w[d].items()}
    sommets = [v for v, s in C.inc.items() if s]
    for b in np.geomspace(beta_min, beta_max, balayages):
        for j in rng.permutation(len(sommets)):
            u = sommets[j]
            h = np.zeros(d)
            for k in C.inc[u]:
                if C.x[k] != 0:
                    a, c = C.aretes[k]
                    h += C.x[k] * lay[c if a == u else a]
            nh = np.linalg.norm(h)
            if nh > 1e-12:
                lay[u] = tirer_vmf(h / nh, b * nh, rng)
    return lay


def resume_classes(a) -> str:
    fr = a["frustration"]
    return " ".join(f"{k}: δ̄={v['defaut_moyen']:.3f}, >τ={v['fraction_violee']:.2f} ({v['cycles']})"
                    for k, v in fr.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", type=int, default=1000)
    ap.add_argument("--nu", type=float, default=0.0)
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--balayages", type=int, default=200)
    ap.add_argument("--h0", choices=("configuration", "germe"), default="configuration")
    a = ap.parse_args()

    P = Parametres(regles="v11", n0=a.n0, nu=a.nu, dims=(2,), graine=a.graine, h0_mode=a.h0)
    m = ModeleA(P)
    issue = m.simuler(200 * a.n0, max_sommets=20 * a.n0)
    rng = random.Random(a.graine)
    adj = obs.graphe_primal(m.C)
    print(f"Simulation : {issue}, {len(m.journal)} événements, {len(adj)} sommets actifs")

    # --- H-geo
    tailles, geante = composantes(adj)
    print("\nH-geo — fragmentation")
    print(f"   composantes : {len(tailles)}, géante = {tailles[0]} sommets "
          f"({tailles[0] / len(adj):.1%}), suivantes : {tailles[1:6]}")
    deg = Counter(len(adj[v]) for v in adj)
    print(f"   distribution des degrés : {dict(sorted(deg.items()))}")
    V = volume_sur(adj, geante, rng)
    r = np.arange(len(V))
    sel = (r >= 3) & (V < 0.3 * len(geante))
    print(f"   V(r) sur la géante : {np.round(V[:21:2], 1).tolist()} ...")
    if sel.sum() >= 3:
        dH = np.polyfit(np.log(r[sel]), np.log(V[sel]), 1)[0]
        texp = np.polyfit(r[sel], np.log(V[sel]), 1)[0]
        print(f"   ajustements sur la géante (r = {r[sel][0]}..{r[sel][-1]}) : "
              f"d_H = {dH:.2f}, taux exponentiel = {texp:.2f}")
    print("   Lecture : géante minoritaire ⇒ H-geo confirmée ; V(r) en loi de puissance sur la "
          "géante ⇒ la géométrie locale est bonne et seule la connexité fait défaut.")

    # --- H-ctx
    print("\nH-ctx — retard de la couche B")
    avant = mb.analyse_contextuelle(m.C, 2)
    print(f"   avant relaxation globale : {resume_classes(avant)}")
    m.C.w[2] = relaxation_globale(m.C, 2, balayages=a.balayages, graine=a.graine)
    apres = mb.analyse_contextuelle(m.C, 2)
    print(f"   après relaxation globale : {resume_classes(apres)}")
    alea = mb.analyse_contextuelle(m.C, 2, w=mb.vecteurs_aleatoires(m.C, 2, graine=a.graine))
    print(f"   vecteurs aléatoires      : {resume_classes(alea)}")
    print(f"   densité de frustration   : {apres['densite_frustration']:.2f}")
    print("   Lecture : après recuit, la proposition 3 prédit δ̄ ≈ 0 dans les composantes équilibrées et "
          "δ̄ proche de 2√2 - 2 ≈ 0,83 sur les 4-cycles frustrés ; l'écart entre « avant » et « après » "
          "mesure le retard de la couche B sur le modèle A.")


if __name__ == "__main__":
    main()
