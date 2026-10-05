"""
Validation du moteur C++ contre l'implémentation Python de référence (modele_A.py, règles v11).

Trois niveaux :
  1. exactitude interne : la reconstruction depuis le journal complet redonne l'état final à
     l'identique ; les coupes produites sont fermées vers le passé ;
  2. exactitude des observables C++ : sur un même état, obs_contexte et obs_cyclicite donnent les
     mêmes valeurs que modele_B.py et observables.py ;
  3. équivalence statistique de la dynamique : les deux moteurs n'utilisent pas le même générateur
     aléatoire, on compare donc des moyennes sur plusieurs graines.
Puis une mesure de vitesse.

Usage :  python valider_moteur.py [--graines 6] [--n0 300] [--evenements 20000]
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from moteur_rapide import MoteurRapide, Parametres as ParamsC
from modele_A import ModeleA, Parametres as ParamsPy
import modele_B as mb
import observables as obs


def ecart(a, b) -> str:
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 1 else float("nan")
    z = (a.mean() - b.mean()) / se if se > 0 else 0.0
    return f"{a.mean():9.4f} | {b.mean():9.4f} | z = {z:+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graines", type=int, default=6)
    ap.add_argument("--n0", type=int, default=300)
    ap.add_argument("--evenements", type=int, default=20000)
    a = ap.parse_args()
    kw = dict(n0=a.n0, nu=0.3, dims=(1, 2))

    print("1. Exactitude interne")
    m = MoteurRapide(ParamsC(graine=1, **kw))
    m.simuler(a.evenements, max_sommets=20 * a.n0)
    n = m.n_evenements
    print(f"   reconstruction depuis le journal complet identique : {m.egal(m.reconstruire(np.arange(n)))}")
    cp = m.coupe_par_profondeur(n // 2)
    print(f"   coupe par profondeur fermée vers le passé          : {m.est_fermee(cp)}")
    print(f"   préfixe en t fermé vers le passé                   : {m.est_fermee(m.coupe_par_temps(n // 2))}")

    print("\n2. Observables C++ contre Python, sur le même état")
    C = m.configuration()
    for k in (0, 1):
        oc = m.contexte(d_index=k)
        op = mb.analyse_contextuelle(C, m.dims[k])
        ok = all(oc["frustration"][c]["cycles"] == op["frustration"][c]["cycles"]
                 and abs(oc["frustration"][c]["defaut_moyen"] - op["frustration"][c]["defaut_moyen"]) < 1e-9
                 for c in mb_classes(op) if op["frustration"][c]["cycles"])
        print(f"   contexte d = {m.dims[k]} : classes et défauts identiques : {ok}")
    adj = obs.graphe_primal(C)
    n_c, _, _ = m.composantes()
    print(f"   composantes : C++ = {n_c}, Python = {len(obs.composantes(adj))}")

    print("\n3. Équivalence statistique (moyenne C++ | moyenne Python | écart réduit)")
    res = {"cpp": {}, "py": {}}
    t_cpp = t_py = 0.0
    ev_cpp = ev_py = 0
    for g in range(a.graines):
        t0 = time.perf_counter()
        mc = MoteurRapide(ParamsC(graine=100 + g, **kw))
        mc.simuler(a.evenements, max_sommets=20 * a.n0)
        t_cpp += time.perf_counter() - t0
        ev_cpp += mc.n_evenements
        t0 = time.perf_counter()
        mp = ModeleA(ParamsPy(regles="v11", graine=100 + g, **kw))
        mp.simuler(a.evenements, max_sommets=20 * a.n0)
        t_py += time.perf_counter() - t0
        ev_py += len(mp.journal)
        for nom, C in (("cpp", mc.configuration()), ("py", mp.C)):
            adj_t = obs.graphe_primal(C)
            comps = obs.composantes(adj_t)
            adj = obs.restreindre(adj_t, comps[0])
            rng = random.Random(g)
            ctx = mb.analyse_contextuelle(C, 2)
            for cle, val in (
                ("sommets actifs", len(adj_t)),
                ("arêtes", len(C.aretes)),
                ("fraction ouverte", sum(1 for x in C.x.values() if x == 0) / max(1, len(C.aretes))),
                ("composantes", len(comps)),
                ("β1(B3)/|B3| (géante)", obs.cyclicite(adj, rng, n_racines=200, rayons=(3,))["beta1_par_sommet_r3"]),
                ("degré moyen", np.mean([len(adj_t[v]) for v in adj_t])),
                ("densité de frustration", ctx["densite_frustration"]),
                ("δ̄ cycles frustrés (d=2)", ctx["frustration"]["frustres"]["defaut_moyen"]),
            ):
                res[nom].setdefault(cle, []).append(val)
    for cle in res["cpp"]:
        print(f"   {cle:28s} {ecart(res['cpp'][cle], res['py'][cle])}")
    print("   (|z| > 3 sur plusieurs grandeurs signalerait une différence de dynamique)")

    print("\n4. Vitesse")
    print(f"   C++    : {ev_cpp / t_cpp:12,.0f} événements/s")
    print(f"   Python : {ev_py / t_py:12,.0f} événements/s")
    print(f"   accélération : × {(ev_cpp / t_cpp) / (ev_py / t_py):.0f}")


def mb_classes(op):
    return list(op["frustration"].keys())


if __name__ == "__main__":
    main()
