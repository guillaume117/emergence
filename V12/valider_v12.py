"""
Validation déterministe des règles v12 (note_front_causal_v12.tex, proposition 3).

Croissance pure depuis le germe en zigzag (μ_□ = 1, sans suppression ni fluctuation, complétion
sans condition de concordance) : la structure de hauteur au plus T doit être exactement le
cylindre plat, quel que soit l'ordre des événements, donc quelle que soit la graine.

Vérifie : invariant de hauteur (aucune arête entre hauteurs non consécutives) ; longueur L0 de
toutes les tranches achevées ; degré 4 à l'intérieur ; aucun pont ; identité des structures
obtenues avec des graines différentes ; reconstruction exacte depuis le journal ; fermeture vers
le passé des coupes à hauteur constante.

Usage :  python valider_v12.py [--L0 40] [--graines 4]
"""

from __future__ import annotations

import argparse

import numpy as np

from moteur_rapide import MoteurRapide, Parametres


def structure_canonique(m, h_lim):
    """Profil par hauteur (nombre de sommets, histogramme des degrés) jusqu'à h_lim."""
    h = m.hauteurs()
    ab, _, viv = m.aretes()
    e = ab[viv]
    deg = np.bincount(e.ravel(), minlength=len(h))
    prof = []
    for t in range(h_lim + 1):
        sel = (h == t) & (deg > 0)
        prof.append((int(sel.sum()), tuple(np.bincount(deg[sel], minlength=5)[:5])))
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L0", type=int, default=40)
    ap.add_argument("--graines", type=int, default=4)
    a = ap.parse_args()
    profils, ok_global = [], True
    for g in range(a.graines):
        P = Parametres(h0_mode="zigzag", L0=a.L0, causal=True, mu_carre=1.0, delta=0.0, delta_carre=0.0,
                       r3_concordance=False, zeta_plus=0.0, zeta_moins=0.0, graine=g, journal=True, dims=(2,))
        m = MoteurRapide(P)
        m.simuler(10 ** 7, max_sommets=40 * a.L0 * 2)
        t = m.tranches()
        L = np.array(t["longueur_par_hauteur"])
        # tranches achevées : avant la première tranche incomplète
        incompletes = np.nonzero(L < a.L0)[0]
        h_ach = int(incompletes[0]) - 2 if len(incompletes) else len(L) - 1
        prof = structure_canonique(m, h_ach)
        interieur_ok = all(n == a.L0 and d[4] == a.L0 for n, d in prof[1:])
        recon = m.egal(m.reconstruire(np.arange(m.n_evenements)))
        coupe = m.coupe_hauteur(h_ach // 2)
        ferme = m.est_fermee(coupe)
        ponts = m.ponts()["fraction_ponts"]
        ok = t["violations_invariant"] == 0 and interieur_ok and recon and ferme and ponts == 0.0
        ok_global &= ok
        profils.append(prof)
        print(f"graine {g} : invariant {t['violations_invariant'] == 0}, tranches achevées jusqu'à h = {h_ach}, "
              f"intérieur plat (L0 sommets de degré 4) {interieur_ok}, ponts {ponts:.3f}, "
              f"reconstruction {recon}, coupe de hauteur fermée {ferme} -> {'OK' if ok else 'ÉCHEC'}")
    h_commun = min(len(p) for p in profils)
    identiques = all(p[:h_commun] == profils[0][:h_commun] for p in profils)
    print(f"structures identiques entre graines jusqu'à h = {h_commun - 1} : {identiques}")
    print("VALIDATION v12 :", "RÉUSSIE" if ok_global and identiques else "ÉCHEC")


if __name__ == "__main__":
    main()
