"""
Balayage de criticité pour le modèle v12 : à ζ+ fixé, on fait varier ζ- et l'on mesure le taux de
branchement spatial m = L(h+1)/L(h) dans la région achevée (sous le front de croissance).

    m > 1 : l'espace gonfle (géométrie de type hyperbolique)
    m < 1 : l'espace s'effondre
    m = 1 : critique, seul régime où une géométrie de dimension 2 est attendue

Le script interpole le rapport ζ-/ζ+ pour lequel m = 1, et trace m ainsi que l'exposant de L(h).
Les simulations sont rapides (sans couche contextuelle ni journal) et tournent en parallèle.

Note : avec R7± actif partout, aucune tranche n'est figée ; le critère retenu est la dérive temporelle
D des longueurs de tranches dans le corps de la structure (D = 0 : stationnaire, critique).

Usage :
    python balayage_criticite.py
    python balayage_criticite.py --zeta_plus 0.02 --rapports 1 2 3 4 6 8 --taille 300000 --graines 3

Sorties : resultats_rapide/criticite/criticite.json et criticite.png
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SORTIE = Path(__file__).parent / "resultats_rapide" / "criticite"


def une(tache):
    from moteur_rapide import MoteurRapide, Parametres
    from experiences_rapide import branchement
    P = Parametres(h0_mode="zigzag", n0=4, causal=True, L0=tache["L0"], zeta_plus=tache["zp"],
                   zeta_moins=tache["zm"], delta=0.0, delta_carre=0.0, r3_concordance=False,
                   nu=0.3, dims=(), journal=False, graine=tache["graine"])
    m = MoteurRapide(P)
    # croissance par étapes ; les profils successifs indiquent quelles tranches sont figées
    profils, evs, issue = [], [], "budget"
    for frac in (0.4, 0.6, 0.8, 1.0):
        issue = m.simuler(tache["max_ev"] // 4, max_sommets=int(tache["taille"] * frac))
        profils.append(m.tranches()["longueur_par_hauteur"])
        evs.append(m.n_evenements)
    L = profils[-1]
    h_pic = int(np.argmax(L))
    h_fig = m.hauteur_figee(profils)
    n_final = int(sum(L))
    b = branchement(L, 2, h_fig) if h_fig >= 6 else {}
    # Dérive des tranches : dans les règles v12, R7± agit partout, si bien qu'aucune tranche n'est
    # jamais figée ; la géométrie fluctue dans tout le volume, comme dans une simulation de
    # triangulations causales. Le critère pertinent est alors la stationnarité dans le temps :
    # variation relative de L(h) entre les deux derniers profils, par événement et par sommet,
    # médiane sur le corps de la structure (hauteurs 3 à 0,6 × pic). D > 0 : l'espace gonfle.
    a_, b_ = np.asarray(profils[-2], float), np.asarray(profils[-1], float)
    n_prev = max(1.0, a_.sum())
    dev = max(1, evs[-1] - evs[-2]) / n_prev
    corps = [h for h in range(3, max(4, int(0.6 * h_pic))) if h < len(a_) and h < len(b_) and a_[h] > 20]
    derive = float(np.median([(b_[h] - a_[h]) / a_[h] for h in corps]) / dev) if corps else float("nan")
    hs = np.arange(len(L))
    sel = (hs >= 2) & (hs <= h_fig) & (np.asarray(L) > 0)
    expo = float(np.polyfit(np.log(hs[sel]), np.log(np.asarray(L, float)[sel]), 1)[0]) if sel.sum() >= 4 else float("nan")
    return {"zp": tache["zp"], "zm": tache["zm"], "rapport": tache["zm"] / tache["zp"], "graine": tache["graine"],
            "issue": issue, "sommets": n_final, "h_max": len(L) - 1, "h_pic": h_pic, "h_figee": h_fig,
            "m": b.get("m_moyen", float("nan")), "derive": derive,
            "exposant_L": expo, "L": list(map(int, L))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeta_plus", type=float, default=0.02)
    ap.add_argument("--rapports", type=float, nargs="+", default=[1, 1.5, 2, 3, 4, 6, 8])
    ap.add_argument("--L0", type=int, default=200)
    ap.add_argument("--taille", type=int, default=200_000)
    ap.add_argument("--max_ev", type=int, default=50_000_000)
    ap.add_argument("--graines", type=int, default=3)
    ap.add_argument("--processus", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    SORTIE.mkdir(parents=True, exist_ok=True)
    print("Critère : dérive D des longueurs de tranches entre les deux derniers profils (D = 0 : critique).\n"
          "N = sommets vivants (les sommets créés puis retirés par R7- ne sont pas comptés).")
    taches = [{"zp": a.zeta_plus, "zm": a.zeta_plus * r, "L0": a.L0, "taille": a.taille, "max_ev": a.max_ev,
               "graine": g} for r in a.rapports for g in range(a.graines)]
    print(f"{len(taches)} simulations sur {a.processus} processus")
    with mp.get_context("spawn").Pool(a.processus) as pool:
        res = list(pool.imap_unordered(une, taches))
    res.sort(key=lambda r: (r["rapport"], r["graine"]))
    for r in res:
        print(f"   ζ-/ζ+ = {r['rapport']:4.1f}  g={r['graine']}  {r['issue']:9s} N={r['sommets']:7d} "
              f"h_max={r['h_max']:4d} figées ≤ {r['h_figee']:4d}  dérive D = {r['derive']:+.4f}")
    rap = sorted(set(r["rapport"] for r in res))
    mm = np.array([np.nanmean([r["derive"] for r in res if r["rapport"] == x]) for x in rap]) + 1.0
    ee = np.array([np.nanmean([r["exposant_L"] for r in res if r["rapport"] == x]) for x in rap])
    critique = None
    for i in range(len(rap) - 1):
        if (mm[i] - 1) * (mm[i + 1] - 1) <= 0 and mm[i] != mm[i + 1]:
            critique = rap[i] + (1 - mm[i]) * (rap[i + 1] - rap[i]) / (mm[i + 1] - mm[i])
            break
    print("Rapport critique estimé ζ-/ζ+ :", f"{critique:.2f}" if critique else "non encadré par le balayage")

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
    axs[0].plot(rap, mm, marker="o")
    axs[0].axhline(1, color="k", ls="--", lw=1)
    if critique:
        axs[0].axvline(critique, color="tab:red", ls=":", lw=1, label=f"critique ≈ {critique:.2f}")
        axs[0].legend()
    axs[0].set_xlabel("ζ- / ζ+"); axs[0].set_ylabel("1 + D (dérive des tranches)"); axs[0].set_title("Stationnarité des tranches : 1 = critique")
    axs[1].plot(rap, ee, marker="o")
    axs[1].axhline(1, color="k", ls="--", lw=1, label="L ∝ h (plat, d = 2, croissance depuis un point)")
    axs[1].axhline(0, color="gray", ls=":", lw=1, label="L constant (cylindre)")
    axs[1].set_xlabel("ζ- / ζ+"); axs[1].set_ylabel("exposant de L(h)"); axs[1].set_title("Forme du profil des tranches")
    axs[1].legend(fontsize=7)
    for x in rap:
        r = next(r for r in res if r["rapport"] == x)
        ln, = axs[2].plot(r["L"], lw=1, label=f"ζ-/ζ+ = {x:g}")
        if r["h_figee"] > 0:
            axs[2].axvline(r["h_figee"], color=ln.get_color(), ls=":", lw=0.8)
    axs[2].set_yscale("log"); axs[2].set_xlabel("hauteur h"); axs[2].set_ylabel("L(h)")
    axs[2].set_title("Profils (pointillés : limite des tranches figées)"); axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(SORTIE / "criticite.png", dpi=140)
    with open(SORTIE / "criticite.json", "w", encoding="utf-8") as f:
        json.dump({"arguments": vars(a), "rapport_critique": critique, "simulations": res}, f, indent=1)
    print(f"Résultats écrits dans {SORTIE}")


if __name__ == "__main__":
    main()
