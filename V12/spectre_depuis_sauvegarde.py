"""
Calcule la dimension spectrale à partir des structures sauvegardées par experiences_rapide.py
(fichiers resultats_rapide/structure_nu*_g*.npz), sans relancer les simulations.

Usage :  python spectre_depuis_sauvegarde.py --dossier germe_r4-strict [--s_max 3000] [--racines 64]
Sorties : resultats_rapide/spectre.json et resultats_rapide/spectre.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import observables_gpu as og

DOSSIER = Path(__file__).parent / "resultats_rapide"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_max", type=int, default=3000)
    ap.add_argument("--racines", type=int, default=64)
    ap.add_argument("--sans_gpu", action="store_true")
    ap.add_argument("--dossier", default="", help="sous-dossier de resultats_rapide (variante)")
    a = ap.parse_args()
    global DOSSIER
    if a.dossier:
        DOSSIER = DOSSIER / a.dossier
    gpu = getattr(og, "GPU", False) and not a.sans_gpu
    print("Calcul sur", "GPU" if gpu else f"CPU ({getattr(og, 'RAISON_SANS_GPU', '')})")
    res = {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f in sorted(DOSSIER.glob("structure_nu*_g*.npz")):
        nu, g = re.match(r"structure_nu(.+)_g(\d+)\.npz", f.name).groups()
        z = np.load(f)
        out = og.dimension_spectrale_tableaux(z["ab"], z["viv"].astype(bool), int(z["n"]),
                                              n_racines=a.racines, s_max=a.s_max, utiliser_gpu=gpu)
        ds = out["d_s"]
        s = np.arange(1, len(ds) + 1)
        fen = (s >= 30) & (s <= a.s_max // 3)
        res[f.stem] = {"nu": float(nu), "graine": int(g), "P_retour": out["P_retour"].tolist(),
                       "d_s": ds.tolist(), "d_s_moyen_fenetre": float(ds[fen].mean()),
                       "d_s_ecart_type_fenetre": float(ds[fen].std()), "gpu": out["gpu"]}
        print(f"   {f.stem} : d_s moyen sur s = 30..{a.s_max // 3} : "
              f"{ds[fen].mean():.3f} ± {ds[fen].std():.3f}")
        ax.semilogx(s, ds, lw=1, label=f"ν = {nu}, g = {g}")
    ax.set_xlabel("s (paramètre de diffusion)")
    ax.set_ylabel("d_s(s)")
    ax.set_title("Dimension spectrale (plateau attendu)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(DOSSIER / "spectre.png", dpi=140)
    with open(DOSSIER / "spectre.json", "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1)
    print(f"Résultats écrits dans {DOSSIER}")


if __name__ == "__main__":
    main()
