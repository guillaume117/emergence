"""
Expériences à grande échelle avec le moteur C++ (règles v11).

  * une simulation par processus (graines × valeurs de ν), via multiprocessing ;
  * géométrie mesurée en C++ sur la composante géante : V(r) jusqu'à r_max, cyclicité ;
  * dimension spectrale dans le processus principal, sur GPU si CuPy est disponible ;
  * contextualité en C++ : classes de frustration, défaut moyen, référence aléatoire ;
  * invariance de section (optionnelle, --section) : demande le journal, donc plus de mémoire.

Usage :
    python experiences_rapide.py --taille 200000 --graines 4 --nus 0 0.3 0.6 --processus 8
    python experiences_rapide.py --taille 1000000 --h0 germe --graines 2 --section
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SORTIE = Path(__file__).parent / "resultats_rapide"


def ajustements(V: np.ndarray, n_geante: int) -> dict:
    r = np.arange(len(V))
    sel = (r >= 3) & (V < 0.3 * n_geante)
    if sel.sum() < 4:
        return {}
    lr, lv = np.log(r[sel]), np.log(V[sel])
    pente_locale = np.gradient(lv, lr)                    # d log V / d log r
    return {"d_H_puissance": float(np.polyfit(lr, lv, 1)[0]),
            "taux_exponentiel": float(np.polyfit(r[sel], lv, 1)[0]),
            "d_H_local_fin_fenetre": float(np.mean(pente_locale[-3:])),
            "fenetre": [int(r[sel][0]), int(r[sel][-1])]}


def degres(ab, viv, n):
    return np.bincount(ab[viv].ravel(), minlength=n)


def tv(a: np.ndarray, b: np.ndarray) -> float:
    m = max(len(a), len(b))
    pa = np.bincount(a, minlength=m) / len(a)
    pb = np.bincount(b, minlength=m) / len(b)
    return 0.5 * float(np.abs(pa - pb).sum())


def une_simulation(tache: dict) -> dict:
    from moteur_rapide import MoteurRapide, Parametres
    P = Parametres(**tache["parametres"])
    m = MoteurRapide(P)
    t0 = time.time()
    issue = m.simuler(tache["max_evenements"], max_sommets=tache["taille"])
    duree = time.time() - t0
    n_comp, lab, g = m.composantes()
    n_geante = int((lab == g).sum())
    actifs = int((lab >= 0).sum())
    V = m.volume(n_racines=tache["racines"], r_max=tache["r_max"], graine=tache["graine_obs"])
    res = {"nu": P.nu, "graine": P.graine, "issue": issue, "evenements": m.n_evenements,
           "duree_s": duree, "evenements_par_s": m.n_evenements / max(duree, 1e-9),
           "sommets_actifs": actifs, "n_composantes": int(n_comp), "fraction_geante": n_geante / max(1, actifs),
           "V": V.tolist(), "trace": m.trace().tolist()}
    res.update(ajustements(V, n_geante))
    for r in (2, 3, 4):
        res[f"beta1_r{r}"] = m.cyclicite(r=r, n_racines=tache["racines"], graine=tache["graine_obs"])
    res["contexte"] = {int(d): m.contexte(d_index=k) for k, d in enumerate(P.dims)}
    if 2 in P.dims:
        res["contexte_aleatoire_d2"] = m.contexte(d_index=list(P.dims).index(2), aleatoire=True,
                                                  graine=tache["graine_obs"])
    if tache["section"]:
        res["section"] = test_section(m, tache)
    ab, _, viv = m.aretes()
    res["_ab"], res["_viv"], res["_n"] = ab, viv, len(lab)
    return res


def test_section(m, tache) -> dict:
    """Coupe biaisée (région de H0 en avance) contre préfixe en t de même cardinal."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path
    rng = np.random.default_rng(tache["graine_obs"])
    # région spatiale : la moitié de la structure finale la plus proche d'un sommet tiré au hasard
    # (distances de graphe) ; un événement est localisé par l'hyperarête qui a engendré son bloc
    ab, _, viv = m.aretes()
    n_s = int(ab.max()) + 1
    e = ab[viv]
    A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_s, n_s))
    _, lab, g = m.composantes()
    racine = int(rng.choice(np.nonzero(lab == g)[0]))
    dist = shortest_path(A, directed=False, unweighted=True, indices=racine)
    seuil = np.median(dist[np.isfinite(dist)])
    proche = dist <= seuil
    region = set(np.nonzero(viv & proche[ab[:, 0]] & proche[ab[:, 1]])[0].tolist())
    cb = m.coupe_biaisee(region)
    n = len(cb)
    if n < 100:
        return {}
    ct = m.coupe_par_temps(n)
    rec = len(np.intersect1d(ct, cb)) / n
    St, Sb = m.reconstruire(ct), m.reconstruire(cb)
    out = {"evenements_coupe": int(n), "recouvrement": float(rec), "test_informatif": rec < 0.5}
    dt = []
    for S in (St, Sb):
        ab, _, viv = S.aretes()
        d = degres(ab, viv, int(ab.max()) + 1 if len(ab) else 1)
        dt.append(d[d > 0])
    sous = rng.choice(dt[0], size=len(dt[0]) // 2, replace=False)
    out["TV_degre_entre_coupes"] = tv(dt[0], dt[1])
    out["TV_degre_bruit"] = tv(sous, dt[0])
    out["V_coupe_t"] = St.volume(n_racines=200, r_max=10).tolist()
    out["V_coupe_biaisee"] = Sb.volume(n_racines=200, r_max=10).tolist()
    return out


def figures(lignes, nus):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    ax = axs[0, 0]
    for nu in nus:
        L = [l for l in lignes if l["nu"] == nu]
        V = np.mean([l["V"] for l in L], axis=0)
        r = np.arange(1, len(V))
        ax.loglog(r, V[1:], marker=".", label=f"ν = {nu}")
    ax.set_xlabel("r"); ax.set_ylabel("V(r)"); ax.set_title("Volume des boules (composante géante)")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    for nu in nus:
        L = [l for l in lignes if l["nu"] == nu and "d_s" in l]
        if L:
            ds = np.mean([l["d_s"] for l in L], axis=0)
            ax.semilogx(np.arange(1, len(ds) + 1), ds, label=f"ν = {nu}")
    ax.set_xlabel("s"); ax.set_ylabel("d_s(s)"); ax.set_title("Dimension spectrale")
    ax.legend(fontsize=8)

    ax = axs[0, 2]
    for r in (2, 3, 4):
        ax.plot(nus, [np.mean([l[f"beta1_r{r}"] for l in lignes if l["nu"] == nu]) for nu in nus],
                marker="o", label=f"r = {r}")
    ax.set_xlabel("ν"); ax.set_ylabel("β1(B_r)/|B_r|"); ax.set_title("Cyclicité"); ax.legend()

    ax = axs[1, 0]
    for l in lignes:
        tr = np.array(l["trace"])
        if len(tr):
            ax.plot(tr[:, 0], tr[:, 1], lw=0.8, alpha=0.7)
    ax.set_xlabel("événements"); ax.set_ylabel("sommets actifs"); ax.set_title("Stationnarité")

    ax = axs[1, 1]
    classes = (("frustres", "frustrés"), ("non_frustres_desequilibres", "non frustrés, déséq."),
               ("non_frustres_equilibres", "non frustrés, équil."))
    x = np.arange(len(nus))
    for j, (cle, lab) in enumerate(classes):
        vals = [np.nanmean([l["contexte"][2]["frustration"][cle]["defaut_moyen"]
                            for l in lignes if l["nu"] == nu and 2 in l["contexte"]]) for nu in nus]
        ax.bar(x + (j - 1.5) * 0.2, vals, 0.2, label=lab)
    vals = [np.nanmean([l["contexte_aleatoire_d2"]["frustration"]["non_frustres_equilibres"]["defaut_moyen"]
                        for l in lignes if l["nu"] == nu and "contexte_aleatoire_d2" in l]) for nu in nus]
    ax.bar(x + 1.5 * 0.2, vals, 0.2, color="k", alpha=0.5, label="équil., aléatoire")
    ax.axhline(2 * np.sqrt(2) - 2, color="r", ls="--", lw=1, label="2√2 − 2")
    ax.set_xticks(x, [str(n) for n in nus]); ax.set_xlabel("ν"); ax.set_ylabel("défaut moyen δ")
    ax.set_title("Défaut contextuel par classe (d = 2)"); ax.legend(fontsize=7)

    ax = axs[1, 2]
    ax.axis("off")
    txt = []
    for nu in nus:
        L = [l for l in lignes if l["nu"] == nu]
        txt.append(f"ν = {nu} : d_H(fin) = {np.nanmean([l.get('d_H_local_fin_fenetre', np.nan) for l in L]):.2f}, "
                   f"géante = {np.mean([l['fraction_geante'] for l in L]):.0%}, "
                   f"{np.mean([l['evenements_par_s'] for l in L]):,.0f} év./s")
    ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(SORTIE / "synthese_rapide.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taille", type=int, default=200_000, help="nombre maximal de sommets créés")
    ap.add_argument("--evenements", type=int, default=10 ** 9)
    ap.add_argument("--graines", type=int, default=4)
    ap.add_argument("--nus", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    ap.add_argument("--h0", choices=("configuration", "germe"), default="configuration")
    ap.add_argument("--n0", type=int, default=3000)
    ap.add_argument("--processus", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--section", action="store_true", help="test d'invariance de section (journal requis)")
    ap.add_argument("--r_max", type=int, default=60)
    ap.add_argument("--s_max", type=int, default=1000)
    ap.add_argument("--racines", type=int, default=400)
    ap.add_argument("--sans_gpu", action="store_true")
    a = ap.parse_args()
    SORTIE.mkdir(exist_ok=True)

    taches = [{"parametres": dict(n0=a.n0, nu=nu, h0_mode=a.h0, graine=g, dims=(1, 2, 3),
                                  journal=a.section),
               "taille": a.taille, "max_evenements": a.evenements, "section": a.section,
               "r_max": a.r_max, "racines": a.racines, "graine_obs": 1000 + g}
              for nu in a.nus for g in range(a.graines)]
    print(f"{len(taches)} simulations sur {a.processus} processus")
    lignes = []
    with mp.get_context("spawn").Pool(a.processus) as pool:
        for res in pool.imap_unordered(une_simulation, taches):
            lignes.append(res)
            c2 = res["contexte"].get(2, {}).get("frustration", {})
            print(f"   ν={res['nu']:.2f} g={res['graine']} {res['issue']:9s} "
                  f"V={res['sommets_actifs']:8d} géante={res['fraction_geante']:.0%} "
                  f"d_H(fin)={res.get('d_H_local_fin_fenetre', float('nan')):.2f} "
                  f"taux_exp={res.get('taux_exponentiel', float('nan')):.2f} "
                  f"β1(r3)={res['beta1_r3']:.3f} | δ̄ frustrés={c2.get('frustres', {}).get('defaut_moyen', float('nan')):.3f} "
                  f"équil.={c2.get('non_frustres_equilibres', {}).get('defaut_moyen', float('nan')):.3f} | "
                  f"{res['evenements_par_s']:,.0f} év./s")

    print("Dimension spectrale" + (" (GPU désactivé)" if a.sans_gpu else ""))
    from observables_gpu import dimension_spectrale_tableaux, GPU
    for res in lignes:
        ds = dimension_spectrale_tableaux(res.pop("_ab"), res.pop("_viv"), res.pop("_n"),
                                          s_max=a.s_max, utiliser_gpu=not a.sans_gpu)
        res["P_retour"], res["d_s"] = ds["P_retour"].tolist(), ds["d_s"].tolist()
    print(f"   calculée sur {'GPU' if (GPU and not a.sans_gpu) else 'CPU'}")

    figures(lignes, a.nus)
    with open(SORTIE / "resultats.json", "w", encoding="utf-8") as f:
        json.dump({"arguments": vars(a), "simulations": lignes}, f, ensure_ascii=False, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"Résultats écrits dans {SORTIE}")


if __name__ == "__main__":
    main()
