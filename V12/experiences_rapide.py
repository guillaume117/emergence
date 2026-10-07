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
    # paliers de taille : les diagnostics de phase sont mesurés à plusieurs tailles d'une même
    # croissance, pour suivre leur évolution avec N (test de mise à l'échelle)
    paliers = []
    profils = []
    issue = "budget"
    for frac in tache["paliers"]:
        issue = m.simuler(tache["max_evenements"], max_sommets=max(5, int(tache["taille"] * frac)))
        if P.causal:
            profils.append(m.tranches()["longueur_par_hauteur"])
        n_c, lab_p, g_p = m.composantes()
        paliers.append({"sommets": int((lab_p >= 0).sum()), "evenements": m.n_evenements,
                        **m.ponts(), **{k: v for k, v in m.degres_courbure().items()
                                       if k != "histogramme_degres"}})
        if issue == "absorbant":
            break
    duree = time.time() - t0
    n_comp, lab, g = m.composantes()
    n_geante = int((lab == g).sum())
    actifs = int((lab >= 0).sum())
    if P.causal:
        # v12 : racines dans la région achevée, loin du front inachevé et du germe
        h_fig = m.hauteur_figee(profils)
        sel, h_pic, h_lim = m.region_achevee(marge=5, h_lim=h_fig if h_fig >= 15 else None)
        m.fixer_racines(sel)
    V = m.volume(n_racines=tache["racines"], r_max=tache["r_max"], graine=tache["graine_obs"])
    res = {"nu": P.nu, "graine": P.graine, "issue": issue, "evenements": m.n_evenements,
           "duree_s": duree, "evenements_par_s": m.n_evenements / max(duree, 1e-9),
           "sommets_actifs": actifs, "n_composantes": int(n_comp), "fraction_geante": n_geante / max(1, actifs),
           "V": V.tolist(), "trace": m.trace().tolist()}
    res.update(ajustements(V, n_geante))
    for r in (2, 3, 4, 10, 20, 40):
        res[f"beta1_r{r}"] = m.cyclicite(r=r, n_racines=tache["racines"], graine=tache["graine_obs"])
    res["paliers"] = paliers
    res["marche"] = m.marche(n_racines=32, marcheurs=64, s_max=tache["s_marche"], graine=tache["graine_obs"])
    if P.causal:
        res["region_achevee"] = {"h_pic": h_pic, "h_lim": h_lim, "h_figee": h_fig, "sommets": int(len(sel)),
                                 "definition": "profils figés" if h_fig >= 15 else "repli : bande 0,2–0,6 × hauteur max"}
        res["_racines"] = sel
        m.fixer_racines(None)
    # pente locale du volume en fin de fenêtre, pour la relation d_s = 2 d_H / d_w
    rr = np.arange(len(V))
    pentes = np.gradient(np.log(np.maximum(V[1:], 1e-12)), np.log(rr[1:]))
    res["pentes_locales_V"] = pentes.tolist()
    res["ponts"] = m.ponts()
    res["degres_courbure"] = m.degres_courbure()
    res["contexte"] = {int(d): m.contexte(d_index=k) for k, d in enumerate(P.dims)}
    if 2 in P.dims:
        res["contexte_aleatoire_d2"] = m.contexte(d_index=list(P.dims).index(2), aleatoire=True,
                                                  graine=tache["graine_obs"])
    if P.causal:
        res["tranches"] = analyse_tranches(m.tranches(), h_lim)
        c2 = res["contexte"].get(2, {})
        res["diamants"] = {k: c2.get(k) for k in ("frustres_4_diamants", "defaut_moyen_diamants",
                                                  "frustres_4_autres", "defaut_moyen_autres")}
    if tache["section"]:
        res["section"] = test_section_hauteur(m, tache) if P.causal else test_section(m, tache)
    ab, _, viv = m.aretes()
    res["_ab"], res["_viv"], res["_n"] = ab, viv, len(lab)
    return res


def branchement(L, h0, h1) -> dict:
    """Taux de branchement spatial m(h) = L(h+1)/L(h) sur [h0, h1] : m > 1 l'espace gonfle,
    m < 1 il s'effondre, m = 1 critique (géométrie de dimension 2 attendue)."""
    L = np.asarray(L, float)
    hs = [h for h in range(max(1, h0), min(h1, len(L) - 1)) if L[h] > 0]
    if len(hs) < 2:
        return {}
    m = np.array([L[h + 1] / L[h] for h in hs])
    # moyenne géométrique, robuste à l'alternance de parité des tranches
    return {"m_moyen": float(np.exp(np.mean(np.log(m)))), "m_par_hauteur": m.tolist(), "fenetre_m": [hs[0], hs[-1]]}


def analyse_tranches(t: dict, h_lim=None) -> dict:
    """Forme du profil L(h) dans la région achevée (hauteurs avant le maximum, le reste étant le
    front en cours de croissance) : L ∝ h correspond à une géométrie plate de dimension 2 (la
    circonférence croît comme le rayon), L constant à un cylindre (dimension 1 à grande échelle),
    L exponentiel à une géométrie hyperbolique."""
    L = np.array(t["longueur_par_hauteur"], float)
    out = dict(t)
    if len(L) < 10:
        return out
    out.update(branchement(L, 2, h_lim if h_lim is not None else int(0.6 * int(np.argmax(L)))))
    hpic = int(np.argmax(L))
    h = np.arange(len(L))
    h_fin = h_lim if h_lim is not None else max(3, int(0.8 * hpic))
    sel = (h >= 2) & (h <= h_fin) & (L > 0)
    if sel.sum() >= 4:
        out["exposant_L_h"] = float(np.polyfit(np.log(h[sel]), np.log(L[sel]), 1)[0])
        out["taux_exponentiel_L_h"] = float(np.polyfit(h[sel], np.log(L[sel]), 1)[0])
        out["fenetre_h"] = [int(h[sel][0]), int(h[sel][-1])]
    return out


def test_section_hauteur(m, tache) -> dict:
    """v12 : coupe à hauteur constante T contre coupe non simultanée (une moitié spatiale de la
    structure poussée jusqu'à T + Δ, le reste à T - Δ), de cardinaux comparables."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path
    rng = np.random.default_rng(tache["graine_obs"])
    t = m.tranches()
    T = (t["h_min"] + t["h_max"]) // 2
    D = max(2, (t["h_max"] - t["h_min"]) // 8)
    ab, _, viv = m.aretes()
    n_s = int(ab.max()) + 1
    e = ab[viv]
    A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_s, n_s))
    _, lab, g = m.composantes()
    racine = int(rng.choice(np.nonzero(lab == g)[0]))
    dist = shortest_path(A, directed=False, unweighted=True, indices=racine)
    proche = dist <= np.median(dist[np.isfinite(dist)])
    region = set(np.nonzero(viv & proche[ab[:, 0]] & proche[ab[:, 1]])[0].tolist())
    ca = m.coupe_hauteur(T)
    cb = m.coupe_hauteur(T - D, region_aretes=region, T_region=T + D)
    n = min(len(ca), len(cb))
    if n < 100:
        return {}
    rec = len(np.intersect1d(ca, cb)) / max(len(ca), len(cb))
    Sa, Sb = m.reconstruire(ca), m.reconstruire(cb)
    out = {"T": int(T), "Delta": int(D), "evenements_coupe_hauteur": int(len(ca)),
           "evenements_coupe_biaisee": int(len(cb)), "recouvrement": float(rec),
           "test_informatif": rec < 0.5}
    dt = []
    for S in (Sa, Sb):
        a2, _, v2 = S.aretes()
        d = degres(a2, v2, int(a2.max()) + 1 if len(a2) else 1)
        dt.append(d[d > 0])
    sous = rng.choice(dt[0], size=len(dt[0]) // 2, replace=False)
    out["TV_degre_entre_coupes"] = tv(dt[0], dt[1])
    out["TV_degre_bruit"] = tv(sous, dt[0])
    out["V_coupe_hauteur"] = Sa.volume(n_racines=200, r_max=10).tolist()
    out["V_coupe_biaisee"] = Sb.volume(n_racines=200, r_max=10).tolist()
    return out


def figure_tranches(lignes, etiquette):
    L_ = [l for l in lignes if "tranches" in l]
    if not L_:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for l in L_:
        L = np.array(l["tranches"]["longueur_par_hauteur"], float)
        h = np.arange(len(L))
        lab = f"ν = {l['nu']}, g = {l['graine']}"
        axs[0].plot(h, L, lw=1, label=lab)
        axs[1].loglog(h[1:], np.maximum(L[1:], 1e-1), lw=1, label=lab)
    hh = np.arange(2, max(len(l["tranches"]["longueur_par_hauteur"]) for l in L_))
    axs[1].loglog(hh, hh * (axs[1].get_ylim()[1] / hh[-1]) * 0.3, "k--", lw=1, label="pente 1 (plat, d = 2)")
    axs[0].set_xlabel("hauteur h"); axs[0].set_ylabel("sommets à la hauteur h")
    axs[0].set_title("Longueur des tranches (le pic marque le front)")
    axs[1].set_xlabel("h"); axs[1].set_ylabel("L(h)"); axs[1].set_title("Échelle log-log")
    axs[0].legend(fontsize=6); axs[1].legend(fontsize=6)
    fig.suptitle(f"Structure causale — {etiquette}")
    fig.tight_layout()
    fig.savefig(SORTIE / "tranches.png", dpi=140)
    plt.close(fig)


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


def figure_phase(lignes, nus, etiquette):
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
    ax = axs[0]
    for l in lignes:
        Ns = [p["sommets"] for p in l["paliers"]]
        ax.semilogx(Ns, [p["fraction_plus_grande_2ec"] for p in l["paliers"]], marker="o",
                    label=f"ν = {l['nu']}, g = {l['graine']}")
    ax.set_xlabel("N (sommets)"); ax.set_ylabel("fraction dans la plus grande comp. 2-arête-connexe")
    ax.set_title("Surface : reste finie ; polymère branché : → 0"); ax.legend(fontsize=6)

    ax = axs[1]
    for l in lignes:
        Ns = [p["sommets"] for p in l["paliers"]]
        ax.semilogx(Ns, [p["fraction_ponts"] for p in l["paliers"]], marker="o")
    ax.set_xlabel("N (sommets)"); ax.set_ylabel("fraction de ponts"); ax.set_title("Ponts (composante géante)")

    ax = axs[2]
    for nu in nus:
        L = [l for l in lignes if l["nu"] == nu]
        h = np.mean([l["degres_courbure"]["histogramme_degres"][:9] for l in L], axis=0)
        ax.plot(range(9), h / h.sum(), marker="o", label=f"ν = {nu}")
    ax.set_xlabel("degré"); ax.set_ylabel("fraction"); ax.set_title("Distribution des degrés"); ax.legend(fontsize=7)

    ax = axs[3]
    rs = (2, 3, 4, 10, 20, 40)
    for nu in nus:
        L = [l for l in lignes if l["nu"] == nu]
        ax.semilogx(rs, [np.mean([l[f"beta1_r{r}"] for l in L]) for r in rs], marker="o", label=f"ν = {nu}")
    ax.set_xlabel("r"); ax.set_ylabel("β1(B_r)/|B_r|"); ax.set_title("Densité de cycles selon le rayon")
    ax.legend(fontsize=7)
    fig.suptitle(f"Diagnostic de phase — {etiquette}")
    fig.tight_layout()
    fig.savefig(SORTIE / "diagnostic_phase.png", dpi=140)
    plt.close(fig)


def figure_marche(lignes, nus, etiquette):
    """Déplacement quadratique moyen et relation d'Alexander–Orbach d_s = 2 d_H / d_w :
    d_H = pente locale de V(r) en fin de fenêtre, d_w = dimension de marche, d_s mesurée sur
    la fenêtre s = 30..s_max/3 de la dimension spectrale."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axs[0]
    for l in lignes:
        ax.loglog(l["marche"]["s"], l["marche"]["msd"], lw=1, label=f"ν = {l['nu']}, g = {l['graine']}")
    ax.set_xlabel("s (pas de la marche)"); ax.set_ylabel("⟨d²⟩")
    ax.set_title("Déplacement quadratique moyen : pente = 2 / d_w"); ax.legend(fontsize=6)
    ax = axs[1]
    for l in lignes:
        dH = float(np.mean(l["pentes_locales_V"][-5:]))
        dw = l["marche"]["d_w"]
        l["alexander_orbach"] = {"d_H": dH, "d_w": dw, "d_s_predit": 2 * dH / dw if dw > 0 else float("nan")}
        if "d_s" in l:
            ds = np.array(l["d_s"]); sv = np.arange(1, len(ds) + 1)
            fen = (sv >= 30) & (sv <= max(31, len(ds) // 3))
            l["alexander_orbach"]["d_s_mesure"] = float(ds[fen].mean())
            ax.scatter(l["alexander_orbach"]["d_s_predit"], l["alexander_orbach"]["d_s_mesure"],
                       label=f"ν = {l['nu']}, g = {l['graine']}")
    lim = [1.0, 3.0]
    ax.plot(lim, lim, "k--", lw=1, label="d_s = 2 d_H / d_w")
    ax.set_xlabel("2 d_H / d_w"); ax.set_ylabel("d_s mesurée"); ax.set_title("Relation d'Alexander–Orbach")
    ax.legend(fontsize=6)
    fig.suptitle(f"Diffusion — {etiquette}")
    fig.tight_layout()
    fig.savefig(SORTIE / "marche.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taille", type=int, default=200_000, help="nombre maximal de sommets créés")
    ap.add_argument("--evenements", type=int, default=10 ** 9)
    ap.add_argument("--graines", type=int, default=4)
    ap.add_argument("--nus", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    ap.add_argument("--h0", choices=("configuration", "germe", "zigzag"), default="configuration")
    ap.add_argument("--modele", choices=("v11", "v12"), default="v11",
                    help="v12 : croissance par front causal (germe en zigzag, R3c, R4c, R7±)")
    ap.add_argument("--L0", type=int, default=100, help="v12 : demi-longueur du germe en zigzag")
    ap.add_argument("--zeta_plus", type=float, default=0.02, help="v12 : taux d'insertion R7+")
    ap.add_argument("--zeta_moins", type=float, default=0.02, help="v12 : taux de contraction R7-")
    ap.add_argument("--n0", type=int, default=3000)
    ap.add_argument("--processus", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--section", action="store_true", help="test d'invariance de section (journal requis)")
    ap.add_argument("--r_max", type=int, default=60)
    ap.add_argument("--s_max", type=int, default=1000)
    ap.add_argument("--racines", type=int, default=400)
    ap.add_argument("--sans_gpu", action="store_true")
    ap.add_argument("--r4", choices=("strict", "sans_concordance", "libre"), default="strict",
                    help="R4' : strict (v11), sans condition de concordance, ou libre (ni concordance "
                         "ni appartenance à la frontière)")
    ap.add_argument("--r3_sans_concordance", action="store_true")
    ap.add_argument("--paliers", type=float, nargs="+", default=[0.0625, 0.125, 0.25, 0.5, 1.0],
                    help="fractions de --taille auxquelles mesurer les diagnostics de phase")
    ap.add_argument("--param", nargs="*", default=[], metavar="NOM=VALEUR",
                    help="surcharge de paramètres du moteur, ex. --param delta_carre=0 epsilon=0")
    ap.add_argument("--s_marche", type=int, default=20000,
                    help="nombre de pas des marches pour la dimension de marche")
    ap.add_argument("--etiquette", default="",
                    help="sous-dossier de sortie (par défaut construit à partir des options)")
    a = ap.parse_args()
    global SORTIE
    from moteur_rapide import Parametres as _P
    surcharges = {}
    for kv in a.param:
        nom, val = kv.split("=", 1)
        defaut = getattr(_P(), nom)            # erreur explicite si le nom est inconnu
        surcharges[nom] = type(defaut)(float(val)) if not isinstance(defaut, str) else val
    etiquette = a.etiquette or (f"{a.h0}_r4-{a.r4}" + ("_r3-libre" if a.r3_sans_concordance else "")
                                + "".join(f"_{k}-{v}" for k, v in surcharges.items()))
    SORTIE = SORTIE / etiquette
    SORTIE.mkdir(parents=True, exist_ok=True)
    variantes = dict(r4_concordance=a.r4 == "strict", r4_frontiere=a.r4 != "libre",
                     r3_concordance=not a.r3_sans_concordance)
    if a.modele == "v12":
        # préréglage v12 : pas de suppression R6 (la croissance n'a plus besoin de casser des
        # carrés), complétion causale sans condition de concordance ; tout reste surchargeable
        # par --param
        a.h0 = "zigzag"
        variantes.update(causal=True, L0=a.L0, zeta_plus=a.zeta_plus, zeta_moins=a.zeta_moins,
                         delta=0.0, delta_carre=0.0, r3_concordance=False)
        variantes.update(surcharges)
        surcharges = {}
        if not a.etiquette:
            etiquette = f"v12_L0-{a.L0}_zp-{a.zeta_plus}_zm-{a.zeta_moins}" + "".join(
                f"_{k}-{v}" for k, v in variantes.items() if k not in (
                    "causal", "L0", "zeta_plus", "zeta_moins", "delta", "delta_carre", "r3_concordance",
                    "r4_concordance", "r4_frontiere"))
            try:
                SORTIE.rmdir()                     # dossier provisoire vide créé plus haut
            except OSError:
                pass
            SORTIE = SORTIE.parent / etiquette
            SORTIE.mkdir(parents=True, exist_ok=True)

    n0 = 4 if a.h0 in ("germe", "zigzag") else a.n0     # ces germes n'utilisent pas n0
    taches = [{"parametres": dict(n0=n0, nu=nu, h0_mode=a.h0, graine=g, dims=(1, 2, 3),
                                  journal=a.section, **variantes, **surcharges),
               "taille": a.taille, "max_evenements": a.evenements, "section": a.section,
               "r_max": a.r_max, "racines": a.racines, "graine_obs": 1000 + g,
               "paliers": sorted(a.paliers), "s_marche": a.s_marche}
              for nu in a.nus for g in range(a.graines)]
    print(f"{len(taches)} simulations sur {a.processus} processus — variante : {etiquette}")
    lignes = []
    with mp.get_context("spawn").Pool(a.processus) as pool:
        for res in pool.imap_unordered(une_simulation, taches):
            lignes.append(res)
            c2 = res["contexte"].get(2, {}).get("frustration", {})
            print(f"   ν={res['nu']:.2f} g={res['graine']} {res['issue']:9s} "
                  f"V={res['sommets_actifs']:8d} géante={res['fraction_geante']:.0%} "
                  f"d_H(fin)={res.get('d_H_local_fin_fenetre', float('nan')):.2f} "
                  f"taux_exp={res.get('taux_exponentiel', float('nan')):.2f} "
                  f"β1(r3)={res['beta1_r3']:.3f} 2EC={res['ponts']['fraction_plus_grande_2ec']:.3f} "
                  f"ponts={res['ponts']['fraction_ponts']:.3f} plats={res['degres_courbure']['fraction_plats']:.3f} "
                  f"d_w={res['marche']['d_w']:.2f} "
                  + (f"m={res['tranches'].get('m_moyen', float('nan')):.3f} "
                     f"L∝h^{res['tranches'].get('exposant_L_h', float('nan')):.2f} "
                     f"viol={res['tranches']['violations_invariant']} " if "tranches" in res else "")
                  + f"| δ̄ frustrés={c2.get('frustres', {}).get('defaut_moyen', float('nan')):.3f} "
                  f"équil.={c2.get('non_frustres_equilibres', {}).get('defaut_moyen', float('nan')):.3f} | "
                  f"{res['evenements_par_s']:,.0f} év./s")

    # sauvegarde intermédiaire : un échec ultérieur ne fait pas perdre les simulations
    def sauver(nom):
        with open(SORTIE / nom, "w", encoding="utf-8") as f:
            json.dump({"arguments": vars(a),
                       "simulations": [{k: v for k, v in l.items() if not k.startswith("_")} for l in lignes]},
                      f, ensure_ascii=False, indent=1,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    sauver("resultats_sans_spectre.json")
    # structures finales sauvegardées : la dimension spectrale peut être (re)calculée plus tard
    # avec spectre_depuis_sauvegarde.py, sans relancer les simulations
    for res in lignes:
        np.savez_compressed(SORTIE / f"structure_nu{res['nu']}_g{res['graine']}.npz",
                            ab=res["_ab"], viv=res["_viv"], n=res["_n"])

    import observables_gpu
    dimension_spectrale_tableaux = observables_gpu.dimension_spectrale_tableaux
    GPU = getattr(observables_gpu, "GPU", False)
    RAISON_SANS_GPU = getattr(observables_gpu, "RAISON_SANS_GPU",
                              "version ancienne de observables_gpu.py : mettez-la à jour")
    utiliser = GPU and not a.sans_gpu
    print("Dimension spectrale sur " + ("GPU" if utiliser else "CPU")
          + ("" if utiliser or a.sans_gpu else f" (GPU indisponible : {RAISON_SANS_GPU})"))
    for res in lignes:
        try:
            ds = dimension_spectrale_tableaux(res["_ab"], res["_viv"], res["_n"],
                                              s_max=a.s_max, utiliser_gpu=utiliser,
                                              candidats=res.get("_racines"))
            res["P_retour"], res["d_s"], res["spectre_gpu"] = ds["P_retour"].tolist(), ds["d_s"].tolist(), ds["gpu"]
        except Exception as exc:
            print(f"   dimension spectrale impossible pour ν={res['nu']} g={res['graine']} : {exc}")
        for k in ("_ab", "_viv", "_n", "_racines"):
            res.pop(k, None)

    figures(lignes, a.nus)
    figure_phase(lignes, a.nus, etiquette)
    figure_marche(lignes, a.nus, etiquette)
    figure_tranches(lignes, etiquette)
    sauver("resultats.json")
    print(f"Résultats écrits dans {SORTIE}")


if __name__ == "__main__":
    main()
