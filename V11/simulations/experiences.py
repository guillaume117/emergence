"""
Expériences sur les modèles jouets révisés (v11), avec comparaison possible à la v10.

  Expérience 1 — modèle A
    1a. diagramme de non-terminaison en (βJ, croissance)
    1b. cyclicité, croissance volumique, dimension spectrale selon ν
    1c. invariance de section : préfixe en t, coupe par profondeur, coupe biaisée ;
        le test n'est déclaré informatif que si le recouvrement est inférieur au seuil (0,5)
    1d. reconstruction exacte depuis le journal
    1e. (v11) stationnarité : évolution de la taille au fil des événements

  Expérience 2 — modèle B
    2a. contrôles mathématiques
    2b. contextualité selon d et ν, référence à vecteurs aléatoires
    2c. (v11) contextualité selon la frustration des cycles dans A : frustrés, non frustrés voisins
        d'un cycle frustré, non frustrés isolés
    2d. (v11) balayage en βJ : densité de frustration de A contre fraction de cycles violant
        l'inégalité. Prédiction C3 : la contextualité suit la densité de frustration et s'annule
        avec elle, alors que la référence à vecteurs aléatoires en est indépendante

Usage :
    python experiences.py                       # v11, version rapide
    python experiences.py --complet             # v11, version complète
    python experiences.py --regles v10          # règles d'origine, pour comparaison
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modele_A import ModeleA, Parametres
import observables as obs
import modele_B as mb

SEUIL_RECOUVREMENT = 0.5
REGLES = "v11"
SORTIE = Path(__file__).parent / "resultats_v11"


def base(**kw) -> Parametres:
    if REGLES == "v10":
        P = Parametres(regles="v10", n0=1000, beta_J=0.6, mu=0.5, k_max=5, nu=0.3,
                       h0_biparti=False, dims=(1, 2, 3))
    else:
        P = Parametres(regles="v11", n0=1000, beta_J=0.3, k_max=4, nu=0.3, mu_carre=0.3,
                       mu_pendant=0.0, delta=0.1, delta_carre=0.05, epsilon=0.05,
                       h0_biparti=True, dims=(1, 2, 3),
                       beta_B=50.0, iterations_B=5, bruit_initial=0.05)
    for k, v in kw.items():
        setattr(P, k, v)
    return P


def nom_croissance() -> str:
    return "mu" if REGLES == "v10" else "mu_carre"


# ----------------------------------------------------------------------------
# Expérience 1
# ----------------------------------------------------------------------------

def exp_1a(graines, n0) -> dict:
    print("1a — non-terminaison")
    bJs = [0.3, 0.6, 1.0, 1.5]
    mus = [0.1, 0.2, 0.3, 0.5, 0.7]
    grille = np.zeros((len(bJs), len(mus)))
    for i, bJ in enumerate(bJs):
        for j, mu in enumerate(mus):
            absorbes = 0
            for g in graines:
                m = ModeleA(base(n0=n0, beta_J=bJ, dims=(), graine=g, **{nom_croissance(): mu}))
                absorbes += m.simuler(60 * n0, max_sommets=6 * n0) == "absorbant"
            grille[i, j] = absorbes / len(graines)
            print(f"   βJ={bJ:.1f} {nom_croissance()}={mu:.1f}  P(absorbant)={grille[i, j]:.2f}")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    im = ax.imshow(grille, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mus)), [str(m) for m in mus])
    ax.set_yticks(range(len(bJs)), [str(b) for b in bJs])
    ax.set_xlabel(nom_croissance())
    ax.set_ylabel("βJ")
    ax.set_title(f"Probabilité d'état absorbant ({REGLES})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(SORTIE / "1a_non_terminaison.png", dpi=140)
    plt.close(fig)
    return {"beta_J": bJs, nom_croissance(): mus, "P_absorbant": grille.tolist()}


def mesures_geometriques(C, rng) -> dict:
    """Les observables géométriques sont mesurées sur la composante géante : sur une structure
    fragmentée, les boules saturent à la taille des composantes et écrasent V(r), d_H et d_s."""
    adj_total = obs.graphe_primal(C)
    comps = obs.composantes(adj_total)
    adj = obs.restreindre(adj_total, comps[0])
    out = obs.stats_structure(C)
    out["n_composantes"] = len(comps)
    out["fraction_geante"] = len(comps[0]) / max(1, len(adj_total))
    out.update(obs.cyclicite(adj, rng))
    out.update(obs.croissance_volumique(adj, rng))
    out.update(obs.dimension_spectrale(adj, rng))
    return out


def exp_1c(m: ModeleA, rng) -> dict:
    """Invariance de section sur trois coupes admissibles de même cardinal."""
    C0 = m.C0
    e_racine = rng.choice(list(C0.aretes))
    region, front = {e_racine}, [e_racine]
    while front and len(region) < len(C0.aretes) // 2:
        nouveau = []
        for e in front:
            for f in C0.voisines(e):
                if f not in region:
                    region.add(f)
                    nouveau.append(f)
        front = nouveau
    cb = m.coupe_biaisee(region)
    n = len(cb)
    if n < 10:
        return {}
    ct = m.coupe_par_temps(n)
    cp = m.coupe_par_profondeur(n)
    assert m.est_fermee(ct) and m.est_fermee(cp) and m.est_fermee(cb)
    At = obs.graphe_primal(m.configuration_sur(ct))
    Ab = obs.graphe_primal(m.configuration_sur(cb))
    Ad = obs.graphe_primal(m.configuration_sur(cp))
    lt1, lt2, lb = obs.lois_locales(At, rng), obs.lois_locales(At, rng), obs.lois_locales(Ab, rng)
    rec = len(set(ct) & set(cb)) / n
    return {
        "evenements_coupe": n,
        "recouvrement_t_biaisee": rec,
        "recouvrement_t_profondeur": len(set(ct) & set(cp)) / n,
        "test_informatif": rec < SEUIL_RECOUVREMENT,
        "TV_degre_t_profondeur": obs.distance_variation_totale(
            obs.lois_locales(At, rng)["degre"], obs.lois_locales(Ad, rng)["degre"]),
        "TV_degre_entre_coupes": obs.distance_variation_totale(lt1["degre"], lb["degre"]),
        "TV_degre_bruit": obs.distance_variation_totale(lt1["degre"], lt2["degre"]),
        "TV_B2_entre_coupes": obs.distance_variation_totale(lt1["taille_B2"], lb["taille_B2"]),
        "TV_B2_bruit": obs.distance_variation_totale(lt1["taille_B2"], lt2["taille_B2"]),
        "beta1_r3_coupe_t": obs.cyclicite(At, rng, rayons=(3,))["beta1_par_sommet_r3"],
        "beta1_r3_coupe_biaisee": obs.cyclicite(Ab, rng, rayons=(3,))["beta1_par_sommet_r3"],
    }


def exp_1b_2b(nus, graines, n0, budget, v_max) -> list:
    print("1b/2b — géométrie et contextualité selon ν")
    lignes = []
    for nu in nus:
        for g in graines:
            t0 = time.time()
            m = ModeleA(base(n0=n0, nu=nu, graine=g))
            issue = m.simuler(budget, max_sommets=v_max)
            rng = random.Random(1000 + g)
            ligne = {"nu": nu, "graine": g, "issue": issue, "evenements": len(m.journal),
                     "profondeur_max": m.journal[-1].profondeur if m.journal else 0,
                     "trace": m.trace}
            if not m.journal or sum(1 for s in m.C.inc.values() if s) < 50:
                print(f"   ν={nu:.2f} g={g} {issue} — structure trop petite, ignorée")
                continue
            ligne.update(mesures_geometriques(m.C, rng))
            Crec = m.configuration_sur(range(len(m.journal)))
            ligne["reconstruction_exacte"] = (Crec.x == m.C.x and Crec.rho == m.C.rho
                                              and Crec.aretes == m.C.aretes)
            ligne["contexte"] = {d: mb.analyse_contextuelle(m.C, d) for d in (1, 2, 3)}
            ligne["contexte_aleatoire_d2"] = mb.analyse_contextuelle(
                m.C, 2, w=mb.vecteurs_aleatoires(m.C, 2, graine=g))
            ligne["section"] = exp_1c(m, rng)
            lignes.append(ligne)
            c2 = ligne["contexte"][2]
            fr = c2["frustration"]
            print(f"   ν={nu:.2f} g={g} {issue:9s} V={ligne['sommets']:5d} "
                  f"β1/V(r3)={ligne['beta1_par_sommet_r3']:.3f} "
                  f"dH={ligne.get('d_H_puissance', float('nan')):.2f} "
                  f"taux_exp={ligne.get('taux_exponentiel', float('nan')):.2f} "
                  f"ds={ligne['d_s_10_40']:.2f} | "
                  f"géante={ligne['fraction_geante']:.0%} | δ moyen : frustrés="
                  f"{fr['frustres']['defaut_moyen']:.3f} déséq.={fr['non_frustres_desequilibres']['defaut_moyen']:.3f} "
                  f"équil.={fr['non_frustres_equilibres']['defaut_moyen']:.3f} | "
                  f"recouvrement={ligne['section'].get('recouvrement_t_biaisee', float('nan')):.2f} "
                  f"({time.time() - t0:.0f}s)")
    return lignes


def exp_2d(graines, n0, budget, v_max, betas=(0.2, 0.3, 0.45, 0.6, 0.8)) -> list:
    """Balayage en βJ (ν fixé) : la densité de frustration de A pilote-t-elle la contextualité de B ?"""
    print("2d — densité de frustration contre contextualité")
    points = []
    for bJ in betas:
        for g in graines:
            m = ModeleA(base(n0=n0, beta_J=bJ, nu=0.6, dims=(2,), graine=g))
            issue = m.simuler(budget, max_sommets=v_max)
            if sum(1 for s in m.C.inc.values() if s) < 50:
                continue
            a = mb.analyse_contextuelle(m.C, 2)
            r = mb.analyse_contextuelle(m.C, 2, w=mb.vecteurs_aleatoires(m.C, 2, graine=g))
            points.append({"beta_J": bJ, "graine": g, "issue": issue,
                           "densite_frustration": a["densite_frustration"],
                           "violation_R5": a["fraction_violee_determines"],
                           "violation_aleatoire": r["fraction_violee_determines"]})
            print(f"   βJ={bJ:.2f} g={g} {issue:9s} frustration={a['densite_frustration']:.2f} "
                  f"violations R5′={a['fraction_violee_determines']:.2f} "
                  f"aléatoire={r['fraction_violee_determines']:.2f}")
    if points:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter([p["densite_frustration"] for p in points], [p["violation_R5"] for p in points],
                   label="relaxation R5′")
        ax.scatter([p["densite_frustration"] for p in points], [p["violation_aleatoire"] for p in points],
                   marker="x", c="k", label="vecteurs aléatoires")
        ax.set_xlabel("densité de frustration (cycles courts frustrés)")
        ax.set_ylabel("fraction de cycles violant l'inégalité")
        ax.set_title("Prédiction C3 : contextualité ∝ frustration")
        ax.legend()
        fig.tight_layout()
        fig.savefig(SORTIE / "2d_frustration_contextualite.png", dpi=140)
        plt.close(fig)
    return points


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def figures(lignes, nus):
    if not lignes:
        return
    nus = [nu for nu in nus if any(l["nu"] == nu for l in lignes)]

    def moy(f, nu):
        vals = [f(l) for l in lignes if l["nu"] == nu]
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    ax = axs[0, 0]
    for r in (2, 3, 4):
        m_s = [moy(lambda l: l[f"beta1_par_sommet_r{r}"], nu) for nu in nus]
        ax.errorbar(nus, [a for a, _ in m_s], [b for _, b in m_s], marker="o", label=f"r = {r}")
    ax.set_xlabel("ν")
    ax.set_ylabel("β1(B_r) / |B_r|")
    ax.set_title("Cyclicité locale")
    ax.legend()

    ax = axs[0, 1]
    for nu in nus:
        V = np.mean([l["V"] for l in lignes if l["nu"] == nu], axis=0)
        r = np.arange(1, len(V))
        ax.loglog(r, V[1:], marker=".", label=f"ν = {nu}")
    ax.set_xlabel("r")
    ax.set_ylabel("V(r)")
    ax.set_title("Croissance volumique (log-log : droite = loi de puissance)")
    ax.legend(fontsize=8)

    ax = axs[0, 2]
    for nu in nus:
        P = np.mean([l["P_retour"] for l in lignes if l["nu"] == nu], axis=0)
        s = np.arange(1, len(P))
        ax.semilogx(s, -2 * np.gradient(np.log(P[1:]), np.log(s)), label=f"ν = {nu}")
    ax.set_xlabel("s")
    ax.set_ylabel("d_s(s)")
    ax.set_title("Dimension spectrale (plateau attendu)")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    for l in lignes:
        if l["trace"]:
            tr = np.array(l["trace"])
            ax.plot(tr[:, 0], tr[:, 1], lw=0.8, alpha=0.7)
    ax.set_xlabel("événements (indice computationnel)")
    ax.set_ylabel("sommets actifs")
    ax.set_title("Stationnarité de la taille")

    ax = axs[1, 1]
    classes = (("frustres", "frustrés"), ("non_frustres_desequilibres", "non frustrés, comp. déséquilibrée"),
               ("non_frustres_equilibres", "non frustrés, comp. équilibrée"))
    largeur = 0.2
    x = np.arange(len(nus))
    for j, (cle, lab) in enumerate(classes):
        vals = [moy(lambda l: l["contexte"][2]["frustration"][cle]["defaut_moyen"], nu)[0]
                for nu in nus]
        ax.bar(x + (j - 1.5) * largeur, vals, largeur, label=f"{lab} (R5′)")
    vals = [moy(lambda l: l["contexte_aleatoire_d2"]["frustration"]["non_frustres_equilibres"]["defaut_moyen"], nu)[0]
            for nu in nus]
    ax.bar(x + 1.5 * largeur, vals, largeur, color="k", alpha=0.5, label="comp. équilibrée, aléatoire")
    ax.axhline(2 * np.sqrt(2) - 2, color="r", ls="--", lw=1, label="2√2 − 2 (4-cycle frustré isolé)")
    ax.set_xticks(x, [str(nu) for nu in nus])
    ax.set_xlabel("ν")
    ax.set_ylabel("défaut moyen δ")
    ax.set_title("Prédiction C3 : défaut porté par la frustration (d = 2)")
    ax.legend(fontsize=7)

    ax = axs[1, 2]
    for d, mk in ((1, "s"), (2, "o"), (3, "^")):
        ax.scatter([l["beta1_par_sommet_r3"] for l in lignes],
                   [l["contexte"][d]["fraction_sommets_defaut_positif"] for l in lignes],
                   marker=mk, label=f"d = {d}, R5′")
    ax.scatter([l["beta1_par_sommet_r3"] for l in lignes],
               [l["contexte_aleatoire_d2"]["fraction_sommets_defaut_positif"] for l in lignes],
               marker="x", c="k", label="d = 2, aléatoire")
    ax.set_xlabel("cyclicité β1(B_3)/|B_3|")
    ax.set_ylabel("fraction de sommets avec δ° > τ")
    ax.set_title("Co-émergence")
    ax.legend(fontsize=8)

    fig.suptitle(f"Modèles jouets — règles {REGLES}")
    fig.tight_layout()
    fig.savefig(SORTIE / "1b_2c_synthese.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------

def main():
    global REGLES, SORTIE
    ap = argparse.ArgumentParser()
    ap.add_argument("--complet", action="store_true")
    ap.add_argument("--regles", choices=("v10", "v11"), default="v11")
    a = ap.parse_args()
    REGLES = a.regles
    SORTIE = Path(__file__).parent / f"resultats_{REGLES}"
    SORTIE.mkdir(exist_ok=True)
    nus = [0.0, 0.1, 0.3, 0.6, 1.0]
    if a.complet:
        graines, n0, graines_1a, n0_1a = range(6), 3000, range(6), 600
    else:
        graines, n0, graines_1a, n0_1a = range(3), 1000, range(4), 300
    budget, v_max = 200 * n0, 20 * n0

    res = {"regles": REGLES, "parametres_base": vars(base())}
    print("2a — contrôles mathématiques du modèle B")
    res["2a"] = {"foret": mb.controle_foret(), "inegalites_de_cycle": mb.controle_inegalites_de_cycle(),
                 "tsirelson": mb.controle_tsirelson(), "relaxation_C3": mb.controle_relaxation()}
    print("   ", res["2a"])
    res["1a"] = exp_1a(graines_1a, n0_1a)
    lignes = exp_1b_2b(nus, graines, n0, budget, v_max)
    res["1b_2b"] = lignes
    figures(lignes, nus)
    if REGLES == "v11":
        res["2d"] = exp_2d(graines, n0, budget // 2, v_max)
    with open(SORTIE / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: str(o))
    print(f"Résultats écrits dans {SORTIE}")


if __name__ == "__main__":
    main()
