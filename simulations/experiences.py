"""
Les deux expériences annoncées dans le résumé :

  Expérience 1 — modèle A
    1a. diagramme de non-terminaison en (βJ, μ)                       [test 6]
    1b. cyclicité, croissance volumique et dimension spectrale selon ν [test 9, expérience G]
    1c. invariance de section : coupe par t contre coupe par l'ordre dérivé [critère 3, expérience F2]
    1d. contrôle de reconstruction : rejouer tout le journal redonne la configuration finale

  Expérience 2 — modèle B (sur les structures produites par 1b)
    2a. contrôles mathématiques : forêts, inégalités de cycle (programme linéaire), Tsirelson
    2b. contextualité locale selon d (d = 1 : contrôle classique) et selon ν
    2c. co-émergence : capacité contextuelle contre cyclicité, d'une phase à l'autre

Usage :  python experiences.py            (version rapide, quelques minutes)
         python experiences.py --complet  (tailles et graines plus grandes)
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

SORTIE = Path(__file__).parent / "resultats"


def base(**kw) -> Parametres:
    """Point de fonctionnement retenu : régime actif (non absorbant) du balayage 1a."""
    P = Parametres(n0=1000, beta_J=0.6, mu=0.5, k_max=5, nu=0.3, eta=0.15,
                   rho0=0.5, p=0.25, dims=(1, 2, 3), kappa=0.2)
    for k, v in kw.items():
        setattr(P, k, v)
    return P


# ----------------------------------------------------------------------------
# Expérience 1
# ----------------------------------------------------------------------------

def exp_1a(graines, n0) -> dict:
    print("1a — non-terminaison")
    bJs, mus = [0.3, 0.6, 1.0, 1.5], [0.1, 0.2, 0.3, 0.5, 0.7]
    grille = np.zeros((len(bJs), len(mus)))
    for i, bJ in enumerate(bJs):
        for j, mu in enumerate(mus):
            absorbes = 0
            for g in graines:
                m = ModeleA(base(n0=n0, beta_J=bJ, mu=mu, dims=(), graine=g))
                absorbes += m.simuler(60 * n0, max_sommets=6 * n0) == "absorbant"
            grille[i, j] = absorbes / len(graines)
            print(f"   βJ={bJ:.1f} μ={mu:.1f}  P(absorbant)={grille[i, j]:.2f}")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    im = ax.imshow(grille, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mus)), [str(m) for m in mus])
    ax.set_yticks(range(len(bJs)), [str(b) for b in bJs])
    ax.set_xlabel("μ (croissance R3)")
    ax.set_ylabel("βJ")
    ax.set_title("Probabilité d'état absorbant U = ∅")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(SORTIE / "1a_non_terminaison.png", dpi=140)
    plt.close(fig)
    return {"beta_J": bJs, "mu": mus, "P_absorbant": grille.tolist()}


def mesures_geometriques(C, rng) -> dict:
    adj = obs.graphe_primal(C)
    out = obs.stats_structure(C)
    out.update(obs.cyclicite(adj, rng))
    out.update(obs.croissance_volumique(adj, rng))
    out.update(obs.dimension_spectrale(adj, rng))
    return out


def exp_1b_2b(nus, graines, n0, v_max) -> list:
    print("1b/2b — géométrie et contextualité selon ν")
    lignes = []
    for nu in nus:
        for g in graines:
            t0 = time.time()
            m = ModeleA(base(n0=n0, nu=nu, graine=g))
            issue = m.simuler(200 * n0, max_sommets=v_max)
            rng = random.Random(1000 + g)
            ligne = {"nu": nu, "graine": g, "issue": issue, "evenements": len(m.journal),
                     "profondeur_max": m.journal[-1].profondeur if m.journal else 0}
            ligne.update(mesures_geometriques(m.C, rng))
            # 1d — reconstruction exacte sur la coupe totale
            Crec = m.configuration_sur(range(len(m.journal)))
            ligne["reconstruction_exacte"] = (Crec.x == m.C.x and Crec.rho == m.C.rho
                                              and Crec.aretes == m.C.aretes)
            # 2b — contextualité, couches d = 1, 2, 3, et état initial pour d = 2
            ligne["contexte"] = {d: mb.analyse_contextuelle(m.C, d) for d in (1, 2, 3)}
            ligne["contexte_initial_d2"] = mb.analyse_contextuelle(m.C0, 2)
            ligne["contexte_aleatoire_d2"] = mb.analyse_contextuelle(
                m.C, 2, w=mb.vecteurs_aleatoires(m.C, 2, graine=g))
            # 1c — invariance de section
            ligne["section"] = exp_1c(m, rng)
            lignes.append(ligne)
            c2 = ligne["contexte"][2]
            print(f"   ν={nu:.2f} g={g} {issue:9s} V={ligne['sommets']:5d} "
                  f"β1/V(r3)={ligne['beta1_par_sommet_r3']:.3f} "
                  f"taux_exp={ligne.get('taux_exponentiel', float('nan')):.2f} "
                  f"ds={ligne['d_s_10_40']:.2f}  "
                  f"4-cycles={c2['nb_4cycles']} viol.CHSH={c2['fraction_CHSH_viole']:.2f} "
                  f"Smax={c2['S_max']:.3f}  ({time.time() - t0:.0f}s)")
    return lignes


def exp_1c(m: ModeleA, rng) -> dict:
    """Compare les lois locales sur des coupes admissibles contenant le même nombre d'événements :
    préfixe en t (feuilletage computationnel), coupe par profondeur dans l'ordre dérivé, et
    coupe biaisée où une moitié de H0 est très en avance sur l'autre. La comparaison principale
    (t contre biaisée) porte sur deux coupes qui se recouvrent peu. Le bruit d'échantillonnage
    est estimé en comparant deux tirages de racines sur la même coupe."""
    # coupe biaisée : une région de H0 (boule autour d'une hyperarête) avancée, le reste en retard
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
    Ap = obs.graphe_primal(m.configuration_sur(cb))   # comparaison principale : t contre biaisée
    Ad = obs.graphe_primal(m.configuration_sur(cp))
    lt1, lt2 = obs.lois_locales(At, rng), obs.lois_locales(At, rng)
    lp = obs.lois_locales(Ap, rng)
    return {
        "evenements_coupe": n,
        "recouvrement_t_biaisee": len(set(ct) & set(cb)) / n,
        "recouvrement_t_profondeur": len(set(ct) & set(cp)) / n,
        "TV_degre_t_profondeur": obs.distance_variation_totale(
            obs.lois_locales(At, rng)["degre"], obs.lois_locales(Ad, rng)["degre"]),
        "TV_degre_entre_coupes": obs.distance_variation_totale(lt1["degre"], lp["degre"]),
        "TV_degre_bruit": obs.distance_variation_totale(lt1["degre"], lt2["degre"]),
        "TV_B2_entre_coupes": obs.distance_variation_totale(lt1["taille_B2"], lp["taille_B2"]),
        "TV_B2_bruit": obs.distance_variation_totale(lt1["taille_B2"], lt2["taille_B2"]),
        "beta1_r3_coupe_t": obs.cyclicite(At, rng, rayons=(3,))["beta1_par_sommet_r3"],
        "beta1_r3_coupe_biaisee": obs.cyclicite(Ap, rng, rayons=(3,))["beta1_par_sommet_r3"],
    }


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def figures(lignes, nus):
    def moy(cle, nu, f=lambda l, c: l[c]):
        vals = [f(l, cle) for l in lignes if l["nu"] == nu]
        return np.mean(vals), np.std(vals)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    # cyclicité
    ax = axs[0, 0]
    for r in (2, 3, 4):
        m_s = [moy(f"beta1_par_sommet_r{r}", nu) for nu in nus]
        ax.errorbar(nus, [a for a, _ in m_s], [b for _, b in m_s], marker="o", label=f"r = {r}")
    ax.set_xlabel("ν (fermeture R4)")
    ax.set_ylabel("β1(B_r) / |B_r|")
    ax.set_title("Cyclicité locale")
    ax.legend()
    # croissance volumique
    ax = axs[0, 1]
    for nu in nus:
        V = np.mean([l["V"] for l in lignes if l["nu"] == nu], axis=0)
        ax.semilogy(np.arange(len(V)), V, marker=".", label=f"ν = {nu}")
    ax.set_xlabel("r")
    ax.set_ylabel("V(r) (échelle log)")
    ax.set_title("Croissance volumique : droite = exponentielle")
    ax.legend(fontsize=8)
    # dimension spectrale
    ax = axs[1, 0]
    for nu in nus:
        P = np.mean([l["P_retour"] for l in lignes if l["nu"] == nu], axis=0)
        s = np.arange(1, len(P))
        ax.semilogx(s, -2 * np.gradient(np.log(P[1:]), np.log(s)), label=f"ν = {nu}")
    ax.set_xlabel("s (paramètre de diffusion)")
    ax.set_ylabel("d_s(s)")
    ax.set_title("Dimension spectrale")
    ax.legend(fontsize=8)
    # co-émergence
    ax = axs[1, 1]
    for d, mk in ((1, "s"), (2, "o"), (3, "^")):
        xs = [l["beta1_par_sommet_r3"] for l in lignes]
        ys = [l["contexte"][d]["fraction_sommets_defaut_positif"] for l in lignes]
        ax.scatter(xs, ys, marker=mk, label=f"d = {d}, dynamique R5")
    ax.scatter([l["beta1_par_sommet_r3"] for l in lignes],
               [l["contexte_aleatoire_d2"]["fraction_sommets_defaut_positif"] for l in lignes],
               marker="x", c="k", label="d = 2, vecteurs aléatoires")
    ax.set_xlabel("cyclicité β1(B_3)/|B_3|")
    ax.set_ylabel("fraction de sommets avec δ° > 0")
    ax.set_title("Co-émergence : contextualité locale / cyclicité")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SORTIE / "1b_2c_geometrie_coemergence.png", dpi=140)
    plt.close(fig)

    # distribution des valeurs CHSH sur les 4-cycles (d = 2), état final vs initial
    fig, ax = plt.subplots(figsize=(6, 3.8))
    nu_max = max(nus)
    for l in lignes:
        if l["nu"] == nu_max:
            break
    vals_f = l["contexte"][2]
    bins = np.linspace(0, 3, 61)
    ax.hist(l["contexte_initial_d2"]["S_echantillon"], bins=bins, alpha=0.5, density=True,
            label="état initial (H0)")
    ax.hist(vals_f["S_echantillon"], bins=bins, alpha=0.5, density=True, label="état final (R5)")
    ax.hist(l["contexte_aleatoire_d2"]["S_echantillon"], bins=bins, histtype="step", lw=1.5,
            density=True, color="k", label="même graphe, vecteurs aléatoires")
    ax.axvline(2, color="k", ls="--", lw=1, label="borne classique 2")
    ax.axvline(2 * np.sqrt(2), color="r", ls="--", lw=1, label="Tsirelson 2√2")
    ax.set_title(f"4-cycles, d = 2, ν = {nu_max} : S_max = {vals_f['S_max']:.3f}, "
                 f"violations = {vals_f['fraction_CHSH_viole']:.2f}")
    ax.set_xlim(0, 3)
    ax.set_xlabel("S (valeur maximale du 4-cycle)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SORTIE / "2b_reperes_CHSH.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complet", action="store_true")
    a = ap.parse_args()
    SORTIE.mkdir(exist_ok=True)
    if a.complet:
        graines, n0, v_max, nus = range(6), 3000, 20000, [0.0, 0.1, 0.3, 0.6, 1.0]
        graines_1a, n0_1a = range(6), 600
    else:
        graines, n0, v_max, nus = range(3), 1000, 6000, [0.0, 0.1, 0.3, 0.6, 1.0]
        graines_1a, n0_1a = range(4), 300

    res = {"parametres_base": vars(base())}
    print("2a — contrôles mathématiques du modèle B")
    res["2a"] = {"foret": mb.controle_foret(), "inegalites_de_cycle": mb.controle_inegalites_de_cycle(),
                 "tsirelson": mb.controle_tsirelson()}
    print("   ", res["2a"])
    res["1a"] = exp_1a(graines_1a, n0_1a)
    lignes = exp_1b_2b(nus, graines, n0, v_max)
    res["1b_2b"] = lignes
    figures(lignes, nus)
    with open(SORTIE / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: str(o))
    print(f"Résultats écrits dans {SORTIE}")


if __name__ == "__main__":
    main()
