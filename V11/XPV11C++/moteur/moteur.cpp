// Moteur C++ des modèles jouets — règles v11 (voir corrections_modeles_v11.tex).
//
// Réimplémentation fidèle de modele_A.py (regles="v11") :
//   blocs de couplage, spécification de Gibbs locale, R1, R2 (plancher ε), R3' (carrés),
//   croissance pendante résiduelle, R4' (fermeture de 4-cycles), R6 (suppression sans pont),
//   R5' (relaxation de Gibbs O(d) de la couche B), horloges de taux λ0|Λ|,
//   dépendance lecture/écriture (C1), journal optionnel et reconstruction sur coupe admissible.
// Observables lourdes en C++ : composantes, volume des boules, cyclicité, contextualité locale.
//
// Interface C (extern "C") appelée depuis Python par ctypes : appels grossiers (simuler N
// événements, exporter des tableaux), donc aucun surcoût d'interface mesurable.
//
// Compilation : voir CMakeLists.txt ou Makefile.

#include <algorithm>
#include <array>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

using std::vector;
using u64 = uint64_t;

// ---------------------------------------------------------------------------
// Paramètres (disposition identique à la Structure ctypes de moteur_rapide.py)
// ---------------------------------------------------------------------------
struct Params {
    int32_t n0, k0, h0_germe, k_max, taille_exacte, balayages_gibbs, iterations_B, rayon_B;
    int32_t L_alternatif, trace_tous_les, journal, ndims;
    int32_t dims[4];
    double p, rho0, beta_J, eta, lambda0, nu, mu_carre, mu_pendant, delta, delta_carre,
        epsilon, beta_B, bruit_initial, lambda_reouverture;
    u64 graine;
    // variantes de règles (diagnostic de phase) : 1 = condition active (règle v11 stricte)
    int32_t r3_concordance, r4_concordance, r4_frontiere, reserve;
};

static inline u64 cle_paire(int e, int f) {
    if (e > f) std::swap(e, f);
    return ((u64)(uint32_t)e << 32) | (uint32_t)f;
}

// Items de lecture/écriture (C1) : 2 bits de type + 62 bits de charge utile
enum : u64 { IT_X = 0, IT_R = 1, IT_INC = 2, IT_W = 3 };
static inline u64 item(u64 type, u64 charge) { return (type << 62) | (charge & ((1ULL << 62) - 1)); }
static inline u64 item_r(int e, int f) {
    if (e > f) std::swap(e, f);
    return item(IT_R, ((u64)(uint32_t)e << 31) | (uint32_t)f);
}

// ---------------------------------------------------------------------------
// État
// ---------------------------------------------------------------------------
struct Etat {
    vector<int32_t> ea, eb;        // extrémités des arêtes
    vector<int8_t> x;              // issue : -1, +1, 0 (ouverte)
    vector<uint8_t> vivante;
    vector<vector<int32_t>> inc;   // sommet -> arêtes
    std::unordered_map<u64, double> rho;
    int ndims = 0;
    int dims[4] = {0, 0, 0, 0};
    vector<double> w[4];           // vecteurs de la couche B, nV x d
    int64_t actifs = 0;            // sommets de degré > 0
    int64_t n_vivantes = 0;

    int nV() const { return (int)inc.size(); }
    int nE() const { return (int)ea.size(); }

    int autre(int e, int v) const { return ea[e] == v ? eb[e] : ea[e]; }
    double r(int e, int f) const {
        auto it = rho.find(cle_paire(e, f));
        return it == rho.end() ? 0.0 : it->second;
    }
    void set_rho(int e, int f, double val) {
        if (val > 0) rho[cle_paire(e, f)] = val;
        else rho.erase(cle_paire(e, f));
    }
    int nouveau_sommet() {
        inc.emplace_back();
        for (int k = 0; k < ndims; ++k) w[k].resize((size_t)inc.size() * dims[k], 0.0);
        return (int)inc.size() - 1;
    }
    void ajouter_inc(int v, int e) {
        if (inc[v].empty()) ++actifs;
        inc[v].push_back(e);
    }
    void retirer_inc(int v, int e) {
        auto& L = inc[v];
        for (size_t i = 0; i < L.size(); ++i)
            if (L[i] == e) { L[i] = L.back(); L.pop_back(); break; }
        if (L.empty()) --actifs;
    }
    int ajouter_arete(int a, int b, int eid = -1) {
        if (eid < 0) eid = nE();
        if (eid >= nE()) {
            ea.resize(eid + 1, -1); eb.resize(eid + 1, -1);
            x.resize(eid + 1, 0); vivante.resize(eid + 1, 0);
        }
        ea[eid] = a; eb[eid] = b; x[eid] = 0; vivante[eid] = 1;
        ajouter_inc(a, eid); ajouter_inc(b, eid);
        ++n_vivantes;
        return eid;
    }
    template <class F> void pour_voisines(int e, F f) const {
        for (int v : {ea[e], eb[e]})
            for (int k : inc[v]) if (k != e) f(k);
    }
    void supprimer_arete(int e) {
        vector<int> vs;
        pour_voisines(e, [&](int f) { vs.push_back(f); });
        for (int f : vs) rho.erase(cle_paire(e, f));
        retirer_inc(ea[e], e); retirer_inc(eb[e], e);
        vivante[e] = 0; x[e] = 0;
        --n_vivantes;
    }
    bool adjacents(int u, int v) const {
        for (int k : inc[u]) if (autre(k, u) == v) return true;
        return false;
    }
    // voisins(a) ∩ voisins(c) == {b}
    bool voisins_communs_egal(int a, int c, int b) const {
        bool vu_b = false;
        for (int k : inc[a]) {
            int z = autre(k, a);
            if (adjacents(c, z)) {
                if (z != b) return false;
                vu_b = true;
            }
        }
        return vu_b;
    }
    bool dans_un_carre(int e) const {
        int a = ea[e], b = eb[e];
        for (int k1 : inc[a]) {
            int c = autre(k1, a);
            if (c == b) continue;
            for (int k2 : inc[c]) {
                int d = autre(k2, c);
                if (d == a || d == b) continue;
                if (adjacents(b, d)) return true;
            }
        }
        return false;
    }
    double* wv(int k, int v) { return &w[k][(size_t)v * dims[k]]; }
    const double* wv(int k, int v) const { return &w[k][(size_t)v * dims[k]]; }
};

// ---------------------------------------------------------------------------
// Journal
// ---------------------------------------------------------------------------
enum : uint8_t { W_X = 0, W_RHO = 1, W_ARETE = 2, W_SUPPR = 3, W_SOMMET = 4, W_VEC = 5 };
struct Ecriture {
    uint8_t type;
    int32_t a, b;        // selon le type : (e, val) / (e, f) / (e, sommet a, sommet b via c) ...
    int32_t c;
    double val;
    int64_t pool;        // décalage dans le réservoir de vecteurs (W_SOMMET, W_VEC)
};

struct Journal {
    vector<double> t;
    vector<int32_t> profondeur, graine_bloc, taille_bloc;
    vector<int64_t> pred_off{0};
    vector<int32_t> preds;
    vector<int64_t> ecr_off{0};
    vector<Ecriture> ecr;
    vector<double> pool;
};

// ---------------------------------------------------------------------------
// Moteur
// ---------------------------------------------------------------------------
struct Moteur {
    Params P;
    Etat C, C0;
    std::mt19937_64 rng;
    double t = 0;
    vector<int32_t> U, posU;
    Journal J;
    std::unordered_map<u64, int32_t> dernier_ecrivain;
    std::unordered_map<u64, vector<int32_t>> lecteurs;
    vector<int64_t> trace;       // (événement, actifs, arêtes, |U|) x n
    int64_t n_evenements = 0;
    // marqueurs réutilisables
    vector<int32_t> marque_e, marque_b;
    int32_t tampon_e = 1, tampon_b = 1;

    std::uniform_real_distribution<double> unif{0.0, 1.0};
    std::normal_distribution<double> norm{0.0, 1.0};

    double U01() { return unif(rng); }

    explicit Moteur(const Params& p) : P(p), rng(p.graine) {
        C.ndims = P.ndims;
        for (int k = 0; k < P.ndims; ++k) C.dims[k] = P.dims[k];
        etat_initial();
        C0 = C;
        for (int e = 0; e < C.nE(); ++e) if (C.vivante[e] && C.x[e] == 0) ouvrir(e);
    }

    // --- vecteurs
    void vecteur_initial(int k, double* out) {
        int d = C.dims[k];
        if (d == 1) { out[0] = 1.0; return; }
        double n2 = 0;
        for (int i = 0; i < d; ++i) {
            out[i] = (i == 0 ? 1.0 : 0.0) + P.bruit_initial * norm(rng);
            n2 += out[i] * out[i];
        }
        double n = std::sqrt(n2);
        for (int i = 0; i < d; ++i) out[i] /= n;
    }
    int nouveau_sommet(const vector<int>& parents) {
        int v = C.nouveau_sommet();
        for (int k = 0; k < C.ndims; ++k) {
            int d = C.dims[k];
            double* o = C.wv(k, v);
            if (parents.empty()) { vecteur_initial(k, o); continue; }
            vector<double> m(d, 0.0);
            for (int u : parents) { const double* wu = C.wv(k, u); for (int i = 0; i < d; ++i) m[i] += wu[i]; }
            for (int i = 0; i < d; ++i) m[i] /= parents.size();
            if (d == 1) { o[0] = m[0] >= 0 ? 1.0 : -1.0; continue; }
            double n2 = 0;
            for (int i = 0; i < d; ++i) { m[i] += P.bruit_initial * norm(rng); n2 += m[i] * m[i]; }
            double n = std::sqrt(n2);
            if (n > 1e-12) for (int i = 0; i < d; ++i) o[i] = m[i] / n;
            else vecteur_initial(k, o);
        }
        return v;
    }

    // --- état initial
    void etat_initial() {
        for (int i = 0; i < P.n0; ++i) nouveau_sommet({});
        std::unordered_set<u64> vues;
        auto ajouter = [&](int a, int b) {
            if (a == b) return;
            u64 k = cle_paire(a, b);
            if (vues.count(k)) return;
            vues.insert(k);
            C.ajouter_arete(std::min(a, b), std::max(a, b));
        };
        if (P.h0_germe) {
            ajouter(0, 1); ajouter(1, 2); ajouter(2, 3); ajouter(3, 0);
        } else {
            int h = P.n0 / 2;
            vector<int> dg, dd;
            for (int v = 0; v < h; ++v) for (int j = 0; j < P.k0; ++j) dg.push_back(v);
            for (int v = h; v < P.n0; ++v) for (int j = 0; j < P.k0; ++j) dd.push_back(v);
            std::shuffle(dg.begin(), dg.end(), rng);
            std::shuffle(dd.begin(), dd.end(), rng);
            for (size_t i = 0; i < std::min(dg.size(), dd.size()); ++i) ajouter(dg[i], dd[i]);
        }
        for (int e = 0; e < C.nE(); ++e) {
            vector<int> vs;
            C.pour_voisines(e, [&](int f) { if (e < f) vs.push_back(f); });
            for (int f : vs) if (U01() < P.p) C.set_rho(e, f, P.rho0);
        }
    }

    // --- U
    void ouvrir(int e) {
        if (e >= (int)posU.size()) posU.resize(e + 1, -1);
        if (posU[e] >= 0) return;
        posU[e] = (int)U.size();
        U.push_back(e);
    }
    void fermer(int e) {
        if (e >= (int)posU.size() || posU[e] < 0) return;
        int i = posU[e];
        int dernier = U.back();
        U.pop_back();
        if (dernier != e) { U[i] = dernier; posU[dernier] = i; }
        posU[e] = -1;
    }

    // --- marqueurs d'arêtes
    void nouveau_tampon() {
        if ((int)marque_e.size() < C.nE()) marque_e.resize(C.nE() * 2 + 16, 0);
        if (++tampon_e == INT32_MAX) { std::fill(marque_e.begin(), marque_e.end(), 0); tampon_e = 1; }
    }

    // --- von Mises–Fisher (algorithme de Wood)
    void tirer_vmf(const double* mu, double kappa, int d, double* out) {
        if (d == 1) {
            out[0] = U01() < 1.0 / (1.0 + std::exp(-2.0 * kappa * mu[0])) ? 1.0 : -1.0;
            return;
        }
        if (kappa < 1e-9) {
            double n2 = 0;
            for (int i = 0; i < d; ++i) { out[i] = norm(rng); n2 += out[i] * out[i]; }
            double n = std::sqrt(n2);
            for (int i = 0; i < d; ++i) out[i] /= n;
            return;
        }
        double b = (-2 * kappa + std::sqrt(4 * kappa * kappa + (double)(d - 1) * (d - 1))) / (d - 1);
        double x0 = (1 - b) / (1 + b);
        double c = kappa * x0 + (d - 1) * std::log(1 - x0 * x0);
        std::gamma_distribution<double> g((d - 1) / 2.0, 1.0);
        double W;
        while (true) {
            double g1 = g(rng), g2 = g(rng);
            double z = g1 / (g1 + g2);
            W = (1 - (1 + b) * z) / (1 - (1 - b) * z);
            if (kappa * W + (d - 1) * std::log(1 - x0 * W) - c >= std::log(U01())) break;
        }
        double v[16];
        double dot = 0, n2 = 0;
        for (int i = 0; i < d; ++i) { v[i] = norm(rng); dot += v[i] * mu[i]; }
        for (int i = 0; i < d; ++i) { v[i] -= dot * mu[i]; n2 += v[i] * v[i]; }
        double n = std::sqrt(n2), s = std::sqrt(std::max(0.0, 1 - W * W));
        for (int i = 0; i < d; ++i) out[i] = W * mu[i] + s * v[i] / n;
    }

    // --- Gibbs local du bloc
    bool dans_bord(int f) const { return f < (int)marque_b.size() && marque_b[f] == tampon_b; }

    void tirer_bloc(const vector<int>& lam, std::unordered_map<int, int>& idx, vector<int>& sigma) {
        int n = (int)lam.size();
        vector<double> Jm((size_t)n * n, 0.0), h(n, 0.0);
        for (int i = 0; i < n; ++i) {
            int e = lam[i];
            C.pour_voisines(e, [&](int f) {
                auto it = idx.find(f);
                if (it != idx.end()) Jm[(size_t)i * n + it->second] = C.r(e, f);
                else if (dans_bord(f)) h[i] += C.r(e, f) * C.x[f];
            });
        }
        sigma.assign(n, 1);
        double bJ = P.beta_J;
        if (n <= P.taille_exacte) {
            int64_t N = 1LL << n;
            vector<double> E(N);
            double Emin = 1e300;
            vector<int> s(n);
            for (int64_t m = 0; m < N; ++m) {
                for (int i = 0; i < n; ++i) s[i] = (m >> i) & 1 ? 1 : -1;
                double en = 0;
                for (int i = 0; i < n; ++i) {
                    en -= h[i] * s[i];
                    for (int j = i + 1; j < n; ++j) en -= Jm[(size_t)i * n + j] * s[i] * s[j];
                }
                E[m] = en;
                Emin = std::min(Emin, en);
            }
            double Z = 0;
            for (int64_t m = 0; m < N; ++m) { E[m] = std::exp(-bJ * (E[m] - Emin)); Z += E[m]; }
            double u = U01() * Z, acc = 0;
            int64_t choix = N - 1;
            for (int64_t m = 0; m < N; ++m) { acc += E[m]; if (acc >= u) { choix = m; break; } }
            for (int i = 0; i < n; ++i) sigma[i] = (choix >> i) & 1 ? 1 : -1;
        } else {
            for (int i = 0; i < n; ++i) sigma[i] = U01() < 0.5 ? 1 : -1;
            vector<int> perm(n);
            std::iota(perm.begin(), perm.end(), 0);
            for (int sw = 0; sw < P.balayages_gibbs; ++sw) {
                std::shuffle(perm.begin(), perm.end(), rng);
                for (int i : perm) {
                    double champ = h[i];
                    for (int j = 0; j < n; ++j) champ += Jm[(size_t)i * n + j] * sigma[j];
                    sigma[i] = U01() < 1.0 / (1.0 + std::exp(-2 * bJ * champ)) ? 1 : -1;
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Événement composite a_Λ
    // ------------------------------------------------------------------------
    vector<u64> R, Wt;
    vector<Ecriture> ecr_ev;
    vector<double> pool_ev;

    // sans journal, le suivi des lectures et écritures est inutile : il est désactivé
    void lire(u64 it) { if (P.journal) R.push_back(it); }
    void ecrire(u64 it) { if (P.journal) Wt.push_back(it); }
    void lire_sommet(int v) { lire(item(IT_INC, v)); }
    void lire_arete(int e) { lire(item(IT_X, e)); lire_sommet(C.ea[e]); lire_sommet(C.eb[e]); }
    void lire_voisinage(int e) {
        if (!P.journal) return;
        lire_arete(e);
        C.pour_voisines(e, [&](int f) { lire(item(IT_X, f)); lire(item_r(e, f)); });
    }
    void ecrire_x(int e, int val) {
        C.x[e] = (int8_t)val;
        ecrire(item(IT_X, e));
        if (P.journal) ecr_ev.push_back({W_X, e, val, 0, 0.0, 0});
        if (val == 0) ouvrir(e); else fermer(e);
    }
    void ecrire_rho(int e, int f, double val) {
        C.set_rho(e, f, val);
        ecrire(item_r(e, f));
        if (P.journal) ecr_ev.push_back({W_RHO, e, f, 0, val, 0});
    }
    int creer_arete(int a, int b, const vector<int>& couples) {
        int eid = C.ajouter_arete(std::min(a, b), std::max(a, b));
        ouvrir(eid);
        if (P.journal) ecr_ev.push_back({W_ARETE, eid, std::min(a, b), std::max(a, b), 0.0, 0});
        ecrire(item(IT_X, eid)); ecrire(item(IT_INC, a)); ecrire(item(IT_INC, b));
        for (int g : couples) if (g < C.nE() && C.vivante[g]) ecrire_rho(eid, g, P.rho0);
        return eid;
    }
    int creer_sommet(const vector<int>& parents) {
        int v = nouveau_sommet(parents);
        ecrire(item(IT_INC, v)); ecrire(item(IT_W, v));
        if (P.journal) {
            int64_t off = (int64_t)pool_ev.size();
            for (int k = 0; k < C.ndims; ++k) { const double* wv = C.wv(k, v); pool_ev.insert(pool_ev.end(), wv, wv + C.dims[k]); }
            ecr_ev.push_back({W_SOMMET, v, 0, 0, 0.0, off});
        }
        return v;
    }

    bool chemin_alternatif(int e) {
        int a = C.ea[e], b = C.eb[e];
        std::unordered_set<int> vus{a};
        vector<int> front{a}, suiv;
        for (int prof = 0; prof < P.L_alternatif; ++prof) {
            suiv.clear();
            for (int u : front) {
                lire_sommet(u);
                for (int k : C.inc[u]) {
                    if (k == e) continue;
                    int v = C.autre(k, u);
                    if (v == b) return true;
                    if (vus.insert(v).second) suiv.push_back(v);
                }
            }
            front.swap(suiv);
        }
        return false;
    }

    // R0 — réouverture spontanée : chaque hyperarête actualisée porte sa propre horloge
    // exponentielle de taux lambda_reouverture. Règle locale et homogène ; elle empêche
    // l'extinction de l'activité (état absorbant U = ∅) dans les petites structures.
    void reouverture_spontanee() {
        R.clear(); Wt.clear(); ecr_ev.clear(); pool_ev.clear();
        int e;
        do { e = (int)(rng() % (u64)C.nE()); } while (!C.vivante[e] || C.x[e] == 0);
        lire(item(IT_X, e));
        ecrire_x(e, 0);
        if (P.journal) consigner(e, 0);
    }

    bool pas() {
        int64_t n_act = C.n_vivantes - (int64_t)U.size();
        double taux_blocs = P.lambda0 * (double)U.size();
        double taux_r0 = P.lambda_reouverture * (double)n_act;
        if (taux_blocs + taux_r0 <= 0) return false;
        std::exponential_distribution<double> ex(taux_blocs + taux_r0);
        t += ex(rng);
        if (U01() * (taux_blocs + taux_r0) >= taux_blocs) {
            reouverture_spontanee();
            if (P.trace_tous_les > 0 && n_evenements % P.trace_tous_les == 0) {
                trace.push_back(n_evenements); trace.push_back(C.actifs);
                trace.push_back(C.n_vivantes); trace.push_back((int64_t)U.size());
            }
            ++n_evenements;
            return true;
        }
        int e0 = U[(size_t)(U01() * U.size()) % U.size()];
        R.clear(); Wt.clear(); ecr_ev.clear(); pool_ev.clear();

        // bloc
        nouveau_tampon();
        vector<int> lam{e0}, pile{e0};
        marque_e[e0] = tampon_e;
        while (!pile.empty()) {
            int e = pile.back(); pile.pop_back();
            C.pour_voisines(e, [&](int f) {
                if (marque_e[f] != tampon_e && C.x[f] == 0 && C.r(e, f) > 0) {
                    marque_e[f] = tampon_e; lam.push_back(f); pile.push_back(f);
                }
            });
        }
        std::sort(lam.begin(), lam.end());
        int tamp_lam = tampon_e;
        for (int e : lam) lire_voisinage(e);
        vector<int> bord;
        if ((int)marque_b.size() < C.nE()) marque_b.resize(C.nE() * 2 + 16, 0);
        if (++tampon_b == INT32_MAX) { std::fill(marque_b.begin(), marque_b.end(), 0); tampon_b = 1; }
        for (int e : lam)
            C.pour_voisines(e, [&](int f) {
                if (marque_e[f] != tamp_lam && !dans_bord(f) && C.x[f] != 0) {
                    bool couple = false;
                    for (int g : lam) if (C.r(g, f) > 0) { couple = true; break; }
                    if (couple) { marque_b[f] = tampon_b; bord.push_back(f); }
                }
            });

        // tirage conjoint
        std::unordered_map<int, int> idx;
        for (int i = 0; i < (int)lam.size(); ++i) idx[lam[i]] = i;
        vector<int> sigma;
        tirer_bloc(lam, idx, sigma);
        for (size_t i = 0; i < lam.size(); ++i) ecrire_x(lam[i], sigma[i]);

        // R1
        std::unordered_set<u64> paires;
        for (int e : lam) C.pour_voisines(e, [&](int f) { if (C.x[f] != 0) paires.insert(cle_paire(e, f)); });
        for (u64 k : paires) {
            int e = (int)(k >> 32), f = (int)(k & 0xffffffffu);
            double ancien = C.r(e, f);
            double nv = std::min(1.0, std::max(0.0, ancien + P.eta * C.x[e] * C.x[f]));
            if (nv != ancien) ecrire_rho(e, f, nv);
        }

        // R2
        vector<int> lam_bord = lam;
        lam_bord.insert(lam_bord.end(), bord.begin(), bord.end());
        vector<int> a_rouvrir;
        for (int e : lam_bord) {
            lire_voisinage(e);
            double he = 0;
            C.pour_voisines(e, [&](int f) { if (C.x[f] != 0) he += C.r(e, f) * C.x[f]; });
            double re = 1.0 / (1.0 + std::exp(2 * P.beta_J * C.x[e] * he));
            re = P.epsilon + (1 - P.epsilon) * re;
            if (U01() < re) a_rouvrir.push_back(e);
        }
        for (int e : a_rouvrir) ecrire_x(e, 0);

        vector<int> actualisees;
        for (int e : lam) if (C.x[e] != 0) actualisees.push_back(e);

        // R3' — complétion de carrés
        {
            vector<u64> ordre;
            std::unordered_map<u64, std::array<int, 3>> coins;
            for (int e : actualisees) {
                for (int sens = 0; sens < 2; ++sens) {
                    int a = sens ? C.eb[e] : C.ea[e], b = sens ? C.ea[e] : C.eb[e];
                    lire_sommet(b);
                    for (int g : C.inc[b]) {
                        if (g == e) continue;
                        lire(item(IT_X, g));
                        if (C.x[g] == 0 || (P.r3_concordance && C.x[g] != C.x[e])) continue;
                        int c = C.autre(g, b);
                        lire_sommet(a); lire_sommet(c);
                        for (int k : C.inc[a]) lire_sommet(C.autre(k, a));
                        for (int k : C.inc[c]) lire_sommet(C.autre(k, c));
                        if (C.voisins_communs_egal(a, c, b)) {
                            u64 cl = cle_paire(a, c);
                            if (!coins.count(cl)) { coins[cl] = {e, g, b}; ordre.push_back(cl); }
                        }
                    }
                }
            }
            vector<u64> retenus;
            for (u64 cl : ordre) if (U01() < P.mu_carre) retenus.push_back(cl);
            std::shuffle(retenus.begin(), retenus.end(), rng);
            for (u64 cl : retenus) {
                int a = (int)(cl >> 32), c = (int)(cl & 0xffffffffu);
                auto [e, g, b] = coins[cl];
                if ((int)C.inc[a].size() < P.k_max && (int)C.inc[c].size() < P.k_max &&
                    C.voisins_communs_egal(a, c, b)) {
                    int vs = creer_sommet({a, c});
                    int n1 = creer_arete(a, vs, {e});
                    int n2 = creer_arete(c, vs, {g});
                    if (C.r(n1, n2) == 0) ecrire_rho(n1, n2, P.rho0);
                }
            }
        }

        // croissance pendante résiduelle
        if (P.mu_pendant > 0)
            for (int e : actualisees)
                for (int v : {C.ea[e], C.eb[e]}) {
                    lire_sommet(v);
                    if ((int)C.inc[v].size() < P.k_max && U01() < P.mu_pendant) {
                        int vs = creer_sommet({v});
                        creer_arete(v, vs, {e});
                    }
                }

        // R4' — fermeture de 4-cycles
        {
            vector<u64> ordre;
            std::unordered_map<u64, std::array<int, 4>> gen;
            for (int e : actualisees) {
                int a = C.ea[e], b = C.eb[e];
                for (int f : C.inc[a]) {
                    if (f == e || (P.r4_frontiere && !dans_bord(f)) || C.x[f] == 0 ||
                        (P.r4_concordance && C.x[f] != C.x[e])) continue;
                    int u = C.autre(f, a);
                    for (int g : C.inc[b]) {
                        if (g == e || (P.r4_frontiere && !dans_bord(g)) || C.x[g] == 0 ||
                            (P.r4_concordance && C.x[g] != C.x[e])) continue;
                        int w = C.autre(g, b);
                        for (int v : {a, b, u, w}) lire_sommet(v);
                        for (int k : C.inc[u]) lire_sommet(C.autre(k, u));
                        for (int k : C.inc[w]) lire_sommet(C.autre(k, w));
                        if (u != w && !C.adjacents(u, w) && C.voisins_communs_egal(u, b, a) &&
                            C.voisins_communs_egal(a, w, b)) {
                            u64 cl = cle_paire(u, w);
                            if (!gen.count(cl)) { gen[cl] = {f, g, a, b}; ordre.push_back(cl); }
                        }
                    }
                }
            }
            vector<u64> retenues;
            for (u64 cl : ordre) if (U01() < P.nu) retenues.push_back(cl);
            std::shuffle(retenues.begin(), retenues.end(), rng);
            for (u64 cl : retenues) {
                int u = (int)(cl >> 32), w = (int)(cl & 0xffffffffu);
                auto [f, g, a, b] = gen[cl];
                if ((int)C.inc[u].size() < P.k_max && (int)C.inc[w].size() < P.k_max && !C.adjacents(u, w) &&
                    C.voisins_communs_egal(u, b, a) && C.voisins_communs_egal(a, w, b))
                    creer_arete(u, w, {f, g});
            }
        }

        // R6 — suppression sans pont
        {
            vector<int> cand;
            for (int e : lam_bord) {
                if (!C.vivante[e] || C.x[e] == 0) continue;
                int a = C.ea[e], b = C.eb[e];
                lire_sommet(a); lire_sommet(b);
                for (int k : C.inc[a]) lire_sommet(C.autre(k, a));
                for (int k : C.inc[b]) lire_sommet(C.autre(k, b));
                double proba = C.dans_un_carre(e) ? P.delta_carre : P.delta;
                if (U01() < proba) cand.push_back(e);
            }
            std::shuffle(cand.begin(), cand.end(), rng);
            for (int e : cand) {
                if (!C.vivante[e] || !chemin_alternatif(e)) continue;
                C.pour_voisines(e, [&](int f) { ecrire(item_r(e, f)); });
                ecrire(item(IT_INC, C.ea[e])); ecrire(item(IT_INC, C.eb[e])); ecrire(item(IT_X, e));
                fermer(e);
                C.supprimer_arete(e);
                if (P.journal) ecr_ev.push_back({W_SUPPR, e, 0, 0, 0.0, 0});
            }
        }

        // R5' — relaxation de Gibbs O(d)
        if (C.ndims > 0) {
            vector<int> sites;
            for (int e : lam) if (C.vivante[e]) { sites.push_back(C.ea[e]); sites.push_back(C.eb[e]); }
            std::sort(sites.begin(), sites.end());
            sites.erase(std::unique(sites.begin(), sites.end()), sites.end());
            for (int rr = 0; rr < P.rayon_B; ++rr) {
                vector<int> ext = sites;
                for (int u : sites) for (int k : C.inc[u]) ext.push_back(C.autre(k, u));
                std::sort(ext.begin(), ext.end());
                ext.erase(std::unique(ext.begin(), ext.end()), ext.end());
                sites.swap(ext);
            }
            for (int u : sites) {
                lire_sommet(u);
                for (int k : C.inc[u]) { lire(item(IT_X, k)); lire(item(IT_W, C.ea[k])); lire(item(IT_W, C.eb[k])); }
            }
            for (int kd = 0; kd < C.ndims; ++kd) {
                int d = C.dims[kd];
                vector<char> modifie(sites.size(), 0);
                vector<int> perm(sites.size());
                std::iota(perm.begin(), perm.end(), 0);
                double h[16], mu[16];
                for (int it = 0; it < P.iterations_B; ++it) {
                    std::shuffle(perm.begin(), perm.end(), rng);
                    for (int j : perm) {
                        int u = sites[j];
                        std::fill(h, h + d, 0.0);
                        for (int k : C.inc[u]) {
                            if (C.x[k] == 0) continue;
                            const double* wv = C.wv(kd, C.autre(k, u));
                            for (int i = 0; i < d; ++i) h[i] += C.x[k] * wv[i];
                        }
                        double nh = 0;
                        for (int i = 0; i < d; ++i) nh += h[i] * h[i];
                        nh = std::sqrt(nh);
                        if (nh < 1e-12) continue;
                        for (int i = 0; i < d; ++i) mu[i] = h[i] / nh;
                        tirer_vmf(mu, P.beta_B * nh, d, C.wv(kd, u));
                        modifie[j] = 1;
                    }
                }
                for (size_t j = 0; j < sites.size(); ++j) {
                    if (!modifie[j]) continue;
                    ecrire(item(IT_W, sites[j]));
                    if (P.journal) {
                        int64_t off = (int64_t)pool_ev.size();
                        const double* wv = C.wv(kd, sites[j]);
                        pool_ev.insert(pool_ev.end(), wv, wv + d);
                        ecr_ev.push_back({W_VEC, sites[j], kd, 0, 0.0, off});
                    }
                }
            }
        }

        // journal : dépendance lecture/écriture
        if (P.journal) consigner(e0, (int)lam.size());
        if (P.trace_tous_les > 0 && n_evenements % P.trace_tous_les == 0) {
            trace.push_back(n_evenements); trace.push_back(C.actifs);
            trace.push_back(C.n_vivantes); trace.push_back((int64_t)U.size());
        }
        ++n_evenements;
        return true;
    }

    void consigner(int e0, int taille) {
        std::sort(R.begin(), R.end()); R.erase(std::unique(R.begin(), R.end()), R.end());
        std::sort(Wt.begin(), Wt.end()); Wt.erase(std::unique(Wt.begin(), Wt.end()), Wt.end());
        int32_t i = (int32_t)J.t.size();
        vector<int32_t> pr;
        auto ajouter_pred = [&](u64 it) {
            auto f = dernier_ecrivain.find(it);
            if (f != dernier_ecrivain.end()) pr.push_back(f->second);
        };
        for (u64 it : R) ajouter_pred(it);
        for (u64 it : Wt) {
            ajouter_pred(it);
            auto l = lecteurs.find(it);
            if (l != lecteurs.end()) pr.insert(pr.end(), l->second.begin(), l->second.end());
        }
        std::sort(pr.begin(), pr.end());
        pr.erase(std::unique(pr.begin(), pr.end()), pr.end());
        int32_t prof = 1;
        for (int32_t j : pr) prof = std::max(prof, J.profondeur[j] + 1);
        for (u64 it : Wt) { dernier_ecrivain[it] = i; lecteurs.erase(it); }
        for (u64 it : R)
            if (!std::binary_search(Wt.begin(), Wt.end(), it)) lecteurs[it].push_back(i);
        J.t.push_back(t);
        J.profondeur.push_back(prof);
        J.graine_bloc.push_back(e0);
        J.taille_bloc.push_back(taille);
        J.preds.insert(J.preds.end(), pr.begin(), pr.end());
        J.pred_off.push_back((int64_t)J.preds.size());
        int64_t base = (int64_t)J.pool.size();
        for (auto& w : ecr_ev) { if (w.type == W_SOMMET || w.type == W_VEC) w.pool += base; J.ecr.push_back(w); }
        J.pool.insert(J.pool.end(), pool_ev.begin(), pool_ev.end());
        J.ecr_off.push_back((int64_t)J.ecr.size());
    }

    // --- reconstruction sur un ensemble d'événements fermé vers le passé
    Etat reconstruire(const vector<int32_t>& indices) const {
        Etat S = C0;
        vector<int32_t> ids = indices;
        std::sort(ids.begin(), ids.end());
        for (int32_t i : ids) {
            for (int64_t q = J.ecr_off[i]; q < J.ecr_off[i + 1]; ++q) {
                const Ecriture& w = J.ecr[q];
                switch (w.type) {
                    case W_X: if (w.a < S.nE() && S.vivante[w.a]) S.x[w.a] = (int8_t)w.b; break;
                    case W_RHO: S.set_rho(w.a, w.b, w.val); break;
                    case W_ARETE: {
                        while (S.nV() <= std::max(w.b, w.c)) S.nouveau_sommet();
                        S.ajouter_arete(w.b, w.c, w.a);
                        break;
                    }
                    case W_SUPPR: S.supprimer_arete(w.a); break;
                    case W_SOMMET: {
                        while (S.nV() <= w.a) S.nouveau_sommet();
                        int64_t off = w.pool;
                        for (int k = 0; k < S.ndims; ++k) {
                            std::memcpy(S.wv(k, w.a), &J.pool[off], sizeof(double) * S.dims[k]);
                            off += S.dims[k];
                        }
                        break;
                    }
                    case W_VEC: std::memcpy(S.wv(w.b, w.a), &J.pool[w.pool], sizeof(double) * S.dims[w.b]); break;
                }
            }
        }
        return S;
    }
};

// ---------------------------------------------------------------------------
// Observables sur un état
// ---------------------------------------------------------------------------
static vector<int32_t> etiquettes_composantes(const Etat& S, int32_t* n_comp, int32_t* geante) {
    vector<int32_t> lab(S.nV(), -1);
    vector<int64_t> tailles;
    vector<int> file;
    for (int s = 0; s < S.nV(); ++s) {
        if (S.inc[s].empty() || lab[s] >= 0) continue;
        int id = (int)tailles.size();
        lab[s] = id; file.assign(1, s);
        int64_t n = 0;
        for (size_t q = 0; q < file.size(); ++q) {
            int u = file[q]; ++n;
            for (int k : S.inc[u]) { int v = S.autre(k, u); if (lab[v] < 0) { lab[v] = id; file.push_back(v); } }
        }
        tailles.push_back(n);
    }
    *n_comp = (int32_t)tailles.size();
    *geante = tailles.empty() ? -1 : (int32_t)(std::max_element(tailles.begin(), tailles.end()) - tailles.begin());
    return lab;
}

static void bfs_boule(const Etat& S, int o, int r, vector<int>& dist_marque, int tampon,
                      vector<int>& dist, vector<int>& visites) {
    visites.assign(1, o);
    dist_marque[o] = tampon; dist[o] = 0;
    for (size_t q = 0; q < visites.size(); ++q) {
        int u = visites[q];
        if (dist[u] == r) continue;
        for (int k : S.inc[u]) {
            int v = S.autre(k, u);
            if (dist_marque[v] != tampon) { dist_marque[v] = tampon; dist[v] = dist[u] + 1; visites.push_back(v); }
        }
    }
}

static vector<int> racines(const Etat& S, int n, u64 graine, bool geante_seule) {
    int32_t nc, g;
    vector<int32_t> lab = etiquettes_composantes(S, &nc, &g);
    vector<int> cand;
    for (int v = 0; v < S.nV(); ++v)
        if (!S.inc[v].empty() && (!geante_seule || lab[v] == g)) cand.push_back(v);
    vector<int> out;
    if (cand.empty()) return out;
    std::mt19937_64 rg(graine);
    for (int i = 0; i < n; ++i) out.push_back(cand[rg() % cand.size()]);
    return out;
}

// Contextualité : cycles de longueur ≤ L, classes par frustration et équilibre de composante
static void analyse_contexte(const Etat& S, int kd, int L, double tau, bool aleatoire, u64 graine,
                             double* out) {
    int d = S.dims[kd];
    vector<double> wloc;
    const double* W = S.w[kd].data();
    if (aleatoire) {
        std::mt19937_64 rg(graine);
        std::normal_distribution<double> nd(0, 1);
        wloc.resize((size_t)S.nV() * d);
        for (int v = 0; v < S.nV(); ++v) {
            double n2 = 0;
            for (int i = 0; i < d; ++i) { wloc[(size_t)v * d + i] = nd(rg); n2 += wloc[(size_t)v * d + i] * wloc[(size_t)v * d + i]; }
            double n = std::sqrt(n2);
            for (int i = 0; i < d; ++i) wloc[(size_t)v * d + i] /= n;
        }
        W = wloc.data();
    }
    // équilibre (Harary) par union-find avec parité
    vector<int> par(S.nV()), pa(S.nV(), 0);
    std::iota(par.begin(), par.end(), 0);
    // trouver itératif (pas de récursion profonde sur les grandes structures)
    vector<int> pile_uf;
    auto trouver = [&](int v) -> int {
        pile_uf.clear();
        while (par[v] != v) { pile_uf.push_back(v); v = par[v]; }
        int racine = v, acc = 0;
        for (auto it = pile_uf.rbegin(); it != pile_uf.rend(); ++it) {
            acc ^= pa[*it];
            pa[*it] = acc;
            par[*it] = racine;
        }
        return racine;
    };
    vector<char> deseq(S.nV(), 0);
    for (int e = 0; e < S.nE(); ++e) {
        if (!S.vivante[e] || S.x[e] == 0) continue;
        int u = S.ea[e], v = S.eb[e], q = S.x[e] == 1 ? 0 : 1;
        int ru = trouver(u), rv = trouver(v);
        int pu = (par[u] == u) ? 0 : pa[u], pv = (par[v] == v) ? 0 : pa[v];
        if (ru != rv) { par[ru] = rv; pa[ru] = pu ^ pv ^ q; deseq[rv] |= deseq[ru]; }
        else if ((pu ^ pv) != q) deseq[ru] = 1;
    }
    auto racine_deseq = [&](int v) { return deseq[trouver(v)] != 0; };

    // énumération des cycles (sommet minimal en tête, un seul sens)
    double cl[4][3] = {{0}};   // frustrés, déséquilibrés, équilibrés, indéterminés : [cycles, violations, Σδ]
    double S4max = -1e9, n4 = 0, viol4 = 0;
    vector<double> defaut_sommet(S.nV(), 0.0);
    vector<int> chemin, arete_chemin;
    vector<char> sur_chemin(S.nV(), 0);
    auto evaluer = [&](int ferm) {
        int n = (int)chemin.size();
        int prod = 1;
        vector<int> ar = arete_chemin; ar.push_back(ferm);
        double somme = 0, minabs = 1e9;
        int neg = 0;
        for (int i = 0; i < n; ++i) {
            int u = chemin[i], v = chemin[(i + 1) % n];
            double dot = 0;
            for (int j = 0; j < d; ++j) dot += W[(size_t)u * d + j] * W[(size_t)v * d + j];
            somme += std::fabs(dot); minabs = std::min(minabs, std::fabs(dot)); neg += dot < 0;
            prod *= S.x[ar[i]];
        }
        double val = somme - ((neg % 2 == 0) ? 2 * minabs : 0.0);
        double dl = std::max(0.0, val - (n - 2));
        int classe = prod == 0 ? 3 : (prod == -1 ? 0 : (racine_deseq(chemin[0]) ? 1 : 2));
        cl[classe][0] += 1; cl[classe][1] += dl > tau; cl[classe][2] += dl;
        if (n == 4) { n4 += 1; viol4 += val > 2 + tau; S4max = std::max(S4max, val); }
        for (int u : chemin) defaut_sommet[u] = std::max(defaut_sommet[u], dl);
    };
    std::function<void(int, int)> dfs = [&](int s, int u) {
        for (int k : S.inc[u]) {
            int v = S.autre(k, u);
            if (v == s && chemin.size() >= 3) {
                if (chemin[1] < chemin.back()) evaluer(k);
            } else if (v > s && !sur_chemin[v] && (int)chemin.size() < L) {
                chemin.push_back(v); arete_chemin.push_back(k); sur_chemin[v] = 1;
                dfs(s, v);
                chemin.pop_back(); arete_chemin.pop_back(); sur_chemin[v] = 0;
            }
        }
    };
    for (int s = 0; s < S.nV(); ++s) {
        if (S.inc[s].empty()) continue;
        chemin.assign(1, s); arete_chemin.clear(); sur_chemin[s] = 1;
        dfs(s, s);
        sur_chemin[s] = 0;
    }
    // sortie : 4 classes x [cycles, violations, défaut moyen], puis densité, fraction violée,
    //          n 4-cycles, fraction CHSH > 2 + τ, S max, fraction de sommets δ° > τ
    for (int k = 0; k < 4; ++k) {
        out[3 * k] = cl[k][0];
        out[3 * k + 1] = cl[k][1];
        out[3 * k + 2] = cl[k][0] > 0 ? cl[k][2] / cl[k][0] : NAN;
    }
    double det = cl[0][0] + cl[1][0] + cl[2][0];
    out[12] = det > 0 ? cl[0][0] / det : NAN;
    out[13] = det > 0 ? (cl[0][1] + cl[1][1] + cl[2][1]) / det : NAN;
    out[14] = n4;
    out[15] = n4 > 0 ? viol4 / n4 : NAN;
    out[16] = n4 > 0 ? S4max : NAN;
    double nb = 0, pos = 0;
    for (int v = 0; v < S.nV(); ++v) if (!S.inc[v].empty()) { nb += 1; pos += defaut_sommet[v] > tau; }
    out[17] = nb > 0 ? pos / nb : NAN;
}

// ---------------------------------------------------------------------------
// Interface C
// ---------------------------------------------------------------------------
struct Poignee {
    Moteur* M = nullptr;     // moteur (simulation)
    Etat* S = nullptr;       // état seul (reconstruction)
    const Etat& etat() const { return M ? M->C : *S; }
};

EXPORT void* moteur_creer(const Params* p) { auto* h = new Poignee; h->M = new Moteur(*p); return h; }
EXPORT void moteur_detruire(void* hp) { auto* h = (Poignee*)hp; delete h->M; delete h->S; delete h; }

// 0 : budget épuisé, 1 : état absorbant, 2 : taille maximale atteinte
EXPORT int moteur_simuler(void* hp, int64_t max_ev, int64_t max_sommets) {
    Moteur& M = *((Poignee*)hp)->M;
    for (int64_t i = 0; i < max_ev; ++i) {
        if (max_sommets > 0 && M.C.nV() >= max_sommets) return 2;
        if (!M.pas()) return 1;
    }
    return 0;
}

EXPORT int64_t moteur_nb_sommets(void* hp) { return ((Poignee*)hp)->etat().nV(); }
EXPORT int64_t moteur_nb_aretes_total(void* hp) { return ((Poignee*)hp)->etat().nE(); }
EXPORT int64_t moteur_nb_couplages(void* hp) { return (int64_t)((Poignee*)hp)->etat().rho.size(); }
EXPORT int64_t moteur_nb_evenements(void* hp) { return ((Poignee*)hp)->M->n_evenements; }
EXPORT int64_t moteur_taille_trace(void* hp) { return (int64_t)((Poignee*)hp)->M->trace.size(); }
EXPORT int64_t moteur_nb_preds(void* hp) { return (int64_t)((Poignee*)hp)->M->J.preds.size(); }
EXPORT int64_t moteur_taille_ouverte(void* hp) { return (int64_t)((Poignee*)hp)->M->U.size(); }

EXPORT void moteur_exporter_aretes(void* hp, int32_t* ab, int8_t* x, uint8_t* vivante) {
    const Etat& S = ((Poignee*)hp)->etat();
    for (int e = 0; e < S.nE(); ++e) { ab[2 * e] = S.ea[e]; ab[2 * e + 1] = S.eb[e]; x[e] = S.x[e]; vivante[e] = S.vivante[e]; }
}
EXPORT void moteur_exporter_couplages(void* hp, int32_t* ef, double* val) {
    const Etat& S = ((Poignee*)hp)->etat();
    int64_t i = 0;
    for (auto& kv : S.rho) { ef[2 * i] = (int32_t)(kv.first >> 32); ef[2 * i + 1] = (int32_t)(kv.first & 0xffffffffu); val[i] = kv.second; ++i; }
}
EXPORT void moteur_exporter_vecteurs(void* hp, int kd, double* out) {
    const Etat& S = ((Poignee*)hp)->etat();
    std::memcpy(out, S.w[kd].data(), sizeof(double) * S.w[kd].size());
}
EXPORT void moteur_exporter_trace(void* hp, int64_t* out) {
    auto& T = ((Poignee*)hp)->M->trace; std::memcpy(out, T.data(), sizeof(int64_t) * T.size());
}
EXPORT void moteur_exporter_journal(void* hp, double* t, int32_t* prof, int32_t* graine_bloc,
                                    int32_t* taille_bloc, int64_t* pred_off, int32_t* preds) {
    auto& J = ((Poignee*)hp)->M->J;
    size_t n = J.t.size();
    std::memcpy(t, J.t.data(), sizeof(double) * n);
    std::memcpy(prof, J.profondeur.data(), sizeof(int32_t) * n);
    std::memcpy(graine_bloc, J.graine_bloc.data(), sizeof(int32_t) * n);
    std::memcpy(taille_bloc, J.taille_bloc.data(), sizeof(int32_t) * n);
    std::memcpy(pred_off, J.pred_off.data(), sizeof(int64_t) * (n + 1));
    std::memcpy(preds, J.preds.data(), sizeof(int32_t) * J.preds.size());
}

// fermeture vers le passé d'un ensemble d'événements (masque de sortie, taille nb_evenements)
EXPORT void moteur_fermeture(void* hp, const int32_t* ids, int64_t n, uint8_t* masque) {
    auto& J = ((Poignee*)hp)->M->J;
    std::memset(masque, 0, J.t.size());
    vector<int32_t> pile(ids, ids + n);
    for (int32_t i : pile) masque[i] = 1;
    while (!pile.empty()) {
        int32_t i = pile.back(); pile.pop_back();
        for (int64_t q = J.pred_off[i]; q < J.pred_off[i + 1]; ++q)
            if (!masque[J.preds[q]]) { masque[J.preds[q]] = 1; pile.push_back(J.preds[q]); }
    }
}

EXPORT void* moteur_reconstruire(void* hp, const int32_t* ids, int64_t n) {
    Moteur& M = *((Poignee*)hp)->M;
    auto* h = new Poignee;
    h->S = new Etat(M.reconstruire(vector<int32_t>(ids, ids + n)));
    return h;
}

EXPORT int moteur_etats_egaux(void* h1, void* h2) {
    const Etat& A = ((Poignee*)h1)->etat();
    const Etat& B = ((Poignee*)h2)->etat();
    if (A.nE() != B.nE() || A.nV() != B.nV()) return 0;
    for (int e = 0; e < A.nE(); ++e) {
        if (A.vivante[e] != B.vivante[e]) return 0;
        if (A.vivante[e] && (A.ea[e] != B.ea[e] || A.eb[e] != B.eb[e] || A.x[e] != B.x[e])) return 0;
    }
    if (A.rho.size() != B.rho.size()) return 0;
    for (auto& kv : A.rho) { auto it = B.rho.find(kv.first); if (it == B.rho.end() || it->second != kv.second) return 0; }
    for (int k = 0; k < A.ndims; ++k) if (A.w[k] != B.w[k]) return 0;
    return 1;
}

EXPORT int32_t obs_composantes(void* hp, int32_t* lab_out, int32_t* geante) {
    int32_t nc;
    vector<int32_t> lab = etiquettes_composantes(((Poignee*)hp)->etat(), &nc, geante);
    std::memcpy(lab_out, lab.data(), sizeof(int32_t) * lab.size());
    return nc;
}

EXPORT void obs_volume(void* hp, int n_racines, int r_max, u64 graine, int geante, double* V) {
    const Etat& S = ((Poignee*)hp)->etat();
    std::fill(V, V + r_max + 1, 0.0);
    vector<int> rac = racines(S, n_racines, graine, geante != 0);
    vector<int> marque(S.nV(), 0), dist(S.nV(), 0), vis;
    int tampon = 0;
    vector<double> compte(r_max + 1);
    for (int o : rac) {
        bfs_boule(S, o, r_max, marque, ++tampon, dist, vis);
        std::fill(compte.begin(), compte.end(), 0.0);
        for (int v : vis) compte[dist[v]] += 1;
        double cum = 0;
        for (int r = 0; r <= r_max; ++r) { cum += compte[r]; V[r] += cum; }
    }
    if (!rac.empty()) for (int r = 0; r <= r_max; ++r) V[r] /= rac.size();
}

EXPORT double obs_cyclicite(void* hp, int n_racines, int r, u64 graine, int geante) {
    const Etat& S = ((Poignee*)hp)->etat();
    vector<int> rac = racines(S, n_racines, graine, geante != 0);
    vector<int> marque(S.nV(), 0), dist(S.nV(), 0), vis;
    int tampon = 0;
    double somme = 0;
    for (int o : rac) {
        bfs_boule(S, o, r, marque, ++tampon, dist, vis);
        double nv = (double)vis.size(), ne2 = 0;
        for (int u : vis) for (int k : S.inc[u]) if (marque[S.autre(k, u)] == tampon) ne2 += 1;
        somme += (ne2 / 2 - nv + 1) / nv;
    }
    return rac.empty() ? NAN : somme / rac.size();
}

// Ponts et composantes 2-arête-connexes de la composante géante (Tarjan itératif).
// Une surface a une composante 2-arête-connexe géante ; un polymère branché d'amas de carrés n'a
// que des composantes 2-arête-connexes finies, reliées par des ponts.
// out : [fraction de ponts parmi les arêtes de la géante, fraction des sommets de la géante dans la
//        plus grande composante 2-arête-connexe, nombre de composantes 2-arête-connexes,
//        taille moyenne de ces composantes, nombre d'arêtes de la géante]
EXPORT void obs_ponts(void* hp, double* out) {
    const Etat& S = ((Poignee*)hp)->etat();
    int32_t nc, g;
    vector<int32_t> lab = etiquettes_composantes(S, &nc, &g);
    int n = S.nV();
    vector<int32_t> disc(n, -1), bas(n, 0);
    vector<char> pont(S.nE(), 0);
    int32_t temps = 0;
    struct Cadre { int v, arete_parent; size_t i; };
    vector<Cadre> pile;
    for (int s = 0; s < n; ++s) {
        if (lab[s] != g || disc[s] >= 0) continue;
        pile.push_back({s, -1, 0});
        disc[s] = bas[s] = temps++;
        while (!pile.empty()) {
            Cadre& c = pile.back();
            if (c.i < S.inc[c.v].size()) {
                int k = S.inc[c.v][c.i++];
                if (k == c.arete_parent) continue;
                int w = S.autre(k, c.v);
                if (disc[w] < 0) {
                    disc[w] = bas[w] = temps++;
                    pile.push_back({w, k, 0});
                } else {
                    bas[c.v] = std::min(bas[c.v], disc[w]);
                }
            } else {
                int v = c.v, kp = c.arete_parent;
                pile.pop_back();
                if (!pile.empty()) {
                    int u = pile.back().v;
                    bas[u] = std::min(bas[u], bas[v]);
                    if (bas[v] > disc[u]) pont[kp] = 1;
                }
            }
        }
    }
    // composantes 2-arête-connexes : composantes après suppression des ponts
    vector<int32_t> comp(n, -1);
    vector<int64_t> tailles;
    int64_t aretes_geante = 0, ponts = 0;
    for (int e = 0; e < S.nE(); ++e)
        if (S.vivante[e] && lab[S.ea[e]] == g) { ++aretes_geante; ponts += pont[e]; }
    vector<int> file;
    for (int s = 0; s < n; ++s) {
        if (lab[s] != g || comp[s] >= 0) continue;
        int id = (int)tailles.size();
        comp[s] = id; file.assign(1, s);
        for (size_t q = 0; q < file.size(); ++q) {
            int u = file[q];
            for (int k : S.inc[u]) {
                if (pont[k]) continue;
                int w = S.autre(k, u);
                if (comp[w] < 0) { comp[w] = id; file.push_back(w); }
            }
        }
        tailles.push_back((int64_t)file.size());
    }
    int64_t ng = 0;
    for (int v = 0; v < n; ++v) ng += lab[v] == g;
    int64_t plus_grande = tailles.empty() ? 0 : *std::max_element(tailles.begin(), tailles.end());
    out[0] = aretes_geante ? (double)ponts / aretes_geante : NAN;
    out[1] = ng ? (double)plus_grande / ng : NAN;
    out[2] = (double)tailles.size();
    out[3] = tailles.empty() ? NAN : (double)ng / tailles.size();
    out[4] = (double)aretes_geante;
}

// Degrés et courbure combinatoire sur la composante géante.
// Un « coin » en v est une paire de voisins (a, c) ayant un voisin commun autre que v (le coin
// appartient à un carré). Courbure combinatoire d'un complexe de carrés : κ(v) = 1 - deg/2 + coins/4.
// Sommet plat : degré 4, quatre coins, chaque voisin dans exactement deux coins (motif de Z²).
// hist : histogramme des degrés 0..16 ; out : [fraction plate, κ moyen, fraction κ = 0,
//        fraction κ < 0, fraction κ > 0, fraction de degré 2, fraction de degré 4]
EXPORT void obs_degres_courbure(void* hp, int64_t* hist, double* out) {
    const Etat& S = ((Poignee*)hp)->etat();
    int32_t nc, g;
    vector<int32_t> lab = etiquettes_composantes(S, &nc, &g);
    std::fill(hist, hist + 17, 0);
    double n = 0, plats = 0, ksum = 0, k0 = 0, kneg = 0, kpos = 0;
    vector<int> vois;
    for (int v = 0; v < S.nV(); ++v) {
        if (lab[v] != g) continue;
        vois.clear();
        for (int k : S.inc[v]) vois.push_back(S.autre(k, v));
        int deg = (int)vois.size();
        hist[std::min(deg, 16)] += 1;
        int coins = 0;
        int dans[16] = {0};
        for (int i = 0; i < deg; ++i)
            for (int j = i + 1; j < deg; ++j) {
                bool carre = false;
                for (int k : S.inc[vois[i]]) {
                    int z = S.autre(k, vois[i]);
                    if (z != v && S.adjacents(z, vois[j])) { carre = true; break; }
                }
                if (carre) { ++coins; if (i < 16) ++dans[i]; if (j < 16) ++dans[j]; }
            }
        double kappa = 1.0 - deg / 2.0 + coins / 4.0;
        bool plat = deg == 4 && coins == 4;
        for (int i = 0; plat && i < 4; ++i) plat = dans[i] == 2;
        n += 1; plats += plat; ksum += kappa;
        k0 += std::fabs(kappa) < 1e-12; kneg += kappa < -1e-12; kpos += kappa > 1e-12;
    }
    out[0] = n ? plats / n : NAN;
    out[1] = n ? ksum / n : NAN;
    out[2] = n ? k0 / n : NAN;
    out[3] = n ? kneg / n : NAN;
    out[4] = n ? kpos / n : NAN;
    out[5] = n ? hist[2] / n : NAN;
    out[6] = n ? hist[4] / n : NAN;
}

EXPORT void obs_contexte(void* hp, int kd, int L, double tau, int aleatoire, u64 graine, double* out) {
    analyse_contexte(((Poignee*)hp)->etat(), kd, L, tau, aleatoire != 0, graine, out);
}
