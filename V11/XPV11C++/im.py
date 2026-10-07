import json, numpy as np
d = json.load(open("resultats_rapide/germe_r4-libre_r3-libre/resultats.json"))
for s in d["simulations"]:
    V = np.array(s["V"]); r = np.arange(len(V))
    pente = np.gradient(np.log(V[1:]), np.log(r[1:]))
    print(f"ν={s['nu']} g={s['graine']} : pente à r=10, 20, 40, 60 :",
          np.round(pente[[9, 19, 39, 59]], 2))