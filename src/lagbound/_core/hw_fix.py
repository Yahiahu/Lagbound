"""diagnose / repair a BM point stuck at saddles: rows with v_i anti-aligned to their gradient G_i (rgd cannot escape,
mixing can). Jacobi mixing on bad rows, accepted only if dv increases. Writes {"V": V} for --initV."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import certify_lifted_vc as CV
tag, ck, out = sys.argv[1], sys.argv[2], sys.argv[3]; nit = int(sys.argv[4]) if len(sys.argv) > 4 else 30
initV = sys.argv[5] if len(sys.argv) > 5 else ""
P = CV.load_problem(tag, ck, os.environ["DATA_DIR"], initV=initV)
op = CV.Op(P); d = P["d"]; L = P["L"]; const = P["C0"] + P["lam"].sum() - P["mu"].sum()
V = P["V"]
def stats(V):
    CVV = op(V); G = CVV + (d**2/L)[:, None]*V; g = np.einsum("ij,ij->i", CVV, V)
    nG = np.linalg.norm(G, axis=1); cos = np.einsum("ij,ij->i", G, V)/np.maximum(nG, 1e-300)
    return (const + g.sum())/L, G, nG, cos
dv, G, nG, cos = stats(V)
print(f"start dv {dv:.6f}  rows cos<0: {(cos<0).sum()}  cos<0.5: {(cos<0.5).sum()}  cos<0.99: {(cos<0.99).sum()}", flush=True)
for it in range(nit):
    bad = cos < 0.999
    Vn = V.copy(); Vn[bad] = G[bad] + 0.5*nG[bad, None]*V[bad]; Vn /= np.linalg.norm(Vn, axis=1, keepdims=True)
    dvn, Gn, nGn, cosn = stats(Vn)
    if dvn < dv:   # damp harder: only the worst rows
        bad = cos < 0
        Vn = V.copy(); Vn[bad] = G[bad] + 0.5*nG[bad, None]*V[bad]; Vn /= np.linalg.norm(Vn, axis=1, keepdims=True)
        dvn, Gn, nGn, cosn = stats(Vn)
        if dvn < dv: print("no ascent; stop"); break
    V, dv, G, nG, cos = Vn, dvn, Gn, nGn, cosn
    print(f"it {it} dv {dv:.6f}  cos<0: {(cos<0).sum()}  cos<0.99: {(cos<0.99).sum()}", flush=True)
np.savez(out, V=V.astype(np.float32))
