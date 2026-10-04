#!/usr/bin/env python3
"""One Leiden run on a big tol5-format graph. usage: leiden_one.py <data_root> <tag> <seed> <resolution> <n_iter>
writes <tag>/leiden_r<res>_s<seed>.npy and .json (modularity at resolution 1)."""
import sys, os, json, random, time, numpy as np, igraph as ig
root, tag, seed, res, nit = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
D = os.path.join(root, tag); n = json.load(open(D + "/graph.json"))["n"]; k = np.load(D + "/keys.npy"); t0 = time.time()
g = ig.Graph(n=n, edges=np.c_[k // n, k % n]); del k
ig.set_random_number_generator(random.Random(seed))
p = g.community_leiden(objective_function="modularity", resolution=res, n_iterations=nit)
lab = np.array(p.membership, dtype=np.int64); q = g.modularity(lab)
pref = D + f"/leiden_r{res:g}_s{seed}"; np.save(pref + ".npy", lab)
json.dump({"Q": q, "clusters": int(lab.max() + 1), "seed": seed, "res": res, "n_iter": nit, "sec": round(time.time() - t0, 1)}, open(pref + ".json", "w"))
print(pref, q, lab.max() + 1, round(time.time() - t0, 1))
