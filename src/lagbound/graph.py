"""Graph input handling.

Every public entry point accepts a graph in any of these forms and converts it with :func:`as_graph`:

* :class:`Graph` (returned unchanged),
* a NetworkX graph,
* an igraph graph,
* a SciPy sparse matrix or dense NumPy adjacency matrix,
* an edge array of shape ``(m, 2)`` or an iterable of node pairs,
* a path to a whitespace-separated edge-list file (``#`` and ``%`` start comments; ``.gz`` is read transparently).

Modularity is computed on the simple undirected graph: directions and edge weights are dropped, self-loops and
duplicate edges are removed, and isolated nodes are set aside (they cannot change modularity and are reported as
singleton clusters in returned partitions).
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Graph:
    """Simple undirected graph on nodes ``0..n-1`` with edges ``u[e] < v[e]``.

    ``labels`` holds the caller's node identifiers (``labels[i]`` is the original name of node ``i``) and
    ``isolated`` the identifiers of isolated nodes that were removed.
    """

    n: int
    u: np.ndarray
    v: np.ndarray
    labels: list = field(default_factory=list)
    isolated: list = field(default_factory=list)

    def __post_init__(self):
        self.u = np.asarray(self.u, dtype=np.int64)
        self.v = np.asarray(self.v, dtype=np.int64)
        if not self.labels:
            self.labels = list(range(self.n))

    @property
    def m(self) -> int:
        return len(self.u)

    @property
    def degrees(self) -> np.ndarray:
        return np.bincount(np.r_[self.u, self.v], minlength=self.n).astype(np.int64)

    def adjacency(self) -> sp.csr_matrix:
        """Symmetric 0/1 adjacency matrix in CSR format with sorted indices."""
        A = sp.csr_matrix((np.ones(2 * self.m), (np.r_[self.u, self.v], np.r_[self.v, self.u])), shape=(self.n, self.n))
        A.sort_indices()
        return A

    def partition_dict(self, membership: np.ndarray) -> dict:
        """Map original node identifiers to cluster ids; isolated nodes get their own clusters."""
        membership = np.asarray(membership)
        out = {self.labels[i]: int(membership[i]) for i in range(self.n)}
        nxt = int(membership.max()) + 1 if self.n else 0
        for node in self.isolated:
            out[node] = nxt
            nxt += 1
        return out


def _from_pairs(pairs, names=None) -> Graph:
    pairs = np.asarray(pairs)
    if pairs.size == 0:
        raise ValueError("the graph has no edges")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("edges must have shape (m, 2)")
    if names is None:
        names, inv = np.unique(pairs.ravel(), return_inverse=True)
        names = names.tolist()
        pairs = inv.reshape(-1, 2)
    else:
        pairs = pairs.astype(np.int64)
    a = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    b = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    keep = a != b
    a, b = a[keep], b[keep]
    n_all = len(names)
    keys = np.unique(a * n_all + b)
    a, b = keys // n_all, keys % n_all
    used = np.zeros(n_all, dtype=bool)
    used[a] = True
    used[b] = True
    new_id = np.cumsum(used) - 1
    kept_names = [names[i] for i in np.flatnonzero(used)]
    isolated = [names[i] for i in np.flatnonzero(~used)]
    if not kept_names:
        raise ValueError("the graph has no edges after removing self-loops")
    return Graph(n=len(kept_names), u=new_id[a], v=new_id[b], labels=kept_names, isolated=isolated)


def _read_edge_list(path: str) -> Graph:
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#%":
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
    if not rows:
        raise ValueError(f"no edges found in {path}")
    arr = np.array(rows, dtype=object)
    try:
        arr = arr.astype(np.int64)
    except ValueError:
        arr = arr.astype(str)
    return _from_pairs(arr)


def as_graph(obj) -> Graph:
    """Convert any supported graph representation to a :class:`Graph` (see module docstring)."""
    if isinstance(obj, Graph):
        return obj
    if isinstance(obj, (str, os.PathLike)):
        return _read_edge_list(os.fspath(obj))
    mod = type(obj).__module__
    if mod.startswith("networkx"):
        nodes = list(obj.nodes())
        index = {node: i for i, node in enumerate(nodes)}
        pairs = np.array([(index[a], index[b]) for a, b in obj.edges()], dtype=np.int64).reshape(-1, 2)
        return _from_pairs(pairs, names=nodes)
    if mod.startswith("igraph"):
        names = obj.vs["name"] if "name" in obj.vs.attributes() else list(range(obj.vcount()))
        return _from_pairs(np.array(obj.get_edgelist(), dtype=np.int64).reshape(-1, 2), names=list(names))
    if sp.issparse(obj):
        C = sp.coo_matrix(obj)
        if C.shape[0] != C.shape[1]:
            raise ValueError("adjacency matrix must be square")
        mask = C.data != 0
        return _from_pairs(np.c_[C.row[mask], C.col[mask]], names=list(range(C.shape[0])))
    arr = np.asarray(obj)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[1] != 2:
        r, c = np.nonzero(arr)
        return _from_pairs(np.c_[r, c], names=list(range(arr.shape[0])))
    return _from_pairs(arr)
