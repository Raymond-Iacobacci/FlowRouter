"""Initial qubit -> (x, y) placements on the patch grid.

Logical qubits sit on the coarse lattice of cells spaced `QUBIT_DISPLACEMENT` apart,
i.e. x in {d-1, 2d-1, ...} and likewise for y. Three placements are available:

  * snake_mapping   walks that lattice in boustrophedon order, qubit i -> i-th slot;
  * random_mapping  draws a uniformly random permutation of the very same slots, so it
                    differs from the snake only in which qubit lands where -- never in
                    how much room the patch has;
  * lsmap_fast      fits a placement to the circuit's CNOT pairs: spectral and random
                    seeds, each polished by 2-opt swaps, best score wins.

`patch_dimensions` sizes the patch for a qubit count and `initial_qubit_mapping` picks
between the placements for the compilers.
"""

from __future__ import annotations

import itertools
import os
from collections import defaultdict
from random import Random

import numpy as np

QUBIT_DISPLACEMENT = 2


def patch_dimensions(qubit_count):
    """(x_slots, y_slots, nx, ny): the slot grid that holds `qubit_count` qubits and the cell grid around it."""
    x_slots = int(np.ceil(np.sqrt(qubit_count)))
    y_slots = int(np.ceil(qubit_count / x_slots))
    return x_slots, y_slots, 2 * x_slots + 1, 2 * y_slots + 1


def snake_mapping(qubit_count, nx, ny):
    traversal = []
    qubit_i = 0
    for row_idx, j in enumerate(range(QUBIT_DISPLACEMENT - 1, ny, QUBIT_DISPLACEMENT)):
        xs = list(range(QUBIT_DISPLACEMENT - 1, nx, QUBIT_DISPLACEMENT))
        if row_idx % 2 == 1:
            xs.reverse()
        for i in xs:
            if qubit_i >= qubit_count:
                break
            traversal.append((qubit_i, (i, j)))
            qubit_i += 1
        if qubit_i >= qubit_count:
            break
    if qubit_i < qubit_count:
        raise RuntimeError(f'Cannot fit all qubits inside the patch with a snake pattern and a spacing of {QUBIT_DISPLACEMENT}')
    return {q: xy for q, xy in traversal}


def random_mapping(qubit_count, nx, ny, displacement, seed):
    """A uniformly random assignment of qubits to slots, reproducible per seed."""
    slots = [(i, j)
             for j in range(displacement - 1, ny, displacement)
             for i in range(displacement - 1, nx, displacement)]
    if len(slots) < qubit_count:
        raise RuntimeError(f'Cannot fit {qubit_count} qubits inside the patch: '
                           f'only {len(slots)} slots at a spacing of {displacement}')
    Random(seed).shuffle(slots)
    return {q: slots[q] for q in range(qubit_count)}


def lsmap_fast(ops, nx=4, ny=4, disp=2, mu=1.0, n_random=6, seed=0):
    """Place the qubits of `ops` (a list of (control, target) pairs) on an `nx` x `ny` slot grid.

    -> (placement {qubit: (x, y)}, its score, the scoring model). The model's `.sites` lists every slot,
    which callers use to seat qubits that appear in no pair."""

    def sites_of(nx, ny, disp=2, margin=1):
        return [(margin + i * disp, margin + j * disp)
                for j in range(ny - 1, -1, -1) for i in range(nx)]

    class Model:
        """Vectorised closed-form score."""
        def __init__(self, ops, sites, disp=2, mu=1.0):
            self.qubits = sorted({q for op in ops for q in op})
            self.qi = {q: i for i, q in enumerate(self.qubits)}
            self.n = len(self.qubits); self.sites = sites
            self.disp = disp; self.mu = mu
            W = defaultdict(float)
            for a, b in ops:
                if a != b: W[(min(a, b), max(a, b))] += 1.0
            self.E = np.array([[self.qi[a], self.qi[b]] for a, b in W], dtype=int)
            self.w = np.array(list(W.values()), dtype=float)
            self.adj = defaultdict(set)
            for a, b in W:
                self.adj[a].add(b); self.adj[b].add(a)

        def coords(self, pos):
            return np.array([pos[q] for q in self.qubits], dtype=float)

        def score(self, P):
            u, v = self.E[:, 0], self.E[:, 1]
            pu, pv = P[u], P[v]
            d = np.abs(pu - pv).sum(1)
            dil = np.maximum(0.0, d - self.disp)
            x0 = np.minimum(pu[:, 0], pv[:, 0]); x1 = np.maximum(pu[:, 0], pv[:, 0])
            y0 = np.minimum(pu[:, 1], pv[:, 1]); y1 = np.maximum(pu[:, 1], pv[:, 1])
            sx = P[:, 0][None, :]; sy = P[:, 1][None, :]
            inside = ((sx >= x0[:, None]) & (sx <= x1[:, None]) &
                      (sy >= y0[:, None]) & (sy <= y1[:, None]))
            idx = np.arange(len(u))
            inside[idx, u] = False; inside[idx, v] = False
            occ = inside.sum(1).astype(float)
            return float((self.w * (dil + self.mu * occ)).sum())

    def seed_spectral(M):
        from scipy.optimize import linear_sum_assignment
        seen, comps = set(), []
        for q in M.qubits:
            if q in seen: continue
            stack, c = [q], {q}; seen.add(q)
            while stack:
                v = stack.pop()
                for u in M.adj[v]:
                    if u not in seen:
                        seen.add(u); c.add(u); stack.append(u)
            comps.append(sorted(c))
        curve = list(M.sites) # We assume that the whole circuit is connected, for simplicity
        pos, cursor = {}, 0
        for comp in sorted(comps, key=len, reverse=True):
            m = len(comp)
            block = curve[cursor:cursor + m]; cursor += m
            if m == 1:
                pos[comp[0]] = block[0]; continue
            idx = {q: i for i, q in enumerate(comp)}
            Lc = np.zeros((m, m))
            for (i, j), wt in zip(M.E, M.w):
                a, b = M.qubits[i], M.qubits[j]
                if a not in idx: continue
                ia, ib = idx[a], idx[b]
                Lc[ia, ib] -= wt; Lc[ib, ia] -= wt; Lc[ia, ia] += wt; Lc[ib, ib] += wt
            vals, vecs = np.linalg.eigh(Lc)
            nz = [i for i in range(m) if vals[i]]
            vx = vecs[:, nz[0]] if len(nz) > 0 else np.zeros(m)
            vy = vecs[:, nz[1]] if len(nz) > 1 else np.zeros(m)
            def rank(a):
                o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a))
                return r / max(1, len(a) - 1)
            rx, ry = rank(vx), rank(vy)
            bxs = sorted({p[0] for p in block}); bys = sorted({p[1] for p in block})
            B = [(bxs.index(p[0]) / max(1, len(bxs) - 1), bys.index(p[1]) / max(1, len(bys) - 1))
                 for p in block]
            C = np.array([[abs(rx[i] - B[j][0]) + abs(ry[i] - B[j][1])
                           for j in range(m)] for i in range(m)])
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                pos[comp[i]] = block[j]
        return [pos]

    def seed_random(M, k, seed=0):
        rng = Random(seed); out = []
        for _ in range(k):
            sh = M.sites[:]; rng.shuffle(sh)
            out.append({q: sh[i] for i, q in enumerate(M.qubits)})
        return out

    def two_opt(M, pos, max_rounds=50):
        P = M.coords(pos)
        used = {tuple(map(int, p)) for p in P}
        spare = np.array([s for s in M.sites if s not in used], dtype=float).reshape(-1, 2)
        cur = M.score(P)
        for _ in range(max_rounds):
            improved = False
            for i, j in itertools.combinations(range(M.n), 2):
                P[[i, j]] = P[[j, i]]
                s = M.score(P)
                if s < cur - 1e-9: cur = s; improved = True
                else: P[[i, j]] = P[[j, i]]
            for i in range(M.n):
                for k in range(len(spare)):
                    old = P[i].copy(); P[i] = spare[k]
                    s = M.score(P)
                    if s < cur - 1e-9:
                        cur = s; spare[k] = old; improved = True
                    else:
                        P[i] = old
            if not improved: break
        return {q: tuple(map(int, P[i])) for i, q in enumerate(M.qubits)}, cur

    sites = sites_of(nx, ny, disp)
    M = Model(ops, sites, disp, mu)
    cands = seed_spectral(M) + seed_random(M, n_random, seed)
    # Every seed is polished. Ranking the seeds and polishing only the best few was measured (on r01) to
    # correlate poorly, with all four optimal basins ranked 10th or worse -- that prune discarded the optimum.
    cands.sort(key=lambda p: M.score(M.coords(p)))
    best, bestc = None, float('inf')
    for p in cands:
        p, c = two_opt(M, p)
        if c < bestc: best, bestc = p, c

    return best, bestc, M


def initial_qubit_mapping(circuit, use_lsmap):
    """-> (qubit_to_xy, nx, ny): the placement a compiler starts from and the cell grid around it.

    use_lsmap=True fits the placement to the circuit's CNOT pairs with lsmap_fast. Otherwise the qubits go to
    the snake, or -- with TOPOLS_MAPPING=random (seed TOPOLS_MAPPING_SEED, default 0) -- a seeded random
    permutation of the same slots. Those two environment variables are ignored when use_lsmap is True."""
    qubit_count = circuit.qubits
    x_slots, y_slots, nx, ny = patch_dimensions(qubit_count)
    if not use_lsmap:
        if os.environ.get('TOPOLS_MAPPING') == 'random':
            return random_mapping(qubit_count, nx, ny, QUBIT_DISPLACEMENT, int(os.environ.get('TOPOLS_MAPPING_SEED', '0'))), nx, ny
        return snake_mapping(qubit_count = qubit_count, nx = nx, ny = ny), nx, ny
    cnot_pairs = [(g.control, g.target) for g in circuit.gates if g.name == 'CNOT']
    # slot counts, not the cell dimensions nx/ny: sites_of() steps i*disp over range(nx), so passing nx=9
    # would lay sites out to x=17 on a 9-cell-wide patch
    qubit_to_xy, _cost, model = lsmap_fast(cnot_pairs, nx = x_slots, ny = y_slots, disp = QUBIT_DISPLACEMENT)
    for q in range(qubit_count):
        if q not in qubit_to_xy:
            qubit_to_xy[q] = next(s for s in model.sites if s not in qubit_to_xy.values())
    return qubit_to_xy, nx, ny
