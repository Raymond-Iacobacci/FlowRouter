from collections import deque
import networkx as nx
from typing import List, Dict, Hashable, Tuple
from pipe_rules import RULES, RULE_S, ORI_MAP
try:
    from topols_rust_accel import NativeDinic
except ImportError:  # standalone copies of this helper retain the Python path
    NativeDinic = None
Node = Hashable
FlowPath = List[Node]
Augment = Tuple[FlowPath, float]
MAX_COLOR_TRIALS : int = 3
qubit_to_xy : Dict # Basic initialization when the import comes
AXIS = {
    (1, 0, 0): "i",
    (-1, 0, 0): "i",
    (0, 1, 0): "j",
    (0, -1, 0): "j",
    (0, 0, 1): "k",
    (0, 0, -1): "k",
}
def flow_path_rank(path):
    return (max(path[2][1][2], path[-3][1][2]), path[-3][1][2], path[2][1][2], len(path))
def direction(a, b):
    return AXIS[(b[0] - a[0], b[1] - a[1], b[2] - a[2])]
def endpoint_orientation(direction):
    return {"i": "j", "j": "i"}.get(direction)
def binary_pipe_bfs_path(
    *,
    control,
    target,
    G,
    nx,
    ny,
    window_base_height,
    window_size,
    qubit_to_xy,
    qubit_column_spiders,
    spider_counts,
    routed_cubes,
    hadamard_cubes=(),
):
    def residual_open(u, v):
        return G.has_edge(u, v) and G[u][v]["capacity"] - G[u][v]["flow"] > 1e-9
    def split_path(coords):
        path = ["__SRC__", ("H_SRC_COL", control), ("H_SRC_NODE", coords[0]), ("in", coords[0]), ("out", coords[0])]
        for coord in coords[1:-1]:
            path.extend([("in", coord), ("out", coord)])
        path.extend([("in", coords[-1]), ("out", coords[-1]), ("H_SNK_NODE", coords[-1]), ("H_SNK_COL", target), "__SNK__"])
        return path
    data_xy = set(qubit_to_xy.values())
    src_xy, snk_xy = qubit_to_xy[control], qubit_to_xy[target]
    def external_open(coord):
        x, y, z = coord
        return 0 <= x < nx and 0 <= y < ny and window_base_height <= z < window_base_height + window_size and coord not in routed_cubes and coord[:2] not in data_xy
    def source_open(cube):
        x, y, z = cube
        return not any((x, y, z0) in hadamard_cubes and z0 != z for z0 in range(z, window_base_height + window_size)) and not any((color := qubit_column_spiders.get((x, y, z0))) not in (None, "blue") and not (z0 == z and color == "yellow") for z0 in range(z, window_base_height + window_size)) and spider_counts.get(cube, 0) < 2
    def sink_open(cube):
        x, y, z = cube
        return not any((x, y, z0) in hadamard_cubes and z0 != z for z0 in range(z, window_base_height + window_size)) and not any((color := qubit_column_spiders.get((x, y, z0))) not in (None, "red") and not (z0 == z and color == "yellow") for z0 in range(z, window_base_height + window_size)) and spider_counts.get(cube, 0) < 2
    def side_neighbors(cube):
        x, y, z = cube
        return ((x - 1, y, z), (x + 1, y, z), (x, y - 1, z), (x, y + 1, z))

    sink_entries = []
    for sink_cube in ((snk_xy[0], snk_xy[1], z) for z in range(window_base_height, window_base_height + window_size)):
        if not sink_open(sink_cube):
            continue
        for nbr in side_neighbors(sink_cube):
            if not external_open(nbr):
                continue
            sink_orientation = endpoint_orientation(direction(nbr, sink_cube))
            if sink_orientation is not None and residual_open(("out", nbr), ("in", sink_cube)) and residual_open(("in", sink_cube), ("out", sink_cube)) and residual_open(("out", sink_cube), ("H_SNK_NODE", sink_cube)):
                sink_entries.append((sink_cube, nbr, sink_orientation))

    q = deque()
    seen = set()
    for src_cube in ((src_xy[0], src_xy[1], z) for z in range(window_base_height, window_base_height + window_size)):
        if not source_open(src_cube):
            continue
        for nbr in side_neighbors(src_cube):
            if not external_open(nbr) or not (residual_open(("H_SRC_NODE", src_cube), ("in", src_cube)) and residual_open(("in", src_cube), ("out", src_cube)) and residual_open(("out", src_cube), ("in", nbr))):
                continue
            first_dir = direction(src_cube, nbr)
            source_orientation = endpoint_orientation(first_dir)
            pipe_type = None if source_orientation is None else RULE_S.get((0, source_orientation, first_dir))
            if pipe_type is not None and (nbr, first_dir, pipe_type) not in seen:
                seen.add((nbr, first_dir, pipe_type))
                q.append((nbr, first_dir, pipe_type, (src_cube, nbr)))

    while q:
        coord, previous_dir, pipe_type, coords = q.popleft()
        for sink_cube, entry, sink_orientation in sink_entries:
            if coord != entry:
                continue
            final_dir = direction(coord, sink_cube)
            final_type = pipe_type if final_dir == previous_dir else RULES.get((pipe_type, previous_dir, final_dir))
            path = split_path(coords + (sink_cube,))
            if final_type is not None and ORI_MAP.get((final_dir, final_type, 1)) == sink_orientation and all(residual_open(u, v) for u, v in zip(path, path[1:])):
                return path
        for nbr in ((coord[0] + 1, coord[1], coord[2]), (coord[0] - 1, coord[1], coord[2]), (coord[0], coord[1] + 1, coord[2]), (coord[0], coord[1] - 1, coord[2]), (coord[0], coord[1], coord[2] + 1), (coord[0], coord[1], coord[2] - 1)):
            if not external_open(nbr) or nbr in coords:
                continue
            next_dir = direction(coord, nbr)
            next_type = pipe_type if next_dir == previous_dir else RULES.get((pipe_type, previous_dir, next_dir))
            state = (nbr, next_dir, next_type)
            if next_type is not None and state not in seen:
                seen.add(state)
                q.append((nbr, next_dir, next_type, coords + (nbr,)))
    return None
def dinics_with_repairing_paths(
    G: nx.DiGraph,
    source: Node,
    sink: Node,
) -> Tuple[float, List[Augment], Dict[Node, Dict[Node, float]]]:
    """Dinic-only route packing followed by the existing color/legalization repair pass."""

    cube_node_names = frozenset(('H_SRC_NODE', 'H_SNK_NODE', 'in', 'out'))
    blocked_node_names = frozenset(('in', 'out'))
    kind_ranks = {'H_SRC_NODE': 0, 'in': 1, 'out': 2, 'H_SNK_NODE': 3}
    blocked_cubes: set[tuple[int, int, int]] = set()
    accepted_paths: List[Augment] = []

    def node_rank(node):
        if node[0] not in cube_node_names:
            return (2 if node == sink else 1, 0, 0, 0, 0)
        kind = node[0] if isinstance(node, tuple) and len(node) == 2 else ""
        kind_rank = kind_ranks.get(kind, 4)
        return (0, node[1][2], node[1][1], node[1][0], kind_rank)

    def can_commit(path, amount=1.0):
        for node in path:
            if node[0] in blocked_node_names and node[1] in blocked_cubes:
                return False
        return all(G.has_edge(u, v) and G[u][v]["capacity"] - G[u][v]["flow"] + 1e-9 >= amount for u, v in zip(path, path[1:]))

    class Edge:
        # Millions of these are allocated per circuit (one per graph edge, per
        # Dinic construction) and their attributes are read in the innermost
        # BFS/sendFlow loops. __slots__ drops the per-instance __dict__, which
        # makes both the allocation and every attribute access cheaper. No code
        # ever sets an attribute outside this list, so behaviour is unchanged.
        __slots__ = ('v', 'flow', 'C', 'rev')

        def __init__(self, v, flow, C, rev):
            self.v = v
            self.flow = flow
            self.C = C
            self.rev = rev

    class Dinic:
        def __init__(self, graph, source, sink):
            self.nodes = tuple(graph.nodes)
            self.node_index = {node: i for i, node in enumerate(self.nodes)}
            self.source = self.node_index[source]
            self.sink = self.node_index[sink]
            self.adj = [[] for _ in self.nodes]
            self.edges = []
            self.edge_index = {}
            self.level = [-1 for _ in self.nodes]
            # ---- accelerated, semantics-preserving rewrite of the three loops below ----
            # `graph is G` at every call site, so the per-edge data dict reached by
            # walking `_succ` is the very same object `G[u][v]` returned. Iterating
            # `_succ` directly yields edges in exactly the order `graph.edges()` does
            # (outer node order, then insertion order within each node), so edge
            # indices, adjacency order and therefore path selection are unchanged --
            # it just skips the OutEdgeView plus two adapter-object lookups per edge.
            node_index = self.node_index
            edge_list = self.edges
            edge_index = self.edge_index
            adj = self.adj
            # edge_index is keyed by the integer node indices rather than the node
            # objects themselves. Nodes are nested tuples like ('in', (x, y, z)),
            # so hashing a (node, node) key means hashing two of those on every
            # insert and every reverse lookup; a pair of small ints is far cheaper.
            # The mapping never escapes __init__, so the key type is unobservable.
            for u, nbrs in graph._succ.items():
                ui = node_index[u]
                adj_u = adj[ui]
                for v, data in nbrs.items():
                    vi = node_index[v]
                    edge_i = len(edge_list)
                    edge_index[(ui, vi)] = edge_i
                    edge_list.append(Edge(vi, data["flow"], data["capacity"], None))
                    adj_u.append(edge_i)
            for (ui, vi), edge_i in edge_index.items():
                edge_list[edge_i].rev = edge_index[(vi, ui)]
            # node_rank is a pure function of the node, so evaluate it once per node
            # instead of once per incident edge, and sort on the precomputed value.
            # Identical keys + a stable sort == identical ordering.
            node_ranks = [node_rank(node) for node in self.nodes]
            edge_target_rank = [node_ranks[edge.v] for edge in edge_list]
            for edges in adj:
                edges.sort(key=edge_target_rank.__getitem__)

        def usable(self, u, edge):
            u_node = self.nodes[u]
            v_node = self.nodes[edge.v]
            v_cube = v_node[1] if v_node[0] in ("in", "out") else None

            return (
                edge.flow < edge.C
                and v_cube not in blocked_cubes
            )

        def BFS(self, s, t):
            for i in range(len(self.nodes)):
                self.level[i] = -1
            self.level[s] = 0
            q = []
            q.append(s)
            while q:
                u = q.pop(0)
                for i in range(len(self.adj[u])):
                    edge_i = self.adj[u][i]
                    edge = self.edges[edge_i]
                    if self.level[edge.v] < 0 and self.usable(u, edge):
                        self.level[edge.v] = self.level[u] + 1
                        q.append(edge.v)
            return False if self.level[t] < 0 else True

        def sendFlow(self, u, path, start):
            if u == self.sink:
                return path
            while start[u] < len(self.adj[u]):
                edge_i = self.adj[u][start[u]]
                edge = self.edges[edge_i]
                v = edge.v
                if self.level[v] == self.level[u] + 1 and self.usable(u, edge):
                    found = self.sendFlow(v, path + [self.nodes[v]], start)
                    if found is not None:
                        edge.flow += 1.0
                        self.edges[edge.rev].flow -= 1.0
                        return found
                start[u] += 1
            return None

        def DinicMaxflow(self):
            paths = []
            while self.BFS(self.source, self.sink):
                start = [0 for _ in self.nodes]
                while True:
                    path = self.sendFlow(self.source, [source], start)
                    if path is None:
                        break
                    for u, v in zip(path, path[1:]):
                        G[u][v]["flow"] += 1
                        G[v][u]["flow"] -= 1
                    paths.append(path)
            return paths

    def flow_dict_from_residual():
        # `v` ranges over G.successors(u), so G.has_edge(u, v) is True by
        # construction and the guard can never take its 0.0 branch; walking
        # `_succ` hands back the same data dicts without the redundant
        # membership test or the second G[u][v] adapter lookup. `_succ` has an
        # entry for every node, so nodes with no successors still yield {}.
        return {
            u: {
                v: data["flow"] if data["flow"] > 0.0 else 0.0
                for v, data in nbrs.items()
            }
            for u, nbrs in G._succ.items()
        }

    def candidate_flow_dict(paths):
        # Repair only needs positive support.  The old route built a dense
        # dictionary containing every zero/reverse edge, copied the whole
        # structure, and then NetworkX immediately discarded those entries.
        # Filtering while reading G keeps the same u/v insertion order and the
        # same positive amounts without allocating two dense residual copies.
        out = {}
        for u, nbrs in G._succ.items():
            positive = {
                v: data["flow"] for v, data in nbrs.items()
                if data["flow"] > 1e-9
            }
            if positive:
                out[u] = positive
        for path, amount in paths:
            for u, v in zip(path, path[1:]):
                if v in out.get(u, {}):
                    out[u][v] = max(0.0, out[u][v] - amount)
        return out

    def decompose(flow_dict):
        support = nx.DiGraph()
        for u, nbrs in flow_dict.items():
            for v, amount in nbrs.items():
                if amount > 1e-9:
                    support.add_edge(u, v, flow=float(amount))
        paths = []
        while True:
            try:
                path = nx.shortest_path(support, source, sink)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return paths
            amount = int(round(min(support[u][v]["flow"] for u, v in zip(path, path[1:]))))
            paths.append((path, amount))
            for u, v in zip(path, path[1:]):
                support[u][v]["flow"] -= amount
                if support[u][v]["flow"] <= 1e-9:
                    support.remove_edge(u, v)

    def route_coords(path):
        coords = []
        for node in path:
            if node[0] in cube_node_names and (not coords or coords[-1] != node[1]):
                coords.append(node[1])
        return coords

    def axis(a, b):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        if dx:
            return "X"
        if dy:
            return "Y"
        if dz:
            return "Z"
        return None

    def axis_runs(coords):
        runs = []
        for a, b in zip(coords, coords[1:]):
            move_axis = axis(a, b)
            if move_axis is not None and (not runs or runs[-1] != move_axis):
                runs.append(move_axis)
        return runs

    def color_switchiness(path):
        runs = axis_runs(route_coords(path))
        return sum(
            1
            for a, b, c in zip(runs, runs[1:], runs[2:])
            if b == "Z" and {a, c} == {"X", "Y"}
        )

    def coord_path_to_split_path(coords, src_col_hub, snk_col_hub):
        if len(coords) < 3:
            return None
        path = [source, src_col_hub, ("H_SRC_NODE", coords[0]), ('in', coords[0]), ('out', coords[0])]
        for coord in coords[1:-1]:
            path.extend([("in", coord), ("out", coord)])
        path.extend([('in', coords[-1]), ('out', coords[-1]), ("H_SNK_NODE", coords[-1]), snk_col_hub, sink])
        if all(node in G for node in path) and all(G.has_edge(u, v) for u, v in zip(path, path[1:])):
            return path
        return None

    def shift_z(coord, dz):
        return (coord[0], coord[1], coord[2] + dz)

    def xy_turns(coords):
        return [i for i in range(1, len(coords) - 1) if {axis(coords[i - 1], coords[i]), axis(coords[i], coords[i + 1])} == {"X", "Y"}]

    def vertical_steps(coords):
        return [i for i in range(len(coords) - 1) if axis(coords[i], coords[i + 1]) == "Z"]

    def simple(coords):
        return len(coords) == len(set(coords))

    def add_vertical_shift_suffix(coords, dz):
        for cut in xy_turns(coords):
            candidate = coords[: cut + 1] + [shift_z(coords[cut], dz)] + [shift_z(coord, dz) for coord in coords[cut + 1:]]
            if simple(candidate):
                yield candidate

    def add_vertical_shift_prefix(coords, dz):
        for cut in xy_turns(coords):
            candidate = [shift_z(coord, dz) for coord in coords[: cut + 1]] + [coords[cut]] + coords[cut + 1:]
            if simple(candidate):
                yield candidate

    def delete_vertical_shift_suffix(coords):
        for i in vertical_steps(coords):
            dz = coords[i + 1][2] - coords[i][2]
            candidate = coords[: i + 1] + [shift_z(coord, -dz) for coord in coords[i + 2:]]
            if len(candidate) >= 2 and simple(candidate):
                yield candidate

    def valid_repair(candidate_coords, amount, src_col_hub, snk_col_hub):
        path = coord_path_to_split_path(candidate_coords, src_col_hub, snk_col_hub)
        if path is not None and can_commit(path, amount):
            return path
        return None

    def repair_paths(path, amount):
        coords = route_coords(path)
        src_col_hub = next((node for node in path if node[0] == "H_SRC_COL"), None)
        snk_col_hub = next((node for node in path if node[0] == "H_SNK_COL"), None)
        if src_col_hub is None or snk_col_hub is None:
            return
        candidate_iters = (
            add_vertical_shift_suffix(coords, -1),
            add_vertical_shift_prefix(coords, -1),
            delete_vertical_shift_suffix(coords),
            add_vertical_shift_suffix(coords, 1),
            add_vertical_shift_prefix(coords, 1),
        )
        for candidates in candidate_iters:
            for candidate_coords in candidates:
                candidate_path = valid_repair(candidate_coords, amount, src_col_hub, snk_col_hub)
                if candidate_path is not None and color_switchiness(candidate_path) % 2 == 1:
                    yield candidate_path, amount

    native_dinic = NativeDinic.create(G, source, sink, node_rank) if NativeDinic is not None else None
    for trial_i in range(MAX_COLOR_TRIALS):
        native_count = native_dinic.max_flow(blocked_cubes) if native_dinic is not None else None
        if native_count is None:
            native_dinic = None
            new_paths = Dinic(G, source, sink).DinicMaxflow()
        else:
            new_paths = native_count

        candidate_paths = decompose(candidate_flow_dict(accepted_paths))
        if not candidate_paths:
            if not new_paths:
                break
            continue

        for path, amount in candidate_paths:
            for u, v in zip(path, path[1:]):
                G[u][v]["flow"] -= amount
                G[v][u]["flow"] += amount
            if native_dinic is not None:
                native_dinic.adjust_path(path, -amount)

        ranked_candidates = []
        for candidate_i, (path, amount) in enumerate(candidate_paths):
            if color_switchiness(path) % 2 == 1:
                ranked_candidates.append((flow_path_rank(path), candidate_i, path, amount))
                continue
            for repaired_path, repaired_amount in repair_paths(path, amount):
                ranked_candidates.append((flow_path_rank(repaired_path), candidate_i, repaired_path, repaired_amount))

        instantiated = 0
        used_candidate_indices = set()
        seen_paths = set()
        for _, candidate_i, path, amount in sorted(ranked_candidates, key=lambda item: (item[0], item[1])):
            path_key = tuple(path)
            if candidate_i in used_candidate_indices or path_key in seen_paths or not can_commit(path, amount):
                continue
            for u, v in zip(path, path[1:]):
                G[u][v]["flow"] += amount
                G[v][u]["flow"] -= amount
            if native_dinic is not None:
                native_dinic.adjust_path(path, amount)
            accepted_paths.append((path, amount))
            blocked_cubes.update(route_coords(path))
            used_candidate_indices.add(candidate_i)
            seen_paths.add(path_key)
            instantiated += 1

        if instantiated == 0 and not new_paths:
            break

    flow_dict = flow_dict_from_residual()
    finalized_augmentations = decompose(flow_dict)
    instantiated_flow_value = float(sum(amount for _, amount in finalized_augmentations))
    return instantiated_flow_value, finalized_augmentations, flow_dict


