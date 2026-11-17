"""
The main function should take in the circuit, spatial constraints and information about the algorithm and return a block graph. It should also take in some form of hook (as a persistent pass-by-reference object) to store the pbp frames and update that with said pbp frames.
First, the algorithm should get the active frontier and check if CNOTs are available. If it's all H/T gates, place them and rerun the function. In fact, whenever there are H/T gates, those should be placed immediately. This function should only end up running when the entire frontier is made up of CNOT gates.
Then the algorithm actually runs the process. This should consist of first finding the appropriate source/sink cubes (maybe that's one function) and then running MF on them (another function presumeably). If we do parallelization, the other function should have multiple threads running at once.
"""

import argparse
import os
import pickle
import sys
from bisect import bisect_right, insort
from collections import deque
from itertools import product
from pathlib import Path
from typing import List

import networkx
import pyzx
from tqdm import tqdm


def _find_source_root():
    return next(p for p in Path(__file__).resolve().parents if (p / 'maxflow_accelerator.py').exists())


_SRC = _find_source_root()
sys.path.insert(0, str(_SRC))

import maxflow_accelerator
from canonical_pipe_faces import canonical_node_tqec
from initial_mappings import initial_qubit_mapping
from pipe_rules import RULES, RULE_S, ORI_MAP
from tqec_block_adapter import block_graph_from_placed_graph, export_block_graph, pipe_kind_from_axis_type

WINDOW_SIZE = 3
_AXIS_INDEX = {'i': 0, 'j': 1, 'k': 2}


def visible(gate, visible_gate_ids):
    if gate.name == 'CNOT':
        gate_id = id(gate)
        return gate_id in visible_gate_ids[gate.control] and gate_id in visible_gate_ids[gate.target]
    return id(gate) in visible_gate_ids[gate.target]

def cube_path(path):
    coords = []
    for node in path:
        if isinstance(node, tuple) and node[0] in ('H_SRC_NODE', 'in', 'out', 'H_SNK_NODE') and (not coords or coords[-1] != node[1]):
            coords.append(node[1])
    return tuple(coords)

def descent_success(baseline, trial):
    return len(trial) > 0 and len(trial) * 2 >= len(baseline)


class Compiler:
    """Part A compiler, basic variant. All placement state lives on the instance; one Compiler compiles one circuit.
    FreeMagicCompiler overrides the few places where the free-magic variant differs."""

    USE_LSMAP = True
    ROUTED_CUBES_TYPE = set
    S_SPIDER_TYPE = 'blue'
    S_ROLE = 'source'
    S_FACE = 0
    S_LEG_AXIS_BY_PARITY = ('i', 'j')
    S_JUNCTION_LETTER = 'Z'

    def __init__(self, circuit, module, write_html = False):
        self.circuit, self.module, self.write_html = circuit, module, write_html
        self.qubit_to_xy, self.nx, self.ny = initial_qubit_mapping(circuit, use_lsmap = self.USE_LSMAP)
        self.xy_to_qubit = {xy: q for q, xy in self.qubit_to_xy.items()}
        self.unplaced_gates = [gate for gate in circuit.gates.copy() if gate.name in ("CNOT", "HAD", "T", "S")]
        self.unplaced_gate_ids = {id(gate) for gate in self.unplaced_gates}
        self.progress = tqdm(total = len(self.unplaced_gates), desc = 'Placing gates', unit = 'gate')
        self._build_events()
        self._build_spiders()
        self._visible_heads = [0] * circuit.qubits
        self.active_columns = list(range(circuit.qubits))
        self.window_base_height, self.round_index = 0, 0
        self._descent_floor = None
        self.frontier_gates, self._last_endpoint_columns = [], []
        self._round_placed_z = {}
        self.routed_cubes = self.ROUTED_CUBES_TYPE()
        self.hadamard_cubes = set()
        self.qubit_column_spiders, self.spider_counts, self._hadamard_z_by_xy = {}, {}, {}
        self.spider_endpoint_metadata, self.routed_segment_pipe_types = {}, {}
        self.s_gate_positions, self.s_machinery_cubes = {}, {}
        self.display_paths, self.t_gate_cubes, self.s_machinery_pipes, self.s_gate_squares = [], [], [], []

    def _build_events(self):
        """One event per placed gate kind, in circuit order (the pickle and Part B index into this list)."""
        self.circuit_events, self.cnot_event_by_gate_id, self.t_gate_positions = [], {}, {}
        self.t_event_index_by_gate_id, self.s_event_index_by_gate_id, h_count = {}, {}, 0
        for gate_index, gate in enumerate(self.circuit.gates):
            if id(gate) not in self.unplaced_gate_ids:
                continue
            index, name = len(self.circuit_events), gate.name
            if name == 'CNOT':
                event = {'kind': 'cnot', 'event_index': index, 'gate_index': gate_index, 'pair': None,
                         'qubits': (int(gate.control), int(gate.target))}
                self.cnot_event_by_gate_id[id(gate)] = event
            elif name == 'HAD':
                event = {'kind': 'h', 'id': h_count, 'event_index': index, 'qubit': int(gate.target), 'gate_index': gate_index}
                h_count += 1
            else:
                (self.t_event_index_by_gate_id if name == 'T' else self.s_event_index_by_gate_id)[id(gate)] = index
                event = {'kind': 't' if name == 'T' else 's', 'event_index': index, 'gate_index': gate_index, 'qubit': int(gate.target)}
                if name == 'S':
                    event['adjoint'] = bool(getattr(gate, 'adjoint', False))
            self.circuit_events.append(event)

    def _build_spiders(self):
        """The colour each gate leaves on each qubit column; adjacent same-coloured spiders commute."""
        colours = {'T': (('target', 'blue'),), 'CNOT': (('target', 'red'), ('control', 'blue')),
                   'HAD': (('target', 'yellow'),), 'S': (('target', self.S_SPIDER_TYPE),)}
        self.spiders_by_qubit = [[] for _ in range(self.circuit.qubits)]
        for gate in self.unplaced_gates:
            for role, colour in colours[gate.name]:
                self.spiders_by_qubit[getattr(gate, role)].append({'type': colour, 'gate': gate})

    def run(self):
        self.place_all_gates()
        self.write_outputs()

    def _window_lo(self):
        return max(0, self.window_base_height - WINDOW_SIZE)

    def _window_hi(self):
        return self.window_base_height + WINDOW_SIZE

    def _effective_lo(self):
        return min(self._window_lo(), max(0, self._descent_floor)) if self._descent_floor is not None else self._window_lo()

    def hadamard_parity(self, x, y, z):
        return bisect_right(self._hadamard_z_by_xy.get((x, y), ()), z) % 2

    def hadamard_blocked_above(self, x, y, z):
        zs = self._hadamard_z_by_xy.get((x, y), ())
        first_above = bisect_right(zs, z)
        return first_above < len(zs) and zs[first_above] < self._window_hi()

    def _note_placed(self, gate, z):
        prev = self._round_placed_z.get(id(gate))
        self._round_placed_z[id(gate)] = z if prev is None else min(prev, z)

    def _retire(self, gate, z = None):
        """Take a placed gate out of the frontier and the unplaced set, record its height (if any), tick the progress bar."""
        self.frontier_gates.remove(gate)
        self.unplaced_gates.remove(gate)
        self.unplaced_gate_ids.discard(id(gate))
        if z is not None:
            self._note_placed(gate, z)
        self.progress.update(1)

    def _cell_free_test(self, ceiling):
        """A predicate for 'this cell is inside the box, off the qubit columns and not already routed over'."""
        column_squares = set(self.qubit_to_xy.values())
        def cell_free(cell):
            return (0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny and 0 <= cell[2] <= ceiling
                    and cell[:2] not in column_squares
                    and cell not in self.routed_cubes)
        return cell_free

    def _visible_gate_ids_by_qubit(self):
        """Compute every column's visible gate group in one pass.

        The old frontier test rebuilt the same visible-index list once per gate (and
        restarted every scan at index zero).  Gate identity and list order are kept
        exactly; this only shares the per-column result across one frontier refresh.
        """
        groups = []
        for qubit, events in enumerate(self.spiders_by_qubit):
            start = self._visible_heads[qubit]
            while start < len(events) and id(events[start]['gate']) not in self.unplaced_gate_ids:
                start += 1
            self._visible_heads[qubit] = start
            if start >= len(events):
                groups.append(set())
                continue
            visible_color = events[start]['type']
            visible = set()
            for i in range(start, len(events)):
                gate_id = id(events[i]['gate'])
                if gate_id not in self.unplaced_gate_ids:
                    continue
                if events[i]['type'] != visible_color:
                    break
                visible.add(gate_id)
            groups.append(visible)
        return groups

    def _column_open(self, role, x, y, z):
        if role == 'H':
            return self._h_column_open(x, y, z)
        elif self._blocked_by_t_gate(role, x, y, z):
            return False
        allowed = 'red' if role == 'sink' else 'blue'
        return (
            not self.hadamard_blocked_above(x, y, z)
            and not any(
                (color := self.qubit_column_spiders.get((x, y, z0), None)) not in (None, allowed)
                and not (z0 == z and color == 'yellow')
                for z0 in range(z, self._window_hi())
            )
            and self.spider_counts.get((x, y, z), 0) < 2
        )

    def _h_column_open(self, x, y, z):
        """Is a Hadamard allowed at this height? Nothing may sit above it on the column."""
        return not any(coord[:2] == (x, y) and coord[2] >= z for coord in self.qubit_column_spiders)

    def _blocked_by_t_gate(self, role, x, y, z):
        """Does a T-gate already hold this cube, or (for a T) is there nowhere to inject it from?"""
        return (role == 'T' and self.available_t_gate_space(x, y, z) is None) or (x, y, z) in self.t_gate_cubes

    def column_floor(self, column, role):
        x, y = self.qubit_to_xy[column]
        for z in range(1 if role == 'H' else 0, self._window_hi()):
            if self._column_open(role, x, y, z):
                return z
        return self._window_hi()

    def compute_frontier_gates_and_apply_one_qubit_gates(self):
        before = len(self.unplaced_gates)
        visible_gate_ids = self._visible_gate_ids_by_qubit()
        self.frontier_gates = [unplaced_gate for unplaced_gate in self.unplaced_gates if visible(unplaced_gate, visible_gate_ids)]
        frontier_ts = [unplaced_gate for unplaced_gate in self.frontier_gates if unplaced_gate.name == 'T']
        frontier_ss = [unplaced_gate for unplaced_gate in self.frontier_gates if unplaced_gate.name == 'S']
        frontier_hs = [unplaced_gate for unplaced_gate in self.frontier_gates if unplaced_gate.name == 'HAD']
        for t in frontier_ts:
            x, y = self.qubit_to_xy[t.target]
            t_floor = self.column_floor(t.target, 'T')
            for z in range(t_floor, self._window_hi()):
                if self._column_open('T', x, y, z):
                    cube = (x, y, z)
                    count = self.spider_counts.get(cube, 0)
                    self._reserve_t_gate(t, x, y, z, cube, count)
                    self.t_gate_cubes.append(cube)
                    self.t_gate_positions[self.t_event_index_by_gate_id[id(t)]] = cube
                    self._retire(t, z)
                    break
        for s in frontier_ss:
            x, y = self.qubit_to_xy[s.target]
            s_floor = self.column_floor(s.target, self.S_ROLE)
            for z in range(s_floor, max([s_floor, self._window_hi(), *(cell[2] for cell in self.routed_cubes), *(cell[2] for cell in self.qubit_column_spiders)]) + 6):
                if not self._column_open(self.S_ROLE, x, y, z):
                    continue
                cube = (x, y, z)
                _, orientation, _ = self.s_gate_touch_kind(x, y, z)
                if self.spider_endpoint_metadata.get(cube, (orientation, self.S_FACE)) != (orientation, self.S_FACE):
                    continue
                machinery = self.bake_s_gate_machinery(x, y, z)
                if machinery is None:
                    continue
                self.qubit_column_spiders[cube] = self.S_SPIDER_TYPE
                self.spider_counts[cube] = self.spider_counts.get(cube, 0) + 1
                self.spider_endpoint_metadata[cube] = (orientation, self.S_FACE)
                self.routed_cubes.update(machinery['cubes'])
                self.s_machinery_cubes.update(machinery['cubes'])
                self.s_machinery_pipes.extend(machinery['pipes'])
                self.s_gate_squares.append(machinery['anchor'])
                self.s_gate_positions[self.s_event_index_by_gate_id[id(s)]] = cube
                self._retire(s, z)
                break
            else:
                raise RuntimeError(f'S-gate on qubit {s.target} found no legal height below the guard; the walk is supposed to succeed in open space')
        for h in frontier_hs:
            x, y = self.qubit_to_xy[h.target]
            h_floor = self.column_floor(h.target, 'H')
            for z in range(h_floor, self._window_hi()):
                if self._column_open('H', x, y, z):
                    cube = (x, y, z)
                    self.qubit_column_spiders[cube] = 'yellow'
                    self.hadamard_cubes.add(cube)
                    insort(self._hadamard_z_by_xy.setdefault((x, y), []), z)
                    self._retire(h)
                    break
        return len(self.unplaced_gates) < before

    def _reserve_t_gate(self, t, x, y, z, cube, count):
        """Claim the cube for T-gate `t` and hold the cell its injection will use."""
        route = self.available_t_gate_space(x, y, z)
        self.qubit_column_spiders[cube] = 'blue'
        self.spider_counts[cube] = count + 1
        self.routed_cubes.update(route)

    def frontier_src_snk_assignment(self, columns: List[int]):
        '''
        Frontier-only qubit-column implementation: rather than splitting every column in the patch into a
        random source/sink half, only implement the columns the frontier actually asks for, and take each
        column's role from the gate that wants it (control -> source, target -> sink). This is the
        generalization of the len(active_columns) == 2 exact case below, which already reads the roles off
        the gate instead of guessing them.
        Gates are taken in circuit order and must be disjoint, since a column can only be a source or a sink
        in a given round. A frontier one-qubit gate still claims its column even though it never becomes an
        endpoint -- a T that failed to place is holding that column and must not be routed over.
        '''
        pool, used, sources, sinks = set(columns), set(), [], []
        for gate in self.frontier_gates:
            is_cnot = gate.name == 'CNOT'
            qubits = [int(gate.control), int(gate.target)] if is_cnot else [int(gate.target)]
            if any(qubit in used for qubit in qubits):
                continue
            if is_cnot and not all(qubit in pool for qubit in qubits):
                continue
            used.update(qubits)
            if is_cnot:
                sources.append(int(gate.control))
                sinks.append(int(gate.target))
        return sources, sinks

    def available_t_gate_space(self, x, y, z):
        """The cells a T-gate at (x, y, z) reserves for its injection, or None when it has no legal place here:
        the first free lateral neighbour."""
        injection_axis = 'y' if self.hadamard_parity(x, y, z) == 1 else 'x'
        candidates = (
            ((x - 1, y, z), (x + 1, y, z))
            if injection_axis == 'x'
            else ((x, y - 1, z), (x, y + 1, z))
        )
        for space in candidates:
            if space not in self.routed_cubes:
                return [space]
        return None

    def s_gate_touch_kind(self, x, y, z):
        '''The axis, orientation and tqec cube kind of the qubit-column cube an S-gate is defined on.

        The S is a CNOT target (sink face 1) whose leg leaves on the sink axis, so the kind is
        ORI_MAP[(leg axis, pipe type, 1)] fed through canonical_node_tqec. Of the two pipe types, 0 gives
        orientation 'k' and hence ZZX, which QUBIT_COLUMN_KINDS in tqec_block_adapter refuses on a column
        cube because its first two letters match -- _choose_cube_kind would find no candidate and raise. So
        the leg's pipe type is 1 and the kind is forced: XZZ where the sink axis is y, ZXZ where it is x.
        Either way the leg attaches through the cube's Z (blue) face, which is what makes this a red spider.
        '''
        leg_axis = self.S_LEG_AXIS_BY_PARITY[self.hadamard_parity(x, y, z)]
        orientation = ORI_MAP[(leg_axis, self.S_FACE, self.S_FACE)]
        return _AXIS_INDEX[leg_axis], orientation, canonical_node_tqec(orientation, self.S_FACE)

    def bake_s_gate_machinery(self, x, y, z):
        '''Walk out from the S-gate's cube at (x, y, z) and return every cube and pipe its machinery needs.

        The shape: one lateral arm off the column, a chain of free cells, then a BLUE junction carrying the
        Y half-cube directly below it and the Z-measurement square on the top face of the leg above it. The
        junction is found first rather than the Y cell: a Y found next to the column would leave the S's own
        cube as the only branch point, and that is a red spider on a worldline that continues upwards, so a
        Y-first walk would have to back out of it. Junction-first cannot get into that state.

        The Y half-cube attaches only through a vertical pipe, and the square occupies +z, so the junction's
        three arms are forced to be {lateral in, -z Y, +z square-or-leg} -- two axes, which is what the
        planarity rule everywhere else in this codebase allows. A blue cube that can take those vertical
        pipes is exactly XZX or ZXX (two X letters, and the first two differing so a z-pipe is well formed),
        the same kinds a CNOT control and a T-gate take.

        The Z-measurement square is stamped on the TOP face of its anchor, never the bottom, even though
        either is sound geometry: every consumer of a square assumes the top. ts-best-inj-ws.py re-derives
        _marker['cell'] as anchor + (0, 0, 1) through the whole-circuit fold, its html stamps the mesh at
        anchor_z * 3 + 0.5 + 1.55, and its pin test accepts an anchor whose only pipe neighbour is BELOW it.
        A bottom-face square would need a sign threaded through all three, so the walk refuses one here
        rather than paying for it there. The junction's arity therefore counts a virtual pipe in +z, and the
        height rule reduces to anchor z >= the Y cell and the touch, since the square sits at anchor z + 0.5.

        There is deliberately no available_s_gate_space() alongside available_t_gate_space(): that function
        holds cells so the timeseries can route an injection it has not planned yet, and an S has no
        decision left over for anything downstream to make. This walk is the only test of whether a height
        works, and the cells it returns are occupancy, not a reservation.
        '''
        axis, orientation, touch_kind = self.s_gate_touch_kind(x, y, z)
        ceiling = max([z, self._window_hi(), *(cell[2] for cell in self.routed_cubes), *(cell[2] for cell in self.qubit_column_spiders)]) + 4
        cell_free = self._cell_free_test(ceiling)
        queue, seen = deque(), set()
        for sign in (-1, 1):
            first = tuple(c + (sign if a == axis else 0) for a, c in enumerate((x, y, z)))
            if not cell_free(first):
                continue
            pipe_kind = ''.join('O' if a == axis else touch_kind[a] for a in range(3))
            for free_letter in ('X', 'Z'):
                kind = ''.join(free_letter if a == axis else touch_kind[a] for a in range(3))
                queue.append((first, kind, frozenset({(axis, -sign)}), (((x, y, z), first, pipe_kind, kind),)))
        while queue:
            cell, kind, dirs_here, chain = queue.popleft()
            chain_cells = {step[1] for step in chain}
            axes_here = {a for a, _ in dirs_here}
            y_cell = (cell[0], cell[1], cell[2] - 1)
            if (kind.count(self.S_JUNCTION_LETTER) == 2 and kind[0] != kind[1]
                    and len(axes_here | {2}) <= 2
                    and (2, -1) not in dirs_here and (2, 1) not in dirs_here
                    and cell_free(y_cell) and y_cell not in chain_cells):
                leg, anchor = [], cell
                while anchor[2] < z:
                    above = (anchor[0], anchor[1], anchor[2] + 1)
                    if not cell_free(above) or above in chain_cells or above == y_cell:
                        break
                    leg.append((anchor, above, kind[:2] + 'O', kind))
                    anchor = above
                if anchor[2] >= z:
                    cubes = {step[1]: step[3] for step in chain}
                    cubes.update({step[1]: step[3] for step in leg})
                    cubes[y_cell] = 'Y'
                    pipes = [(step[0], step[1], step[2]) for step in chain]
                    pipes.extend((step[0], step[1], step[2]) for step in leg)
                    pipes.append((y_cell, cell, kind[:2] + 'O'))
                    return {'cubes': cubes, 'pipes': pipes, 'anchor': anchor, 'orientation': orientation}
            for step_axis in range(3):
                if len(axes_here | {step_axis}) > 2:
                    continue
                transverse = [a for a in range(3) if a != step_axis]
                if kind[transverse[0]] == kind[transverse[1]]:
                    continue
                step_pipe_kind = ''.join('O' if a == step_axis else kind[a] for a in range(3))
                for sign in (-1, 1):
                    if (step_axis, sign) in dirs_here:
                        continue
                    new_cell = tuple(cell[a] + (sign if a == step_axis else 0) for a in range(3))
                    if not cell_free(new_cell) or new_cell in chain_cells:
                        continue
                    for free_letter in ('X', 'Z'):
                        new_kind = ''.join(free_letter if a == step_axis else kind[a] for a in range(3))
                        state = (new_cell, new_kind, (step_axis, -sign))
                        if state in seen:
                            continue
                        seen.add(state)
                        queue.append((new_cell, new_kind, frozenset({(step_axis, -sign)}), chain + ((cell, new_cell, step_pipe_kind, new_kind),)))
        return None

    def cnot_path_metadata(self, path, sink_face = 1):
        dirs = [self.module.direction(a, b) for a, b in zip(path, path[1:])]
        source_face = 0
        source_orientation = self.module.endpoint_orientation(dirs[0])
        if source_orientation is None:
            return None
        pipe_type = RULE_S.get((source_face, source_orientation, dirs[0]))
        if pipe_type is None:
            return None
        specs = {}
        for index, (a, b) in enumerate(zip(path, path[1:])):
            if index > 0 and dirs[index] != dirs[index - 1]:
                pipe_type = RULES.get((pipe_type, dirs[index - 1], dirs[index]))
                if pipe_type is None:
                    return None
            specs[tuple(sorted((tuple(a), tuple(b))))] = int(pipe_type)
        sink_orientation = ORI_MAP.get((dirs[-1], pipe_type, sink_face))
        if sink_orientation is None:
            return None
        return (source_orientation, source_face), (sink_orientation, sink_face), specs

    def run_iteration(self):
        self._round_placed_z.clear()
        self._last_endpoint_columns = []

        placed_oqg = self.compute_frontier_gates_and_apply_one_qubit_gates()
        if not any(gate.name == 'CNOT' for gate in self.frontier_gates):
            if not placed_oqg:
                self.window_base_height, self.active_columns = self.window_base_height + 1, list(range(self.circuit.qubits))
            return dict(self._round_placed_z)
        uncommitted_graph = networkx.DiGraph()
        uncommitted_graph.add_node('__SRC__')
        uncommitted_graph.add_node('__SNK__')
        _succ = uncommitted_graph._succ
        _pred = uncommitted_graph._pred
        _nodes = uncommitted_graph._node
        def _add_edge(u, v, capacity, flow=None):
            if u not in _succ:
                _succ[u] = {}
                _pred[u] = {}
                _nodes[u] = {}
            if v not in _succ:
                _succ[v] = {}
                _pred[v] = {}
                _nodes[v] = {}
            data = _succ[u].get(v)
            if data is None:
                data = {}
                _succ[u][v] = data
                _pred[v][u] = data
            data['capacity'] = capacity
            if flow is not None:
                data['flow'] = flow
        cnots = [g for g in self.frontier_gates if g.name == 'CNOT']
        if len(self.active_columns) == 2 and (gate := next((g for g in cnots if g.control in self.active_columns and g.target in self.active_columns), None)) is not None:
            source_columns, sink_columns = [gate.control], [gate.target]
        else:
            source_columns, sink_columns = self.frontier_src_snk_assignment(self.active_columns)
        self._last_endpoint_columns = [(c, 'source') for c in source_columns] + [(c, 'sink') for c in sink_columns]
        effective_lo = self._effective_lo()
        window_hi = self._window_hi()
        data_cubes = {(x, y, z) for x, y in self.qubit_to_xy.values() for z in range(effective_lo, window_hi)}
        for x, y, z in product(range(self.nx), range(self.ny), range(effective_lo, window_hi)):
            if (x, y, z) not in self.routed_cubes:
                _add_edge(('in', (x, y, z)), ('out', (x, y, z)), capacity = 1)
        for x, y, z in product(range(self.nx), range(self.ny), range(effective_lo, window_hi)):
            if (x, y, z) in self.routed_cubes:
                continue
            for nbr in (
                (x + 1, y, z),
                (x - 1, y, z),
                (x, y + 1, z),
                (x, y - 1, z),
                (x, y, z + 1),
                (x, y, z - 1),
            ):
                if not (0 <= nbr[0] < self.nx and 0 <= nbr[1] < self.ny and effective_lo <= nbr[2] < window_hi):
                    continue
                if nbr in self.routed_cubes:
                    continue
                if (x, y, z) in data_cubes or nbr in data_cubes:
                    continue
                _add_edge(('out', (x, y, z)), (('in', nbr)), capacity = 1)
        for column in source_columns:
            x, y = self.qubit_to_xy[column]
            _add_edge('__SRC__', ('H_SRC_COL', column), capacity = 2 * (window_hi - effective_lo))
            source_opened = 0
            for z in range(effective_lo, window_hi):
                _add_edge(('H_SRC_COL', column), ('H_SRC_NODE', (x, y, z)), capacity = 1)
                source_axis = 'x' if self.hadamard_parity(x, y, z) == 0 else 'y'
                if (source_open := self._column_open('source', x, y, z)):
                    _add_edge(('H_SRC_NODE', (x, y, z)), ('in', (x, y, z)), capacity = 1)
                for nbr in ( ((x - 1, y, z), (x + 1, y, z)) if source_axis == 'x' else ((x, y - 1, z), (x, y + 1, z)) ):
                    if 0 <= nbr[0] < self.nx and 0 <= nbr[1] < self.ny and nbr not in self.routed_cubes:
                        _add_edge(('out', (x, y, z)), ('in', nbr), capacity = 1)
                if source_open:
                    source_opened += 1
                if source_opened >= WINDOW_SIZE:
                    break
        for column in sink_columns:
            x, y = self.qubit_to_xy[column]
            _add_edge(('H_SNK_COL', column), '__SNK__', capacity = 2 * (window_hi - effective_lo))
            sink_opened = 0
            for z in range(effective_lo, window_hi):
                _add_edge(('H_SNK_NODE', (x, y, z)), ('H_SNK_COL', column), capacity = 1)
                sink_axis = 'y' if self.hadamard_parity(x, y, z) == 0 else 'x'
                if (sink_open := self._column_open('sink', x, y, z)):
                    _add_edge(('out', (x, y, z)), ('H_SNK_NODE', (x, y, z)), capacity = 1)
                for nbr in ( ((x - 1, y, z), (x + 1, y, z)) if sink_axis == 'x' else ((x, y - 1, z), (x, y + 1, z)) ):
                    if 0 <= nbr[0] < self.nx and 0 <= nbr[1] < self.ny and nbr not in self.routed_cubes:
                        _add_edge(('out', nbr), ('in', (x, y, z)), capacity = 1)
                if sink_open:
                    sink_opened += 1
                if sink_opened >= WINDOW_SIZE:
                    break
        for u, nbrs in _succ.items():
            for v, forward_data in nbrs.items():
                if 'flow' not in forward_data:
                    forward_data['flow'] = 0
                reverse_data = _succ[v].get(u)
                if reverse_data is None:
                    _add_edge(v, u, capacity = 0, flow = 0)
                else:
                    if 'capacity' not in reverse_data:
                        reverse_data['capacity'] = 0
                    if 'flow' not in reverse_data:
                        reverse_data['flow'] = 0
        self.module.qubit_to_xy = self.qubit_to_xy
        _, finalized_augmentations, _ = self.module.dinics_with_repairing_paths(uncommitted_graph, '__SRC__', '__SNK__')
        any_CNOT_gates_placed = False
        rebuild_flow_graph = False
        while self.frontier_gates:
            placed = False
            for gate in (gate for gate in self.frontier_gates if gate.name == 'CNOT'):
                finalized_augmentation = min(((path, count) for path, count in finalized_augmentations if count > 0 and (self.xy_to_qubit[path[2][1][:2]], self.xy_to_qubit[path[-3][1][:2]]) == (gate.control, gate.target) ), key=lambda pc: self.module.flow_path_rank(pc[0]), default=None)
                if finalized_augmentation is None and len(self.active_columns) == 2 and source_columns == [gate.control] and sink_columns == [gate.target]:
                    fallback_path = self.module.binary_pipe_bfs_path( control=gate.control, target=gate.target, G=uncommitted_graph, nx=self.nx, ny=self.ny, window_base_height=effective_lo, window_size=window_hi - effective_lo, qubit_to_xy=self.qubit_to_xy, qubit_column_spiders=self.qubit_column_spiders, spider_counts=self.spider_counts, routed_cubes=self.routed_cubes, hadamard_cubes=self.hadamard_cubes )
                    if fallback_path is not None:
                        finalized_augmentation = (fallback_path, 1)
                        finalized_augmentations.append(finalized_augmentation)
                if finalized_augmentation is None:
                    continue
                path, count = finalized_augmentation
                display_path = cube_path(path)
                path_metadata = self.cnot_path_metadata(display_path)
                if path_metadata is None:
                    continue
                source_metadata, sink_metadata, pipe_types = path_metadata
                for segment, pipe_type in pipe_types.items():
                    self.routed_segment_pipe_types[segment] = pipe_type
                self.routed_cubes.update(node[1] for node in path if node[0] in ('in', 'out') and node[1][:2] not in self.qubit_to_xy.values())
                self.qubit_column_spiders[path[2][1]] = 'blue'
                self.qubit_column_spiders[path[-3][1]] = 'red'
                self.spider_endpoint_metadata[path[2][1]] = source_metadata
                self.spider_endpoint_metadata[path[-3][1]] = sink_metadata
                self.spider_counts[path[2][1]] = self.spider_counts.get(path[2][1], 0) + 1
                self.spider_counts[path[-3][1]] = self.spider_counts.get(path[-3][1], 0) + 1
                self.display_paths.append(display_path)
                self.cnot_event_by_gate_id[id(gate)]['pair_coords'] = (display_path[0], display_path[-1])
                self._retire(gate, min(path[2][1][2], path[-3][1][2]))
                finalized_augmentations.remove(finalized_augmentation)
                if count > 1:
                    finalized_augmentations.append((path, count - 1))
                routed_cube_count_before_oqg = len(self.routed_cubes)
                self.compute_frontier_gates_and_apply_one_qubit_gates()
                rebuild_flow_graph = len(self.routed_cubes) != routed_cube_count_before_oqg
                placed = True
                any_CNOT_gates_placed = True
                self.active_columns = list(range(self.circuit.qubits))
                break
            if rebuild_flow_graph:
                break
            if not placed:
                break
        if not any_CNOT_gates_placed:
            if len(self.active_columns) == 2:
                self.window_base_height += 1
                self.active_columns = list(range(self.circuit.qubits))
                return dict(self._round_placed_z)
            else:
                ordered = list(dict.fromkeys(q for g in self.unplaced_gates if g.name == 'CNOT' for q in (g.control, g.target) if q in self.active_columns))
                self.active_columns = ordered[:max(len(ordered) // 2, 2)]
        return dict(self._round_placed_z)

    def _tracked(self):
        """(attribute, copy kind) for everything a descent trial can change and a revert must undo."""
        return (('window_base_height', 'value'), ('active_columns', 'list_rebind'), ('frontier_gates', 'list_rebind'),
                ('routed_cubes', 'set'), ('qubit_column_spiders', 'dict'),
                ('spider_counts', 'dict'), ('hadamard_cubes', 'set'), ('_hadamard_z_by_xy', 'dict_of_lists'),
                ('unplaced_gates', 'list'), ('unplaced_gate_ids', 'set'), ('_visible_heads', 'list'),
                ('display_paths', 'list'), ('t_gate_cubes', 'list'),
                ('t_gate_positions', 'dict'), ('s_gate_positions', 'dict'), ('s_machinery_cubes', 'dict'),
                ('s_machinery_pipes', 'list'), ('s_gate_squares', 'list'), ('spider_endpoint_metadata', 'dict'),
                ('routed_segment_pipe_types', 'dict'), ('cnot_event_by_gate_id', 'dict_of_dicts'))

    def _snapshot(self):
        copiers = {'value': lambda v: v, 'list': list, 'list_rebind': list, 'dict': dict, 'set': set,
                   'dict_of_lists': lambda d: {k: list(v) for k, v in d.items()},
                   'dict_of_dicts': lambda d: {k: dict(v) for k, v in d.items()}}
        return {name: copiers[kind](getattr(self, name)) for name, kind in self._tracked()}

    def _restore(self, s):
        """Put every tracked attribute back. Containers are refilled in place so that anything holding a reference
        (cnot_event_by_gate_id's dicts are shared with circuit_events) sees the restored contents."""
        for name, kind in self._tracked():
            saved, current = s[name], getattr(self, name)
            if kind == 'value':
                setattr(self, name, saved)
            elif kind == 'list_rebind':
                setattr(self, name, list(saved))
            elif kind == 'list':
                current[:] = saved
            elif kind in ('dict', 'set'):
                current.clear(); current.update(saved)
            elif kind == 'dict_of_lists':
                current.clear(); current.update((xy, list(zs)) for xy, zs in saved.items())
            else:
                for key, inner in saved.items():
                    current[key].clear(); current[key].update(inner)

    def place_all_gates(self):
        while self.unplaced_gates:
            self.round_index += 1
            pre = self._snapshot()
            self._descent_floor = None
            baseline = self.run_iteration()
            endpoint_cols = list(self._last_endpoint_columns)
            committed = self._snapshot()
            if endpoint_cols and len(baseline) > 0:
                self._restore(pre)
                target = min(self.column_floor(c, role) for c, role in endpoint_cols)
                G = self._window_lo() - target
                self._restore(committed)
                gap = G
                while G > 0 and gap > 0:
                    gap = gap // 2
                    self._restore(pre)
                    self._descent_floor = target + gap
                    trial = self.run_iteration()
                    if descent_success(baseline, trial):
                        committed = self._snapshot()
                    else:
                        self._restore(committed)
            self._descent_floor = None
            if self.progress.total is not None:
                self.progress.n = self.progress.total - len(self.unplaced_gates)
                self.progress.refresh()

    def write_outputs(self):
        self.progress.close()
        segment_hadamards = {frozenset(((coord[0], coord[1], coord[2] - 1), coord)): 1 for coord in self.hadamard_cubes}
        final_z = max([1, *(coord[2] for path in self.display_paths for coord in path), *(coord[2] for coord in self.qubit_column_spiders)])
        placed_graph = networkx.DiGraph()
        placed_graph.graph["display_paths"] = tuple(self.display_paths) + tuple(tuple((x, y, z) for z in range(final_z + 1)) for x, y in self.qubit_to_xy.values())
        spider_positions = {f"{color}_{i}": coord for i, (coord, color) in enumerate(self.qubit_column_spiders.items())}
        spider_node_by_coord = {coord: node_id for node_id, coord in spider_positions.items()}
        resolved_circuit_events = []
        for event in self.circuit_events:
            event = dict(event)
            if event['kind'] == 'cnot':
                event['pair'] = tuple(spider_node_by_coord[coord] for coord in event.pop('pair_coords'))
            resolved_circuit_events.append(event)
        payload_spider_positions = dict(spider_positions)
        t_spider_entries = self._t_spider_entries(spider_positions, resolved_circuit_events, payload_spider_positions)
        placed_graph.graph["spider_positions"] = spider_positions
        placed_graph.graph["spider_orientations"] = {node_id: self.spider_endpoint_metadata[coord][0] for node_id, coord in spider_positions.items() if coord in self.spider_endpoint_metadata}
        placed_graph.graph["spider_faces"] = {node_id: self.spider_endpoint_metadata[coord][1] for node_id, coord in spider_positions.items() if coord in self.spider_endpoint_metadata}
        placed_graph.graph["segment_hadamards"] = segment_hadamards
        placed_graph.graph["allow_inferred_hadamards"] = bool(self.t_gate_cubes)
        placed_graph.graph["qubit_to_xy"] = self.qubit_to_xy
        placed_graph.graph["num_qubits"] = self.circuit.qubits
        placed_graph.graph["circuit_events"] = tuple(resolved_circuit_events)
        log_path = _SRC.parent / 'result' / 'pbp_log.pkl'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'wb') as fp:
            pickle.dump({
                "circuit_events": tuple(resolved_circuit_events),
                "qubit_to_xy": dict(self.qubit_to_xy),
                "spider_positions": payload_spider_positions,
                **t_spider_entries,
                "display_paths": tuple(placed_graph.graph["display_paths"]),
                "segment_hadamards": dict(segment_hadamards),
                "num_qubits": int(self.circuit.qubits),
                "t_gate_positions": {int(k): tuple(int(c) for c in v) for k, v in self.t_gate_positions.items()},
                "s_gate_positions": {int(k): tuple(int(c) for c in v) for k, v in self.s_gate_positions.items()},
                "s_gate_squares": [tuple(int(c) for c in anchor) for anchor in self.s_gate_squares],
                **self._free_magic_pickle_entries(),
            }, fp)
        placed_graph.graph["routed_segment_pipe_types"] = dict(self.routed_segment_pipe_types)
        self._add_machinery(placed_graph.graph)
        export_block_graph(block_graph_from_placed_graph(placed_graph, name='a.tqec', add_io_ports=True), _SRC.parent/'result', 'a_clean', formats=('json', 'html') if self.write_html else ('json',))

    def _t_spider_entries(self, spider_positions, resolved_circuit_events, payload_spider_positions):
        """Give every T event its own blue-spider record (a T and a CNOT control can share one cube) and return the
        pickle entry for it. Adds virtual records to payload_spider_positions where a T has no spider of its own."""
        _cnot_spider_ids = {
            spider_id
            for event in resolved_circuit_events if event.get('pair')
            for spider_id in event['pair']
        }
        _s_cells = {tuple(position) for position in self.s_gate_positions.values()}
        _unpaired_blue_by_cell = {}
        for _spider_id, _position in spider_positions.items():
            _cell = tuple(_position)
            if (_spider_id.startswith('blue_') and _spider_id not in _cnot_spider_ids
                    and _cell not in _s_cells):
                _unpaired_blue_by_cell.setdefault(_cell, []).append(_spider_id)
        t_spider_by_event = {}
        for _event_index, _position in self.t_gate_positions.items():
            _cell = tuple(_position)
            _available = _unpaired_blue_by_cell.get(_cell, [])
            if _available:
                _spider_id = _available.pop(0)
            else:
                _spider_id = f'blue_t_{_event_index}'
                payload_spider_positions[_spider_id] = _cell
            t_spider_by_event[int(_event_index)] = _spider_id
        return {"t_spiders": t_spider_by_event}

    def _free_magic_pickle_entries(self):
        """Extra pickle entries, added at the end of the payload. The basic variant has none."""
        return {}

    def _add_machinery(self, graph):
        """Hand the S-gate machinery to the block-graph adapter as finished cubes and pipes."""
        graph["machinery_cubes"] = dict(self.s_machinery_cubes)
        graph["machinery_pipes"] = tuple(self.s_machinery_pipes)


_AXIS_LETTERS = ('i', 'j', 'k')


class _VersionedSet(set):
    '''A set that stamps a global version every time it changes.

    routed_cubes is the only thing the free-magic BFS reads out of the placement state, so a counter
    that moves whenever it does is enough to know when the cached free-magic field is stale. _restore() swaps
    the whole set for a snapshot, which can leave the size unchanged, so size is not a usable stamp.'''
    version = 0


def _bumping(name):
    method = getattr(set, name)
    def wrapper(self, *args):
        _VersionedSet.version += 1
        return method(self, *args)
    return wrapper


for _name in ('add', 'update', 'clear', 'discard', 'remove', 'pop', 'difference_update', '__ior__'):
    setattr(_VersionedSet, _name, _bumping(_name))


def _face0_orientation(kind):
    return _AXIS_LETTERS[kind.index('Z')]

def _bake_free_magic_geometry(t_cube, t_kind, chain):
    '''Turn a walked chain of (from, to, axis, pipe type) steps into the cubes, pipes and port to instantiate.

    A cube's kind is read off the pipes touching it: a pipe on one axis pins the cube's other two letters,
    so a route cube -- which always has exactly two pipes, the one it came in on and the one it leaves by
    -- is fully determined where those pipes turn, and free in one letter where they do not. That free
    letter is the axis both pipes run along, where the cube is open at both ends, so 'X' is picked and
    nothing reads it. The port takes no kind at all; tqec infers its pipe from the cube it attaches to.'''
    port_cell = chain[-1][1]
    letters = {}
    pipes = []
    for u, v, axis, pipe_type in chain:
        kind = pipe_kind_from_axis_type('xyz'[axis], pipe_type)
        for cell in (u, v):
            if cell == t_cube or cell == port_cell:
                continue
            for index, letter in enumerate(kind):
                if letter == 'O':
                    continue
                if letters.setdefault(cell, {}).setdefault(index, letter) != letter:
                    return None
        if v != port_cell:
            pipes.append((u, v, kind))
    first_kind = pipe_kind_from_axis_type('xyz'[chain[0][2]], chain[0][3])
    if any(letter != 'O' and t_kind[index] != letter for index, letter in enumerate(first_kind)):
        return None
    return {
        'cubes': {cell: ''.join(faces.get(index, 'X') for index in range(3)) for cell, faces in letters.items()},
        'pipes': pipes,
        'port': (port_cell, chain[-1][0]),
        'route': tuple(v for _u, v, _a, _t in chain[:-1]),
        'orientation': _face0_orientation(t_kind),
    }


class FreeMagicCompiler(Compiler):
    """Free-magic variant: every T-gate needs an instantiable route from its red face out of the patch, qubits start on the
    snake (or a seeded random permutation), and S-gates keep the pre-fix colours."""

    USE_LSMAP = False
    ROUTED_CUBES_TYPE = _VersionedSet
    S_SPIDER_TYPE = 'red'
    S_ROLE = 'sink'
    S_FACE = 1
    S_LEG_AXIS_BY_PARITY = ('j', 'i')
    S_JUNCTION_LETTER = 'X'

    def __init__(self, circuit, module, write_html = False):
        super().__init__(circuit, module, write_html)
        self.t_free_magic_cubes, self.t_free_magic_pipes, self.t_free_magic_ports = {}, [], []
        self.free_magic_port_cells = set()
        self.t_gate_free_magic_paths, self._free_magic_field_cache, self._free_magic_plan_cache = {}, {}, {}

    def _tracked(self):
        return super()._tracked() + (('t_gate_free_magic_paths', 'dict'), ('t_free_magic_cubes', 'dict'), ('t_free_magic_pipes', 'list'),
                                     ('t_free_magic_ports', 'list'), ('free_magic_port_cells', 'set'))

    def _h_column_open(self, x, y, z):
        return not any(self.qubit_column_spiders.get((x, y, z0), None) for z0 in range(z, self._window_hi()))

    def _blocked_by_t_gate(self, role, x, y, z):
        return role == 'T' and self.available_t_gate_space(x, y, z) is None

    def _reserve_t_gate(self, t, x, y, z, cube, count):
        """Claim the cube and instantiate the whole free-magic route: its cells, pipes and the open port at its end."""
        route_plan = self.t_gate_free_magic_plan(x, y, z)
        self.qubit_column_spiders[cube] = 'blue'
        self.spider_counts[cube] = count + 1
        self.spider_endpoint_metadata[cube] = (route_plan['orientation'], 0)
        self.routed_cubes.update(route_plan['route'])
        self.t_free_magic_cubes.update(route_plan['cubes'])
        self.t_free_magic_pipes.extend(route_plan['pipes'])
        self.t_free_magic_ports.append((f'tinj_{len(self.t_free_magic_ports)}', route_plan['port'][0], route_plan['port'][1]))
        self.free_magic_port_cells.add(route_plan['port'][0])
        self.t_gate_free_magic_paths[self.t_event_index_by_gate_id[id(t)]] = route_plan['route']

    def available_t_gate_space(self, x, y, z):
        """The whole route out of the patch, or None when no instantiable route exists from this cube."""
        plan = self.t_gate_free_magic_plan(x, y, z)
        return None if plan is None else list(plan['route'])

    def _program_ceiling(self):
        '''The top of the box the free-magic BFS runs inside: the highest z anything has been committed at.'''
        return max([self._window_hi() - 1, *(cell[2] for cell in self.routed_cubes), *(cell[2] for cell in self.qubit_column_spiders)])

    def _free_magic_field(self):
        '''Parent pointers of a BFS grown inward from the patch's four vertical walls over free space.

        A T-state is cultivated off-patch, so the injection can only attach to a red face that free space
        connects to the outside of the program. "Outside" is the four vertical walls of the box that bounds
        the patch over all times -- x == 0, x == nx - 1, y == 0, y == ny - 1, at every height -- and NOT the
        two horizontal faces at z == 0 and the ceiling, which are the start and the end of the computation
        rather than somewhere a factory can stand. The route itself is free to move in z on its way out.

        Growing the BFS from the walls inward rather than from each T-gate outward answers every candidate
        cell in one pass, and since the moves are symmetric the parent chain out of a cell IS the route that
        cell's own BFS would have found. Blocked: cells already routed over, and the qubit columns, whose
        worldlines occupy their square at every height.

        This is free space only -- it knows nothing about pipe types, so it can say a cell has no way out but
        never that a route is instantiable. It is the cheap filter in front of bake_t_gate_free_magic(), which is
        what actually decides a placement.'''
        ceiling = self._program_ceiling()
        key = (_VersionedSet.version, ceiling)
        cached = self._free_magic_field_cache.get(key)
        if cached is not None:
            return cached
        cell_free = self._cell_free_test(ceiling)
        parents = {}
        queue = deque()
        for x, y, z in product(range(self.nx), range(self.ny), range(ceiling + 1)):
            if (x == 0 or x == self.nx - 1 or y == 0 or y == self.ny - 1) and cell_free((x, y, z)):
                parents[(x, y, z)] = None
                queue.append((x, y, z))
        while queue:
            cell = queue.popleft()
            for nbr in ((cell[0] - 1, cell[1], cell[2]), (cell[0] + 1, cell[1], cell[2]),
                        (cell[0], cell[1] - 1, cell[2]), (cell[0], cell[1] + 1, cell[2]),
                        (cell[0], cell[1], cell[2] - 1), (cell[0], cell[1], cell[2] + 1)):
                if nbr not in parents and cell_free(nbr):
                    parents[nbr] = cell
                    queue.append(nbr)
        self._free_magic_field_cache.clear()
        self._free_magic_field_cache[key] = parents
        return parents

    def t_gate_free_magic_route(self, cell):
        '''The route from a red-face cell to the boundary, or None when free space does not reach it.'''
        parents = self._free_magic_field()
        if cell not in parents:
            return None
        route, step = [cell], parents[cell]
        while step is not None:
            route.append(step)
            step = parents[step]
        return tuple(route)

    def bake_t_gate_free_magic(self, x, y, z):
        '''Walk a T-gate's injection route from its red face out of the patch, and return the geometry it takes.

        The route is not an annotation on the placement, it IS part of it: the cubes and pipes below are
        instantiated in the block graph and the cells are held in routed_cubes, so nothing else can ever be
        routed through them. A T-gate that cannot be given such a route is not placed at this height.

        A T-gate is the same spider as a CNOT control -- face 0, orientation endpoint_orientation(injection
        axis), hence kind XZX at even Hadamard parity and ZXX at odd -- so the route leaves on the lateral
        axis a source endpoint uses and its first pipe type is RULE_S[(0, orientation, that axis)]. From
        there the walk is the same rewrite cnot_path_metadata applies to a max-flow path: the pipe type only
        changes where the route turns, through RULES. Carrying the type in the BFS state (rather than
        checking it after the fact) is what makes every route the walk returns instantiable -- a cell reached
        with the wrong type is simply a different state, still open to being reached with the right one.

        The route ends on one of the four vertical walls and takes one more step through it, to a Port cube
        outside the patch. That port is the externality: it is where the factory that cultivated the T-state
        hands it over, and it is why the top and the bottom of the box are not exits -- those are the start
        and the end of the computation, not somewhere a factory can stand.'''
        injection_axis = 1 if self.hadamard_parity(x, y, z) == 1 else 0
        orientation = self.module.endpoint_orientation(_AXIS_LETTERS[injection_axis])
        t_kind = canonical_node_tqec(orientation, 0)
        start_type = RULE_S.get((0, orientation, _AXIS_LETTERS[injection_axis]))
        if t_kind is None or start_type is None:
            return None
        if self.spider_endpoint_metadata.get((x, y, z), (orientation, 0)) != (orientation, 0):
            return None
        ceiling = self._program_ceiling()
        cell_free = self._cell_free_test(ceiling)
        def step(cell, axis, sign):
            return tuple(c + (sign if a == axis else 0) for a, c in enumerate(cell))
        def turned(pipe_type, axis, next_axis):
            return pipe_type if axis == next_axis else RULES.get((pipe_type, _AXIS_LETTERS[axis], _AXIS_LETTERS[next_axis]))
        def exits(cell, axis, pipe_type):
            '''The ways this cell can leave the box, straight on first so a route does not turn for no reason.'''
            walls = [(a, -1) for a in (0, 1) if cell[a] == 0] + [(a, 1) for a in (0, 1) if cell[a] == (self.nx, self.ny)[a] - 1]
            for exit_axis, sign in sorted(walls, key = lambda wall: wall[0] != axis):
                exit_type = turned(pipe_type, axis, exit_axis)
                port_cell = step(cell, exit_axis, sign)
                if exit_type is not None and port_cell not in self.free_magic_port_cells:
                    yield (cell, port_cell, exit_axis, exit_type)
        queue, seen = deque(), set()
        for sign in (-1, 1):
            first = step((x, y, z), injection_axis, sign)
            if not cell_free(first):
                continue
            state = (first, injection_axis, start_type)
            seen.add(state)
            queue.append((state, (((x, y, z), first, injection_axis, start_type),)))
        while queue:
            (cell, axis, pipe_type), chain = queue.popleft()
            for exit_step in exits(cell, axis, pipe_type):
                baked = _bake_free_magic_geometry((x, y, z), t_kind, chain + (exit_step,))
                if baked is not None:
                    return baked
            for next_axis in range(3):
                next_type = turned(pipe_type, axis, next_axis)
                if next_type is None:
                    continue
                for sign in (-1, 1):
                    nbr = step(cell, next_axis, sign)
                    state = (nbr, next_axis, next_type)
                    if state in seen or not cell_free(nbr) or any(nbr == u for u, _v, _a, _t in chain):
                        continue
                    seen.add(state)
                    queue.append((state, chain + ((cell, nbr, next_axis, next_type),)))
        return None

    def t_gate_free_magic_plan(self, x, y, z):
        '''The baked free-magic geometry for a T-gate at this cube, or None. Cached per routed_cubes version, since
        a column scan asks about many heights and the placement asks again about the one it settles on.'''
        key = (_VersionedSet.version, x, y, z, len(self.free_magic_port_cells))
        if key not in self._free_magic_plan_cache:
            self._free_magic_plan_cache.clear()
            injection_axis = 1 if self.hadamard_parity(x, y, z) == 1 else 0
            red_faces = ((x, y - 1, z), (x, y + 1, z)) if injection_axis else ((x - 1, y, z), (x + 1, y, z))
            if all(self.t_gate_free_magic_route(cell) is None for cell in red_faces):
                self._free_magic_plan_cache[key] = None
            else:
                self._free_magic_plan_cache[key] = self.bake_t_gate_free_magic(x, y, z)
        return self._free_magic_plan_cache[key]

    def _t_spider_entries(self, spider_positions, resolved_circuit_events, payload_spider_positions):
        """The free-magic variant never wrote per-T-event spider records."""
        return {}

    def _free_magic_pickle_entries(self):
        return {"t_gate_free_magic_paths": {int(event): tuple(tuple(int(c) for c in cell) for cell in route)
                                        for event, route in self.t_gate_free_magic_paths.items() if route},
                "t_gate_free_magic_ports": {str(label): (tuple(int(c) for c in port), tuple(int(c) for c in neighbour))
                                        for label, port, neighbour in self.t_free_magic_ports}}

    def _add_machinery(self, graph):
        """S-gate machinery and T-gate free-magic routes go to the adapter together, plus the open ports."""
        if set(self.s_machinery_cubes) & set(self.t_free_magic_cubes):
            raise RuntimeError(f'S-gate machinery and a T-gate free-magic route claim the same cubes: {sorted(set(self.s_machinery_cubes) & set(self.t_free_magic_cubes))}')
        graph["machinery_cubes"] = {**self.s_machinery_cubes, **self.t_free_magic_cubes}
        graph["machinery_pipes"] = tuple(self.s_machinery_pipes) + tuple(self.t_free_magic_pipes)
        graph["extra_ports"] = tuple(self.t_free_magic_ports)


VARIANTS = {'basic': Compiler, 'free_magic': FreeMagicCompiler}


def parse_arguments(argv = None):
    cli = argparse.ArgumentParser(description = "Part A compiler: cultivation-aware behaviour by default, free-magic with --variant free_magic.")
    cli.add_argument('--variant', choices = tuple(VARIANTS), default = 'basic', help = 'which original compiler to reproduce (default: basic)')
    cli.add_argument('--circuit', default = None, help = 'benchmark name (read from benchmark/<name>.qasm) or the path of a .qasm file (default: adder_4q for basic, graphstate_100 for free_magic)')
    cli.add_argument('--html', action = 'store_true', help = 'also write the block-graph HTML (default: off; pickle and JSON are always written)')
    return cli.parse_args(argv)


def main(argv = None):
    start_dir = Path.cwd()
    args = parse_arguments(argv)
    compiler_class = VARIANTS[args.variant]
    circuit_name = args.circuit
    qasm_path = start_dir / circuit_name if circuit_name.endswith('.qasm') else _SRC.parent / 'benchmark' / f'{circuit_name}.qasm'
    os.chdir(_SRC)
    compiler_class(pyzx.Circuit.load(qasm_path), maxflow_accelerator, write_html = args.html).run()


if __name__ == '__main__':
    main()
