import os, sys
from pathlib import Path
_SRC = next(p for p in Path(__file__).resolve().parents if (p / 'maxflow_accelerator.py').exists()) # the directory holding the shared modules (src/), found by search so this file works sitting in src/ or one folder below it
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent / 'augment'))
os.chdir(_SRC) # both originals resolve most paths against the CWD; re-root so this copy behaves like src/
import math
import pickle
import json
from collections import deque, defaultdict
import collections as _c
from random import random
import random as rnd
from itertools import product
import numpy as np
from tqec import BlockGraph
from tqdm import tqdm
import argparse
from enum import Enum
from cultivation_rate_calculation import RESCALED_CULTIVATION_PARAMETER as _RESCALED


class CubeDistance(Enum):
    """The code distances this model runs at, with the values that vary with each: the CUBE_DISTANCE itself,
    the rescaled mean of the memoryless cultivation draw, and the default trial seed."""
    D11 = (11, _RESCALED[11], 10)
    D13 = (13, _RESCALED[13], 10)
    D15 = (15, _RESCALED[15], 11)

    def __init__(self, cube_distance, rescaled_cultivation_parameter, trial_seed):
        self.cube_distance = cube_distance
        self.rescaled_cultivation_parameter = rescaled_cultivation_parameter
        self.trial_seed = trial_seed


_cli = argparse.ArgumentParser(description = "Part B timeseries run: reads ../result/pbp_log.pkl and a_clean.json, writes the routed block graph.")
_cli.add_argument('--distance', type = int, required = True, choices = [m.cube_distance for m in CubeDistance], help = 'CUBE_DISTANCE to run at')
_distance = next(m for m in CubeDistance if m.cube_distance == _cli.parse_args().distance)


TRIAL_SEED = int(os.environ.get('TOPOLS_TRIAL_SEED', str(_distance.trial_seed)))
rnd.seed(TRIAL_SEED)  # per-trial cultivation draws
LARGE_INT = 100000
RESCALED_CULTIVATION_PARAMETER = _distance.rescaled_cultivation_parameter # rescaled mean of the memoryless exponential cultivation draw, in cubes
VARIANT = 'b_branching' # A delaying T-gate drags the 1-qubit gates above it up its column; when the stack jams against a 2-qubit spider the stack is taken OFF the column into a per-column queue, letting the T-gate keep sliding until it reaches the 2-qubit spider itself. Once the T-gate is delivered the queue drains one gate per height step ahead of that spider. Variant 'a' instead freezes the stack in place -- see timeseries-modeling-a.py
MAX_HEIGHT = int(os.environ.get('TOPOLS_MAX_HEIGHT', '2000')) # safety valve: parking changes the loop's termination argument, and this loop has a history of livelocking
with open(Path('../result/pbp_log.pkl'), 'rb') as f:
    payload = pickle.load(f)
# We'll pass nx and ny through the results package somewhere
_xs = int(math.ceil(math.sqrt(payload['num_qubits'])))
_ys = int(math.ceil(payload['num_qubits'] / _xs))
nx = 2 * _xs + 1

ny = 2 * _ys + 1
CUBE_DISTANCE = _distance.cube_distance
cnot_spiders = {s for e in payload['circuit_events'] if e.get('pair') for s in e['pair']}
s_gate_cells = {tuple(c) for c in payload['s_gate_positions'].values()}

routedness = {q: [v, ''] for q, v in payload['t_gate_positions'].items()} # Should be combined with t_gate_positions but that's some programming for later; NOTE: as it currently stands, v will become incorrect as we change the height of T-gates throughout the program

def _build_t_spiders():
    """T event -> the distinct blue-spider key that moves with that T-gate.

    New Part A payloads carry this map explicitly because a T may share its physical cube with a
    CNOT endpoint. For old payloads, consume the available unpaired blue keys cell-by-cell and create
    an equivalent logical key when the coordinate-only geometry collapsed two spider identities."""
    explicit = payload.get('t_spiders')
    if explicit is not None:
        keys = {int(event): str(spider) for event, spider in explicit.items()}
        assert set(keys) == set(payload['t_gate_positions']), 'explicit T-spider map does not cover every T event'
        for event, spider in keys.items():
            assert spider in payload['spider_positions'], f'T event {event} references missing spider {spider!r}'
            assert tuple(payload['spider_positions'][spider]) == tuple(payload['t_gate_positions'][event]), \
                f'T event {event} and spider {spider!r} occupy different cells'
        return keys

    on_cell = defaultdict(list)
    for key, position in payload['spider_positions'].items():
        if (key.split('_')[0] == 'blue' and key not in cnot_spiders
                and tuple(position) not in s_gate_cells):
            on_cell[tuple(position)].append(key)
    keys = {}
    for event, position in payload['t_gate_positions'].items():
        here = on_cell[tuple(position)]
        if here:
            keys[event] = here.pop(0)
        else:
            key = f'blue_t_{event}'
            payload['spider_positions'][key] = tuple(position)
            keys[event] = key
    return keys
t_spider = _build_t_spiders()

cultivation_round = {(i,j): False for i, j in product(range(nx), range(ny))} # True when a finished magic state is sitting on this square waiting to be routed out


def cultivated(square):
    """Is a finished magic state sitting on this square right now?"""
    return cultivation_round[square]


graph = json.loads(Path('../result/a_clean.json').read_text())

height = 0 # We define the height of an ECC as the height at its base

delay = 0
undelivered_fractional_ticks = 0

new_fractional_delay_found = False
delay_planes = []
paid_height = None # the layer whose cycles have been advanced already
prepaid_layers = 0 # layers a fold inserted whose cycles the fractional loop has already run
# --> bloated code
baked_cubes = set()
# <-- bloated code
backup_t_injection_reserved = {} # T event -> the (x, y) square held for its injection route. A square rather than a cell: a gate and its backup T-injection cube rise together, and it is the square whose column has to stay clear for the cultivation under it to finish
machinery_cells = {} # cell -> the T event whose injection machinery occupies it. Every membership test reads it exactly as the old set did; the value is what lets _eliminate find a deleted gate's machinery, which a bare set could never answer
conditional_markers = []
def terminal_cube_kind(pipe_kind):
    axis = pipe_kind.find('O')
    reference_letter = pipe_kind[0] if axis == 2 else pipe_kind[1 - axis]
    return pipe_kind.replace('O', 'X' if reference_letter == 'Z' else 'Z')

def available_reservation_blocks(gate_pos, is_free, kinds):
    kind = kinds.get(tuple(gate_pos))
    allowed_axes = [0, 1]
    if kind is not None and set(kind) <= set('XZ'):
        allowed_axes = [a for a in allowed_axes if kind[a] == 'X']

    blocks = []
    for axis in allowed_axes:
        for sign in (-1, 1):
            cell = tuple(gate_pos[i] + (sign if i == axis else 0) for i in range(3))
            if is_free(cell):
                blocks.append(cell)
    return blocks

def reserved_cells(exclude_events):
    """The cells other pending T-gates hold for their own injection routes.

    A held square is read at its owner's current z, because a gate and its backup T-injection cube rise together. The
    caller passes its own events in `exclude_events`: a gate has to see its own backup T-injection cube as free, or
    available_reservation_blocks would report it boxed in by its own reservation."""
    exclude = set(exclude_events)
    return {square + (tuple(payload['t_gate_positions'][ev])[2],)
            for ev, square in backup_t_injection_reserved.items()
            if ev not in exclude and ev in payload['t_gate_positions']}

def reserve_backup_t_injection_cubes():
    """One square per pending T-gate, held until that gate delivers.

    This replaces the old rule that protected a cell only once it was some gate's LAST free
    neighbour. Protecting at the last moment protects nothing: a gate loses its backup T-injection cubes one at a
    time to other gates' routes, and by the time the rule notices, the column under the survivor has
    already been built on.

    A SQUARE is held, not a cell. The walk only ever grows the circuit upward, so a square nothing
    may build on keeps the column under it clear while `height` climbs, and the cultivation there
    then finishes once that column is deep enough. Re-picking a square every pass would restart that
    accumulation, so a held one is kept for as long as it is still a legal backup T-injection cube.

    Gates are served in event order, so the assignment is re-derived identically on every pass."""
    kinds = {tuple(c['position']): c['kind'] for c in graph['cubes']}
    for ev in [e for e in backup_t_injection_reserved if e not in payload['t_gate_positions']]:
        backup_t_injection_reserved.pop(ev) # delivered or deleted: the square goes back before anyone else asks
    geometry_free = lambda c: (0 <= c[0] < nx and 0 <= c[1] < ny and c not in kinds and c not in baked_cubes and c not in machinery_cells)
    candidates = {ev: [tuple(b) for b in available_reservation_blocks(tuple(position), geometry_free, kinds)] for ev, position in payload['t_gate_positions'].items()}
    # A cell that is some gate's ONLY candidate is that gate's last one. Handing it to anybody else
    # boxes that gate in, which is the deadlock. Lowest event wins so the map is re-derived identically.
    last_cube = {}
    for ev in sorted(candidates):
        if len(candidates[ev]) == 1:
            last_cube.setdefault(candidates[ev][0], ev)
    _res_col = {e: tuple(v)[:2] for e, v in payload['t_gate_positions'].items()}
    _res_z = {e: tuple(v)[2] for e, v in payload['t_gate_positions'].items()}
    _res_total = _c.Counter()  # cell -> how many events hold it
    _res_bycol = _c.Counter()  # (cell, column) -> how many events of that column hold it
    def _res_add(e):
        cell = tuple(backup_t_injection_reserved[e]) + (_res_z[e],)
        _res_total[cell] += 1
        _res_bycol[(cell, _res_col[e])] += 1
    def _res_drop(e):
        cell = tuple(backup_t_injection_reserved[e]) + (_res_z[e],)
        _res_total[cell] -= 1
        _res_bycol[(cell, _res_col[e])] -= 1
    for _e in backup_t_injection_reserved: # every key is in t_gate_positions: the stale ones were popped above
        _res_add(_e)
    for ev, position in sorted((e, tuple(v)) for e, v in payload['t_gate_positions'].items()):
        mine = candidates[ev]
        _mycol = _res_col[ev]
        # Two gates must never hold the same cube: a held cube IS the guarantee, and one shared
        # between two gates guarantees neigher. So `c not in taken` is not negotable, and the only
        # rung below it drops the weaker rule -- do not take a cell that is some other gate's last
        # one. A gate left with nothing here holds nothing, try owes it a
        # cell that does not exist yet, and the walk's answer delay.
        pick = ([c for c in mine if _res_total[c] == _res_bycol[(c, _mycol)] and last_cube.get(c, ev) == ev]
                or [c for c in mine if _res_total[c] == _res_bycol[(c, _mycol)]])
        held = backup_t_injection_reserved.get(ev)
        if held is not None and any(c[:2] == held for c in pick):
            continue # still legal where the gate now stands: keep it, and keep the column under it
        if pick:
            if ev in backup_t_injection_reserved:
                _res_drop(ev)
            backup_t_injection_reserved[ev] = tuple(min(pick)[:2]) # exactly one cube, never a set
            _res_add(ev)
        else:
            if ev in backup_t_injection_reserved:
                _res_drop(ev)
            backup_t_injection_reserved.pop(ev, None) # no free geometry at all: a delay has to open a cell first

def _column_hadamard_walls(xy):
    """{upper-endpoint z -> segment} for every Hadamard wall in this column. The upper
    endpoint is the convention the blocking-height expression already uses (max over the
    segment's two endpoints), and it is the z the wall actually blocks: a gate at z - 1
    cannot step onto z through it."""
    return {max(e[2] for e in seg): seg
            for seg in payload['segment_hadamards']
            if all(tuple(e[:2]) == xy for e in seg)}

def _spiders_at(xy, z):
    return [k for k, v in payload['spider_positions'].items()
            if tuple(v[:2]) == xy and v[2] == z]


def _cube_at(cell): # General helper function that is quite useful
    return next((c for c in graph['cubes'] if tuple(c['position']) == tuple(cell)), None)

def commutative_stack(xy, gate_z):
    """Walk up from the T-gate absorbing every 1-qubit gate that blocks its commutation.

    Returns (stack, ceiling):
      stack   ascending [(z, [non-CNOT spider keys at z], wall_segment_or_None)] -- the
              1-qubit gates that ride up with the T-gate.
      ceiling z of the first 2-qubit spider some member of the stack cannot commute past,
              or LARGE_INT if the column is clear all the way up."""

    walls = _column_hadamard_walls(xy)
    column_cells = {
        tuple(c['position']): c for c in graph['cubes']
        if tuple(c['position'])[:2] == tuple(xy)
    }

    def _pins(key, cell):
        # possible BUG: this says color-neutral, but T-gates commuting through CNOT controls are fine...we do this because the "stack" almost always contains Hadamards...technically, we should commute part of the stack that only contains T-gates upwards, so this function should become more complicated because it might have to split the stack up
        return key in cnot_spiders or cell in s_gate_cells

    def _leg_colour(cell):
        c = column_cells.get(tuple(cell))
        if c is None:
            return None
        return 'blue' if c['kind'].count('Z') == 1 else 'red'
    def _blocks(key, cell):
        if cell in s_gate_cells:
            return True
        if key not in cnot_spiders:
            return False
        return not _leg_colour(cell) == 'blue'
    pushes = []
    frm = gate_z
    while True:
        cz = min((v[2] for k, v in payload['spider_positions'].items() if tuple(v[:2]) == xy and v[2] > frm and _blocks(k, tuple(v))), default = LARGE_INT)
        if cz != LARGE_INT and _leg_colour(xy + (cz,)) == 'red' and walls.get(cz) is not None:
            pushes.append((cz, walls[cz]))
            frm = cz
            continue
        ceiling = cz
        break
    pushed_wall_tops = {cz for cz, _ in pushes}
    
    stack = []
    for z in sorted({v[2] for v in payload['spider_positions'].values() if tuple(v[:2]) == xy} | set(walls)):
        if not gate_z < z <= ceiling:
            continue
        keys = [k for k in _spiders_at(xy, z) if not _pins(k, xy + (z,))]
        wall = None if z in pushed_wall_tops else walls.get(z)
        if keys or wall is not None:
            stack.append((z, keys, wall))
    return stack, ceiling, pushes
    

def _cell_legal(cell, column_cells):
    """The three predicates the solo slide has always used for a landing cell: a real
    (non-PORT) cube in this column, not reserved machinery, and room for the blue
    spider about to land there under the data cube's capacity of 2."""
    return (column_cells.get(cell, {'kind': 'PORT'})['kind'] != 'PORT'
            and cell not in machinery_cells
            and len([k for k, v in payload['spider_positions'].items()
                     if tuple(v) == cell and k.split('_')[0] == 'blue']) < 2)

def stack_reflow(xy, gate_z): # Combines both commutative_stack and xy since commutative_stack contains some malformed data structures and engineering choices
    # --> commutative_stack portion
    walls = {i - 0.5: j for i, j in _column_hadamard_walls(xy).items() if i > gate_z}
    def _leg_color(cell): # Helpful function which takes advantage of the verticality of the qubit-column and the restrictions on wall coloring
        c = _cube_at(cell)
        if c is None: # Not sure why this would activate
            return None
        return 'blue' if c['kind'].count('Z') == 1 else 'red'
    # No _blocks definition because that doesn't take into account free Hadamards
    stack_front = 'blue'
    z_walk = gate_z
    column_cells = {tuple(c['position']): c for c in graph['cubes'] if tuple(c['position'])[:2] == xy}
    column_top = max((cell[2] for cell in column_cells), default = gate_z)
    pushes = []
    commutative_stack = []
    index = 0

    def flip_cube_basis(cells): # not sure if this breaks the code by repeatedly inserting Hadamards when only one parity should be flipped but we'll see in tests
        def _reinfer_pipe(p):
            u, v = tuple(p['u']), tuple(p['v'])
            axis = next(a for a in range(3) if u[a] != v[a])
            lo, hi = (u, v) if u[axis] < v[axis] else (v, u)
            faces = [None if c is None or c['kind'] == 'PORT' else c['kind'] for c in (_cube_at(lo), _cube_at(hi))]
            base = next(k for k in faces if k is not None)
            seam = all(faces) and any(faces[0][a] != faces[1][a] for a in range(3) if a != axis)
            p['kind'] = ''.join('O' if a == axis else base[a] for a in range(3)) + ('H' if seam else '')
        cells = {tuple(c) for c in cells}
        for cell in cells:
            cube = _cube_at(cell)
            if cube is None or cube['kind'] == 'PORT':
                continue
            cube['kind'] = ''.join('Z' if ch == 'X' else ('X' if ch == 'Z' else ch) for ch in cube['kind'])
        for p in graph['pipes']:
            if cells & {tuple(p['u']), tuple(p['v'])}:
                _reinfer_pipe(p)

    def pair_wall(leg_z):
        z = leg_z
        while z < column_top:
            if z + 0.5 in walls:
                return z + 0.5
            if _spiders_at(xy, z + 1):
                return None
            z += 1
        return None

    def plan_push(leg_z):
        nonlocal index
        wall = commutative_stack.pop()
        index -= 1
        walls.pop(wall['height'])
        pair = pair_wall(leg_z)
        if pair is None:
            walls[leg_z + 0.5] = frozenset({xy + (leg_z,), xy + (leg_z + 1,)})
        else:
            walls.pop(pair)
        pushes.append((wall['height'], leg_z, pair))

    ceiling = column_top
    while z_walk < column_top:
        ceiling = z_walk
        if [ev for ev, pos in payload["t_gate_positions"].items() if tuple(pos) == (xy + (z_walk,)) and events_by_index[ev]["kind"] == 't']:
            stack_front = 'blue'
            commutative_stack.append({'color': 'blue', 'height': z_walk, 'index': index}) # appends the index, 0-indexed
            index += 1
        if z_walk + 0.5 in walls: # Look ahead for Hadamards
            stack_front = 'yellow'
            commutative_stack.append({'color': 'yellow', 'height': z_walk + 0.5, 'index': index})
            index += 1
        spider_ids = [spider_id for spider_id, position in payload["spider_positions"].items() if tuple(position) == (xy + (z_walk + 1,))]
        spider_colors = [spider_id.split("_", 1)[0] for spider_id in spider_ids]
        # We can decide whether to, or not, execute a push every time, but either way the stack stops, no?
        if (xy + (z_walk + 1,)) in s_gate_cells:
            break
        if not spider_colors:
            z_walk += 1
            continue
        on_cnot = any(spider_id in cnot_spiders for spider_id in spider_ids)
        if spider_colors[0] == 'red' and stack_front == 'blue':
            break
        elif len(spider_colors) == 1 and spider_colors[0] == 'red' and stack_front == 'yellow' and on_cnot:
            plan_push(z_walk + 1)
            stack_front = 'blue'
        elif len(spider_colors) == 1 and spider_colors[0] == 'blue' and stack_front == 'yellow' and on_cnot:
            plan_push(z_walk + 1)
            break
        z_walk += 1
    else:
        ceiling = column_top
    # <-- commutative_stack portion
    if not commutative_stack:
        return None
    z_walk = gate_z
    def has_backup_t_injection_cube(cell, protected, kind = None):
        cell = tuple(cell)
        kinds = {tuple(c["position"]): c["kind"] for c in graph["cubes"]}
        kind = kinds.get(cell) if kind is None else kind
        if kind is None or kind == "PORT":
            return None
        used = set(kinds) | set(baked_cubes) | set(machinery_cells)
        # Only claims NOT listed in `protected` block the backup T-injection cube.
        outside_claims = protected
        for axis in (0, 1):
            if kind[axis] != 'X':
                continue
            for sign in (-1, 1):
                backup_cube = tuple(value + (sign if i == axis else 0) for i, value in enumerate(cell))
                if(0 <= backup_cube[0] < nx and 0 <= backup_cube[1] < ny and backup_cube not in used and backup_cube not in outside_claims):
                    return backup_cube
        return None

    assert commutative_stack == sorted(commutative_stack, key=lambda member: member['index'])
    events_at_index = {}

    for member in commutative_stack:
        if member['color'] != 'blue':
            continue
        z = int(member['height'])
        events = [ev for ev, pos in payload['t_gate_positions'].items() if tuple(pos) == xy + (z,)]
        if not events:
            return None
        events_at_index[member['index']] = events
    protected = reserved_cells([ev for evs in events_at_index.values() for ev in evs])

    pushed_legs = {leg_z for _wall_z, leg_z, _pair in pushes}
    leg_zs = {pos[2] for spider, pos in payload['spider_positions'].items() if tuple(pos[:2]) == xy and spider in cnot_spiders}
    red_leg_zs = {z for z in leg_zs if _leg_color(xy + (z,)) != 'blue' and z not in pushed_legs}
    def blue_legal(member, candidate):
        cell = xy + (candidate,)
        if candidate in red_leg_zs or any(member['height'] < leg <= candidate for leg in red_leg_zs):
            return False
        if candidate not in pushed_legs and any(tuple(pos) == cell and spider.split('_', 1)[0] == 'red' for spider, pos in payload['spider_positions'].items()):
            return False
        if not _cell_legal(cell, column_cells):
            return False
        return has_backup_t_injection_cube(cell, protected) is not None
    placements = {}
    floor_z = gate_z + 1
    for member in commutative_stack:
        if member['color'] == 'yellow':
            destination = floor_z - 0.5
            # This check is only necessary if we assume co-placement of T- and S-gates. This shouldn't happen in a well-formed quantum circuit.
            if destination > member['height'] and destination - 0.5 >= ceiling:
                return None
            placements[member['index']] = destination
            floor_z += 1
            continue
        candidate = max(floor_z, member['height'])
        while candidate <= ceiling and not blue_legal(member, candidate):
            candidate += 1
        if candidate > ceiling:
            return None
        placements[member['index']] = candidate
        floor_z = candidate + 1
    delaying = next((m for m in commutative_stack if m['color'] == 'blue' and m['height'] == gate_z), None)
    if delaying is None or placements[delaying['index']] <= gate_z:
        return None # This is <= because if the delaying gate can't be rehoused then the whole thing breaks. NOTE: I'm not sure if this is correct if we have multiple T-gates at the same location -- would they all be forced to undergo this even if only one is delaying (because of resource contention issues)?

    # The wall moves below flip the basis of every cube a wall passes over, so a destination cube's
    # X-faces after the reflow can point at different lanes than the ones has_backup_t_injection_cube just checked.
    # Count how many times each cell is flipped and re-check every blue destination on its post-flip kind.
    flips = _c.Counter()
    for wall_z, leg_z, pair in pushes:
        for z in range(int(wall_z + 0.5), (leg_z if pair is None else int(pair - 0.5)) + 1):
            flips[xy + (z,)] += 1
    for member in commutative_stack:
        if member['color'] == 'yellow':
            shift = int(placements[member['index']] - member['height'])
            if shift:
                upper = int(member['height'] + .5)
                for z in range(min(upper, upper + shift), max(upper, upper + shift)):
                    flips[xy + (z,)] += 1
    for member in commutative_stack:
        if member['color'] != 'blue':
            continue
        cell = xy + (int(placements[member['index']]),)
        if flips[cell] % 2 == 0:
            continue
        cube = _cube_at(cell)
        if cube is None or cube['kind'] == 'PORT':
            continue
        flipped = ''.join('Z' if ch == 'X' else ('X' if ch == 'Z' else ch) for ch in cube['kind'])
        if has_backup_t_injection_cube(cell, protected, kind = flipped) is None:
            return None

    for wall_z, leg_z, pair in pushes:
        payload['segment_hadamards'].pop(frozenset({xy + (int(wall_z - 0.5),), xy + (int(wall_z + 0.5),)}), None)
        if pair is None:
            payload['segment_hadamards'][frozenset({xy + (leg_z,), xy + (leg_z + 1,)})] = 1
        else:
            payload['segment_hadamards'].pop(frozenset({xy + (int(pair - 0.5),), xy + (int(pair + 0.5),)}), None)
        flip_cube_basis(xy + (z,) for z in range(int(wall_z + 0.5), (leg_z if pair is None else int(pair - 0.5)) + 1))

    for member in commutative_stack:
        if member['color'] == 'yellow':
            payload['segment_hadamards'].pop(frozenset(walls[member['height']]), None)
    for member in commutative_stack:
        destination = placements[member['index']]
        if member['color'] == 'blue':
            for ev in events_at_index[member['index']]:
                _seat(ev, xy + (int(destination),))
            continue
        shift = int(destination - member['height'])

        source = frozenset(walls[member['height']])
        destination_wall = frozenset((x, y, z + shift) for x, y, z in source)

        # Two Hadamards at the same segment cancel.
        if destination_wall in payload['segment_hadamards']:
            payload['segment_hadamards'].pop(destination_wall)
        else:
            payload['segment_hadamards'][destination_wall] = 1

        if shift:
            source_upper_z = int(member['height'] + .5)
            flip_cube_basis(xy + (z,) for z in range(min(source_upper_z, source_upper_z + shift), max(source_upper_z, source_upper_z + shift)))

    return placements

def flip_cube_basis(cells):
    def _reinfer_pipe(p):
        u, v = tuple(p['u']), tuple(p['v'])
        axis = next(a for a in range(3) if u[a] != v[a])
        lo, hi = (u, v) if u[axis] < v[axis] else (v, u)
        faces = [None if c is None or c['kind'] == 'PORT' else c['kind'] for c in (_cube_at(lo), _cube_at(hi))]
        base = next(k for k in faces if k is not None)
        seam = all(faces) and any(faces[0][a] != faces[1][a] for a in range(3) if a != axis)
        p['kind'] = ''.join('O' if a == axis else base[a] for a in range(3)) + ('H' if seam else '')
    cells = {tuple(c) for c in cells}
    for cell in cells:
        cube = _cube_at(cell)
        if cube is None or cube['kind'] == 'PORT':
            continue
        cube['kind'] = ''.join('Z' if ch == 'X' else ('X' if ch == 'Z' else ch) for ch in cube['kind'])
    for p in graph['pipes']:
        if cells & {tuple(p['u']), tuple(p['v'])}:
            _reinfer_pipe(p)

def attach_y_half_cube(route, root, owner_event, kinds, pipe_dirs):
    occupied = set(kinds)
    anchors = [route[-1]] + route[1:-1] + [route[0]]
    spider_cells = {tuple(v) for v in payload['spider_positions'].values()}
    gate_x, gate_y, gate_z = route[0]
    column_z = gate_z + 1
    def is_zx_kind(kind):
        return len(kind) == 3 and not set(kind) - set('XZ')
    while column_z < LARGE_INT:
        column_cell = (gate_x, gate_y, column_z)
        if column_cell in spider_cells or column_cell in machinery_cells or not is_zx_kind(kinds.get(column_cell, '')):
            break
        anchors.append(column_cell)
        column_z += 1
    queue = deque()
    seen = set()
    for anchor in anchors:
        anchor_kind = kinds.get(anchor, '')
        if not is_zx_kind(anchor_kind): # This should never fire, no?
            print('FAILED')
            continue
        queue.append((anchor, anchor_kind, frozenset(pipe_dirs.get(anchor,())), []))
    chosen = None

    protected = reserved_cells([owner_event]) # same rule as the delivery route: only the gate whose machinery this is may see its own backup T-injection cube as free. Keyed by cell it exempted every co-located gate too, so the machinery chain was free to bake over their reserved cubes

    def cell_free(cell):
        return 0 <= cell[0] < nx and 0 <= cell[1] < ny and cell[2] >= route[0][2] and cell not in occupied and cell not in protected
    while queue and chosen is None:
        cell, kind, dirs_here, chain = queue.popleft()
        if kind[0] != kind[1] and len({a for a, _ in dirs_here} | {2}) <= 2:
            for dz in (-1, 1): # prefer the pipe running down to the Y half-cube
                y_cell = (cell[0], cell[1], cell[2] + dz)
                if (2, dz) not in dirs_here and cell_free(y_cell):
                    chosen = {'y_anchor': cell, 'y_cell': y_cell, 'chain': chain}
                    break
        if chosen is not None:
            continue
        axes_here = {a for a, _ in dirs_here}
        for axis in range(3):
            if len(axes_here | {axis}) > 2:
                continue
            transverse = [a for a in range(3) if a != axis]
            if kind[transverse[0]] == kind[transverse[1]]:
                continue
            step_pipe_kind = ''.join('O' if a == axis else kind[a] for a in range(3))
            for sign in (-1, 1):
                if (axis, sign) in dirs_here:
                    continue
                new_cell = tuple(cell[a] + (sign if a == axis else 0) for a in range(3))
                if not cell_free(new_cell):
                    continue
                for free_letter in ('X', 'Z'):
                    new_kind = ''.join(free_letter if a == axis else kind[a] for a in range(3))
                    state = (new_cell, new_kind, (axis, -sign))
                    if state in seen:
                        continue
                    seen.add(state)
                    queue.append((new_cell, new_kind, frozenset({(axis, -sign)}), chain + [(cell, new_cell, step_pipe_kind, new_kind)]))
    if chosen is None:
        return False
    for chain_from, chain_to, chain_pipe_kind, chain_cube_kind in chosen['chain']:
        graph['pipes'].append({'u': list(chain_from), 'v': list(chain_to), 'kind': chain_pipe_kind, 'transform': root['transform']})
        connector = {'position': list(chain_to), 'kind': chain_cube_kind, 'label': '', 'transform': root['transform']}
        graph['cubes'].append(connector)
        kinds[chain_to] = chain_cube_kind
        baked_cubes.add(chain_to)
        machinery_cells[chain_to] = owner_event
        if chain_to[2] == height:
            current_height_cubes[chain_to] = connector
        elif chain_to[2] == height + 1:
            next_height_cubes[chain_to] = connector
    y_base = chosen['y_anchor']
    y_base_kind = kinds[y_base]
    y_pipe_ends = sorted([y_base, chosen['y_cell']], key=lambda c: c[2])
    graph['pipes'].append({'u': list(y_pipe_ends[0]), 'v': list(y_pipe_ends[1]), 'kind': y_base_kind[0] + y_base_kind[1] + 'O', 'transform': root['transform']})
    y_cube = {'position': list(chosen['y_cell']), 'kind': 'Y', 'label': '', 'transform': root['transform']}
    graph['cubes'].append(y_cube)
    baked_cubes.add(chosen['y_cell'])
    machinery_cells[chosen['y_cell']] = owner_event
    if chosen['y_cell'][2] == height:
        current_height_cubes[chosen['y_cell']] = y_cube
    elif chosen['y_cell'][2] == height + 1:
        next_height_cubes[chosen['y_cell']] = y_cube
    return True


events_by_index = {e['event_index']: e for e in payload['circuit_events']}


def _seat(ev, cell):
    """The single writer of a T-gate's position: the seat at commit, the delay's move up its column,
    the delivery retry and the worldline slide all move the gate through here, onto a worldline cell
    where the cube already exists.

    It moves the gate's blue spider with it -- looked up through t_spider, since a positional lookup cannot tell
    this gate's spider from another T's on the same cell -- resets routedness, and keeps this pass's
    pos_to_events snapshot in step. It is not the only writer of t_gate_positions: the ordinary column
    path keeps writing its own moves directly, and its destinations are worldline cubes that must not
    be asked to satisfy the rules above.

    Re-seating a gate onto the cell it is already on returns immediately. Every pass re-seats most
    gates onto their own cell, and a reset of routedness there would clear the deliberate 'T' the
    rollback of an unattachable delivery leaves for the fold to read."""
    cell = tuple(cell)
    old = tuple(payload['t_gate_positions'][ev])
    if old == cell:
        return
    payload['t_gate_positions'][ev] = cell
    spider = t_spider.get(ev)
    if spider is not None:
        payload['spider_positions'][spider] = cell
    routedness[ev][1] = ''
    if ev in pos_to_events.get(old, ()):
        pos_to_events[old].remove(ev)
    pos_to_events.setdefault(cell, []).append(ev)


def _delay_gate(position):
    """Move this T-gate to the lowest cell above the one it stands on that still has a free backup T-injection cube.
    -> did it move?

    The candidates are the cells of the gate's own column, bounded by the first 2-qubit spider its
    commuting stack cannot pass. The destination has to have a backup T-injection cube, and that is the whole of the
    guarantee: the walk grows the circuit only upward, so the column under a held square keeps
    clearing and the cultivation there finishes once it is deep enough.

    False means nothing above the gate was legal. The caller delays the circuit, and calls this again
    once the fold has opened the layer -- which is where the cell a worldline gate needs comes from."""
    return stack_reflow(tuple(position[:2]), position[2]) is not None


previous_length = len(payload['t_gate_positions'])
pos_to_events = {}
current_height_cubes, next_height_cubes = {}, {}
pbar = tqdm(total = previous_length)

def step_one_height():
    """One iteration of the height walk, extracted verbatim from the old top-level while body so the
    branch sim can invoke it on a forked state (PLAN §5.1). Returns 'advance' when height moved up
    (the old fall-through past `height += 1`) or 'repeat' when the same height is re-decided (the old
    outer `continue`s at the fractional-delay and post-fold checks -- the ONLY two while-level
    continues; every other continue/break belongs to an inner loop). MAX_HEIGHT/abort and the
    progress bar stay in the driver below."""
    global height, delay, undelivered_fractional_ticks, new_fractional_delay_found, paid_height, prepaid_layers, baked_cubes, machinery_cells, current_height_cubes, next_height_cubes, pos_to_events

    current_height_cubes = {tuple(cube['position']): cube for cube in graph['cubes'] if cube['position'][2] == height} # We used to use takewhile because it would stop as soon as it started reading gates that were too late to save time, but takewhile produces an iterator with other properties that weren't beneficial
    next_height_cubes = {tuple(cube['position']): cube for cube in graph['cubes'] if cube['position'][2] == height + 1}
    _nlc = {}
    for _cube_ in graph['cubes']:
        _p = _cube_['position']
        if _p[2] < height:
            _sq = (_p[0], _p[1])
            if _sq not in _nlc or _nlc[_sq] < _p[2]:
                _nlc[_sq] = _p[2]
    # Pipes do not change until the delivery loop below. Keep the incident axes
    # indexed for this pass and extend it after each committed delivery.
    _pipe_dirs_by_endpoint = defaultdict(set)
    for _pipe in graph['pipes']:
        _u, _v = tuple(_pipe['u']), tuple(_pipe['v'])
        _axis = next(a for a in range(3) if _u[a] != _v[a])
        _sign = 1 if _v[_axis] > _u[_axis] else -1
        _pipe_dirs_by_endpoint[_u].add((_axis, _sign))
        _pipe_dirs_by_endpoint[_v].add((_axis, -_sign))
    pos_to_events = defaultdict(list)
    for ev, pos in payload['t_gate_positions'].items():
        pos_to_events[tuple(pos)].append(ev)
    pos_to_events = dict(pos_to_events)
    if paid_height != height: # a second pass over the same layer is the loop re-deciding, not the machine running: a T-gate that slid up its own column consumed no time
        if prepaid_layers:
            prepaid_layers -= 1
        else:
            # MEMORYLESS-PORT-NOTES: restored from git show cf1910a. Draws the real cultivation
            # sequence on the module RNG. One layer of draws per layer whose
            # CUBE_DISTANCE cycles have not been run yet: this used to sit outside the paid_height
            # check, guarded only by new_fractional_delay_found, so it drew again on every pass that
            # re-decided a layer after a whole-cube delay and again on every delay layer the fold had
            # inserted -- 20 of 42 layers of draws on wstate_16 seed 10 were for cycles the fractional
            # loop below had already spent.
            for square_i in product(range(nx), range(ny)):
                if (square_i + (height,)) in current_height_cubes.keys():
                    cultivation_round[square_i] = False
                elif not cultivated(square_i) and random() < 1 - math.exp(-1 / (RESCALED_CULTIVATION_PARAMETER )): # There is no backtracking because the only things that exist are the T-states that are currently there. If they were there in the past, their either had a route attached to them that then propagates up to the current time or they have been overwritten
                    cultivation_round[square_i] = True
        paid_height = height
    for gate_layer_cubes, gate_z in ((current_height_cubes, height), (next_height_cubes, height + 1)):
        for _seeded_pos in [p for p in pos_to_events if p[2] == gate_z]:
            gate_pos = _seeded_pos
            excluded = defaultdict(set)
            for e in pos_to_events[gate_pos]: # enables re-mapping of states that found cultivations but didn't get anything else -- useful for having a more compact total T-gate by giving extra attachment points to the machinery. Doesn't really break anything since the gate isn't moved and will likely rediscover the same thing. However, there isn't co-optimization between the cultivation search and the machinery search, so the only case in which this is realistically useful is when the cultivation path happens to open up more routes. This isn't guaranteed to happen, and frankly doesn't appear to create any statistical difference in usefulness
                if 'C' not in routedness[e][1]:
                    routedness[e][1] = ''
            while any('C' not in routedness[ev][1] for ev in pos_to_events.get(gate_pos, ())): # TODO: delete. This is erroneous code, we're only adding it in because for some reason VQE ran very quickly with it
                root = gate_layer_cubes.get(gate_pos)
                ev = next(e for e in pos_to_events[gate_pos] if 'C' not in routedness[e][1])

                _route_view_free = lambda c: 0 <= c[0] < nx and 0 <= c[1] < ny and c not in current_height_cubes and c not in next_height_cubes and c not in baked_cubes
                protected_cells = reserved_cells([ev]) # keyed to the gate being DELIVERED, not to its cube. The exemption exists so a gate sees its own backup T-injection cube as free; keyed by cell it also exempted every other ©ate standing on that cube, and a cube can hold more than one T-gate. That is how a delivery ate a co-located gate's reserved backup T-injection cube and left it with nothing to inject through

                _gate_horiz_axes = {a for a, _ in _pipe_dirs_by_endpoint.get(gate_pos, ())} - {2}
                visited = set([gate_pos])
                parent = {gate_pos: (-1, -1, -1)}
                queue = deque([gate_pos])
                target = None
                while queue and target is None:
                    current_node = queue.popleft()
                    neighbors = [(current_node[0] + 1, current_node[1], current_node[2]), (current_node[0], current_node[1] + 1, current_node[2]), (current_node[0] - 1, current_node[1], current_node[2]), (current_node[0], current_node[1] - 1, current_node[2]), (current_node[0], current_node[1], current_node[2] + 1), (current_node[0], current_node[1], current_node[2] - 1)]
                    neighbors = [i for i in neighbors if height <= i[2] <= height + 1 and 0 <= i[1] < ny and 0 <= i[0] < nx]
                    for neighbor in neighbors:
                        if neighbor in visited:
                            continue
                        if current_node == gate_pos:
                            _step_axis = next(a for a in range(3) if neighbor[a] != current_node[a])
                            if root['kind'][_step_axis] != 'X':
                                continue # makes sure that the path root connects from the X-face: kind[a] is the basis of the walls normal to a, and a blue/Z spider (which is what every T-gate cube is) can only take a pipe on its two X-lettered axes -- the Z-lettered one is its normal axis, where no pipe is legal
                            if _gate_horiz_axes and _step_axis != 2 and _step_axis not in _gate_horiz_axes:
                                continue # makes sure that there aren't existing routes heading out from the X-face of this cube
                        if neighbor in current_height_cubes.keys() or neighbor in next_height_cubes.keys() or neighbor in baked_cubes or neighbor in protected_cells:
                            continue
                        if neighbor[2] == height and cultivated(neighbor[:2]) and neighbor not in excluded[gate_pos]:
                            parent[neighbor] = current_node
                            target = neighbor
                            break
                        visited.add(neighbor)
                        parent[neighbor] = current_node
                        queue.append(neighbor)
                if target is None:
                    break
                route = deque()
                node = target # type: ignore
                while node != (-1, -1, -1):
                    route.appendleft(node)
                    node = parent[node]
                route = list(route)

                # Joint check: a single route can wrap a pending gate and take two of its free
                # neighbors at once (each passing the sole-survivor filter individually); reject
                # any route that would zero out another pending gate's free lateral neighbors.
                _route_cells = set(route)
                _zero_kinds = {tuple(c['position']): c['kind'] for c in graph['cubes']}
                # Only gates with two or more free neighbors: the one square each gate holds is in
                # protected_cells, which the BFS cannot cross, so a gate can never be routed down to
                # nothing. This catches the rest -- a route taking several of a gate's UNRESERVED
                # neighbours at once, each of which passes the per-cell filter on its own.
                _zeroed = next((pp for pp in {tuple(v) for v in payload['t_gate_positions'].values()} - {gate_pos}
                                if (lambda blocks: len(blocks) > 1 and all(b in _route_cells for b in blocks))(available_reservation_blocks(pp, _route_view_free, _zero_kinds))), None)
                if _zeroed is not None:
                    excluded[gate_pos].add(target)
                    continue

                cubes_checkpoint = len(graph['cubes'])
                pipes_checkpoint = len(graph['pipes'])
                machinery_checkpoint = dict(machinery_cells)

                markers_checkpoint = len(conditional_markers)

                # Recolor the gate cube's +/-z faces (wall_z) so its exit pipe has differing basis walls.
                # Exit along x: pipe is (O, wall_y, wall_z) -> need wall_z != wall_y.
                # Exit along y: pipe is (wall_x, O, wall_z) -> need wall_z != wall_x.
                # Exit along z: pipe is (wall_x, wall_y, O) -> already legal, leave untouched.
                gate_exit_axis = int(np.argmax(np.abs([route[1][0] - route[0][0], route[1][1] - route[0][1], route[1][2] - route[0][2]])))
                _gate_kind = list(root['kind'])
                if gate_exit_axis == 0:
                    _gate_kind[2] = 'Z' if _gate_kind[1] == 'X' else 'X'
                elif gate_exit_axis == 1:
                    _gate_kind[2] = 'Z' if _gate_kind[0] == 'X' else 'X'
                root['kind'] = ''.join(_gate_kind) # in place: root is the gate cube's dict inside graph['cubes']

                previous_cube_kind = root['kind']
                for i in range(1, len(route) - 1):
                    last_pipe_axis = np.argmax(np.abs([route[i - 1][0] - route[i][0], route[i - 1][1] - route[i][1], route[i - 1][2] - route[i][2]]))
                    next_pipe_axis = np.argmax(np.abs([route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1], route[i + 1][2] - route[i][2]]))
                    raw_pipe_kind = ''.join([letter if i != last_pipe_axis else 'O' for i, letter in enumerate(previous_cube_kind)])
                    graph['pipes'].append({
                        'u': list(route[i - 1]),
                        'v': list(route[i]),
                        'kind': raw_pipe_kind,
                        'transform': root['transform']
                    })
                    raw_cube_kind = ['O']*3
                    if last_pipe_axis == next_pipe_axis:
                        raw_cube_kind = list(terminal_cube_kind(raw_pipe_kind))
                    else:
                        orthogonal_axis = next(a for a in range(3) if a != last_pipe_axis and a != next_pipe_axis)
                        raw_cube_kind[next_pipe_axis] = raw_pipe_kind[next_pipe_axis]
                        raw_cube_kind[orthogonal_axis] = raw_pipe_kind[orthogonal_axis]
                        raw_cube_kind[last_pipe_axis] = 'X' if raw_cube_kind[orthogonal_axis] == 'Z' else 'Z'
                    raw_cube_kind = ''.join(raw_cube_kind)
                    graph['cubes'].append({
                        'position': list(route[i]),
                        'kind': raw_cube_kind,
                        'label': '',
                        'transform': root['transform']
                    })
                    previous_cube_kind = raw_cube_kind
                    baked_cubes.add(route[i])
                final_pipe_axis = np.argmax(np.abs([route[-2][0] - route[-1][0], route[-2][1] - route[-1][1], route[-2][2] - route[-1][2]]))
                final_pipe_kind = ''.join([letter if axis != final_pipe_axis else 'O' for axis, letter in enumerate(previous_cube_kind)])
                graph['pipes'].append({
                    'u': list(route[-2]),
                    'v': list(route[-1]),
                    'kind': final_pipe_kind,
                    'transform': root['transform']
                })
                # Materialize the cultivation (T-state) terminal cube at route[-1] (the reversed end).
                # The cultivation cube's two transverse walls are inherited from its (legal) incoming pipe, so
                # they differ. The face along the pipe axis (where the 'O' was) is free: set it opposite the
                # other in-plane wall so the VERTICAL pipe out of this cube -- the Y-measurement drop, kind
                # (wall_x, wall_y, O) -- has differing basis walls.
                _cultivation_kind = list(final_pipe_kind)
                # The free face (the one the 'O' sat on) is set opposite the other in-plane wall, so the
                # vertical Y-measurement drop out of this cube keeps differing basis walls -- the rule
                # spelled out in the comment above. Hardcoding 'Z' for a lateral approach only held while
                # routes left the T cube through its Z-lettered (normal) face; once the injection uses the
                # legal X face it vetoes ~80% of otherwise-good attachments and the delivery loop then
                # delays the circuit forever. Axis 2 keeps today's letter: both laterals are inherited and
                # already differ, so either is legal there. TODO: confirm 'Z' is what we want to model.
                _cultivation_kind[final_pipe_axis] = (
                    'Z' if final_pipe_axis == 2
                    else ('Z' if _cultivation_kind[1 - final_pipe_axis] == 'X' else 'X')
                )
                cultivation_cube = {'position': list(route[-1]), 'kind': ''.join(_cultivation_kind), 'label': '', 'transform': root['transform']}
                graph['cubes'].append(cultivation_cube)
                current_height_cubes[route[-1]] = cultivation_cube  # keep the in-pass snapshot consistent so later BFS won't tunnel through the new cube
                baked_cubes.add(route[-1])
                # If we have this bake and can't attach a y half cube (and then a conditional measurement) in their legal locations, we'll write a marker for the two cases on this. We don't necessarily roll back the work...we might not be able to attach the measurements because of circuitry in the above circuit. We could make a policy that says that we prioritize the early execution of a T-gate so that we always take the route and then fit the machinery later, and then as an alternative we could be "greedy" and try to fit the whole thing as long as we aren't delaying anything else in the circuit, but realistically that probably doesn't save us too much since we're only looking down a total of 1 cube in this part. As an alternative: when we have lots of T-gates and *some* CNOTs, it becomes much more relevant to have an ahead-of-time compiler so that we can fit the machinery around these inputs in a mathematically-useful way (taking up cubes that have lots of empty cubes above them, so the lost utility is low).
                routedness[ev][1] += 'T'
                _zero_kinds[gate_pos] = root['kind']
                for _cube in graph['cubes'][cubes_checkpoint:]:
                    _zero_kinds[tuple(_cube['position'])] = _cube['kind']
                attached = attach_y_half_cube(
                    route, root, ev, _zero_kinds, _pipe_dirs_by_endpoint
                )
                # (The veto that used to sit here was the mirror of the hardcoded 'Z' cultivation wall above:
                # it discarded every attachment whose approach pipe carried a Z wall on the other lateral axis.
                # With the free wall chosen geometrically that case is legal, so there is nothing to reject.)

                if not attached:
                    for baked_entry in graph['cubes'][cubes_checkpoint:]:
                        baked_position = tuple(baked_entry['position'])
                        baked_cubes.discard(baked_position)
                        current_height_cubes.pop(baked_position, None)
                        next_height_cubes.pop(baked_position, None)
                    del graph['cubes'][cubes_checkpoint:]
                    del graph['pipes'][pipes_checkpoint:]
                    machinery_cells.clear()
                    machinery_cells.update(machinery_checkpoint)
                    del conditional_markers[markers_checkpoint:]
                    routedness[ev][1] = 'T' # logging so that we know whether to move height up or not, but gets rolled back in the earlier check once we've finished deciding what we're going to do with the height variable
                    excluded[gate_pos].add(target)
                    continue
                for _pipe in graph['pipes'][pipes_checkpoint:]:
                    _u, _v = tuple(_pipe['u']), tuple(_pipe['v'])
                    _axis = next(a for a in range(3) if _u[a] != _v[a])
                    _sign = 1 if _v[_axis] > _u[_axis] else -1
                    _pipe_dirs_by_endpoint[_u].add((_axis, _sign))
                    _pipe_dirs_by_endpoint[_v].add((_axis, -_sign))
                # These notes are meant to show whether we promote the T-gate during rest-of-circuit delays or we leave it where it is
                routedness[ev][1] += 'C'
                payload['t_gate_positions'].pop(ev, None)
                pos_to_events[gate_pos].remove(ev)
                cultivation_round[target[:2]] = False # consumed: the square starts a fresh cultivation from scratch
                if undelivered_fractional_ticks:
                    delay += undelivered_fractional_ticks / CUBE_DISTANCE
                    delay_planes.append((height, 'fractional'))
                    undelivered_fractional_ticks = 0
    delay_whole_circuit = False

    # reserve_backup_t_injection_cubes runs again here because this pass has baked geometry since the assignment at the
    # top of it -- deliveries, machinery, the drain above -- and the assignment that matters is the one
    # that holds now.
    reserve_backup_t_injection_cubes()
    # Only a gate the walk is about to act on can be helped by delaying: a gate parked far above the
    # frontier has no geometry around it yet, so "holds nothing" says nothing about it and blocking on
    # it delays forever.
    _unheld = sorted(e for e, v in payload['t_gate_positions'].items() if e not in backup_t_injection_reserved and tuple(v)[2] <= height)
    if _unheld:
        delay_whole_circuit = True
    if delay_whole_circuit or any(position[2] == height and event_idx for position, event_idx in pos_to_events.items()):
        new_fractional_delay_found = False # reflect only THIS pass, so a saturated/unroutable gate falls through to height advance instead of pinning the loop
        for unrouted_active_t_gate in [(i, j) for i, j in pos_to_events.items() if i[2] == height and j]:
            # One delay for every gate, whatever kind of cell it stands on. A gate whose T-state is
            # routed but whose machinery would not fit takes it too: the rollback above has
            # already undone that geometry and left 'T' behind only as a marker, so there is
            # nothing baked here to strand, and it needs a new cell for exactly the same reason.
            if any('T' in routedness[e][1] for e in unrouted_active_t_gate[1]):
                # A routed T-state whose machinery would not fit. Hold the gate where it stands: the
                # layers this fold inserts then accumulate ABOVE it, which is the room the Y half-cube
                # needs, and the 'T' marker survives to say so. Moving it instead goes through _seat,
                # which resets routedness -- so the next pass re-routes from scratch, fails the attach
                # again, and re-excludes the same target, forever.
                delay_whole_circuit = True
                continue
            if _delay_gate(unrouted_active_t_gate[0]):
                continue
            delay_whole_circuit = True # nothing above this gate is legal yet: the circuit delays, and the retry after the fold takes the layer it opens
        if delay_whole_circuit:
            while undelivered_fractional_ticks < CUBE_DISTANCE and not new_fractional_delay_found:
                undelivered_fractional_ticks += 1
                for square_i in product(range(nx), range(ny)):
                    if bool(random() < 1 - math.exp(-1/CUBE_DISTANCE / (RESCALED_CULTIVATION_PARAMETER ))) and not cultivated(square_i) and (square_i + (height,)) not in current_height_cubes.keys() and (square_i + (height,)) not in baked_cubes:
                        cultivation_round[square_i] = True
                        new_fractional_delay_found = True
        if not delay_whole_circuit or undelivered_fractional_ticks < CUBE_DISTANCE:
            return 'repeat' # old while-level continue: re-decide the same height without advancing
        delay += 1
        prepaid_layers += 1 # the fold below inserts one layer, and the CUBE_DISTANCE cycles the fractional loop just spent ARE that layer's cycles -- whichever pass walks onto it must not charge them twice
        delay_planes.append((height, 'whole'))
        undelivered_fractional_ticks = 0
        # The following 19 LOC are to pin cubes and pipes that terminate so that they are not delayed and clog up the graph, i.e. pipes and cubes that form a perfectly vertical path and terminate in Y-basis or conditional X-/Z-basis measurements
        _kinds = {tuple(c['position']): c['kind'] for c in graph['cubes']}
        _touch = defaultdict(set)
        for _p in graph['pipes']:
            _touch[tuple(_p['u'])].add(tuple(_p['v']))
            _touch[tuple(_p['v'])].add(tuple(_p['u']))
        pinned_y = set()
        _marker_anchors = {_m['anchor'] for _m in conditional_markers}
        for _p in graph['pipes']:
            _u, _v = tuple(_p['u']), tuple(_p['v'])
            _bot, _top = sorted((_u, _v), key=lambda c: c[2])
            if _u[:2] != _v[:2] or not (_bot[2] <= height < _top[2]):
                continue
            column = []
            cell = _top
            while _kinds.get(cell, 'Y') != 'Y' and _touch[cell] == {(cell[0], cell[1], cell[2] - 1), (cell[0], cell[1], cell[2] + 1)}:
                column.append(cell)
                cell = (cell[0], cell[1], cell[2] + 1)
            if (_kinds.get(cell) == 'Y' or cell in _marker_anchors) and _touch[cell] == {(cell[0], cell[1], cell[2] - 1)}:
                pinned_y.update(column)
                pinned_y.add(cell)

        updated_segment_hadamards = {}
        for (endpoint_a, endpoint_b), value in payload['segment_hadamards'].items():
            if max(endpoint_a[2], endpoint_b[2]) > height:
                endpoint_a = (endpoint_a[0], endpoint_a[1], endpoint_a[2] + 1)
                endpoint_b = (endpoint_b[0], endpoint_b[1], endpoint_b[2] + 1)
            updated_segment_hadamards[frozenset({endpoint_a, endpoint_b})] = value
        payload['segment_hadamards'] = updated_segment_hadamards
        payload['spider_positions'] = {k: (v[0], v[1], v[2] + 1) if v[2] > height else (v[0], v[1], v[2]) for k, v in payload['spider_positions'].items()}
        payload['t_gate_positions'] = {k: (v[0], v[1], v[2] + 1) if v[2] > height else (v[0], v[1], v[2]) for k, v in payload['t_gate_positions'].items()}

        for ev, v in sorted((e, tuple(w)) for e, w in payload['t_gate_positions'].items()):
            if v[2] != height or 'T' in routedness[ev][1]:
                continue
            if commutative_stack(v[:2], v[2])[1] == v[2]:
                continue
            payload['t_gate_positions'][ev] = (v[0], v[1], v[2] + 1)
            if ev in t_spider and tuple(payload['spider_positions'].get(t_spider[ev], v)) == v:
                payload['spider_positions'][t_spider[ev]] = (v[0], v[1], v[2] + 1) # the gate and its blue spider have to arrive together, or every later capacity count and commutation walk reads the vacated cell

        # First, we'll see if the T-gate can't be implemented because the next blocking height is this height. If so, we delay the circuit including the stuff already implemented at this height (feasible even if we have T-gates because cultivation doesn't fail with extra time) and retry. If the T-gate can't be implemented and the next blocking height is the next height, then we'll delay the circuit *not* including the stuff already at this height (so the next blocking height becomes the current height + 2), promote the T-gate to the next height, 
        graph['pipes'] = [{
            'u': [pipe['u'][0], pipe['u'][1], pipe['u'][2] + 1] if max(pipe['u'][2], pipe['v'][2]) > height and tuple(pipe['u']) not in pinned_y and tuple(pipe['v']) not in pinned_y else [pipe['u'][0], pipe['u'][1], pipe['u'][2]],
            'v': [pipe['v'][0], pipe['v'][1], pipe['v'][2] + 1] if max(pipe['u'][2], pipe['v'][2]) > height and tuple(pipe['u']) not in pinned_y and tuple(pipe['v']) not in pinned_y else [pipe['v'][0], pipe['v'][1], pipe['v'][2]],
            'kind': pipe['kind'],
            'transform': pipe['transform']}
            for pipe in graph['pipes']]
        graph['cubes'] = [{ # TODO: there is a bug in indexing when we're inserting delay cubes...maybe
            'position': [cube['position'][0], cube['position'][1], cube['position'][2] + 1] if cube['position'][2] > height and tuple(cube['position']) not in pinned_y else [cube['position'][0], cube['position'][1], cube['position'][2]],
            'kind': cube['kind'],
            'label': cube['label'],
            'transform': cube['transform']}
            for cube in graph['cubes']]
        baked_cubes = {
            (x, y, z + 1) if z > height and (x, y, z) not in pinned_y else (x, y, z)
            for x, y, z in baked_cubes
        }
        machinery_cells = {
            ((x, y, z + 1) if z > height and (x, y, z) not in pinned_y else (x, y, z)): owner
            for (x, y, z), owner in machinery_cells.items()
        }
        if 'ports' in graph:
            graph['ports'] = {lbl: ([c[0], c[1], c[2] + 1] if c[2] > height and (c[0], c[1], c[2]) not in pinned_y else list(c)) for lbl, c in graph['ports'].items()}
        # Each square rides its anchor cube through the fold and is then re-derived from it, so
        # it can never drift off the face it is stamped on.
        for _marker in conditional_markers:
            _ax, _ay, _az = _marker['anchor']
            if _az > height and (_ax, _ay, _az) not in pinned_y:
                _az += 1
            _marker['anchor'] = (_ax, _ay, _az)
            _marker['cell'] = (_ax, _ay, _az + 1)
        # Now we'll add in the verticals, we'll look for pipes that *(now)* sit above the height and are vertical
        vertical_pipes = [pipe for pipe in graph['pipes'] if max(pipe['u'][2], pipe['v'][2]) == height + 2 and min(pipe['u'][2], pipe['v'][2]) == height + 1 and tuple(pipe['u']) not in pinned_y and tuple(pipe['v']) not in pinned_y]
        for vertical_pipe in vertical_pipes:
            graph['pipes'].append({
                'u': [vertical_pipe['u'][0], vertical_pipe['u'][1], vertical_pipe['u'][2] - 1],
                'v': [vertical_pipe['v'][0], vertical_pipe['v'][1], vertical_pipe['v'][2] - 1],
                'kind': vertical_pipe['kind'][:3],
                'transform': vertical_pipe['transform']
            }) # We only need this singular piece because these are pretty much latent pipes, nothing really happens here
            _passthrough_position = min(vertical_pipe['u'], vertical_pipe['v'], key = lambda e: e[2])
            graph['cubes'].append({
                'position': _passthrough_position,
                'kind': vertical_pipe['kind'][:3].replace('O', 'X'), # Arbitrary since it's a passthrough-cube
                'label': '',
                'transform': vertical_pipe['transform']
            })
            if tuple(vertical_pipe['u'][:3]) in machinery_cells or tuple(vertical_pipe['v'][:3]) in machinery_cells:
                machinery_cells[tuple(_passthrough_position)] = machinery_cells.get(tuple(vertical_pipe['u'][:3]), machinery_cells.get(tuple(vertical_pipe['v'][:3]))) # a passthrough extending a machinery leg is itself machinery, owned by the same T event
        reserve_backup_t_injection_cubes()
        if any(e not in backup_t_injection_reserved for e, v in payload['t_gate_positions'].items() if v[2] <= height + 1):
            return 'repeat'
        # The walk must not advance past a gate it left standing: a cell below the frontier is one
        # nothing ever looks at again. Repeating buys the room -- each fold inserts an empty layer
        # above the gate -- and the backup T-injection cube reservation is what makes that terminate, since every fold
        # raises `height` off a held square whose column nothing may build on.
        if any(v[2] == height for v in payload['t_gate_positions'].values()):
            return 'repeat' # old while-level continue: re-decide this layer without advancing
    else:
        new_fractional_delay_found = False
    if undelivered_fractional_ticks:
        delay += undelivered_fractional_ticks / CUBE_DISTANCE
        delay_planes.append((height, 'fractional'))
        undelivered_fractional_ticks = 0
    for square_i in product(range(nx), range(ny)):
        if cultivated(square_i) and ((square_i + (height,)) in current_height_cubes or (square_i + (height,)) in baked_cubes):
            cultivation_round[square_i] = False
    height += 1
    return 'advance'


MAX_STALL = int(os.environ.get('TOPOLS_MAX_STALL', str(2 * CUBE_DISTANCE))) # passes the walk may re-decide one layer before it is called a stall
_stall, _stall_height = 0, None
while payload['t_gate_positions']: # parking a solo T-gate takes it out of the work queue, so an empty t_gate_positions is not on its own a finished circuit
    if height > MAX_HEIGHT:
        print(f'ABORTED at height {height}: {len(payload["t_gate_positions"])} T-gates remain')
        break
    # A pass that re-decides the same layer buys the gates standing on it more room, and the walk
    # deliberately takes as many as it needs -- but it does not advance `height`, so MAX_HEIGHT above
    # cannot see it. Room only ever grows ABOVE the layer: a gate that instead needs a free neighbour
    # BESIDE it gains nothing from a repeat, because the fold leaves the layer at `height` alone. This
    # is the valve for that case, and it names the gates so the geometry can be read off the abort.
    _stall, _stall_height = (_stall + 1 if height == _stall_height else 0), height
    if _stall > MAX_STALL:
        print(f'ABORTED at height {height}: re-decided for {_stall} passes without advancing; '
              f'{sorted(e for e, v in payload["t_gate_positions"].items() if v[2] == height)} standing on it')
        break
    pbar.update(previous_length - len(payload['t_gate_positions']))
    previous_length = len(payload['t_gate_positions'])
    step_one_height()


if payload['t_gate_positions']: # a non-empty queue means the drain starved and the delay below is not a complete circuit
    print(f'WARNING: {len(payload["t_gate_positions"])} T-gates never made it back into the circuit')
print(f'Total delay: {delay}')


_final_pipe_dirs = defaultdict(set)
for _p in graph['pipes']:
    _u, _v = tuple(_p['u']), tuple(_p['v'])
    _axis = next(a for a in range(3) if _u[a] != _v[a])
    _sign = 1 if _v[_axis] > _u[_axis] else -1
    _final_pipe_dirs[_u].add((_axis, _sign))
    _final_pipe_dirs[_v].add((_axis, -_sign))

out_dir = Path('../result')
out_dir.mkdir(parents=True, exist_ok=True)
json_path = out_dir / f'injections_clean_{VARIANT}.json' # per-variant so the three runs don't clobber each other
html_path = out_dir / f'injections_clean_{VARIANT}.html'
json_path.write_text(json.dumps(graph, indent=2))
block_graph = BlockGraph.from_json(json_path)
block_graph.validate() # a full transplant means a mis-coloured ZZ merge is a hard failure, not a silent one
block_graph.view_as_html(write_html_filepath=html_path)

_plane_w, _plane_d = nx * 3, ny * 3
_plane_cx, _plane_cy = (nx - 1) * 3 / 2 + 0.5, (ny - 1) * 3 / 2 + 0.5
_plane_specs = json.dumps([{'z': (z + 1) * 3 + 0.5 if k == 'whole' else z * 3 + 0.5, 'whole': k == 'whole'} for z, k in delay_planes])
_hook = '        scene.add(collada.scene);\n'
_marker_specs = json.dumps([list(_m['anchor']) for _m in conditional_markers])
_html = html_path.read_text()

import re
_html = re.sub(r'scene\.background = new THREE\.Color\("[^"]*"\);',
                'scene.background = new THREE.Color("#FFFFFF");', _html)
_html = re.sub(
    r'\s*const axesHelper = new THREE\.AxesHelper\(2\);\n'
    r'\s*axesHelper\.rotation\.x = -Math\.PI / 2;\n'
    r'\s*scene\.add\(axesHelper\);\n',
        '\n', _html)

html_path.write_text(_html.replace(_hook, _hook + (
    # The conditional X/Z measurement is a flat black square stamped on its anchor's top face.
    f'        for (const p of {_marker_specs}) {{'
    ' const sq = new THREE.Mesh(new THREE.BoxGeometry(1.0, 1.0, 0.14),'
    ' new THREE.MeshBasicMaterial({color: 0x000000, depthWrite: false}));'
    ' sq.position.set(p[0] * 3 + 0.5, p[1] * 3 + 0.5, p[2] * 3 + 0.5 + 1.55);'
    ' collada.scene.add(sq); }\n'
    f'        for (const p of {_plane_specs}) {{'
    f' const plane = new THREE.Mesh(new THREE.PlaneGeometry({_plane_w}, {_plane_d}),'
    ' new THREE.MeshBasicMaterial({color: p.whole ? 0x2563eb : 0xdb2777, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthWrite: false}));'
    f' plane.position.set({_plane_cx}, {_plane_cy}, p.z);'
    ' collada.scene.add(plane); }\n'
), 1))
print(f'Wrote block graph HTML to {html_path}')