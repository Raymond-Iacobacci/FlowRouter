from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
from tqec import BlockGraph
from tqec.computation.cube import Cube, Port, YHalfCube, ZXCube
from tqec.computation.pipe import Pipe
from tqec.utils.position import Position3D

try:
    from .canonical_pipe_faces import (
        PIPE_BLUE,
        PIPE_ENDCAP,
        PIPE_RED,
        canonical_node_tqec,
        canonical_pipe_face_colors,
    )
except ImportError:
    from canonical_pipe_faces import (
        PIPE_BLUE,
        PIPE_ENDCAP,
        PIPE_RED,
        canonical_node_tqec,
        canonical_pipe_face_colors,
    )


AXES = ("x", "y", "z")
ALLOWED_CUBE_KINDS = tuple(str(kind) for kind in ZXCube.all_kinds())
QUBIT_COLUMN_KINDS = tuple(
    cube_kind for cube_kind in ALLOWED_CUBE_KINDS if cube_kind[0] != cube_kind[1]
)


class TQECBlockAdapterError(RuntimeError):
    """Raised when placed routing metadata cannot be converted to a TQEC block graph."""


@dataclass(frozen=True)
class TQECBlockExportResult:
    json_path: Path | None = None
    dae_path: Path | None = None
    html_path: Path | None = None

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(path for path in (self.json_path, self.dae_path, self.html_path) if path is not None)


@dataclass(frozen=True)
class IOPortSpec:
    label: str
    position: tuple[int, int, int]
    neighbor: tuple[int, int, int]


def pipe_kind_from_axis_type(axis: str, pipe_type: int, *, has_hadamard: bool = False) -> str:
    """Return a TQEC pipe-kind string for a canonical TopoLS pipe state."""
    if axis not in AXES:
        raise TQECBlockAdapterError(f"Unsupported pipe axis {axis!r}.")
    if int(pipe_type) not in {0, 1}:
        raise TQECBlockAdapterError(f"Unsupported pipe type {pipe_type!r}.")

    colors = canonical_pipe_face_colors(axis, int(pipe_type))
    bases: list[str] = []
    for axis_index in range(3):
        first_face_color = colors[axis_index * 2]
        if first_face_color == PIPE_ENDCAP:
            bases.append("O")
        elif first_face_color == PIPE_RED:
            bases.append("X")
        elif first_face_color == PIPE_BLUE:
            bases.append("Z")
        else:
            raise TQECBlockAdapterError(
                f"Cannot map face color {first_face_color!r} to a TQEC basis."
            )
    return "".join(bases) + ("H" if has_hadamard else "")


def block_graph_from_placed_graph(
    placed: nx.DiGraph,
    *,
    name: str = "TopoLS",
    add_io_ports: bool = False,
) -> BlockGraph:
    """Instantiate a TQEC BlockGraph from TopoLS placed-routing metadata."""
    known_kinds, labels = _known_cubes_from_placed_graph(placed)
    explicit_hadamards = _normalized_hadamard_segments(placed.graph.get("segment_hadamards", {}))

    cube_coords: set[tuple[int, int, int]] = set(known_kinds)
    segments: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

    for path in placed.graph.get("display_paths", ()):
        normalized_path = _normalize_path(path)
        if len(normalized_path) < 2:
            continue
        cube_coords.update(normalized_path)
        segments.update(_segment_key(start, end) for start, end in zip(normalized_path, normalized_path[1:]))

    io_ports = _infer_io_ports(placed, cube_coords) if add_io_ports else ()
    qubit_column_coords = _qubit_column_coords(placed, cube_coords)
    cube_kinds, pipe_specs = _direct_tqec_kinds(
        cube_coords=sorted(cube_coords),
        segments=sorted(segments),
        known_kinds=known_kinds,
        explicit_hadamards=explicit_hadamards,
        routed_segment_pipe_types=_normalized_routed_segment_pipe_types(
            placed.graph.get("routed_segment_pipe_types", {})
        ),
        qubit_column_coords=qubit_column_coords,
    )

    block_graph = BlockGraph(name)
    for coord in sorted(cube_coords):
        block_graph.add_cube(_position(coord), cube_kinds[coord], label=labels.get(coord, ""))
    for (start, end), pipe_kind in sorted(pipe_specs.items()):
        block_graph.add_pipe(_position(start), _position(end), pipe_kind)
    for port in io_ports:
        block_graph.add_cube(_position(port.position), Port(), label=port.label) # where are the port labels created in the renderer? Are they even created rendering-side?
        block_graph.add_pipe(_position(port.position), _position(port.neighbor))

    # Machinery baked by the caller with its kinds already decided (S-gate gadgets). It bypasses
    # cube_coords/_direct_tqec_kinds entirely: there is nothing to infer, and a 'Y' half-cube is not a
    # ZXCube, so _inferred_pipe_kind could not walk a pipe touching one. validate() below still sees it.
    for coord, kind in sorted(placed.graph.get("machinery_cubes", {}).items()):
        block_graph.add_cube(_position(_normalize_coord(coord)), YHalfCube() if str(kind) == "Y" else ZXCube.from_str(str(kind)))
    for start, end, pipe_kind in sorted(placed.graph.get("machinery_pipes", ())):
        block_graph.add_pipe(_position(_normalize_coord(start)), _position(_normalize_coord(end)), str(pipe_kind))

    # Additive: ports the caller decided on, outside the inferred input/output ones -- a T-gate injection
    # route ends on one, where the factory hands the state over. Read through .get(), so a graph that does
    # not carry the key exports exactly as it did before. The pipe takes no kind: a Port has none, so tqec
    # infers it from the cube on the other end, which the caller already pinned.
    for label, position, neighbor in sorted(placed.graph.get("extra_ports", ())):
        block_graph.add_cube(_position(_normalize_coord(position)), Port(), label=str(label))
        block_graph.add_pipe(_position(_normalize_coord(position)), _position(_normalize_coord(neighbor)))

    block_graph.validate()
    return block_graph


def export_block_graph(
    block_graph: BlockGraph,
    output_dir: str | Path,
    basename: str,
    *,
    formats: Iterable[str] = ("json", "dae", "html"),
    validate: bool = True,
) -> TQECBlockExportResult:
    """Write native TQEC block graph artifacts and return their paths."""
    selected_formats = {format_name.lower() for format_name in formats}
    unknown_formats = selected_formats - {"json", "dae", "html"}
    if unknown_formats:
        raise TQECBlockAdapterError(f"Unknown TQEC export formats: {sorted(unknown_formats)}")

    if validate:
        block_graph.validate()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"{basename}.json" if "json" in selected_formats else None
    dae_path = output_path / f"{basename}.dae" if "dae" in selected_formats else None
    html_path = output_path / f"{basename}.html" if "html" in selected_formats else None

    if json_path is not None:
        block_graph.to_json(json_path)
    if dae_path is not None:
        block_graph.to_dae_file(dae_path)
    if html_path is not None:
        block_graph.view_as_html(write_html_filepath=html_path)

    return TQECBlockExportResult(json_path=json_path, dae_path=dae_path, html_path=html_path)


def _infer_io_ports(
    placed: nx.DiGraph,
    cube_coords: set[tuple[int, int, int]],
) -> tuple[IOPortSpec, ...]:
    qubit_columns = _infer_qubit_columns(placed, cube_coords)
    ports: list[IOPortSpec] = []
    occupied = set(cube_coords)
    for qubit, column_coords in sorted(qubit_columns.items()):
        if not column_coords:
            continue
        ordered_column = sorted(column_coords, key=lambda coord: coord[2])
        input_neighbor = ordered_column[0]
        output_neighbor = ordered_column[-1]
        input_position = (input_neighbor[0], input_neighbor[1], input_neighbor[2] - 1)
        output_position = (output_neighbor[0], output_neighbor[1], output_neighbor[2] + 1)
        if input_position in occupied:
            raise TQECBlockAdapterError(
                f"Cannot place input port for qubit {qubit} at occupied position {input_position!r}."
            )
        if output_position in occupied:
            raise TQECBlockAdapterError(
                f"Cannot place output port for qubit {qubit} at occupied position {output_position!r}."
            )
        ports.append(IOPortSpec(f"in_{qubit}", input_position, input_neighbor))
        ports.append(IOPortSpec(f"out_{qubit}", output_position, output_neighbor))
    return tuple(ports)


def _infer_qubit_columns(
    placed: nx.DiGraph,
    cube_coords: set[tuple[int, int, int]],
) -> dict[int, set[tuple[int, int, int]]]:
    xy_to_qubit = _xy_to_qubit_from_metadata(placed)

    qubit_columns: dict[int, set[tuple[int, int, int]]] = defaultdict(set)
    for coord in cube_coords:
        qubit = xy_to_qubit.get(coord[:2])
        if qubit is not None:
            qubit_columns[qubit].add(coord)
    return dict(qubit_columns)


def _qubit_column_coords(
    placed: nx.DiGraph,
    cube_coords: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    qubit_column_xys = set(_xy_to_qubit_from_metadata(placed))
    return {coord for coord in cube_coords if coord[:2] in qubit_column_xys}


def _xy_to_qubit_from_metadata(placed: nx.DiGraph) -> dict[tuple[int, int], int]:
    raw_mapping = placed.graph.get("qubit_to_xy", {})
    raw_num_qubits = placed.graph.get("num_qubits")
    num_qubits = int(raw_num_qubits) if raw_num_qubits is not None else None
    xy_to_qubit = {}
    for raw_qubit, raw_xy in dict(raw_mapping or {}).items():
        qubit = int(raw_qubit)
        if num_qubits is not None and not (0 <= qubit < num_qubits):
            continue
        xy = tuple(raw_xy)
        if len(xy) != 2:
            raise TQECBlockAdapterError(f"Expected a 2D qubit-column coordinate, got {raw_xy!r}.")
        xy_to_qubit[(int(xy[0]), int(xy[1]))] = qubit
    return xy_to_qubit


def _is_qcol_node(node_id) -> bool:
    return (
        isinstance(node_id, tuple)
        and len(node_id) == 3
        and node_id[0] == "QCOL"
    )


def _known_cubes_from_placed_graph(placed: nx.DiGraph):
    spider_positions = dict(placed.graph.get("spider_positions", {}))
    spider_orientations = dict(placed.graph.get("spider_orientations", {}))
    spider_faces = dict(placed.graph.get("spider_faces", {}))

    known_kinds: dict[tuple[int, int, int], str] = {}
    labels: dict[tuple[int, int, int], str] = {}

    for node_id, raw_coord in spider_positions.items():
        coord = _normalize_coord(raw_coord)
        labels.setdefault(coord, str(node_id))

        if _is_qcol_node(node_id):
            continue

        orientation = spider_orientations.get(node_id)
        face = spider_faces.get(node_id)
        if orientation is None or face is None:
            continue

        kind = canonical_node_tqec(str(orientation), int(face))
        if kind is None:
            continue
        existing_kind = known_kinds.get(coord)
        if existing_kind is not None and existing_kind != kind:
            raise TQECBlockAdapterError(
                f"Conflicting known cube kinds at {coord}: {existing_kind!r} vs {kind!r}."
            )
        known_kinds[coord] = kind

    return known_kinds, labels


def _normalized_routed_segment_pipe_types(routed_segment_pipe_types) -> dict[
    tuple[tuple[int, int, int], tuple[int, int, int]],
    int,
]:
    normalized = {}
    for raw_segment, raw_pipe_type in dict(routed_segment_pipe_types or {}).items():
        segment = _segment_key(*tuple(raw_segment))
        pipe_type = int(raw_pipe_type)
        if pipe_type not in {0, 1}:
            raise TQECBlockAdapterError(f"Unsupported exported pipe type {pipe_type!r} for segment {segment!r}.")
        existing = normalized.get(segment)
        if existing is not None and existing != pipe_type:
            raise TQECBlockAdapterError(
                f"Conflicting exported pipe types for segment {segment!r}: {existing!r} vs {pipe_type!r}."
            )
        normalized[segment] = pipe_type
    return normalized


def _axis_from_segment(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> str:
    start = _normalize_coord(start)
    end = _normalize_coord(end)
    _validate_neighboring_coords(start, end)
    delta = tuple(end[index] - start[index] for index in range(3))
    if delta[0]:
        return "x"
    if delta[1]:
        return "y"
    return "z"


def _add_cube_basis_constraint(
    constraints: dict[tuple[int, int, int], dict[int, str]],
    coord: tuple[int, int, int],
    axis_index: int,
    basis: str,
) -> None:
    basis = str(basis)
    if basis not in {"X", "Z"}:
        raise TQECBlockAdapterError(f"Unsupported cube basis constraint {basis!r} at {coord!r}.")
    existing = constraints.setdefault(coord, {}).get(axis_index)
    if existing is not None and existing != basis:
        raise TQECBlockAdapterError(
            f"Conflicting cube basis constraints at {coord!r} axis {axis_index}: {existing!r} vs {basis!r}."
        )
    constraints[coord][axis_index] = basis


def _choose_cube_kind(
    coord: tuple[int, int, int],
    constraints: dict[int, str],
    candidates: Iterable[str],
) -> str:
    matching = [
        kind
        for kind in candidates
        if all(kind[axis_index] == basis for axis_index, basis in constraints.items())
    ]
    if not matching:
        raise TQECBlockAdapterError(
            f"No TQEC cube kind at {coord!r} satisfies exported face constraints {constraints!r}."
        )
    return sorted(matching)[0]


def _direct_tqec_kinds(
    *,
    cube_coords: list[tuple[int, int, int]],
    segments: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    known_kinds: dict[tuple[int, int, int], str],
    explicit_hadamards: set[tuple[tuple[int, int, int], tuple[int, int, int]]],
    routed_segment_pipe_types: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int],
    qubit_column_coords: set[tuple[int, int, int]] | None = None,
) -> tuple[
    dict[tuple[int, int, int], str],
    dict[tuple[tuple[int, int, int], tuple[int, int, int]], str],
]:
    qubit_column_coords = set(qubit_column_coords or ())
    constraints: dict[tuple[int, int, int], dict[int, str]] = {
        coord: {} for coord in cube_coords
    }

    for coord, kind in known_kinds.items():
        for axis_index, basis in enumerate(kind):
            _add_cube_basis_constraint(constraints, coord, axis_index, basis)

    for segment, pipe_type in routed_segment_pipe_types.items():
        if segment not in segments:
            raise TQECBlockAdapterError(f"Exported pipe type references missing segment {segment!r}.")
        axis = _axis_from_segment(*segment)
        pipe_kind = pipe_kind_from_axis_type(axis, pipe_type)
        for coord in segment:
            for axis_index, basis in enumerate(pipe_kind[:3]):
                if basis != "O":
                    _add_cube_basis_constraint(constraints, coord, axis_index, basis)

    hadamard_zs_by_xy: dict[tuple[int, int], list[int]] = defaultdict(list)
    for start, end in explicit_hadamards:
        if start[:2] == end[:2] and abs(start[2] - end[2]) == 1:
            hadamard_zs_by_xy[start[:2]].append(min(start[2], end[2]))

    cube_kinds = {}
    for coord in qubit_column_coords:
        hadamard_zs = hadamard_zs_by_xy.get(coord[:2])
        if hadamard_zs:
            xy_basis = "ZX" if sum(hadamard_z < coord[2] for hadamard_z in hadamard_zs) % 2 else "XZ"
            _add_cube_basis_constraint(constraints, coord, 0, xy_basis[0])
            _add_cube_basis_constraint(constraints, coord, 1, xy_basis[1])

    for coord in cube_coords:
        candidates = QUBIT_COLUMN_KINDS if coord in qubit_column_coords else ALLOWED_CUBE_KINDS
        cube_kinds[coord] = _choose_cube_kind(coord, constraints[coord], candidates)

    pipe_specs = {}
    for start, end in segments:
        segment = _segment_key(start, end)
        if segment in routed_segment_pipe_types:
            pipe_kind = pipe_kind_from_axis_type(_axis_from_segment(start, end), routed_segment_pipe_types[segment])
            inferred_pipe_kind = _inferred_pipe_kind(start, end, cube_kinds[start], cube_kinds[end])
            if inferred_pipe_kind != pipe_kind:
                raise TQECBlockAdapterError(
                    f"Exported pipe kind {pipe_kind!r} for segment {segment!r} does not match cube kinds "
                    f"{cube_kinds[start]!r}, {cube_kinds[end]!r} -> {inferred_pipe_kind!r}."
                )
        else:
            pipe_kind = _inferred_pipe_kind(start, end, cube_kinds[start], cube_kinds[end])
        if pipe_kind is None:
            raise TQECBlockAdapterError(
                f"Could not infer a TQEC pipe kind for segment {start!r} -> {end!r}."
            )
        if segment in explicit_hadamards and not pipe_kind.endswith("H"):
            raise TQECBlockAdapterError(f"Expected a Hadamard pipe for segment {segment!r}, got {pipe_kind!r}.")
        pipe_specs[(start, end)] = pipe_kind
    return cube_kinds, pipe_specs


def _inferred_pipe_kind(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    start_kind: str,
    end_kind: str,
) -> str | None:
    try:
        pipe = Pipe.from_cubes(
            Cube(_position(start), ZXCube.from_str(start_kind)),
            Cube(_position(end), ZXCube.from_str(end_kind)),
        )
    except Exception:
        return None
    return str(pipe.kind)


def _normalize_path(path) -> tuple[tuple[int, int, int], ...]:
    normalized = []
    for raw_coord in path:
        coord = _normalize_coord(raw_coord)
        if not normalized or normalized[-1] != coord:
            normalized.append(coord)
    for start, end in zip(normalized, normalized[1:]):
        _validate_neighboring_coords(start, end)
    return tuple(normalized)


def _normalized_hadamard_segments(segment_hadamards) -> set[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    return {
        _segment_key(*tuple(segment))
        for segment, value in dict(segment_hadamards or {}).items()
        if value
    }


def _segment_key(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    start = _normalize_coord(start)
    end = _normalize_coord(end)
    _validate_neighboring_coords(start, end)
    return tuple(sorted((start, end)))


def _normalize_coord(coord) -> tuple[int, int, int]:
    if isinstance(coord, Position3D):
        return coord.as_tuple()
    if not isinstance(coord, tuple):
        coord = tuple(coord)
    if len(coord) != 3:
        raise TQECBlockAdapterError(f"Expected a 3D coordinate, got {coord!r}.")
    normalized = []
    for value in coord:
        rounded = int(round(float(value)))
        if abs(float(value) - rounded) > 1e-9:
            raise TQECBlockAdapterError(f"Coordinate {coord!r} is not integral.")
        normalized.append(rounded)
    return tuple(normalized)


def _validate_neighboring_coords(start: tuple[int, int, int], end: tuple[int, int, int]) -> None:
    distance = sum(abs(start[index] - end[index]) for index in range(3))
    if distance != 1:
        raise TQECBlockAdapterError(
            f"TQEC pipes require neighboring cubes, got segment {start!r} -> {end!r}."
        )


def _position(coord: tuple[int, int, int]) -> Position3D:
    return Position3D(*coord)



# from __future__ import annotations

# from collections import defaultdict, deque
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Iterable

# import networkx as nx
# from tqec import BlockGraph
# from tqec.computation.cube import Cube, Port, ZXCube
# from tqec.computation.pipe import Pipe
# from tqec.utils.position import Position3D

# try:
#     from .canonical_pipe_faces import (
#         PIPE_BLUE,
#         PIPE_ENDCAP,
#         PIPE_RED,
#         canonical_node_tqec,
#         canonical_pipe_face_colors,
#     )
# except ImportError:
#     from canonical_pipe_faces import (
#         PIPE_BLUE,
#         PIPE_ENDCAP,
#         PIPE_RED,
#         canonical_node_tqec,
#         canonical_pipe_face_colors,
#     )


# AXES = ("x", "y", "z")
# ALLOWED_CUBE_KINDS = tuple(str(kind) for kind in ZXCube.all_kinds())
# QUBIT_COLUMN_KINDS = tuple(
#     cube_kind for cube_kind in ALLOWED_CUBE_KINDS if cube_kind[0] != cube_kind[1]
# )


# class TQECBlockAdapterError(RuntimeError):
#     """Raised when placed routing metadata cannot be converted to a TQEC block graph."""


# @dataclass(frozen=True)
# class TQECBlockExportResult:
#     json_path: Path | None = None
#     dae_path: Path | None = None
#     html_path: Path | None = None

#     @property
#     def paths(self) -> tuple[Path, ...]:
#         return tuple(path for path in (self.json_path, self.dae_path, self.html_path) if path is not None)


# @dataclass(frozen=True)
# class IOPortSpec:
#     label: str
#     position: tuple[int, int, int]
#     neighbor: tuple[int, int, int]


# def pipe_kind_from_axis_type(axis: str, pipe_type: int, *, has_hadamard: bool = False) -> str:
#     """Return a TQEC pipe-kind string for a canonical TopoLS pipe state."""
#     if axis not in AXES:
#         raise TQECBlockAdapterError(f"Unsupported pipe axis {axis!r}.")
#     if int(pipe_type) not in {0, 1}:
#         raise TQECBlockAdapterError(f"Unsupported pipe type {pipe_type!r}.")

#     colors = canonical_pipe_face_colors(axis, int(pipe_type))
#     bases: list[str] = []
#     for axis_index in range(3):
#         first_face_color = colors[axis_index * 2]
#         if first_face_color == PIPE_ENDCAP:
#             bases.append("O")
#         elif first_face_color == PIPE_RED:
#             bases.append("X")
#         elif first_face_color == PIPE_BLUE:
#             bases.append("Z")
#         else:
#             raise TQECBlockAdapterError(
#                 f"Cannot map face color {first_face_color!r} to a TQEC basis."
#             )
#     return "".join(bases) + ("H" if has_hadamard else "")


# def block_graph_from_placed_graph(
#     placed: nx.DiGraph,
#     *,
#     name: str = "TopoLS",
#     add_io_ports: bool = False,
# ) -> BlockGraph:
#     """Instantiate a TQEC BlockGraph from TopoLS placed-routing metadata."""
#     known_kinds, labels = _known_cubes_from_placed_graph(placed)
#     explicit_hadamards = _normalized_hadamard_segments(placed.graph.get("segment_hadamards", {}))

#     cube_coords: set[tuple[int, int, int]] = set(known_kinds)
#     segments: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()

#     for path in placed.graph.get("display_paths", ()):
#         normalized_path = _normalize_path(path)
#         if len(normalized_path) < 2:
#             continue
#         cube_coords.update(normalized_path)
#         segments.update(_segment_key(start, end) for start, end in zip(normalized_path, normalized_path[1:]))

#     io_ports = _infer_io_ports(placed, cube_coords) if add_io_ports else ()
#     qubit_column_coords = _qubit_column_coords(placed, cube_coords)
#     cube_kinds, pipe_specs = _solve_tqec_kinds(
#         cube_coords=sorted(cube_coords),
#         segments=sorted(segments),
#         known_kinds=known_kinds,
#         explicit_hadamards=explicit_hadamards,
#         port_constraints=_port_kind_constraints(io_ports),
#         qubit_column_coords=qubit_column_coords,
#     )

#     block_graph = BlockGraph(name)
#     for coord in sorted(cube_coords):
#         block_graph.add_cube(_position(coord), cube_kinds[coord], label=labels.get(coord, ""))
#     for (start, end), pipe_kind in sorted(pipe_specs.items()):
#         block_graph.add_pipe(_position(start), _position(end), pipe_kind)
#     for port in io_ports:
#         block_graph.add_cube(_position(port.position), Port(), label=port.label) # where are the port labels created in the renderer? Are they even created rendering-side?
#         block_graph.add_pipe(_position(port.position), _position(port.neighbor))

#     block_graph.validate()
#     return block_graph


# def export_block_graph(
#     block_graph: BlockGraph,
#     output_dir: str | Path,
#     basename: str,
#     *,
#     formats: Iterable[str] = ("json", "dae", "html"),
#     validate: bool = True,
# ) -> TQECBlockExportResult:
#     """Write native TQEC block graph artifacts and return their paths."""
#     selected_formats = {format_name.lower() for format_name in formats}
#     unknown_formats = selected_formats - {"json", "dae", "html"}
#     if unknown_formats:
#         raise TQECBlockAdapterError(f"Unknown TQEC export formats: {sorted(unknown_formats)}")

#     if validate:
#         block_graph.validate()

#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     json_path = output_path / f"{basename}.json" if "json" in selected_formats else None
#     dae_path = output_path / f"{basename}.dae" if "dae" in selected_formats else None
#     html_path = output_path / f"{basename}.html" if "html" in selected_formats else None

#     if json_path is not None:
#         block_graph.to_json(json_path)
#     if dae_path is not None:
#         block_graph.to_dae_file(dae_path)
#     if html_path is not None:
#         block_graph.view_as_html(write_html_filepath=html_path)

#     return TQECBlockExportResult(json_path=json_path, dae_path=dae_path, html_path=html_path)


# def _infer_io_ports(
#     placed: nx.DiGraph,
#     cube_coords: set[tuple[int, int, int]],
# ) -> tuple[IOPortSpec, ...]:
#     qubit_columns = _infer_qubit_columns(placed, cube_coords)
#     ports: list[IOPortSpec] = []
#     occupied = set(cube_coords)
#     for qubit, column_coords in sorted(qubit_columns.items()):
#         if not column_coords:
#             continue
#         ordered_column = sorted(column_coords, key=lambda coord: coord[2])
#         input_neighbor = ordered_column[0]
#         output_neighbor = ordered_column[-1]
#         input_position = (input_neighbor[0], input_neighbor[1], input_neighbor[2] - 1)
#         output_position = (output_neighbor[0], output_neighbor[1], output_neighbor[2] + 1)
#         if input_position in occupied:
#             raise TQECBlockAdapterError(
#                 f"Cannot place input port for qubit {qubit} at occupied position {input_position!r}."
#             )
#         if output_position in occupied:
#             raise TQECBlockAdapterError(
#                 f"Cannot place output port for qubit {qubit} at occupied position {output_position!r}."
#             )
#         ports.append(IOPortSpec(f"in_{qubit}", input_position, input_neighbor))
#         ports.append(IOPortSpec(f"out_{qubit}", output_position, output_neighbor))
#     return tuple(ports)


# def _infer_qubit_columns(
#     placed: nx.DiGraph,
#     cube_coords: set[tuple[int, int, int]],
# ) -> dict[int, set[tuple[int, int, int]]]:
#     xy_to_qubit = _xy_to_qubit_from_metadata(placed)

#     qubit_columns: dict[int, set[tuple[int, int, int]]] = defaultdict(set)
#     for coord in cube_coords:
#         qubit = xy_to_qubit.get(coord[:2])
#         if qubit is not None:
#             qubit_columns[qubit].add(coord)
#     return dict(qubit_columns)


# def _qubit_column_coords(
#     placed: nx.DiGraph,
#     cube_coords: set[tuple[int, int, int]],
# ) -> set[tuple[int, int, int]]:
#     qubit_column_xys = set(_xy_to_qubit_from_metadata(placed))
#     return {coord for coord in cube_coords if coord[:2] in qubit_column_xys}


# def _xy_to_qubit_from_metadata(placed: nx.DiGraph) -> dict[tuple[int, int], int]:
#     raw_mapping = placed.graph.get("qubit_to_xy", {})
#     raw_num_qubits = placed.graph.get("num_qubits")
#     num_qubits = int(raw_num_qubits) if raw_num_qubits is not None else None
#     xy_to_qubit = {}
#     for raw_qubit, raw_xy in dict(raw_mapping or {}).items():
#         qubit = int(raw_qubit)
#         if num_qubits is not None and not (0 <= qubit < num_qubits):
#             continue
#         xy = tuple(raw_xy)
#         if len(xy) != 2:
#             raise TQECBlockAdapterError(f"Expected a 2D qubit-column coordinate, got {raw_xy!r}.")
#         xy_to_qubit[(int(xy[0]), int(xy[1]))] = qubit
#     return xy_to_qubit


# def _is_qcol_node(node_id) -> bool:
#     return (
#         isinstance(node_id, tuple)
#         and len(node_id) == 3
#         and node_id[0] == "QCOL"
#     )


# def _port_kind_constraints(io_ports: Iterable[IOPortSpec]) -> dict[tuple[int, int, int], set[str]]:
#     constraints: dict[tuple[int, int, int], set[str]] = {}
#     for port in io_ports:
#         allowed = {
#             cube_kind
#             for cube_kind in ALLOWED_CUBE_KINDS
#             if _inferred_port_pipe_kind(port.position, port.neighbor, cube_kind) is not None
#         }
#         existing = constraints.get(port.neighbor)
#         constraints[port.neighbor] = allowed if existing is None else existing & allowed
#     return constraints


# def _known_cubes_from_placed_graph(placed: nx.DiGraph):
#     spider_positions = dict(placed.graph.get("spider_positions", {}))
#     spider_orientations = dict(placed.graph.get("spider_orientations", {}))
#     spider_faces = dict(placed.graph.get("spider_faces", {}))

#     known_kinds: dict[tuple[int, int, int], str] = {}
#     labels: dict[tuple[int, int, int], str] = {}

#     for node_id, raw_coord in spider_positions.items():
#         coord = _normalize_coord(raw_coord)
#         labels.setdefault(coord, str(node_id))

#         if _is_qcol_node(node_id):
#             continue

#         orientation = spider_orientations.get(node_id)
#         face = spider_faces.get(node_id)
#         if orientation is None or face is None:
#             continue

#         kind = canonical_node_tqec(str(orientation), int(face))
#         if kind is None:
#             continue
#         existing_kind = known_kinds.get(coord)
#         if existing_kind is not None and existing_kind != kind:
#             raise TQECBlockAdapterError(
#                 f"Conflicting known cube kinds at {coord}: {existing_kind!r} vs {kind!r}."
#             )
#         known_kinds[coord] = kind

#     return known_kinds, labels


# def _solve_tqec_kinds(
#     *,
#     cube_coords: list[tuple[int, int, int]],
#     segments: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
#     known_kinds: dict[tuple[int, int, int], str],
#     explicit_hadamards: set[tuple[tuple[int, int, int], tuple[int, int, int]]],
#     port_constraints: dict[tuple[int, int, int], set[str]] | None = None,
#     qubit_column_coords: set[tuple[int, int, int]] | None = None,
# ) -> tuple[
#     dict[tuple[int, int, int], str],
#     dict[tuple[tuple[int, int, int], tuple[int, int, int]], str],
# ]:
#     qubit_column_coords = set(qubit_column_coords or ())
#     domains = {
#         coord: _initial_cube_domain(coord, known_kinds, qubit_column_coords)
#         for coord in cube_coords
#     }
#     _constrain_hadamard_column_domains(domains, qubit_column_coords, explicit_hadamards)
#     for coord, allowed_kinds in (port_constraints or {}).items():
#         if coord not in domains:
#             raise TQECBlockAdapterError(f"I/O port is attached to missing cube {coord!r}.")
#         domains[coord] &= allowed_kinds
#         if not domains[coord]:
#             raise TQECBlockAdapterError(
#                 f"No TQEC cube kind at {coord!r} can support its inferred I/O port."
#             )
#     allowed_pairs = _allowed_segment_kind_pairs(segments, domains, explicit_hadamards)
#     neighbors = _segment_neighbors(segments)
#     domains = _propagate_domains(domains, allowed_pairs, neighbors)
#     solution = _search_domain_solution(domains, allowed_pairs, neighbors)
#     if solution is None:
#         raise TQECBlockAdapterError(
#             "Could not infer TQEC cube kinds that satisfy the placed graph topology."
#         )
#     return solution, _pipe_specs_from_solution(segments, solution)


# def _initial_cube_domain(
#     coord: tuple[int, int, int],
#     known_kinds: dict[tuple[int, int, int], str],
#     qubit_column_coords: set[tuple[int, int, int]],
# ) -> set[str]:
#     known_kind = known_kinds.get(coord)
#     if coord not in qubit_column_coords:
#         return {known_kind} if known_kind is not None else set(ALLOWED_CUBE_KINDS)

#     if known_kind is None:
#         return set(QUBIT_COLUMN_KINDS)

#     # Qubit-column cubes can flip their temporal (z-normal) basis without
#     # changing the routed circuit. This lets lateral pipes get differently
#     # colored long-side walls while preserving the recorded x/y bases.
#     same_xy_domain = {
#         candidate
#         for candidate in QUBIT_COLUMN_KINDS
#         if candidate[:2] == known_kind[:2]
#     }
#     if same_xy_domain:
#         return same_xy_domain

#     return set(QUBIT_COLUMN_KINDS)

# def _constrain_hadamard_column_domains(
#     domains: dict[tuple[int, int, int], set[str]],
#     qubit_column_coords: set[tuple[int, int, int]],
#     explicit_hadamards: set[tuple[tuple[int, int, int], tuple[int, int, int]]],
# ) -> None:
#     hadamard_zs_by_xy: dict[tuple[int, int], list[int]] = defaultdict(list)
#     for start, end in explicit_hadamards:
#         if start[:2] == end[:2] and abs(start[2] - end[2]) == 1:
#             hadamard_zs_by_xy[start[:2]].append(min(start[2], end[2]))

#     for coord in qubit_column_coords:
#         hadamard_zs = hadamard_zs_by_xy.get(coord[:2])
#         if not hadamard_zs:
#             continue
#         xy_basis = "ZX" if sum(hadamard_z < coord[2] for hadamard_z in hadamard_zs) % 2 else "XZ"
#         domains[coord] &= {kind for kind in QUBIT_COLUMN_KINDS if kind[:2] == xy_basis}
#         if not domains[coord]:
#             raise TQECBlockAdapterError(f"No Hadamard-compatible cube kind remains possible at {coord!r}.")


# def _allowed_segment_kind_pairs(
#     segments: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
#     domains: dict[tuple[int, int, int], set[str]],
#     explicit_hadamards: set[tuple[tuple[int, int, int], tuple[int, int, int]]],
# ) -> dict[tuple[tuple[int, int, int], tuple[int, int, int]], set[tuple[str, str]]]:
#     allowed_pairs: dict[tuple[tuple[int, int, int], tuple[int, int, int]], set[tuple[str, str]]] = {}
#     for start, end in segments:
#         pairs = set()
#         require_hadamard = (start, end) in explicit_hadamards
#         for start_kind in domains[start]:
#             for end_kind in domains[end]:
#                 pipe_kind = _inferred_pipe_kind(start, end, start_kind, end_kind)
#                 if pipe_kind is None:
#                     continue
#                 if require_hadamard and not pipe_kind.endswith("H"):
#                     continue
#                 pairs.add((start_kind, end_kind))
#         if not pairs:
#             raise TQECBlockAdapterError(
#                 f"No TQEC-compatible cube kind pairs for segment {start!r} -> {end!r}."
#             )
#         allowed_pairs[(start, end)] = pairs
#         allowed_pairs[(end, start)] = {(end_kind, start_kind) for start_kind, end_kind in pairs}
#     return allowed_pairs


# def _segment_neighbors(
#     segments: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
# ) -> dict[tuple[int, int, int], list[tuple[int, int, int]]]:
#     neighbors: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
#     for start, end in segments:
#         neighbors[start].append(end)
#         neighbors[end].append(start)
#     return dict(neighbors)


# def _propagate_domains(
#     domains: dict[tuple[int, int, int], set[str]],
#     allowed_pairs: dict[tuple[tuple[int, int, int], tuple[int, int, int]], set[tuple[str, str]]],
#     neighbors: dict[tuple[int, int, int], list[tuple[int, int, int]]],
# ) -> dict[tuple[int, int, int], set[str]]:
#     domains = {coord: set(values) for coord, values in domains.items()}
#     queue = deque(allowed_pairs)
#     while queue:
#         coord, neighbor = queue.popleft()
#         removed_values = {
#             kind
#             for kind in domains[coord]
#             if not any((kind, neighbor_kind) in allowed_pairs[(coord, neighbor)] for neighbor_kind in domains[neighbor])
#         }
#         if not removed_values:
#             continue
#         domains[coord] -= removed_values
#         if not domains[coord]:
#             raise TQECBlockAdapterError(f"No TQEC cube kind remains possible at {coord!r}.")
#         for other_neighbor in neighbors.get(coord, ()):
#             if other_neighbor != neighbor:
#                 queue.append((other_neighbor, coord))
#     return domains


# def _search_domain_solution(
#     domains: dict[tuple[int, int, int], set[str]],
#     allowed_pairs: dict[tuple[tuple[int, int, int], tuple[int, int, int]], set[tuple[str, str]]],
#     neighbors: dict[tuple[int, int, int], list[tuple[int, int, int]]],
# ) -> dict[tuple[int, int, int], str] | None:
#     if all(len(values) == 1 for values in domains.values()):
#         return {coord: next(iter(values)) for coord, values in domains.items()}

#     coord = min(
#         (coord for coord, values in domains.items() if len(values) > 1),
#         key=lambda candidate: (len(domains[candidate]), candidate),
#     )
#     for kind in _ordered_domain_values(coord, domains, allowed_pairs, neighbors):
#         narrowed_domains = {candidate: set(values) for candidate, values in domains.items()}
#         narrowed_domains[coord] = {kind}
#         try:
#             propagated = _propagate_domains(narrowed_domains, allowed_pairs, neighbors)
#         except TQECBlockAdapterError:
#             continue
#         solution = _search_domain_solution(propagated, allowed_pairs, neighbors)
#         if solution is not None:
#             return solution
#     return None


# def _ordered_domain_values(
#     coord: tuple[int, int, int],
#     domains: dict[tuple[int, int, int], set[str]],
#     allowed_pairs: dict[tuple[tuple[int, int, int], tuple[int, int, int]], set[tuple[str, str]]],
#     neighbors: dict[tuple[int, int, int], list[tuple[int, int, int]]],
# ) -> list[str]:
#     def value_score(kind: str) -> tuple[int, int, str]:
#         forced_hadamards = 0
#         compatible_pairs = 0
#         for neighbor in neighbors.get(coord, ()):
#             for neighbor_kind in domains[neighbor]:
#                 if (kind, neighbor_kind) not in allowed_pairs[(coord, neighbor)]:
#                     continue
#                 compatible_pairs += 1
#                 pipe_kind = _inferred_pipe_kind(coord, neighbor, kind, neighbor_kind)
#                 if pipe_kind is not None and pipe_kind.endswith("H"):
#                     forced_hadamards += 1
#         return forced_hadamards, -compatible_pairs, kind

#     return sorted(domains[coord], key=value_score)


# def _pipe_specs_from_solution(
#     segments: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
#     cube_kinds: dict[tuple[int, int, int], str],
# ) -> dict[tuple[tuple[int, int, int], tuple[int, int, int]], str]:
#     pipe_specs = {}
#     for start, end in segments:
#         pipe_kind = _inferred_pipe_kind(start, end, cube_kinds[start], cube_kinds[end])
#         if pipe_kind is None:
#             raise TQECBlockAdapterError(
#                 f"Could not infer a TQEC pipe kind for segment {start!r} -> {end!r}."
#             )
#         pipe_specs[(start, end)] = pipe_kind
#     return pipe_specs


# def _inferred_pipe_kind(
#     start: tuple[int, int, int],
#     end: tuple[int, int, int],
#     start_kind: str,
#     end_kind: str,
# ) -> str | None:
#     try:
#         pipe = Pipe.from_cubes(
#             Cube(_position(start), ZXCube.from_str(start_kind)),
#             Cube(_position(end), ZXCube.from_str(end_kind)),
#         )
#     except Exception:
#         return None
#     return str(pipe.kind)


# def _inferred_port_pipe_kind(
#     port: tuple[int, int, int],
#     neighbor: tuple[int, int, int],
#     cube_kind: str,
# ) -> str | None:
#     try:
#         pipe = Pipe.from_cubes(
#             Cube(_position(port), Port(), "port"),
#             Cube(_position(neighbor), ZXCube.from_str(cube_kind)),
#         )
#     except Exception:
#         return None
#     return str(pipe.kind)


# def _normalize_path(path) -> tuple[tuple[int, int, int], ...]:
#     normalized = []
#     for raw_coord in path:
#         coord = _normalize_coord(raw_coord)
#         if not normalized or normalized[-1] != coord:
#             normalized.append(coord)
#     for start, end in zip(normalized, normalized[1:]):
#         _validate_neighboring_coords(start, end)
#     return tuple(normalized)


# def _normalized_hadamard_segments(segment_hadamards) -> set[tuple[tuple[int, int, int], tuple[int, int, int]]]:
#     return {
#         _segment_key(*tuple(segment))
#         for segment, value in dict(segment_hadamards or {}).items()
#         if value
#     }


# def _segment_key(
#     start: tuple[int, int, int],
#     end: tuple[int, int, int],
# ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
#     start = _normalize_coord(start)
#     end = _normalize_coord(end)
#     _validate_neighboring_coords(start, end)
#     return tuple(sorted((start, end)))


# def _normalize_coord(coord) -> tuple[int, int, int]:
#     if isinstance(coord, Position3D):
#         return coord.as_tuple()
#     if not isinstance(coord, tuple):
#         coord = tuple(coord)
#     if len(coord) != 3:
#         raise TQECBlockAdapterError(f"Expected a 3D coordinate, got {coord!r}.")
#     normalized = []
#     for value in coord:
#         rounded = int(round(float(value)))
#         if abs(float(value) - rounded) > 1e-9:
#             raise TQECBlockAdapterError(f"Coordinate {coord!r} is not integral.")
#         normalized.append(rounded)
#     return tuple(normalized)


# def _validate_neighboring_coords(start: tuple[int, int, int], end: tuple[int, int, int]) -> None:
#     distance = sum(abs(start[index] - end[index]) for index in range(3))
#     if distance != 1:
#         raise TQECBlockAdapterError(
#             f"TQEC pipes require neighboring cubes, got segment {start!r} -> {end!r}."
#         )


# def _position(coord: tuple[int, int, int]) -> Position3D:
#     return Position3D(*coord)
