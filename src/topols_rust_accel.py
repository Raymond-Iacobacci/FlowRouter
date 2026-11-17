"""Optional Rust acceleration for the compiler's order-sensitive Dinic kernel.

The shared library has no third-party Rust dependencies and is built lazily on
first use.  If Cargo is unavailable, compilation fails, or the native function
rejects its inputs, callers transparently keep using the Python implementation.
Set ``TOPOLS_RUST_ACCEL=0`` to force that reference path for comparisons.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from array import array
from pathlib import Path


_DISABLED = os.environ.get("TOPOLS_RUST_ACCEL", "1") == "0"
_LOAD_ATTEMPTED = False
_LIBRARY = None


def _library_path(crate: Path) -> Path:
    if sys.platform == "darwin":
        filename = "libtopols_dinic.dylib"
    elif os.name == "nt":
        filename = "topols_dinic.dll"
    else:
        filename = "libtopols_dinic.so"
    return crate / "target" / "release" / filename


def _load_library():
    global _LOAD_ATTEMPTED, _LIBRARY
    if _LOAD_ATTEMPTED or _DISABLED:
        return _LIBRARY
    _LOAD_ATTEMPTED = True

    crate = Path(__file__).resolve().parents[1] / "rust" / "topols_dinic"
    library_path = _library_path(crate)
    sources = (crate / "Cargo.toml", crate / "src" / "lib.rs")
    needs_build = not library_path.exists() or any(
        source.stat().st_mtime_ns > library_path.stat().st_mtime_ns for source in sources
    )
    cargo = shutil.which("cargo")
    if needs_build and cargo:
        try:
            subprocess.run(
                [cargo, "build", "--release", "--manifest-path", str(crate / "Cargo.toml")],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
    if not library_path.exists():
        return None

    try:
        library = ctypes.CDLL(str(library_path))
        native_run = library.topols_dinic_run
    except (AttributeError, OSError):
        return None
    native_run.argtypes = (
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int64),
    )
    native_run.restype = ctypes.c_size_t
    _LIBRARY = library
    return library


class NativeDinic:
    """Flattened, reusable view of one NetworkX residual graph."""

    def __init__(self, graph, source, sink, node_rank):
        library = _load_library()
        if library is None:
            raise RuntimeError("Rust accelerator unavailable")

        self._library = library
        self.nodes = tuple(graph.nodes)
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.source = self.node_index[source]
        self.sink = self.node_index[sink]

        edge_sources = []
        edge_targets = []
        capacities = []
        flows = []
        self._edge_data = []
        self._edge_index_by_data_id = {}
        self._graph_succ = graph._succ
        for u, neighbours in graph._succ.items():
            ui = self.node_index[u]
            for v, data in neighbours.items():
                edge_index = len(edge_sources)
                edge_sources.append(ui)
                edge_targets.append(self.node_index[v])
                capacities.append(int(data["capacity"]))
                flows.append(int(data["flow"]))
                self._edge_data.append(data)
                # Python adjusts only a few accepted/repaired paths between
                # native trials.  Indexing by the existing edge-data object's
                # integer identity avoids allocating and hashing a nested
                # ((kind, coord), (kind, coord)) tuple for every graph edge.
                self._edge_index_by_data_id[id(data)] = edge_index

        self.edge_count = len(edge_sources)
        # ctypes' ``CArray(*python_list)`` converts every element through the
        # argument marshaller.  ``array`` builds the same contiguous native
        # buffers in C and from_buffer exposes them without a second copy.
        self._source_buffer = array("I", edge_sources)
        self._target_buffer = array("I", edge_targets)
        self._capacity_buffer = array("q", capacities)
        self._flow_buffer = array("q", flows)
        self._blocked_buffer = bytearray(len(self.nodes))
        self._sources = (ctypes.c_uint32 * self.edge_count).from_buffer(self._source_buffer)
        self._targets = (ctypes.c_uint32 * self.edge_count).from_buffer(self._target_buffer)
        self._capacities = (ctypes.c_int64 * self.edge_count).from_buffer(self._capacity_buffer)
        self._flows = (ctypes.c_int64 * self.edge_count).from_buffer(self._flow_buffer)
        self._blocked = (ctypes.c_uint8 * len(self.nodes)).from_buffer(self._blocked_buffer)
        flat_ranks = []
        self._cube_by_node = []
        for node in self.nodes:
            flat_ranks.extend(node_rank(node))
            self._cube_by_node.append(
                node[1]
                if isinstance(node, tuple) and len(node) == 2 and node[0] in ("in", "out")
                else None
            )
        self._rank_buffer = array("q", flat_ranks)
        self._ranks = (ctypes.c_int64 * len(flat_ranks)).from_buffer(self._rank_buffer)
        self._last_blocked_cubes = frozenset()

    @classmethod
    def create(cls, graph, source, sink, node_rank):
        try:
            return cls(graph, source, sink, node_rank)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def max_flow(self, blocked_cubes) -> int | None:
        frozen_blocked = frozenset(blocked_cubes)
        if frozen_blocked != self._last_blocked_cubes:
            for index, cube in enumerate(self._cube_by_node):
                self._blocked[index] = int(cube is not None and cube in frozen_blocked)
            self._last_blocked_cubes = frozen_blocked

        count = self._library.topols_dinic_run(
            len(self.nodes),
            self.edge_count,
            self.source,
            self.sink,
            self._sources,
            self._targets,
            self._capacities,
            self._flows,
            self._blocked,
            self._ranks,
        )
        if count == ctypes.c_size_t(-1).value:
            return None
        for data, flow in zip(self._edge_data, self._flows):
            data["flow"] = int(flow)
        return int(count)

    def adjust_path(self, path, amount) -> None:
        """Mirror a Python-side residual update into the native flow buffer."""
        delta = int(amount)
        for u, v in zip(path, path[1:]):
            forward = self._edge_index_by_data_id[id(self._graph_succ[u][v])]
            reverse = self._edge_index_by_data_id[id(self._graph_succ[v][u])]
            self._flows[forward] += delta
            self._flows[reverse] -= delta
