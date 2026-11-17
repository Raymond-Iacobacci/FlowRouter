from __future__ import annotations

from dataclasses import dataclass

try:
    from .pipe_rules import RULE_S
except ImportError:
    from pipe_rules import RULE_S

EXPORT_NODE_TYPES = {0, 1, 4, 5}
PIPE_RED = "#d0382a"
PIPE_BLUE = "#2b6ee6"
PIPE_ENDCAP = "#d7dbe3"

IJK_TO_CART = {"i": "x", "j": "y", "k": "z"}
CART_TO_IJK = {cart: ijk for ijk, cart in IJK_TO_CART.items()}
FACE_INDEX_PAIRS = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}
STEP_TO_IJK = {
    (1, 0, 0): "i",
    (-1, 0, 0): "i",
    (0, 1, 0): "j",
    (0, -1, 0): "j",
    (0, 0, 1): "k",
    (0, 0, -1): "k",
}
ORI_MAP = {
    ("i", 0, 0): "j",
    ("i", 0, 1): "k",
    ("i", 1, 0): "k",
    ("i", 1, 1): "j",
    ("j", 0, 0): "i",
    ("j", 0, 1): "k",
    ("j", 1, 0): "k",
    ("j", 1, 1): "i",
    ("k", 0, 0): "i",
    ("k", 0, 1): "j",
    ("k", 1, 0): "j",
    ("k", 1, 1): "i",
}


@dataclass(frozen=True)
class TracedPipeSegment:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    axis: str
    pipe_type: int


def face_indices_for_axis(axis: str) -> tuple[int, int]:
    return FACE_INDEX_PAIRS[axis]


def solid_face_colors(color: str) -> tuple[str, str, str, str, str, str]:
    return (color,) * 6


def axis_from_points(start, end) -> str:
    delta = tuple(int(round(end[index] - start[index])) for index in range(3))
    try:
        return IJK_TO_CART[STEP_TO_IJK[delta]]
    except KeyError as exc:
        raise ValueError(f"Invalid pipe segment delta {delta!r}.") from exc


def canonical_pipe_face_colors(
    axis: str,
    pipe_type: int,
    *,
    red_color: str = PIPE_RED,
    blue_color: str = PIPE_BLUE,
    endcap_color: str = PIPE_ENDCAP,
) -> tuple[str, str, str, str, str, str]:
    direction = CART_TO_IJK[axis]
    blue_axis = ORI_MAP[(direction, int(pipe_type), 0)]
    red_axis = next(candidate for candidate in ("i", "j", "k") if candidate not in {direction, blue_axis})

    colors = [endcap_color] * 6
    for face_index in FACE_INDEX_PAIRS[IJK_TO_CART[blue_axis]]:
        colors[face_index] = blue_color
    for face_index in FACE_INDEX_PAIRS[IJK_TO_CART[red_axis]]:
        colors[face_index] = red_color
    return tuple(colors)


def canonical_node_tqec(orientation: str, node_type: int) -> str | None:
    if node_type not in EXPORT_NODE_TYPES:
        return None

    tqec = []
    for axis in ("i", "j", "k"):
        if axis == orientation:
            tqec.append("X" if node_type == 1 else "Z")
        else:
            tqec.append("Z" if node_type == 1 else "X")
    return "".join(tqec)


def canonical_node_face_colors(
    orientation: str,
    node_type: int,
    *,
    red_color: str = PIPE_RED,
    blue_color: str = PIPE_BLUE,
) -> tuple[str, str, str, str, str, str] | None:
    tqec = canonical_node_tqec(orientation, node_type)
    if tqec is None:
        return None
    return canonical_node_face_colors_from_tqec(
        tqec,
        red_color=red_color,
        blue_color=blue_color,
    )


def canonical_node_face_colors_from_tqec(
    tqec: str,
    *,
    red_color: str = PIPE_RED,
    blue_color: str = PIPE_BLUE,
) -> tuple[str, str, str, str, str, str]:
    return tuple(
        color
        for basis in tqec
        for color in ([red_color] if basis == "X" else [blue_color]) * 2
    )


def canonical_pipe_face_colors_from_tqec(
    axis: str,
    tqec: str,
    *,
    red_color: str = PIPE_RED,
    blue_color: str = PIPE_BLUE,
    endcap_color: str = PIPE_ENDCAP,
) -> tuple[str, str, str, str, str, str]:
    colors = [endcap_color] * 6
    for other_axis in ("x", "y", "z"):
        if other_axis == axis:
            continue
        color = red_color if tqec["xyz".index(other_axis)] == "X" else blue_color
        for face_index in FACE_INDEX_PAIRS[other_axis]:
            colors[face_index] = color
    return tuple(colors)


def endpoint_pipe_face_colors(
    axis: str,
    orientation: str,
    node_face: int,
    *,
    red_color: str = PIPE_RED,
    blue_color: str = PIPE_BLUE,
    endcap_color: str = PIPE_ENDCAP,
) -> tuple[str, str, str, str, str, str]:
    pipe_type = RULE_S[(node_face, orientation, CART_TO_IJK[axis])]
    return canonical_pipe_face_colors(
        axis,
        pipe_type,
        red_color=red_color,
        blue_color=blue_color,
        endcap_color=endcap_color,
    )