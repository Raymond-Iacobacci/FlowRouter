from __future__ import annotations

# Initialization rule applied at the first detected direction.
RULE_S = {
    (0, "i", "j"): 0,
    (0, "i", "k"): 0,
    (0, "j", "i"): 0,
    (0, "j", "k"): 1,
    (0, "k", "i"): 1,
    (0, "k", "j"): 1,
    (1, "i", "j"): 1,
    (1, "i", "k"): 1,
    (1, "j", "i"): 1,
    (1, "j", "k"): 0,
    (1, "k", "i"): 0,
    (1, "k", "j"): 0,
}

# General transition table for subsequent direction changes.
RULES = {
    (0, "i", "j"): 0,
    (0, "i", "k"): 1,
    (1, "i", "j"): 1,
    (1, "i", "k"): 0,
    (0, "j", "i"): 0,
    (0, "j", "k"): 0,
    (1, "j", "i"): 1,
    (1, "j", "k"): 1,
    (0, "k", "i"): 1,
    (0, "k", "j"): 0,
    (1, "k", "i"): 0,
    (1, "k", "j"): 1,
}

AXIS_MAP = {
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