from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "floorplan.schema.json"


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def validate(plan: dict) -> None:
    """Raise jsonschema.ValidationError if plan does not match the published schema."""
    jsonschema.validate(plan, load_schema(), cls=jsonschema.Draft202012Validator)
    _check_intervals(plan)


def _check_intervals(node: object, path: str = "$") -> None:
    # Schema can't express ci_low <= value <= ci_high; enforce it here.
    if isinstance(node, dict):
        if {"value", "ci_low", "ci_high"} <= node.keys():
            if not node["ci_low"] <= node["value"] <= node["ci_high"]:
                raise ValueError(f"interval does not contain value at {path}: {node}")
        for key, child in node.items():
            _check_intervals(child, f"{path}.{key}")
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _check_intervals(child, f"{path}[{index}]")
