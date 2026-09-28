"""Pinned FastPLMs checkpoint and tensor report conversion."""
from __future__ import annotations

from typing import Any

import torch

MODEL_ID = "Synthyra/Boltz2"
REVISION = "bce98f7ce914d468182726b5a0fbd23737167875"


def first_float(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.reshape(-1)[0])


def to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return float(value) if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
