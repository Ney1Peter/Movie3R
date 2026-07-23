"""Keep Torch model and hub caches off the small home filesystem."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def default_torch_home() -> Path:
    configured = os.environ.get("TORCH_HOME")
    if configured:
        return Path(configured).expanduser()

    username = os.environ.get("USER") or Path.home().name
    data_home = Path("/data") / username / ".cache" / "torch"
    if data_home.exists() or os.access(data_home.parent, os.W_OK):
        return data_home
    return Path.home() / ".cache" / "torch"


def configure_torch_cache() -> Path:
    torch_home = default_torch_home()
    hub_dir = torch_home / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["TORCH_HUB_DIR"] = str(hub_dir)
    torch.hub.set_dir(str(hub_dir))
    return torch_home
