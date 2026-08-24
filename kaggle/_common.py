"""Shared helpers for the kaggle/ scripts."""
import json
from pathlib import Path


def kaggle_username() -> str:
    """Username from kaggle.json if present, else from the authenticated API
    (access-token auth, kaggle CLI >= 2.x)."""
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api.get_config_value("username")
