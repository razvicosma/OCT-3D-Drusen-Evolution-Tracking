import os
import json
from datetime import datetime

from app.config import SETTINGS_PATH, MAX_RECENT, MODEL_BASE

DEFAULT_SETTINGS = {
    "reconstruction": True,
    "segmentation": True,
    "model": MODEL_BASE,
    "recent": [],
}


def load_settings():

    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
        return {**DEFAULT_SETTINGS, **data}
    return dict(DEFAULT_SETTINGS)


def save_settings(settings):

    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def add_recent(settings, path, kind):

    recent = settings.get("recent", [])
    recent = [r for r in recent if r["path"] != path]
    recent.insert(0, {
        "path": path,
        "kind": kind,
        "timestamp": datetime.now().isoformat(),
    })
    settings["recent"] = recent[:MAX_RECENT]
    save_settings(settings)
