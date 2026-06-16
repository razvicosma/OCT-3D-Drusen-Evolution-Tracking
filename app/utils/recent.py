import json
import os
from datetime import datetime

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")
MAX_RECENT = 3

_defaults = {
    "reconstruction": True,
    "segmentation": True,
    "model": "dinov3",
    "recent": [],
}


def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
        return {**_defaults, **data}
    return dict(_defaults)


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
