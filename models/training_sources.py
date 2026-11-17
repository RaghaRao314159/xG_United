"""Pinned public sources shared by the training downloaders; never used for inference."""
import json
from pathlib import Path

MANIFEST = json.loads((Path(__file__).resolve().parents[1] / "training_data.lock.json").read_text())
IDSSE_REVISION = MANIFEST["idsse"]["revision"]
IDSSE_MIRROR = f"https://huggingface.co/datasets/pysport/idsse-data/resolve/{IDSSE_REVISION}/"
SKILLCORNER_REVISION = MANIFEST["skillcorner"]["revision"]
SKILLCORNER_MATCHES = tuple(MANIFEST["skillcorner"]["matches"])
SKILLCORNER_FILES = tuple(MANIFEST["skillcorner"]["files"])
