"""Runtime configuration (all overridable with environment variables)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "data" / "index"))
MODEL_CACHE = Path(os.getenv("MODEL_CACHE", ROOT / "models"))

# Small quantised ONNX model (~70 MB) that fits Render's 512 MB free instance.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
USE_DENSE = os.getenv("USE_DENSE", "1") == "1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Tried in order; only :free models (plus the free router) so nothing is ever billed.
# Free models rotate often - run `python scripts/list_free_models.py` to refresh this list.
OPENROUTER_MODELS = [m.strip() for m in os.getenv(
    "OPENROUTER_MODELS",
    "google/gemma-4-31b-it:free,nvidia/nemotron-3-ultra-550b-a55b:free,openrouter/free",
).split(",") if m.strip()]
APP_URL = os.getenv("APP_URL", "https://darglobal-chatbot.onrender.com")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "6"))
TOP_K_LISTINGS = int(os.getenv("TOP_K_LISTINGS", "8"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "12"))
