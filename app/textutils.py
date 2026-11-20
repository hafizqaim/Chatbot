"""Text helpers shared by ingestion and the app: normalisation and tokenising."""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "what", "which",
    "with", "me", "show", "tell", "about", "any", "do", "does", "you", "have", "i", "can", "please",
    "there", "at", "by", "it", "its", "be", "from", "that", "this", "how", "much", "many", "some",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(normalize(text))
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_words(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    out, step = [], size - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) < 40 and out:  # fold a tiny tail into the previous chunk
            out[-1] += " " + " ".join(piece)
            break
        out.append(" ".join(piece))
    return out
