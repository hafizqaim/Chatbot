"""Pre-LLM guardrails: block prompt-injection/jailbreak attempts and clearly disallowed
content before a message reaches the model or spends free-tier quota. Defense-in-depth on
top of the system prompt's own rules (app/prompts.py), not a replacement for them."""
from __future__ import annotations

import re

from app.textutils import normalize

_INJECTION_RE = re.compile(
    r"""
    \bignore\s+(all\s+|any\s+|the\s+|previous\s+|prior\s+|above\s+)*(instructions|rules|prompt|context)\b |
    \bdisregard\s+(your\s+|the\s+|all\s+|previous\s+)*(instructions|rules|prompt)\b |
    \boverride\s+(your\s+|the\s+)*(instructions|rules|prompt)\b |
    \byou\s+are\s+(now|no\s+longer)\b |
    \bact\s+as\s+(if\s+you|a)\b |
    \b(reveal|repeat|print|show|output)\s+(your|the)\s+(system\s+prompt|instructions|rules)\b |
    \bsystem\s+prompt\b |
    \bjailbreak\b | \bdan\s+mode\b | \bdeveloper\s+mode\b | \bdo\s+anything\s+now\b |
    \bnew\s+instructions\s*:
    """,
    re.I | re.X,
)

_DISALLOWED_RE = re.compile(
    r"""
    \bhow\s+to\s+(make|build)\s+a\s+(bomb|explosive|weapon)\b |
    \b(kill|hurt)\s+(myself|yourself)\b | \bsuicide\s+method\b |
    \bchild\s+(sexual|abuse|porn) |
    \bhow\s+to\s+hack\s+(into\s+)?(a|someone)\b |
    \bcredit\s+card\s+number\s+generator\b
    """,
    re.I | re.X,
)

INJECTION_REPLY = (
    "I can't follow instructions embedded in a message like that. I'm only able to answer "
    "questions about DarGlobal projects and Wasalt listings -- feel free to ask about those."
)
DISALLOWED_REPLY = (
    "I can't help with that. I'm a property assistant for DarGlobal and Wasalt -- ask me about "
    "projects, listings, prices or areas instead."
)


def check_input(message: str) -> str | None:
    """Return a canned refusal if `message` should never reach the LLM, else None."""
    norm = normalize(message)
    if _DISALLOWED_RE.search(norm):
        return DISALLOWED_REPLY
    if _INJECTION_RE.search(norm):
        return INJECTION_REPLY
    return None
