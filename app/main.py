"""FastAPI app: chat UI, streaming /api/chat endpoint, health check."""
from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from collections import OrderedDict, defaultdict, deque
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import config
from app.filters import Filters, extract, merge_followup
from app.guardrails import check_input
from app.llm import LLMUnavailable, stream_answer
from app.prompts import build_context, build_messages
from app.search import Index
from app.textutils import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")

INDEX: Optional[Index] = None


@asynccontextmanager
async def lifespan(_app):
    global INDEX
    t = time.time()
    INDEX = Index()
    log.info("index loaded: %d chunks, dense=%s in %.1fs", len(INDEX.chunks), INDEX.embedder is not None,
             time.time() - t)
    yield


app = FastAPI(title="DarGlobal & Wasalt Property Assistant", docs_url=None, redoc_url=None, lifespan=lifespan)
STATIC = config.ROOT / "app" / "static"

SUGGESTIONS = [
    "What projects does DarGlobal have in Oman?",
    "Tell me about Tierra Viva",
    "Which DarGlobal projects are partnered with Trump?",
    "3 bedroom apartments for sale in Riyadh under 1.2M",
    "Average rent for villas in Jeddah",
    "What is Wasalt?",
]


# ------------------------------------------------------------------ guards
_hits: dict[str, deque] = defaultdict(deque)


def rate_limited(ip: str) -> bool:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_PER_MIN:
        return True
    q.append(now)
    return False


class AnswerCache:
    """Caches first-turn answers so repeat questions don't spend free-model quota."""

    def __init__(self, size: int = 200, ttl: int = 6 * 3600):
        self.size, self.ttl, self.d = size, ttl, OrderedDict()

    def get(self, key: str):
        v = self.d.get(key)
        if v and time.time() - v[0] < self.ttl:
            self.d.move_to_end(key)
            return v[1]
        self.d.pop(key, None)
        return None

    def put(self, key: str, value):
        self.d[key] = (time.time(), value)
        self.d.move_to_end(key)
        while len(self.d) > self.size:
            self.d.popitem(last=False)


CACHE = AnswerCache()


# ------------------------------------------------------------------ routes
class Turn(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=40)


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(INDEX.chunks) if INDEX else 0}


@app.get("/api/info")
def info():
    m = INDEX.manifest
    return {"snapshot_date": m.get("snapshot_date"), "projects": m.get("projects"), "articles": m.get("articles"),
            "listings": m.get("listings"), "dense": INDEX.embedder is not None,
            "models": config.OPENROUTER_MODELS, "suggestions": SUGGESTIONS,
            "llm_configured": bool(config.OPENROUTER_API_KEY)}


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def plan(message: str, history: list[Turn]):
    """Retrieval for one turn: filters (with follow-up memory), chunks, listings."""
    prev_user = next((t.content for t in reversed(history) if t.role == "user"), None)
    prev_f: Optional[Filters] = extract(prev_user, INDEX.districts) if prev_user else None
    f = merge_followup(extract(message, INDEX.districts), prev_f, message)

    query = message
    if prev_user and len(message.split()) <= 6 and not f.listing_intent:
        query = f"{prev_user} {message}"  # "when is it completed?" needs the previous subject
    prefer = "darglobal" if f.darglobal_intent else ("wasalt" if f.listing_intent else None)
    hits = INDEX.search(query, k=config.TOP_K_CHUNKS if not f.listing_intent else 4, prefer=prefer)
    lr = INDEX.listings(f) if f.listing_intent else None
    return f, hits, lr


def sources_payload(hits, lr, f):
    srcs, seen = [], set()
    for i, h in enumerate(hits, 1):
        c = h.chunk
        srcs.append({"n": i, "title": c["title"], "url": c["url"], "source": c["source"], "kind": c["kind"]})
        seen.add(c["url"])
    listings = []
    if lr and f and f.listing_intent:
        for j, r in enumerate(lr["rows"], 1):
            listings.append({"n": f"L{j}", "title": r["title"], "url": r["url"], "price": r["price"],
                             "rent_period": r.get("rent_period"), "area_sqm": r.get("area_sqm"),
                             "bedrooms": r.get("bedrooms"), "city": r.get("city"), "district": r.get("district")})
    return {"sources": srcs, "listings": listings,
            "filters": f.active() if f and f.listing_intent else None,
            "listing_count": lr["count"] if lr else None, "relaxed": lr["relaxed"] if lr else []}


def fallback_answer(message: str, hits, lr, f) -> str:
    """Used when no API key is set or every free model is down: retrieval-only reply."""
    lines = ["⚠️ The language model is unavailable right now, so here are the closest matches from the data:"]
    if lr and f.listing_intent and lr["rows"]:
        lines.append(f"\n**{lr['count']} matching Wasalt listings** (first {len(lr['rows'])}):")
        for j, r in enumerate(lr["rows"], 1):
            price = f"{r['price']:,.0f} SAR" if r["price"] else "price n/a"
            lines.append(f"- [L{j}] {r['title']} — {price}")
    for i, h in enumerate(hits[:4], 1):
        lines.append(f"- [{i}] **{h.chunk['title']}**: {h.chunk['text'][:220]}…")
    return "\n".join(lines)


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if rate_limited(ip):
        raise HTTPException(429, "Too many messages, please wait a minute.")
    message = body.message.strip()
    history = body.history[-config.HISTORY_TURNS * 2:]

    blocked = check_input(message)
    if blocked:
        log.info("guardrail blocked a message from %s", ip)

        async def blocked_gen():
            yield sse("meta", {"sources": [], "listings": [], "filters": None, "listing_count": None,
                               "relaxed": []})
            yield sse("delta", {"text": blocked})
            yield sse("done", {})

        return StreamingResponse(blocked_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    f, hits, lr = plan(message, history)
    meta = sources_payload(hits, lr, f)
    cache_key = normalize(message) if not history else None

    async def gen():
        yield sse("meta", meta)
        if cache_key and (cached := CACHE.get(cache_key)):
            yield sse("model", {"model": cached["model"] + " (cached)"})
            yield sse("delta", {"text": cached["text"]})
            yield sse("done", {})
            return
        if not config.OPENROUTER_API_KEY:
            yield sse("delta", {"text": fallback_answer(message, hits, lr, f)})
            yield sse("done", {})
            return
        context = build_context(hits, lr, f)
        msgs = build_messages(message, [t.model_dump() for t in history], context,
                              INDEX.manifest.get("snapshot_date", "recently"))
        parts, model = [], ""
        try:
            async for kind, val in stream_answer(msgs):
                if kind == "model":
                    model = val
                    yield sse("model", {"model": val})
                else:
                    parts.append(val)
                    yield sse("delta", {"text": val})
        except LLMUnavailable as e:
            log.error("all models failed: %s", e)
            yield sse("delta", {"text": fallback_answer(message, hits, lr, f)})
        else:
            if cache_key and parts:
                CACHE.put(cache_key, {"text": "".join(parts), "model": model})
        yield sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
