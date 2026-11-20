"""Retrieval: hybrid BM25 + dense search over text chunks, and SQL over listings."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import statistics
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app import config
from app.filters import Filters
from app.textutils import normalize, tokenize

log = logging.getLogger("app.search")


@dataclass
class Hit:
    chunk: dict
    score: float


class Index:
    def __init__(self, index_dir=config.INDEX_DIR):
        self.dir = index_dir
        self.manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.chunks = [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l]
        self.bm25 = BM25Okapi([tokenize(c["title"] + " " + c["text"]) for c in self.chunks])
        self.project_names = {normalize(c["meta"]["project"]): c["meta"]["project"]
                              for c in self.chunks if c["meta"].get("project")}
        self.db_path = index_dir / "listings.db"
        self.districts = self._load_districts()
        self.emb: Optional[np.ndarray] = None
        self.embedder = None
        if config.USE_DENSE and self.manifest.get("embed_model") and (index_dir / "embeddings.npy").exists():
            try:
                from fastembed import TextEmbedding
                self.embedder = TextEmbedding(self.manifest["embed_model"], cache_dir=str(config.MODEL_CACHE))
                self.emb = np.load(index_dir / "embeddings.npy")
                self._embed_query("warm up")
                log.info("dense search on (%s)", self.manifest["embed_model"])
            except Exception as e:  # noqa: BLE001 - never fail startup over the optional path
                log.warning("dense search disabled: %s", e)
                self.embedder, self.emb = None, None

    # ------------------------------------------------------------- text chunks
    def _embed_query(self, q: str) -> np.ndarray:
        prefix = "query: " if "e5" in self.manifest["embed_model"].lower() else ""
        v = np.array(next(iter(self.embedder.embed([prefix + q]))), dtype="float32")
        return v / (np.linalg.norm(v) + 1e-9)

    def search(self, query: str, k: int = config.TOP_K_CHUNKS, prefer: Optional[str] = None) -> list[Hit]:
        q = query
        qn = normalize(q)
        ranks: dict[int, float] = {}
        bm = self.bm25.get_scores(tokenize(q))
        for r, i in enumerate(np.argsort(-bm)[:40]):
            if bm[i] > 0:
                ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        if self.embedder is not None:
            sims = self.emb @ self._embed_query(q)
            for r, i in enumerate(np.argsort(-sims)[:40]):
                ranks[i] = ranks.get(i, 0) + 1 / (60 + r)

        mentioned = {name for key, name in self.project_names.items() if len(key) > 3 and key in qn}
        for i in list(ranks):
            c = self.chunks[i]
            if c["meta"].get("project") in mentioned:
                ranks[i] *= 1.6
            if prefer and c["source"] == prefer:
                ranks[i] *= 1.25
            if c["kind"] == "catalog" and re.search(r"\b(projects?|portfolio|developments?|which|list|all)\b", qn):
                ranks[i] *= 1.4
            if c["kind"] in ("blog", "press") and mentioned:
                ranks[i] *= 0.8

        out, per_url = [], {}
        for i, s in sorted(ranks.items(), key=lambda kv: -kv[1]):
            url = self.chunks[i]["url"]
            if per_url.get(url, 0) >= 3:
                continue
            per_url[url] = per_url.get(url, 0) + 1
            out.append(Hit(self.chunks[i], s))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------- listings
    def _load_districts(self) -> dict[str, str]:
        if not self.db_path.exists():
            return {}
        con = sqlite3.connect(self.db_path)
        rows = con.execute("SELECT DISTINCT district FROM listings WHERE district IS NOT NULL").fetchall()
        con.close()
        out = {}
        for (d,) in rows:
            key = normalize(d).replace("-", " ")
            out[key] = d
            out[re.sub(r"^al ", "", key)] = d  # "Malaz" matches "Al-Malaz"
        return out

    def listings(self, f: Filters, k: int = config.TOP_K_LISTINGS) -> dict:
        """Filtered listings + stats. Relaxes filters step by step if nothing matches."""
        if not self.db_path.exists():
            return {"rows": [], "count": 0, "relaxed": []}
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        order = {"price_asc": "price ASC", "price_desc": "price DESC",
                 "area_desc": "area_sqm DESC"}.get(f.sort or "", "price IS NULL, scraped_at DESC, price ASC")

        def run(params: dict):
            where, args = ["1=1"], []
            for col in ("purpose", "property_type", "city"):
                if params.get(col):
                    where.append(f"lower({col}) = lower(?)")
                    args.append(params[col])
            if params.get("district"):
                where.append("lower(district) LIKE lower(?)")
                args.append(f"%{params['district'].replace('Al-', '').replace('Al ', '')}%")
            if params.get("bedrooms") is not None:
                where.append("bedrooms = ?"); args.append(params["bedrooms"])
            if params.get("min_bedrooms") is not None:
                where.append("bedrooms >= ?"); args.append(params["min_bedrooms"])
            if params.get("price_max") is not None:
                where.append("price <= ?"); args.append(params["price_max"])
            if params.get("price_min") is not None:
                where.append("price >= ?"); args.append(params["price_min"])
            sql = " AND ".join(where)
            rows = con.execute(f"SELECT * FROM listings WHERE {sql} ORDER BY {order} LIMIT ?", [*args, k]).fetchall()
            prices = [r[0] for r in con.execute(f"SELECT price FROM listings WHERE {sql} AND price IS NOT NULL", args)]
            count = con.execute(f"SELECT COUNT(*) FROM listings WHERE {sql}", args).fetchone()[0]
            return rows, prices, count

        params = f.active()
        relaxed: list[str] = []
        rows, prices, count = run(params)
        for drop in ("bedrooms", "min_bedrooms", "district", "price_min", "price_max", "property_type"):
            if count or drop not in params:
                continue
            params.pop(drop)
            relaxed.append(drop)
            rows, prices, count = run(params)
        con.close()
        stats = None
        if prices:
            stats = {"min": min(prices), "median": statistics.median(prices), "max": max(prices), "n": len(prices)}
        return {"rows": [dict(r) for r in rows], "count": count, "stats": stats, "relaxed": relaxed,
                "applied": params}
