"""Turn data/raw/*.jsonl into the read-only index the app ships with.

Outputs (data/index/):
  chunks.jsonl     text chunks with title/url/source metadata (BM25 + dense search)
  embeddings.npy   L2-normalised float32 matrix aligned with chunks.jsonl (optional)
  listings.db      SQLite table of Wasalt listings for structured filtering
  manifest.json    counts, snapshot date, embedding model

Usage:  python -m ingest.build_index [--no-dense]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from app.config import EMBED_MODEL, INDEX_DIR, MODEL_CACHE
from app.textutils import chunk_words, clean_ws
from ingest.common import RAW_DIR
from ingest.schemas import Article, Listing, Project, now_iso

SOURCE_NAME = {"darglobal": "DarGlobal", "wasalt": "Wasalt"}


def read(path: Path, model):
    if not path.exists():
        return []
    return [model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt_money(x: float) -> str:
    return f"{x:,.0f}"


class ChunkWriter:
    def __init__(self):
        self.chunks: list[dict] = []
        self._seen: set[str] = set()

    def add(self, text: str, *, title: str, url: str, source: str, kind: str, **meta):
        text = clean_ws(text)
        if len(text) < 40 or text in self._seen:
            return
        self._seen.add(text)
        self.chunks.append({"id": len(self.chunks), "text": text, "title": title, "url": url,
                            "source": source, "kind": kind, "meta": meta})


def project_chunks(cw: ChunkWriter, p: Project):
    owner = SOURCE_NAME[p.source]
    head = f"{owner} project: {p.name}."
    card = [head]
    for label, val in [("Country", p.country), ("Location", p.location), ("Brand partner", p.brand_partner),
                       ("Status", p.status), ("Property types", ", ".join(p.property_types))]:
        if val:
            card.append(f"{label}: {val}.")
    if p.facts:
        card.append("Key facts: " + "; ".join(f"{k}: {v}" for k, v in p.facts.items()) + ".")
    if p.summary:
        card.append(p.summary)
    meta = dict(project=p.name, country=p.country)
    cw.add(" ".join(card), title=p.name, url=p.url, source=p.source, kind="project", **meta)
    for s in p.sections:
        if s.heading.lower() == "key facts":
            continue
        prefix = f"{p.name} ({owner}{', ' + p.country if p.country else ''})"
        prefix += f" - {s.heading}: " if s.heading else ": "
        for piece in chunk_words(s.text, 200, 40):
            cw.add(prefix + piece, title=p.name, url=p.url, source=p.source, kind="project", **meta)
    for i in range(0, len(p.faqs), 4):
        qa = " ".join(f"Q: {f.q} A: {f.a}" for f in p.faqs[i:i + 4])
        cw.add(f"{p.name} FAQ. {qa}", title=f"{p.name} FAQ", url=p.url, source=p.source, kind="faq", **meta)


def article_chunks(cw: ChunkWriter, a: Article):
    owner = SOURCE_NAME[a.source]
    for s in a.sections:
        heading = s.heading if s.heading and s.heading != a.title else ""
        prefix = f"{owner} {a.kind}: {a.title}" + (f" - {heading}" if heading else "") + ". "
        for piece in chunk_words(s.text, 200, 40):
            cw.add(prefix + piece, title=a.title, url=a.url, source=a.source, kind=a.kind,
                   published=a.published)
    for i in range(0, len(a.faqs), 4):
        qa = " ".join(f"Q: {f.q} A: {f.a}" for f in a.faqs[i:i + 4])
        cw.add(f"{owner} FAQ ({a.title}). {qa}", title=a.title, url=a.url, source=a.source, kind="faq")


def catalog_chunks(cw: ChunkWriter, projects: list[Project]):
    by_country: dict[str, list[Project]] = defaultdict(list)
    for p in projects:
        if p.source == "darglobal":
            by_country[p.country or "Other / unspecified"].append(p)
    if not by_country:
        return
    lines = []
    for country, ps in sorted(by_country.items()):
        names = "; ".join(f"{p.name}" + (f" (with {p.brand_partner})" if p.brand_partner else "") for p in ps)
        lines.append(f"{country}: {names}.")
        cw.add(f"DarGlobal projects in {country}: {names}.", title=f"DarGlobal projects in {country}",
               url="https://darglobal.co.uk/projects", source="darglobal", kind="catalog", country=country)
    cw.add("Complete list of DarGlobal projects by country. " + " ".join(lines),
           title="DarGlobal project catalogue", url="https://darglobal.co.uk/projects",
           source="darglobal", kind="catalog")


def market_chunks(cw: ChunkWriter, listings: list[Listing], snapshot: str):
    groups: dict[tuple, list[Listing]] = defaultdict(list)
    for l in listings:
        if l.price and l.city:
            groups[(l.city, l.purpose, l.property_type or "property")].append(l)
            groups[(l.city, l.purpose, "property")].append(l) if l.property_type else None
    for (city, purpose, ptype), ls in sorted(groups.items()):
        if len(ls) < 3:
            continue
        prices = sorted(l.price for l in ls)
        areas = [l.area_sqm for l in ls if l.area_sqm]
        ppsqm = [l.price / l.area_sqm for l in ls if l.area_sqm and l.area_sqm > 10]
        label = f"{ptype}s" if ptype != "property" else "properties (all types)"
        period = " (rent figures are as listed; most Saudi rentals are quoted per year)" if purpose == "rent" else ""
        text = (f"Wasalt market snapshot: {label} for {purpose} in {city}. Based on {len(ls)} listings scraped "
                f"on {snapshot}: price range {fmt_money(prices[0])} to {fmt_money(prices[-1])} SAR, "
                f"median {fmt_money(statistics.median(prices))} SAR")
        if areas:
            text += f", median size {statistics.median(areas):,.0f} sqm"
        if ppsqm and purpose == "sale":
            text += f", median price per sqm {fmt_money(statistics.median(ppsqm))} SAR"
        text += f".{period} These are asking prices on listings, not valuations."
        cw.add(text, title=f"Wasalt {label} for {purpose} in {city}",
               url=f"https://wasalt.sa/en/properties-for-{purpose}-in-{city.lower().replace(' ', '-')}",
               source="wasalt", kind="market", city=city, purpose=purpose)


def write_listings_db(path: Path, listings: list[Listing]):
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE listings (
        id TEXT PRIMARY KEY, url TEXT, title TEXT, purpose TEXT, property_type TEXT, city TEXT,
        district TEXT, price REAL, rent_period TEXT, area_sqm REAL, bedrooms INTEGER,
        bathrooms INTEGER, scraped_at TEXT)""")
    con.executemany(
        "INSERT OR REPLACE INTO listings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(l.id, l.url, l.title, l.purpose, l.property_type, l.city, l.district, l.price, l.rent_period,
          l.area_sqm, l.bedrooms, l.bathrooms, l.scraped_at) for l in listings])
    for col in ("purpose", "property_type", "city", "price", "bedrooms"):
        con.execute(f"CREATE INDEX idx_{col} ON listings({col})")
    con.commit()
    con.close()


def embed(chunks: list[dict], model_name: str):
    import numpy as np
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name, cache_dir=str(MODEL_CACHE))
    prefix = "passage: " if "e5" in model_name.lower() else ""
    texts = [prefix + c["text"] for c in chunks]
    vecs = np.array(list(model.embed(texts, batch_size=64)), dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return vecs


def main(no_dense: bool):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    projects = read(RAW_DIR / "darglobal_projects.jsonl", Project) + read(RAW_DIR / "wasalt_projects.jsonl", Project)
    articles = read(RAW_DIR / "darglobal_articles.jsonl", Article) + read(RAW_DIR / "wasalt_articles.jsonl", Article)
    listings = read(RAW_DIR / "wasalt_listings.jsonl", Listing)
    if not (projects or articles or listings):
        raise SystemExit("No raw data in data/raw/. Run the scrapers first (see README).")
    snapshot = max([r.scraped_at for r in [*projects, *articles, *listings]], default=now_iso())

    cw = ChunkWriter()
    catalog_chunks(cw, projects)
    for p in projects:
        project_chunks(cw, p)
    for a in articles:
        article_chunks(cw, a)
    market_chunks(cw, listings, snapshot)

    with (INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in cw.chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    write_listings_db(INDEX_DIR / "listings.db", listings)

    model_used = None
    emb_path = INDEX_DIR / "embeddings.npy"
    if not no_dense:
        import numpy as np
        try:
            np.save(emb_path, embed(cw.chunks, EMBED_MODEL))
            model_used = EMBED_MODEL
        except Exception as e:  # noqa: BLE001
            print(f"Dense embeddings skipped ({e}); the app will use keyword search only.")
    if model_used is None:
        emb_path.unlink(missing_ok=True)

    manifest = {
        "snapshot_date": snapshot, "embed_model": model_used, "chunks": len(cw.chunks),
        "projects": {"darglobal": sum(p.source == "darglobal" for p in projects),
                     "wasalt": sum(p.source == "wasalt" for p in projects)},
        "articles": {"darglobal": sum(a.source == "darglobal" for a in articles),
                     "wasalt": sum(a.source == "wasalt" for a in articles)},
        "listings": len(listings), "listings_with_price": sum(1 for l in listings if l.price),
        "project_names": sorted({p.name for p in projects if p.source == "darglobal"}),
    }
    (INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "project_names"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    main(ap.parse_args().no_dense)
