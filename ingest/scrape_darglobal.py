"""Scrape public DarGlobal content (darglobal.co.uk).

Discovery is sitemap-driven:
  * /projects/properties-in-<country>[/<type>] category pages -> project links + country
  * single-segment pages that look like projects (have an H1 + FAQ/amenities/location)
  * evergreen pages: about, faq, why-invest, hospitality, partners, ...
  * a capped sample of /blog, /insights, /press articles

Usage:  python -m ingest.scrape_darglobal [--max-articles 80] [--browser]
Output: data/raw/darglobal_projects.jsonl, data/raw/darglobal_articles.jsonl
"""
from __future__ import annotations

import argparse
import logging
import re
from urllib.parse import urlparse

from ingest.common import RAW_DIR, Fetcher, absolutize, parse_sitemap, write_jsonl
from ingest.extract import Page, parse
from ingest.schemas import Article, Project

BASE = "https://darglobal.co.uk"
log = logging.getLogger("ingest.darglobal")

COUNTRIES = {"uae": "UAE", "uk": "United Kingdom", "spain": "Spain", "oman": "Oman",
             "qatar": "Qatar", "saudi-arabia": "Saudi Arabia", "greece": "Greece", "usa": "USA",
             "maldives": "Maldives"}
EVERGREEN = ["about", "faq", "why-invest", "hospitality", "partners", "luxury-golf-communities",
             "development-management", "commercial", "investor", "dg-circle", "insignia",
             "exclusive-membership", "one-of-one", "tokenization", "projects",
             "blog", "insights", "press"]  # hub/index pages, not individual project pages
# Pages that are never useful for a property assistant
SKIP = re.compile(r"(privacy|terms|thank-you|success|internship|careers|pay-online|bookYourUnit|"
                  r"broker-contest|landing-page|new-home-page|win-a-trip|become-a|campaigns|"
                  r"aston-martin-media|get-in-touch|catalogue)", re.I)
BRANDS = ["Trump", "Lamborghini", "Pagani", "Aston Martin", "Missoni", "Elie Saab", "W Hotels",
          "Marriott", "Bentley", "Mulliner", "Roberto Cavalli", "Versace", "Zuhair Murad",
          "Swarovski", "Automobili Lamborghini"]
CITIES = ["Dubai", "Abu Dhabi", "Riyadh", "Jeddah", "Muscat", "Doha", "London", "Marbella",
          "Benahavís", "Estepona", "Costa del Sol", "Sohar", "Athens", "Aida"]


def slug(url: str) -> str:
    return urlparse(url).path.strip("/")


def classify(urls: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"category": [], "single": [], "blog": [], "insights": [],
                                    "press": [], "nested": []}
    for u in dict.fromkeys(urls):
        if urlparse(u).netloc and "darglobal" not in urlparse(u).netloc:
            continue
        s = slug(u)
        if not s or SKIP.search(s):
            continue
        first = s.split("/")[0]
        if first == "projects" and "/" in s:
            groups["category"].append(u)
        elif first in ("blog", "insights", "press") and "/" in s:
            groups[first].append(u)
        elif "/" not in s:
            groups["single"].append(u)
        else:
            groups["nested"].append(u)  # /rayana/..., /amaya/..., /partners/..., /one-of-one/...
    return groups


def looks_like_project(page: Page) -> bool:
    if not page.h1:
        return False
    text = page.text.lower()
    signals = sum(k in text for k in ("amenities", "location", "handover", "completion",
                                      "bedroom", "villa", "apartment", "residences", "payment plan",
                                      "why invest", "sqm", "floor plan"))
    return signals >= 3


def infer_country(page: Page, hint: str | None) -> str | None:
    """Content wins over the category-crawl hint: DarGlobal's per-country category pages
    (e.g. /projects/properties-in-uae/villas) all render the same global "related projects"
    widget, so the hint derived from which category page linked to a slug is unreliable."""
    blob = (page.h1 + " " + " ".join(page.facts.values()) + " " + page.text[:3000]).lower()
    for key, name in [("dubai", "UAE"), ("abu dhabi", "UAE"), ("riyadh", "Saudi Arabia"),
                      ("jeddah", "Saudi Arabia"), ("muscat", "Oman"), ("oman", "Oman"),
                      ("qatar", "Qatar"), ("doha", "Qatar"), ("spain", "Spain"),
                      ("marbella", "Spain"), ("london", "United Kingdom"), ("greece", "Greece")]:
        if key in blob:
            return name
    return hint


def to_project(page: Page, country_hint: str | None, types_hint: set[str]) -> Project:
    text = page.text
    brand = next((b for b in BRANDS if b.lower() in (page.h1 + " " + text[:4000]).lower()), None)
    location = page.facts.get("Location") or next(
        (c for c in CITIES if c.lower() in (page.h1 + " " + text[:3000]).lower()), None)
    status = page.facts.get("Status")
    if not status:
        m = re.search(r"\b(under development|under construction|completed|ready|sold out|off[- ]plan|"
                      r"pre-launch|launching soon)\b", text, re.I)
        status = m.group(1).title() if m else None
    types = set(types_hint)
    for t in ("villa", "apartment", "penthouse", "townhouse", "hotel room", "plot", "duplex"):
        if re.search(rf"\b{t}s?\b", text, re.I):
            types.add(t)
    return Project(
        source="darglobal", url=page.url, name=page.h1 or page.title, country=infer_country(page, country_hint),
        location=location, brand_partner=brand, status=status, property_types=sorted(types),
        facts=page.facts, summary=page.description, sections=page.sections, faqs=page.faqs,
    )


def to_article(page: Page, kind: str) -> Article:
    pub = None
    for item in page.jsonld:
        pub = item.get("datePublished") or pub
    return Article(source="darglobal", url=page.url, title=page.h1 or page.title, kind=kind,
                   published=pub, sections=page.sections, faqs=page.faqs)


def run(max_articles: int, browser: bool, no_cache: bool):
    f = Fetcher(delay=1.0, use_cache=not no_cache, browser_fallback=browser)
    urls = parse_sitemap(f, f"{BASE}/sitemap.xml")
    if not urls:
        log.warning("sitemap empty/unreachable; falling back to crawling /projects")
        urls = [f"{BASE}/projects"] + [f"{BASE}/{p}" for p in EVERGREEN]
    groups = classify(urls)
    log.info("sitemap: %s", {k: len(v) for k, v in groups.items()})

    # Category pages tell us which project slugs belong to which country / type.
    country_of: dict[str, str] = {}
    types_of: dict[str, set[str]] = {}
    categories = groups["category"] + [f"{BASE}/projects"]
    for cu in categories:
        html = f.get(cu)
        if not html:
            continue
        parts = slug(cu).split("/")
        country = COUNTRIES.get(parts[1].replace("properties-in-", "")) if len(parts) > 1 else None
        ptype = parts[2].rstrip("s").replace("-", " ") if len(parts) > 2 else None
        page = parse(html, cu)
        for link in absolutize(cu, page.links):
            s = slug(link)
            if link.startswith(BASE) and s and "/" not in s and not SKIP.search(s):
                if country:
                    country_of.setdefault(s.lower(), country)
                if ptype:
                    types_of.setdefault(s.lower(), set()).add(ptype)
                if link not in groups["single"]:
                    groups["single"].append(link)

    projects: list[Project] = []
    articles: list[Article] = []
    evergreen = {e.lower() for e in EVERGREEN}
    for u in groups["single"] + groups["nested"]:
        html = f.get(u)
        if not html:
            continue
        page = parse(html, u)
        s = slug(u).lower()
        if s in evergreen or s.split("/")[0] in evergreen:
            articles.append(to_article(page, "faq" if s == "faq" else "page"))
        elif s in country_of or looks_like_project(page):
            projects.append(to_project(page, country_of.get(s), types_of.get(s, set())))
        elif len(page.text) > 400:
            articles.append(to_article(page, "page"))

    # Capped sample of editorial content: newest first is unknown, so take an even spread.
    for kind in ("insights", "press", "blog"):
        pool = groups[kind]
        cap = max_articles // 2 if kind == "blog" else max_articles // 4
        step = max(1, len(pool) // max(cap, 1))
        for u in pool[::step][:cap]:
            html = f.get(u)
            if html:
                page = parse(html, u)
                if len(page.text) > 300:
                    articles.append(to_article(page, kind.rstrip("s") if kind != "press" else "press"))

    # De-duplicate projects that appear under two slugs (e.g. /amour-sans-detour and -2)
    uniq: dict[str, Project] = {}
    for p in projects:
        key = p.name.lower().strip()
        if key not in uniq or len(p.sections) > len(uniq[key].sections):
            uniq[key] = p
    n1 = write_jsonl(RAW_DIR / "darglobal_projects.jsonl", uniq.values())
    n2 = write_jsonl(RAW_DIR / "darglobal_articles.jsonl", articles)
    f.close()
    log.info("DarGlobal: %d projects, %d articles | http %s", n1, n2, f.stats)
    print(f"DarGlobal done: {n1} projects, {n2} articles. Fetch stats: {f.stats}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-articles", type=int, default=80)
    ap.add_argument("--browser", action="store_true", help="Playwright fallback on 403/429")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(a.max_articles, a.browser, a.no_cache)
