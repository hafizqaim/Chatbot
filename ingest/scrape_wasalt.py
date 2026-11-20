"""Scrape public Wasalt content (wasalt.sa + blog.wasalt.sa).

  * Listings: server-rendered SEO search pages (/en/properties-for-<sale|rent>-in-<place>?page=N).
    Cards are parsed from the page; we never hit /search (disallowed in robots.txt).
  * Projects: /en/projects (+ project detail pages it links to), best effort.
  * Knowledge: wasalt.sa overview pages + Wasalt blog posts via the WordPress REST API.

Usage:  python -m ingest.scrape_wasalt [--max-listings 1500] [--max-posts 120] [--browser]
Output: data/raw/wasalt_listings.jsonl, wasalt_projects.jsonl, wasalt_articles.jsonl
"""
from __future__ import annotations

import argparse
import html as htmllib
import logging
import re
from typing import Any, Iterable, Optional
from urllib.parse import urlparse

from selectolax.lexbor import LexborHTMLParser as HTMLParser, LexborNode as Node

from app.textutils import clean_ws
from ingest.common import RAW_DIR, Fetcher, absolutize, write_jsonl
from ingest.extract import parse
from ingest.schemas import Article, Listing, Project, Section

BASE = "https://wasalt.sa"
BLOG = "https://blog.wasalt.sa"
log = logging.getLogger("ingest.wasalt")

CITIES = ["riyadh", "jeddah", "dammam", "khobar", "makkah", "madinah", "taif", "abha", "tabuk",
          "dhahran", "jubail", "buraidah", "khamis-mushait", "al-ahsa", "hail", "yanbu", "jazan",
          "najran", "qatif", "al-kharj"]
CITY_NAMES = {c.replace("-", " "): c.replace("-", " ").title() for c in CITIES}
CITY_NAMES.update({"mecca": "Makkah", "medina": "Madinah", "al khobar": "Khobar",
                   "hofuf": "Al Ahsa", "al hofuf": "Al Ahsa"})
TYPES = ["apartment", "villa", "floor", "land", "building", "duplex", "townhouse", "office",
         "shop", "warehouse", "rest house", "chalet", "room", "studio", "compound", "farm",
         "showroom", "penthouse", "palace", "istiraha"]
LISTING_HREF = re.compile(r"/property/(sale|rent)/([^/?#]+?)-(\d{4,})(?:[/?#]|$)")
SEO_PAGE = re.compile(r"^/en/[a-z0-9-]+-for-(sale|rent)-in-[a-z0-9-]+/?$")


# ----------------------------------------------------------------- listing cards
def _num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def parse_card_text(text: str, purpose: str, url: str, lid: str, city_hint: Optional[str]) -> Listing:
    t = clean_ws(text)
    low = t.lower()
    title_m = re.search(rf"\b({'|'.join(TYPES)})\b[^|]{{0,80}}?(?:sqm|bedrooms?|width street)[^|]{{0,40}}",
                        t, re.I)
    title = clean_ws(title_m.group(0)) if title_m else t[:90]
    ptype = next((ty for ty in TYPES if re.search(rf"\b{ty}\b", title, re.I)), None) or \
        next((ty for ty in TYPES if re.search(rf"\b{ty}\b", low)), None)
    area_m = re.search(r"([\d][\d,]*(?:\.\d+)?)\s*(?:sqm|sq\.?\s*m|m²|m2)\b", t, re.I)
    area = _num(area_m.group(1)) if area_m else None
    beds_m = re.search(r"(\d{1,2})\s*(?:bed(?:room)?s?|br)\b", t, re.I)
    baths_m = re.search(r"(\d{1,2})\s*(?:bath(?:room)?s?)\b", t, re.I)

    # price: prefer an explicit "SAR 1,200,000"/"1,200,000 SAR", else the largest comma-grouped number
    price = None
    pm = re.search(r"sar\s*([\d,]+(?:\.\d+)?)|([\d,]{5,}(?:\.\d+)?)\s*sar", t, re.I)
    if pm:
        price = _num(pm.group(1) or pm.group(2))
    if price is None:
        cands = [_num(x) for x in re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", t)]
        cands = [c for c in cands if c and (area is None or abs(c - area) > 1)]
        price = max(cands) if cands else None
    period = None
    if purpose == "rent":
        pm2 = re.search(r"\b(yearly|annually|per year|/\s*year|monthly|per month|/\s*month|daily|"
                        r"per night|/\s*night)\b", low)
        if pm2:
            w = pm2.group(1)
            period = "monthly" if "month" in w else "daily" if ("da" in w or "night" in w) else "yearly"

    city, district = None, None
    for name_low, name in sorted(CITY_NAMES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(name_low)}\b", low):
            city = name
            loc_m = re.search(rf"([A-Z][\w'’ -]{{2,40}}),\s*(?:[A-Z][\w'’ -]{{2,40}},\s*)?{re.escape(name)}",
                              t, re.I)
            if loc_m:
                district = clean_ws(loc_m.group(1))
            break
    city = city or city_hint
    # discard obviously broken prices (e.g. per-sqm typed into total, or 1.2bn for a 500 sqm plot)
    if price is not None and (price < 500 or price > 500_000_000):
        price = None
    return Listing(id=lid, url=url, title=title, purpose=purpose, property_type=ptype, city=city,
                   district=district, price=price, rent_period=period, area_sqm=area,
                   bedrooms=int(beds_m.group(1)) if beds_m else None,
                   bathrooms=int(baths_m.group(1)) if baths_m else None, raw_text=t[:600])


def _card_node(a: Node, lid: str) -> Node:
    """Climb from the listing link to the largest ancestor that only covers this one listing."""
    node = a
    while node.parent is not None and node.parent.tag not in ("body", "main", "html"):
        ids = {m.group(3) for x in node.parent.css("a[href]")
               if (m := LISTING_HREF.search(x.attributes.get("href") or ""))}
        if len(ids - {lid}) > 0:
            break
        node = node.parent
    return node


def cards_from_dom(html: str, page_url: str, city_hint: Optional[str]) -> list[Listing]:
    tree = HTMLParser(html)
    for tag in ("script", "style", "svg", "noscript"):
        for n in tree.css(tag):
            n.decompose()
    out, seen = [], set()
    for a in tree.css("a[href]"):
        m = LISTING_HREF.search(a.attributes.get("href") or "")
        if not m or m.group(3) in seen:
            continue
        purpose, lid = m.group(1), m.group(3)
        seen.add(lid)
        card = _card_node(a, lid)
        url = absolutize(page_url, [a.attributes["href"]])[0]
        out.append(parse_card_text(card.text(separator=" | "), purpose, url, lid, city_hint))
    return out


def _walk(obj: Any) -> Iterable[dict]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def cards_from_json(next_data: Any, purpose: str, city_hint: Optional[str]) -> list[Listing]:
    """Best-effort: find listing-like dicts in __NEXT_DATA__ (keys vary, so match loosely)."""
    out, seen = [], set()
    for d in _walk(next_data):
        keys = {k.lower() for k in d}
        if not ({"price", "saleprice", "rentprice", "salesprice"} & keys) or not ({"id", "propertyid"} & keys):
            continue
        lid = str(d.get("propertyId") or d.get("id"))
        if lid in seen:
            continue
        seen.add(lid)
        blob = " | ".join(f"{k}: {v}" for k, v in d.items() if isinstance(v, (str, int, float)))
        slug = d.get("slug") or d.get("seoSlug") or ""
        url = f"{BASE}/en/property/{purpose}/{slug}-{lid}" if slug else f"{BASE}/en/property/{purpose}/{lid}"
        rec = parse_card_text(blob, purpose, url, lid, city_hint)
        price = d.get("price") or d.get("salePrice") or d.get("salesPrice") or d.get("rentPrice")
        if isinstance(price, (int, float)):
            rec.price = float(price)
        title = d.get("title") or d.get("propertyTitle") or d.get("name")
        if isinstance(title, str):
            rec.title = clean_ws(title)
        out.append(rec)
    return out


def scrape_listings(f: Fetcher, max_listings: int, max_pages_per_seed: int) -> list[Listing]:
    listings: dict[str, Listing] = {}
    seeds: list[tuple[str, str, Optional[str]]] = []
    for purpose in ("sale", "rent"):
        seeds.append((f"{BASE}/en/properties-for-{purpose}-in-saudi-arabia", purpose, None))
        for c in CITIES[:8]:
            seeds.append((f"{BASE}/en/properties-for-{purpose}-in-{c}", purpose, CITY_NAMES[c.replace('-', ' ')]))
    # interleave sale/rent and cities so a cap still gives a balanced sample
    seeds.sort(key=lambda s: (CITIES.index(s[0].rsplit("-in-", 1)[1]) if s[2] else -1, s[1]))
    per_seed_cap = max(40, max_listings // max(len(seeds), 1) * 2)

    discovered: set[str] = set()
    queue = list(seeds)
    while queue and len(listings) < max_listings:
        seed, purpose, city = queue.pop(0)
        got = 0
        for page_no in range(1, max_pages_per_seed + 1):
            url = seed if page_no == 1 else f"{seed}?page={page_no}"
            html = f.get(url)
            if not html:
                break
            page = parse(html, url)
            cards = cards_from_json(page.next_data, purpose, city) if page.next_data else []
            if len(cards) < 3:
                cards = cards_from_dom(html, url, city)
            new = [c for c in cards if c.id not in listings]
            for c in new:
                listings[c.id] = c
            got += len(new)
            if page_no == 1:  # discover more SEO pages (districts, property types) from links
                for link in absolutize(url, page.links):
                    p = urlparse(link)
                    if p.netloc.endswith("wasalt.sa") and SEO_PAGE.match(p.path) and link not in discovered:
                        discovered.add(link)
            if not new or got >= per_seed_cap or len(listings) >= max_listings:
                break
        log.info("%s -> %d new (total %d)", seed, got, len(listings))
        if not queue and len(listings) < max_listings:
            seed_urls = {s[0] for s in seeds}
            extra = [(u, "sale" if "-for-sale-" in u else "rent", None) for u in sorted(discovered)
                     if u not in seed_urls]
            seeds += extra[:20]
            queue = extra[:20]
            discovered.clear()
    return list(listings.values())


# ----------------------------------------------------------------- projects & pages
def scrape_projects(f: Fetcher, cap: int = 40) -> list[Project]:
    out: list[Project] = []
    for start in (f"{BASE}/en/projects", f"{BASE}/en/projects-in-saudi-arabia"):
        html = f.get(start)
        if not html:
            continue
        page = parse(html, start)
        links = [u for u in absolutize(start, page.links)
                 if urlparse(u).netloc.endswith("wasalt.sa") and re.search(r"/en/projects?/[^/]+", urlparse(u).path)]
        for u in links[:cap]:
            h = f.get(u)
            if not h:
                continue
            p = parse(h, u)
            if not p.h1 or len(p.text) < 200:
                continue
            out.append(Project(source="wasalt", url=u, name=p.h1, country="Saudi Arabia",
                               location=p.facts.get("Location") or p.facts.get("City"),
                               facts=p.facts, summary=p.description, sections=p.sections, faqs=p.faqs))
        if out:
            break
    return out


def scrape_pages(f: Fetcher) -> list[Article]:
    out = []
    for path in ("/en", "/en/corporate", "/en/s/wasalt-business"):
        u = BASE + path
        h = f.get(u)
        if h:
            p = parse(h, u)
            if len(p.text) > 300:
                out.append(Article(source="wasalt", url=u, title=p.h1 or p.title, kind="page",
                                   sections=p.sections, faqs=p.faqs))
    return out


def scrape_blog(f: Fetcher, max_posts: int) -> list[Article]:
    out: list[Article] = []
    for api in (f"{BLOG}/en/wp-json/wp/v2/posts", f"{BLOG}/wp-json/wp/v2/posts"):
        page_no = 1
        while len(out) < max_posts:
            data = f.get_json(f"{api}?per_page=50&page={page_no}&_fields=link,title,content,date")
            if not isinstance(data, list) or not data:
                break
            for post in data:
                body = (post.get("content") or {}).get("rendered", "")
                text = clean_ws(htmllib.unescape(re.sub(r"<[^>]+>", " ", body)))
                title = clean_ws(htmllib.unescape((post.get("title") or {}).get("rendered", "")))
                if len(text) < 300:
                    continue
                if re.search(r"[؀-ۿ]", title) and "/en/" not in post.get("link", ""):
                    continue  # English-only corpus: skip Arabic-language blog posts
                out.append(Article(source="wasalt", url=post.get("link", ""), title=title, kind="blog",
                                   published=(post.get("date") or "")[:10],
                                   sections=[Section(heading=title, text=text[:15000])]))
            page_no += 1
        if out:
            break
    return out[:max_posts]


def run(max_listings: int, max_posts: int, max_pages: int, browser: bool, no_cache: bool):
    f = Fetcher(delay=1.2, use_cache=not no_cache, browser_fallback=browser)
    listings = scrape_listings(f, max_listings, max_pages)
    projects = scrape_projects(f)
    articles = scrape_pages(f) + scrape_blog(f, max_posts)
    n1 = write_jsonl(RAW_DIR / "wasalt_listings.jsonl", listings)
    n2 = write_jsonl(RAW_DIR / "wasalt_projects.jsonl", projects)
    n3 = write_jsonl(RAW_DIR / "wasalt_articles.jsonl", articles)
    f.close()
    priced = sum(1 for l in listings if l.price)
    print(f"Wasalt done: {n1} listings ({priced} with price), {n2} projects, {n3} articles. "
          f"Fetch stats: {f.stats}")
    if n1 and priced / n1 < 0.5:
        print("WARNING: fewer than half the listings have a price. Inspect a cached page in "
              "data/cache/ and adjust parse_card_text(); or rerun with --browser.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-listings", type=int, default=1500)
    ap.add_argument("--max-posts", type=int, default=120)
    ap.add_argument("--max-pages", type=int, default=8, help="pagination depth per seed page")
    ap.add_argument("--browser", action="store_true", help="Playwright fallback on 403/429")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(a.max_listings, a.max_posts, a.max_pages, a.browser, a.no_cache)
