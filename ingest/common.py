"""Polite HTTP fetching: robots.txt, rate limiting, retries, on-disk cache,
and an optional headless-browser fallback for pages that need JavaScript."""
from __future__ import annotations

import hashlib
import os
import json
import logging
import re
import time
import urllib.robotparser
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger("ingest")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(os.getenv("RAW_DIR", ROOT / "data" / "raw"))
CACHE_DIR = ROOT / "data" / "cache"

BOT_WALL_RE = re.compile(
    r"(_Incapsula_Resource|Incapsula incident|Pardon Our Interruption|Attention Required!|"
    r"cf-browser-verification|Just a moment\.\.\.|cf_chl_opt|Access Denied|Request unsuccessful)",
    re.I,
)


def _looks_blocked(text: str) -> bool:
    """Some bot walls (e.g. Incapsula) answer 200 OK with a tiny JS-challenge page
    instead of a real 403/429/503, so a plain status check doesn't catch them."""
    return len(text) < 4000 and bool(BOT_WALL_RE.search(text))


USER_AGENT = (
    "Mozilla/5.0 (compatible; DarGlobal-Assessment-Bot/1.0; "
    "candidate technical assignment; contact: careers applicant)"
)
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


class Fetcher:
    def __init__(self, delay: float = 1.0, use_cache: bool = True, browser_fallback: bool = False,
                 respect_robots: bool = True):
        self.delay = delay
        self.use_cache = use_cache
        self.browser_fallback = browser_fallback
        self.respect_robots = respect_robots
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._browser = None
        self.client = httpx.Client(
            headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9",
                     "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"},
            follow_redirects=True, timeout=30.0,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.stats = {"fetched": 0, "cached": 0, "failed": 0, "blocked": 0, "browser": 0}

    # ---------- robots ----------
    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        host = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        rp = self._robots.get(host)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.client.get(host + "/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            self._robots[host] = rp
        return rp.can_fetch(USER_AGENT, url)

    def sitemaps(self, host: str) -> list[str]:
        try:
            r = self.client.get(host.rstrip("/") + "/robots.txt")
            return re.findall(r"(?im)^\s*sitemap:\s*(\S+)", r.text)
        except httpx.HTTPError:
            return []

    # ---------- fetching ----------
    def _cache_path(self, url: str) -> Path:
        return CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".html")

    def _throttle(self, url: str):
        host = urlparse(url).netloc
        wait = self.delay - (time.time() - self._last.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()

    def get(self, url: str, retries: int = 3) -> Optional[str]:
        if not self.allowed(url):
            log.warning("robots.txt disallows %s", url)
            self.stats["blocked"] += 1
            return None
        cp = self._cache_path(url)
        if self.use_cache and cp.exists():
            self.stats["cached"] += 1
            return cp.read_text(encoding="utf-8", errors="ignore")
        for attempt in range(retries):
            self._throttle(url)
            try:
                r = self.client.get(url)
            except httpx.HTTPError as e:
                log.info("fetch error %s (%s), attempt %d", url, e, attempt + 1)
                time.sleep(2 ** attempt)
                continue
            blocked = r.status_code == 200 and r.text and _looks_blocked(r.text)
            if r.status_code == 200 and r.text and not blocked:
                cp.write_text(r.text, encoding="utf-8")
                self.stats["fetched"] += 1
                return r.text
            if (blocked or r.status_code in (403, 429, 503)) and self.browser_fallback:
                html = self._browser_get(url)
                if html and not _looks_blocked(html):
                    cp.write_text(html, encoding="utf-8")
                    self.stats["fetched"] += 1
                    return html
            if r.status_code in (404, 410):
                break
            time.sleep(2 ** attempt * (3 if r.status_code == 429 else 1))
        self.stats["failed"] += 1
        log.info("giving up on %s", url)
        return None

    def get_json(self, url: str) -> Optional[object]:
        text = self.get(url)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _browser_get(self, url: str) -> Optional[str]:
        """Render with Playwright (pip install playwright && playwright install chromium).

        Waits for "domcontentloaded", not "networkidle": some sites (e.g. continuous
        analytics beacons) never go idle within the timeout even though the real content
        loaded in the first couple of seconds. Capturing content() outside the goto's own
        try also means a goto timeout doesn't discard an already-loaded page."""
        try:
            if self._browser is None:
                from playwright.sync_api import sync_playwright
                self._pw = sync_playwright().start()
                self._browser = self._pw.chromium.launch(headless=True)
            page = self._browser.new_page(user_agent=BROWSER_UA)
        except Exception as e:  # noqa: BLE001 - optional path, never fatal
            log.warning("browser fallback failed for %s: %s", url, e)
            return None
        try:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45000)
            except Exception as e:
                log.info("goto did not settle for %s (%s); using whatever loaded", url, e)
            page.wait_for_timeout(2500)
            html = page.content()
            self.stats["browser"] += 1
            return html
        except Exception as e:  # noqa: BLE001 - optional path, never fatal
            log.warning("browser fallback failed for %s: %s", url, e)
            return None
        finally:
            page.close()

    def close(self):
        self.client.close()
        if self._browser is not None:
            self._browser.close()
            self._pw.stop()


def parse_sitemap(fetcher: Fetcher, url: str, depth: int = 0) -> list[str]:
    """Return all <loc> URLs, following sitemap indexes (max depth 2)."""
    xml = fetcher.get(url)
    if not xml:
        return []
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)
    if "<sitemapindex" in xml and depth < 2:
        out: list[str] = []
        for child in locs:
            out.extend(parse_sitemap(fetcher, child, depth + 1))
        return out
    return locs


def absolutize(base: str, links: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for href in links:
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        u = urljoin(base, href).split("#")[0]
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def write_jsonl(path: Path, records: Iterable) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write((r.model_dump_json() if hasattr(r, "model_dump_json") else json.dumps(r)) + "\n")
            n += 1
    return n
