"""Generic HTML -> structured content extraction.

Layered, because we can't rely on one site structure:
  1. JSON-LD blocks (schema.org FAQPage, Offer, Residence, ...)
  2. Next.js __NEXT_DATA__ payload (fallback text when the DOM is thin)
  3. DOM walk: headings + paragraphs/list items -> sections, FAQ pairs, label: value facts
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from selectolax.lexbor import LexborHTMLParser as HTMLParser, LexborNode as Node

from app.textutils import clean_ws
from ingest.schemas import FAQ, Section

DROP_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "header", "footer", "nav",
             "button", "select", "option", "template"]
BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th", "dt", "dd", "blockquote",
              "figcaption", "span", "div"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5"}
BOILERPLATE = re.compile(
    r"(cookie|subscribe|newsletter|all rights reserved|privacy policy|terms (and|&) conditions|"
    r"follow us|download (the )?app|register your interest|enter your (email|phone)|"
    r"view all images|show more|read more|book now|get in touch)", re.I)


@dataclass
class Page:
    url: str
    title: str = ""
    h1: str = ""
    description: str = ""
    sections: list[Section] = field(default_factory=list)
    faqs: list[FAQ] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    jsonld: list[dict] = field(default_factory=list)
    next_data: Any = None
    links: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(f"{s.heading}\n{s.text}".strip() for s in self.sections)


def _own_text(node: Node) -> str:
    """Text of a block element, excluding nested block children (avoids duplicates)."""
    parts = []
    child = node.child
    while child is not None:
        if child.tag == "-text":
            parts.append(child.text_content or "")
        elif child.tag in ("a", "strong", "b", "em", "i", "u", "small", "sup", "sub", "br", "label"):
            parts.append(child.text(separator=" ") if child.tag != "br" else " ")
        child = child.next
    return clean_ws(" ".join(parts))


def parse(html: str, url: str) -> Page:
    tree = HTMLParser(html)
    page = Page(url=url)

    t = tree.css_first("title")
    page.title = clean_ws(t.text()) if t else ""
    md = tree.css_first('meta[name="description"]') or tree.css_first('meta[property="og:description"]')
    page.description = clean_ws(md.attributes.get("content", "")) if md else ""

    # 1. JSON-LD
    for s in tree.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(s.text())
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        page.jsonld.extend(i for i in items if isinstance(i, dict))

    # 2. __NEXT_DATA__
    nd = tree.css_first("script#__NEXT_DATA__")
    if nd:
        try:
            page.next_data = json.loads(nd.text())
        except json.JSONDecodeError:
            pass

    page.links = [a.attributes.get("href") or "" for a in tree.css("a[href]")]

    # 3. DOM walk
    for tag in DROP_TAGS:
        for n in tree.css(tag):
            n.decompose()
    body = tree.css_first("main") or tree.body
    if body is None:
        return page
    h1 = body.css_first("h1")
    page.h1 = clean_ws(h1.text(separator=" ")) if h1 else ""

    blocks: list[tuple[str, str]] = []  # (tag, text)
    seen: set[str] = set()
    for node in body.traverse(include_text=False):
        if node.tag not in BLOCK_TAGS:
            continue
        txt = clean_ws(node.text(separator=" ")) if node.tag in HEADINGS else _own_text(node)
        if not txt or len(txt) < 2 or txt in seen:
            continue
        if node.tag in ("span", "div") and len(txt) < 25:
            # short spans/divs are mostly UI chrome, but keep "Label: value" style facts
            if ":" not in txt and not re.search(r"\d", txt):
                continue
        if BOILERPLATE.search(txt) and len(txt) < 120:
            continue
        seen.add(txt)
        blocks.append(("h" if node.tag in HEADINGS else "li" if node.tag == "li" else "p", txt))

    page.sections = _to_sections(blocks)
    page.faqs = _faqs_from_blocks(blocks) or _faqs_from_jsonld(page.jsonld)
    page.facts = _pair_facts(body)
    for k, v in _facts(blocks).items():
        page.facts.setdefault(k, v)
    if page.facts:  # make the key facts searchable text too
        page.sections.insert(0, Section(heading="Key facts",
                                        text="; ".join(f"{k}: {v}" for k, v in page.facts.items())))

    # Thin DOM (client-rendered page)? fall back to strings inside __NEXT_DATA__.
    if len(page.text) < 300 and page.next_data:
        strings = _long_strings(page.next_data)
        if strings:
            page.sections.append(Section(heading="Page data", text=" ".join(strings)[:20000]))
    return page


def _to_sections(blocks: list[tuple[str, str]]) -> list[Section]:
    sections: list[Section] = []
    heading, buf = "", []
    for kind, txt in blocks:
        if kind == "h":
            if buf:
                sections.append(Section(heading=heading, text=" ".join(buf)))
            elif heading and sections is not None and len(heading) > 60:
                # a long "heading" with no body is really content (sites style text as h3)
                sections.append(Section(heading="", text=heading))
            heading, buf = txt, []
        else:
            buf.append(f"- {txt};" if kind == "li" else txt)
    if buf:
        sections.append(Section(heading=heading, text=" ".join(buf)))
    return [s for s in sections if len(s.text) > 15]


def _faqs_from_blocks(blocks: list[tuple[str, str]]) -> list[FAQ]:
    faqs = []
    for i, (kind, txt) in enumerate(blocks):
        is_q = txt.endswith("?") and len(txt) < 200 and (kind == "h" or len(txt.split()) <= 20)
        if not is_q:
            continue
        answer = []
        for k2, t2 in blocks[i + 1:i + 6]:
            if k2 == "h" or (t2.endswith("?") and len(t2) < 200):
                break
            answer.append(t2)
        if answer:
            faqs.append(FAQ(q=txt, a=" ".join(answer)[:2000]))
    return faqs


def _faqs_from_jsonld(items: list[dict]) -> list[FAQ]:
    out = []
    for it in items:
        if it.get("@type") == "FAQPage":
            for q in it.get("mainEntity", []) or []:
                ans = (q.get("acceptedAnswer") or {}).get("text", "")
                if q.get("name") and ans:
                    out.append(FAQ(q=clean_ws(q["name"]), a=clean_ws(re.sub("<[^>]+>", " ", ans))))
    return out


_FACT_RE = re.compile(r"^([A-Za-z][A-Za-z /()&-]{1,30}):\s*(.{1,120})$")
_FACT_LABELS = re.compile(
    r"^(location|city|country|area|size|type|property type|unit types?|status|bedrooms?|"
    r"bathrooms?|price|starting price|price from|handover|completion|developer|payment plan|"
    r"unit type|area \(sqm\)|built up area|plot area)$", re.I)


def _facts(blocks: list[tuple[str, str]]) -> dict[str, str]:
    facts: dict[str, str] = {}
    for i, (_, txt) in enumerate(blocks):
        m = _FACT_RE.match(txt)
        if m:
            facts.setdefault(m.group(1).strip().title(), m.group(2).strip())
            continue
        # "Location" followed by value block
        if _FACT_LABELS.match(txt) and i + 1 < len(blocks) and len(blocks[i + 1][1]) < 120:
            facts.setdefault(txt.strip().title(), blocks[i + 1][1])
    return facts


def _pair_facts(body: Node) -> dict[str, str]:
    """<div><span>Location</span><span>Dubai</span></div> and <dt>/<dd> label-value pairs."""
    facts: dict[str, str] = {}
    for node in body.traverse(include_text=False):
        kids = [c for c in node.iter(include_text=False)]
        if len(kids) != 2:
            continue
        label, value = (clean_ws(k.text(separator=" ")) for k in kids)
        if label and value and len(value) < 120 and _FACT_LABELS.match(label.rstrip(":")):
            facts.setdefault(label.rstrip(":").strip().title(), value)
    return facts


def _long_strings(obj: Any, out: list[str] | None = None, limit: int = 400) -> list[str]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(obj, str):
        s = clean_ws(re.sub("<[^>]+>", " ", obj))
        if len(s) > 40 and not s.startswith(("http", "/", "{")) and " " in s and s not in out:
            out.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            _long_strings(v, out, limit)
    elif isinstance(obj, list):
        for v in obj:
            _long_strings(v, out, limit)
    return out
