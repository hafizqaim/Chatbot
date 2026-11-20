"""Deterministic extraction of listing filters from a user message.

No LLM call: free models are rate-limited and their tool-calling is unreliable,
so "3 bed apartments for rent in Riyadh under 80k" is parsed with rules.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from app.textutils import normalize

CITIES = {
    "riyadh": "Riyadh", "jeddah": "Jeddah", "jiddah": "Jeddah", "dammam": "Dammam", "khobar": "Khobar",
    "al khobar": "Khobar", "makkah": "Makkah", "mecca": "Makkah", "madinah": "Madinah", "medina": "Madinah",
    "taif": "Taif", "abha": "Abha", "tabuk": "Tabuk", "dhahran": "Dhahran", "jubail": "Jubail",
    "buraidah": "Buraidah", "khamis mushait": "Khamis Mushait", "al ahsa": "Al Ahsa", "hail": "Hail",
    "yanbu": "Yanbu", "jazan": "Jazan", "najran": "Najran", "qatif": "Qatif", "al kharj": "Al Kharj",
}
TYPES = {
    "apartment": "apartment", "apartments": "apartment", "flat": "apartment", "flats": "apartment",
    "villa": "villa", "villas": "villa", "land": "land", "plot": "land", "plots": "land", "lands": "land",
    "building": "building", "buildings": "building", "floor": "floor", "floors": "floor",
    "duplex": "duplex", "townhouse": "townhouse", "townhouses": "townhouse", "office": "office",
    "offices": "office", "shop": "shop", "shops": "shop", "studio": "studio", "studios": "studio",
    "warehouse": "warehouse", "chalet": "chalet", "rest house": "rest house", "istiraha": "rest house",
}
# Places Wasalt does not cover -> the question is about DarGlobal, never about Wasalt listings
NON_SAUDI = re.compile(r"\b(dubai|abu dhabi|uae|emirates|oman|muscat|aida|qatar|doha|spain|marbella|"
                       r"benahav[ií]s|costa del sol|london|uk|united kingdom|greece|athens|maldives|usa)\b")
DARGLOBAL = re.compile(r"\b(dar ?global|darglobal|trump|lamborghini|pagani|aston martin|missoni|elie saab|"
                       r"w residences|marriott|mulliner|tierra viva|les vagues|neptune|the astera|dg1|"
                       r"project|projects|off[- ]plan|branded residence|handover|payment plan|developer)\b")
LISTING_WORDS = re.compile(r"\b(listing|listings|for sale|for rent|to rent|to buy|buy|rent|rental|lease|"
                           r"available|cheapest|affordable|budget|bedrooms?|beds?|br|bhk|sqm|under|below|"
                           r"price range|average price|median price|how much)\b")
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


@dataclass
class Filters:
    purpose: Optional[str] = None          # sale | rent
    property_type: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    bedrooms: Optional[int] = None
    min_bedrooms: Optional[int] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    sort: Optional[str] = None             # price_asc | price_desc | area_desc
    listing_intent: bool = False
    darglobal_intent: bool = False
    notes: list[str] = field(default_factory=list)

    def active(self) -> dict:
        d = asdict(self)
        for k in ("listing_intent", "darglobal_intent", "notes"):
            d.pop(k)
        return {k: v for k, v in d.items() if v is not None}


def _money(num: str, unit: Optional[str]) -> Optional[float]:
    try:
        v = float(num.replace(",", ""))
    except ValueError:
        return None
    unit = (unit or "").lower()
    if unit in ("k", "thousand"):
        v *= 1_000
    elif unit in ("m", "mn", "million", "mil"):
        v *= 1_000_000
    elif unit in ("b", "bn", "billion"):
        v *= 1_000_000_000
    return v


_AMT = r"(?:sar|aed|riyals?)?\s*([\d][\d,]*(?:\.\d+)?)\s*(k|m|mn|mil|million|thousand|bn|billion)?\b"


def extract(message: str, known_districts: Optional[dict[str, str]] = None) -> Filters:
    text = re.sub(r"[?!]", " ", normalize(message))
    f = Filters()

    if re.search(r"\b(for rent|to rent|rent|rental|renting|lease|leasing|yearly|monthly)\b", text):
        f.purpose = "rent"
    elif re.search(r"\b(for sale|sale|buy|buying|purchase|own|invest(?:ment)?)\b", text):
        f.purpose = "sale"

    for word, canon in sorted(TYPES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", text):
            f.property_type = canon
            break
    for word, canon in sorted(CITIES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", text):
            f.city = canon
            break
    if known_districts:
        for key, name in known_districts.items():
            if len(key) > 3 and re.search(rf"\b{re.escape(key)}\b", text):
                f.district = name
                break

    m = re.search(r"\b(\d{1,2}|" + "|".join(NUM_WORDS) + r")\s*(?:\+|or more|plus)?\s*[- ]?(?:bed(?:room)?s?|br|bhk|rooms?)\b", text)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUM_WORDS[m.group(1)]
        if re.search(r"(\+|or more|plus|at least|minimum)", m.group(0) + text[m.end():m.end() + 12]) or \
                re.search(r"(at least|minimum|min)\s*$", text[:m.start()]):
            f.min_bedrooms = n
        else:
            f.bedrooms = n

    between = re.search(rf"\bbetween\s+{_AMT}\s+(?:and|to|-)\s+{_AMT}", text)
    if between:
        lo, hi = _money(between.group(1), between.group(2) or between.group(4)), _money(between.group(3), between.group(4))
        f.price_min, f.price_max = lo, hi
    else:
        mx = re.search(rf"\b(?:under|below|less than|max(?:imum)?|up to|upto|within|budget(?: of| is)?|cheaper than|<)\s*{_AMT}", text)
        if mx:
            f.price_max = _money(mx.group(1), mx.group(2))
        mn = re.search(rf"\b(?:over|above|more than|at least|min(?:imum)?|from|starting at|>)\s*{_AMT}", text)
        if mn and not (f.bedrooms or f.min_bedrooms and mn.group(1) == str(f.min_bedrooms)):
            f.price_min = _money(mn.group(1), mn.group(2))
    # A tiny "price" is almost certainly a bedroom/area count misread; drop it
    for attr in ("price_min", "price_max"):
        v = getattr(f, attr)
        if v is not None and v < 1000:
            setattr(f, attr, None)

    if re.search(r"\b(cheapest|lowest price|most affordable|affordable|budget)\b", text):
        f.sort = "price_asc"
    elif re.search(r"\b(most expensive|highest price|priciest|luxur(y|ious))\b", text):
        f.sort = "price_desc"
    elif re.search(r"\b(largest|biggest|most spacious)\b", text):
        f.sort = "area_desc"

    f.darglobal_intent = bool(DARGLOBAL.search(text) or NON_SAUDI.search(text))
    has_shape = bool(f.property_type or f.purpose or f.bedrooms or f.min_bedrooms or f.price_max or f.price_min)
    f.listing_intent = (not NON_SAUDI.search(text) and not re.search(r"\bdar ?global|darglobal\b", text)
                        and has_shape and bool(LISTING_WORDS.search(text) or f.city or f.district))
    return f


def merge_followup(current: Filters, previous: Optional[Filters], message: str) -> Filters:
    """'What about Jeddah?' after 'apartments for rent in Riyadh' keeps purpose/type."""
    if previous is None or not previous.listing_intent or current.darglobal_intent:
        return current
    short = len(message.split()) <= 9
    followup_cue = re.search(r"^(and|what about|how about|any|show|more|cheaper|bigger|only|in |with |"
                             r"same|those|these|ok)", normalize(message).strip())
    if not (short or followup_cue):
        return current
    if current.listing_intent and current.purpose and current.property_type:
        return current
    merged = Filters(**{k: v for k, v in previous.active().items()})
    for k, v in current.active().items():
        setattr(merged, k, v)
    if current.city and current.city != previous.city:
        merged.district = current.district
    merged.listing_intent = True
    merged.notes.append("continued filters from the previous question")
    return merged
