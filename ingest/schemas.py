"""Normalised record types written by the scrapers (one JSON object per line)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class FAQ(BaseModel):
    q: str
    a: str


class Section(BaseModel):
    heading: str = ""
    text: str


class Project(BaseModel):
    """A development project (DarGlobal project page or a Wasalt project page)."""
    source: Literal["darglobal", "wasalt"]
    url: str
    name: str
    country: Optional[str] = None
    location: Optional[str] = None
    brand_partner: Optional[str] = None
    status: Optional[str] = None
    property_types: list[str] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)   # label -> value pairs found on page
    summary: str = ""
    sections: list[Section] = Field(default_factory=list)
    faqs: list[FAQ] = Field(default_factory=list)
    scraped_at: str = Field(default_factory=now_iso)


class Listing(BaseModel):
    """A single property listing (Wasalt)."""
    source: Literal["wasalt"] = "wasalt"
    id: str
    url: str
    title: str
    purpose: Literal["sale", "rent"]
    property_type: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    price: Optional[float] = None          # SAR
    rent_period: Optional[str] = None      # yearly / monthly / daily, rentals only
    area_sqm: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    raw_text: str = ""
    scraped_at: str = Field(default_factory=now_iso)


class Article(BaseModel):
    """Any other page worth answering from: about, FAQ, blog, press, guides."""
    source: Literal["darglobal", "wasalt"]
    url: str
    title: str
    kind: str = "page"  # page | faq | blog | press | insight | guide | category
    published: Optional[str] = None
    sections: list[Section] = Field(default_factory=list)
    faqs: list[FAQ] = Field(default_factory=list)
    scraped_at: str = Field(default_factory=now_iso)
