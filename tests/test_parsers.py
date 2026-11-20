from pathlib import Path

from ingest.extract import parse
from ingest.scrape_darglobal import looks_like_project, to_project
from ingest.scrape_wasalt import cards_from_dom

FIX = Path(__file__).parent / "fixtures"


def test_wasalt_cards():
    html = (FIX / "wasalt_list.html").read_text(encoding="utf-8")
    cards = {c.id: c for c in cards_from_dom(html, "https://wasalt.sa/en/properties-for-sale-in-central-riyadh-al-malaz", "Riyadh")}
    assert set(cards) == {"5926939", "5942463", "5940126"}
    floor = cards["5926939"]
    assert floor.property_type == "floor" and floor.price == 900_000
    assert floor.area_sqm == 137.48 and floor.bedrooms == 3 and floor.bathrooms == 2
    assert floor.city == "Riyadh" and floor.district and "Malaz" in floor.district
    assert floor.url.startswith("https://wasalt.sa/en/property/sale/")
    land = cards["5942463"]
    assert land.property_type == "land" and land.price == 1_500_000 and land.area_sqm == 400
    assert cards["5940126"].price is None  # 1.19bn for 531 sqm is rejected as broken


def test_darglobal_project():
    html = (FIX / "darglobal_project.html").read_text(encoding="utf-8")
    page = parse(html, "https://darglobal.co.uk/tierra-viva")
    assert page.h1.startswith("Tierra Viva")
    assert looks_like_project(page)
    assert page.facts.get("Location") == "Benahavís | Spain"
    assert any("June 2028" in f.a for f in page.faqs)
    p = to_project(page, None, set())
    assert p.country == "Spain" and p.brand_partner and "Lamborghini" in p.brand_partner
    assert p.status == "Under Development" and "villa" in p.property_types
    assert "View All Images" not in page.text and "FOLLOW US" not in page.text
