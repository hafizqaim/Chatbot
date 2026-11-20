"""Builds a small SAMPLE raw dataset from the test fixtures + synthetic listings, for local
testing only (never shipped). Usage: RAW_DIR=/tmp/raw python -m tests.make_sample_data"""
import random
from pathlib import Path

from ingest.common import RAW_DIR, write_jsonl
from ingest.extract import parse
from ingest.schemas import FAQ, Article, Listing, Project, Section
from ingest.scrape_darglobal import to_project
from ingest.scrape_wasalt import cards_from_dom

FIX = Path(__file__).parent / "fixtures"
random.seed(7)
tv = to_project(parse((FIX / "darglobal_project.html").read_text(encoding="utf-8"), "https://darglobal.co.uk/tierra-viva"), None, set())
projects = [tv,
    Project(source="darglobal", url="https://darglobal.co.uk/aida-trump-villas", name="AIDA Trump Villas",
            country="Oman", location="Muscat", brand_partner="Trump", status="Under Development",
            property_types=["villa"], summary="Trump-branded villas at AIDA, a hilltop community in Muscat.",
            sections=[Section(heading="Overview", text="SAMPLE: villas overlooking the Gulf of Oman with golf course access.")],
            faqs=[FAQ(q="Can foreigners buy at AIDA?", a="SAMPLE: AIDA is an Integrated Tourism Complex open to foreign ownership.")]),
    Project(source="darglobal", url="https://darglobal.co.uk/dg1", name="DG1 Living", country="UAE",
            location="Dubai", status="Completed", property_types=["apartment"],
            summary="SAMPLE: apartments on Business Bay waterfront.")]
articles = [Article(source="darglobal", url="https://darglobal.co.uk/about", title="About DarGlobal", kind="page",
                    sections=[Section(heading="About", text="SAMPLE: DarGlobal is a London-listed luxury developer founded in 2017 as the international arm of Dar Al Arkan.")]),
            Article(source="wasalt", url="https://wasalt.sa/en", title="Wasalt overview", kind="page",
                    sections=[Section(heading="Overview", text="SAMPLE: Wasalt is an AI-powered Saudi real-estate platform for verified listings, licensed by REGA.")])]
listings = cards_from_dom((FIX / "wasalt_list.html").read_text(encoding="utf-8"), "https://wasalt.sa/en/x", "Riyadh")
types = ["apartment", "villa", "floor", "land"]
for i in range(300):
    city = random.choice(["Riyadh", "Jeddah", "Dammam", "Khobar"])
    purpose = random.choice(["sale", "rent"]); t = random.choice(types)
    beds = None if t == "land" else random.randint(1, 6)
    area = round(random.uniform(80, 600), 1)
    price = round(area * (random.uniform(3000, 7000) if purpose == "sale" else random.uniform(150, 400)), -3)
    listings.append(Listing(id=str(9000000 + i), url=f"https://wasalt.sa/en/property/{purpose}/sample-{9000000+i}",
        title=f"{t.title()} {area} SQM" + (f" with {beds} Bedrooms" if beds else ""), purpose=purpose,
        property_type=t, city=city, district=random.choice(["Al Narjis", "Al Malqa", "Al Rawdah", "Al Shati"]),
        price=price, rent_period="yearly" if purpose == "rent" else None, area_sqm=area, bedrooms=beds))
print(write_jsonl(RAW_DIR / "darglobal_projects.jsonl", projects), write_jsonl(RAW_DIR / "darglobal_articles.jsonl", articles),
      write_jsonl(RAW_DIR / "wasalt_listings.jsonl", listings))
