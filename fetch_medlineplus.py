"""
fetch_medlineplus.py - Build the medical knowledge base from real, public-domain MedlinePlus data.

Downloads the official MedlinePlus Health Topic XML (U.S. National Library of Medicine),
takes the English health topic summaries (public domain) and writes one Markdown file per
topic into ./medical_kb, in the format medical_rag_demo.py already understands.

Usage (inside the same virtual environment as the demo):
    python fetch_medlineplus.py                 # all English health topics (~1,000+)
    python fetch_medlineplus.py --demo          # only ~30 topics used in the demo (faster ingest)
    python fetch_medlineplus.py --xml mplus_topics_2026-11-17.xml   # use a file you downloaded yourself
Then:
    python medical_rag_demo.py --ingest

Source: MedlinePlus.gov, U.S. National Library of Medicine. Health topic summaries are in the
public domain; the A.D.A.M. Medical Encyclopedia and drug monographs are NOT used.
"""

import argparse
import datetime as dt
import html
import io
import re
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path

KB_DIR = Path("./medical_kb")
BASE = "https://medlineplus.gov/xml/"
INDEX = "https://medlineplus.gov/xml.html"
UA = {"User-Agent": "Mozilla/5.0 (educational RAG demo; contact: student project)"}

# The generated summaries from the first version of the demo (moved to a backup folder)
GENERATED_SEED_FILES = ["cardiology_acs.md", "pulmonology_pe.md", "pulmonology_pneumonia.md",
                        "infectious_influenza.md", "neurology_meningitis.md", "neurology_stroke.md",
                        "neurology_headache.md", "gastro_appendicitis.md", "gastro_gastroenteritis.md",
                        "urology_uti.md"]

# Topics for --demo: covers every demo scenario plus their differential diagnoses
DEMO_TOPICS = ["Meningitis", "Encephalitis", "Headache", "Migraine", "Flu", "COVID-19", "Common Cold",
               "Pneumonia", "Acute Bronchitis", "Pulmonary Embolism", "Heart Attack", "Chest Pain", "Stroke",
               "Transient Ischemic Attack", "Urinary Tract Infections", "Bladder Diseases", "Kidney Stones",
               "Appendicitis", "Gastroenteritis", "Food Poisoning", "Diarrhea", "Abdominal Pain",
               "Sepsis", "Fever", "Asthma", "Dehydration", "Sinusitis", "Streptococcal Infections", "Fainting",
               "Seizures", "Anemia", "Iron", "Dizziness and Vertigo", "Tinnitus", "Fatigue",
               "Low Blood Pressure", "Shortness of Breath"]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def latest_xml_bytes() -> tuple[bytes, str]:
    """Find the newest compressed health-topic file on the XML page; fall back to recent dates."""
    candidates = []
    try:
        page = fetch(INDEX).decode("utf-8", "ignore")
        candidates = re.findall(r"mplus_topics_compressed_(\d{4}-\d{2}-\d{2})\.zip", page)
    except Exception as e:
        print(f"[warn] could not read {INDEX}: {e}")
    if not candidates:   # files are published Tuesday-Saturday: try the last 10 days
        today = dt.date.today()
        candidates = [(today - dt.timedelta(days=i)).isoformat() for i in range(10)]
    for date in sorted(set(candidates), reverse=True):
        url = f"{BASE}mplus_topics_compressed_{date}.zip"
        try:
            print(f"[download] {url}")
            data = fetch(url)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                name = next(n for n in z.namelist() if n.endswith(".xml"))
                return z.read(name), date
        except Exception as e:
            print(f"[skip] {date}: {e}")
    sys.exit("Could not download MedlinePlus XML. Download it manually from "
             "https://medlineplus.gov/xml.html and run:  python fetch_medlineplus.py --xml <file.xml>")


# --------------------------------------------------------------------------- #
# HTML summary -> Markdown
# --------------------------------------------------------------------------- #
class SummaryToMarkdown(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.prefix = [], ""

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "div", "ul", "ol"):
            self.out.append("\n\n")
        elif tag in ("h2", "h3", "h4"):
            self.out.append("\n\n## ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "br":
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("h2", "h3", "h4", "p", "ul", "ol"):
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(re.sub(r"\s+", " ", data))

    def markdown(self) -> str:
        text = "".join(self.out)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def summary_markdown(elem) -> str:
    # The summary may be escaped HTML text or nested elements: handle both.
    raw = (elem.text or "") + "".join(ET.tostring(ch, encoding="unicode") for ch in elem)
    parser = SummaryToMarkdown()
    parser.feed(html.unescape(raw) if "&lt;" in raw else raw)
    return parser.markdown()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


# --------------------------------------------------------------------------- #
# XML -> Markdown files
# --------------------------------------------------------------------------- #
def convert(xml_bytes: bytes, file_date: str, only: list[str] | None) -> int:
    root = ET.fromstring(xml_bytes)
    wanted = {t.lower() for t in only} if only else None
    KB_DIR.mkdir(exist_ok=True)
    for old in KB_DIR.glob("mplus_*.md"):      # replace a previous MedlinePlus import
        old.unlink()
    written, found = 0, set()
    for topic in root.iter("health-topic"):
        if topic.get("language", "English") != "English":
            continue
        title = topic.get("title") or (topic.text or "").strip()
        if not title:
            continue
        also = [a.text.strip() for a in topic.findall("also-called") if a.text]
        if wanted is not None:
            names = {title.lower(), *[a.lower() for a in also]}
            if not names & wanted:
                continue
            found |= names & wanted
        fs = topic.find("full-summary")
        if fs is None:
            continue
        body = summary_markdown(fs)
        if len(body) < 80:
            continue
        groups = [g.text.strip() for g in topic.findall("group") if g.text]
        front = ("---\n"
                 f"specialty: {', '.join(groups) or 'general'}\n"
                 f"last_reviewed: {file_date}\n"
                 "source_type: MedlinePlus health topic (U.S. National Library of Medicine, public domain)\n"
                 f"url: {topic.get('url', '')}\n"
                 "---\n")
        content = (f"{front}# {title}\n\n"
                   + (f"Also called: {', '.join(also)}\n\n" if also else "")
                   + f"{body}\n\nSource: MedlinePlus.gov, U.S. National Library of Medicine.\n")
        (KB_DIR / f"mplus_{topic.get('id', written)}_{slug(title)}.md").write_text(content, encoding="utf-8")
        written += 1
    if wanted is not None:
        missing = sorted(wanted - found)
        if missing:
            print(f"[info] not found under these names: {', '.join(missing)}")
    return written


def backup_generated_seeds() -> None:
    present = [f for f in GENERATED_SEED_FILES if (KB_DIR / f).exists()]
    if present:
        backup = KB_DIR / "_generated_backup"     # sub-folder: ignored by the demo loader
        backup.mkdir(exist_ok=True)
        for f in present:
            shutil.move(str(KB_DIR / f), str(backup / f))
        print(f"[backup] moved {len(present)} generated summaries to {backup}")


def main():
    ap = argparse.ArgumentParser(description="Build medical_kb from MedlinePlus XML")
    ap.add_argument("--demo", action="store_true", help="only the ~30 topics used in the demo")
    ap.add_argument("--xml", help="path to an already downloaded mplus_topics_*.xml or .zip")
    args = ap.parse_args()

    if args.xml:
        p = Path(args.xml)
        data = p.read_bytes()
        if p.suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                data = z.read(next(n for n in z.namelist() if n.endswith(".xml")))
        m = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
        date = m.group(0) if m else dt.date.today().isoformat()
    else:
        data, date = latest_xml_bytes()

    KB_DIR.mkdir(exist_ok=True)
    backup_generated_seeds()
    n = convert(data, date, DEMO_TOPICS if args.demo else None)
    print(f"[done] wrote {n} MedlinePlus topics (file date {date}) to {KB_DIR}/")
    print("Next:  python medical_rag_demo.py --ingest")


if __name__ == "__main__":
    main()
