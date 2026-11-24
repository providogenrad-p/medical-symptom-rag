"""
update.py - puts the newest downloaded project files from your Downloads folder into this folder.

Usage: download the new file(s) from the chat, then in the project folder run:
    python update.py
It finds medical_rag_demo.py, medical_web.py, fetch_medlineplus.py and tech_rag.py in Downloads (also names like
"medical_rag_demo (1).py"), copies the newest one here, shows its version and cleans up Downloads.
"""

import re
import shutil
from pathlib import Path

DOWNLOADS = Path.home() / "Downloads"
HERE = Path(__file__).resolve().parent
FILES = ["medical_rag_demo.py", "medical_web.py", "fetch_medlineplus.py", "tech_rag.py"]

updated = 0
for name in FILES:
    stem = Path(name).stem
    pattern = re.compile(rf"{re.escape(stem)}( \(\d+\))?\.py")
    found = [p for p in DOWNLOADS.glob(f"{stem}*.py") if pattern.fullmatch(p.name)]
    if not found:
        continue
    newest = max(found, key=lambda p: p.stat().st_mtime)
    target = HERE / name
    shutil.copy2(newest, target)
    version = re.search(r'VERSION = "([^"]+)"', target.read_text(encoding="utf-8", errors="ignore"))
    print(f"[updated] {name}" + (f"  -> {version.group(1)}" if version else ""))
    for p in found:          # remove the downloads so the next one keeps the plain name
        p.unlink()
    updated += 1

if not updated:
    print(f"No new files found in {DOWNLOADS}. Download the file from the chat first.")
