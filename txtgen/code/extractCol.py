import csv
import re

input_file = "trump.csv"
output_file = "trump.txt"

def clean_text(text):
    if not text:
        return None

    text = str(text)

    # Remove awkward symbols (keep basic punctuation)
    text = re.sub(r"[^\w\s.,!?;:'\"()-]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None

with open(input_file, "r", encoding="utf-8", errors="ignore", newline="") as in_f, \
     open(output_file, "w", encoding="utf-8") as out_f:

    reader = csv.DictReader(in_f)

    for row in reader:
        text = row.get("text")

        # Skip lines containing "t.co"
        if not text or "t.co" in text.lower():
            continue

        cleaned = clean_text(text)
        if cleaned:
            out_f.write(cleaned + "\n")
