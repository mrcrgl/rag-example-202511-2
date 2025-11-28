import os
import json
import re
import pdfplumber
import fitz  # PyMuPDF
from pathlib import Path
import uuid

PDF_PATH = "data/BMI25028_pks-2024.pdf"  # dein PDF
OUT_DIR = "parsed_pks"

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_text_and_tables(pdf_path: str, out_dir: str):
    ensure_dir(out_dir)
    text_file = Path(out_dir) / "content.txt"

    manifest = {
        "pdf": pdf_path,
        "pages": []
    }

    with pdfplumber.open(pdf_path) as pdf, open(text_file, "w", encoding="utf-8") as f:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_entry = {
                "page": page_num,
                "tables": []
            }

            # -------- Text ----------
            text = page.extract_text() or ""
            f.write(f"\n\n=== PAGE {page_num} TEXT ===\n")
            f.write(text.strip())

            # -------- Tabellen ----------
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables, start=1):
                page_entry["tables"].append({
                    "index": t_idx,
                    "type": "simple_table"
                })

                f.write(f"\n\n=== PAGE {page_num} TABLE {t_idx} ===\n")
                # Tabelle als pipe-getrennte Zeilen ausgeben
                for row in table:
                    # row kann None enthalten, deswegen absichern
                    cells = [(c or "").strip() for c in row]
                    f.write(" | ".join(cells) + "\n")

            manifest["pages"].append(page_entry)

    # Manifest hilft dir später beim Matching
    with open(Path(out_dir) / "manifest_text_tables.json", "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)


def extract_images(pdf_path: str, out_dir: str):
    img_dir = Path(out_dir) / "images"
    ensure_dir(img_dir)

    doc = fitz.open(pdf_path)
    img_manifest = {
        "pdf": pdf_path,
        "images": []
    }

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]  # meist 'png' oder 'jpeg'

            img_name = f"page_{page_index+1}_img_{img_index}.{img_ext}"
            img_path = img_dir / img_name

            with open(img_path, "wb") as img_file:
                img_file.write(img_bytes)

            img_manifest["images"].append({
                "page": page_index + 1,
                "index": img_index,
                "path": str(img_path),
                "width": base_image.get("width"),
                "height": base_image.get("height"),
            })

    with open(Path(out_dir) / "manifest_images.json", "w", encoding="utf-8") as mf:
        json.dump(img_manifest, mf, ensure_ascii=False, indent=2)


def is_headline(line: str) -> bool:
    """Heuristic headline detection from plain text lines."""
    if not line:
        return False
    s = line.strip()
    if len(s) < 3:
        return False
    # Numbered headings like "1", "1.1", "2.3.4 Some Title", "3) Title"
    if re.match(r'^\d+(\.\d+)*(\)|\.)?\s+\S', s):
        return True
    # ALL CAPS or mostly caps short-ish lines
    letters = [ch for ch in s if ch.isalpha()]
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if upper_ratio >= 0.7 and len(s) <= 120:
            return True
    # Title case short lines
    # if len(s) <= 80 and s.istitle():
    #     return True
    # Ends with colon
    if len(s) <= 120 and s.endswith(":"):
        return True
    return False


def build_document_structure(pdf_path: str, out_dir: str):
    """
    Build a document structure with segments grouped by detected headlines.
    Each segment contains:
      - meta: { page, headline }
      - headline: string
      - pages: list[int]
      - text: concatenated text under the headline
      - tables: list of { page, index, rows }
    Also writes the structure to out_dir/document_structure.json and returns it.
    """
    ensure_dir(out_dir)
    structure = {"pdf": pdf_path, "segments": []}

    with pdfplumber.open(pdf_path) as pdf:
        current_segment = None

        for page_num, page in enumerate(pdf.pages, start=1):
            # -------- Text: parse lines and detect headlines ----------
            text = page.extract_text() or ""
            lines = [ln.strip() for ln in text.splitlines()]

            for ln in lines:
                if is_headline(ln):
                    # Commit previous segment if it has any content
                    if current_segment and (current_segment.get("text") or current_segment.get("tables")):
                        structure["segments"].append(current_segment)

                    current_segment = {
                        "headline": ln,
                        "meta": {"page": page_num, "headline": ln},
                        "pages": [page_num],
                        "text": "",
                        "tables": []
                    }
                else:
                    if not current_segment:
                        # Fallback segment before the first detected headline
                        current_segment = {
                            "segment_id": str(uuid.uuid4()),
                            "headline": "UNTITLED",
                            "meta": {"page": page_num, "headline": "UNTITLED"},
                            "pages": [page_num],
                            "text": "",
                            "tables": []
                        }
                    current_segment["text"] = (current_segment["text"] + ("\n" if current_segment["text"] else "") + ln)
                    if page_num not in current_segment["pages"]:
                        current_segment["pages"].append(page_num)

            # -------- Tables: attach to the current segment ----------
            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables, start=1):
                rows = [[(c or "").strip() for c in (row or [])] for row in (table or [])]
                if not current_segment:
                    current_segment = {
                        "segment_id": str(uuid.uuid4()),
                        "headline": "UNTITLED",
                        "meta": {"page": page_num, "headline": "UNTITLED"},
                        "pages": [page_num],
                        "text": "",
                        "tables": []
                    }
                current_segment["tables"].append({
                    "page": page_num,
                    "index": t_idx,
                    "rows": rows
                })
                if page_num not in current_segment["pages"]:
                    current_segment["pages"].append(page_num)

        # Commit the last segment
        if current_segment and (current_segment.get("text") or current_segment.get("tables")):
            structure["segments"].append(current_segment)

    # Persist JSON
    with open(Path(out_dir) / "document_structure.json", "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

    return structure


def main():
    ensure_dir(OUT_DIR)
    extract_text_and_tables(PDF_PATH, OUT_DIR)
    extract_images(PDF_PATH, OUT_DIR)
    build_document_structure(PDF_PATH, OUT_DIR)
    print(f"Fertig. Ausgabe in: {OUT_DIR}")


if __name__ == "__main__":
    main()
