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

            # -------- Text (filtered: exclude table regions and numeric-only lines) ----------
            f.write(f"\n\n=== PAGE {page_num} TEXT ===\n")
            words = _filter_words_outside_bboxes(page)
            lines_info = _group_words_into_lines(words)
            filtered_lines = []
            for li in lines_info:
                ln = (li.get("text") or "").strip()
                if not ln:
                    continue
                # Skip lines that are dominated by digits or contain many numeric tokens (axes/tables)
                if li.get("digit_ratio", 0.0) >= 0.5:
                    continue
                if len(re.findall(r'\b[\d.,]+\b', ln)) >= 3:
                    continue
                filtered_lines.append(ln)
            f.write("\n".join(filtered_lines))

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


def is_headline(line: str, meta: dict | None = None) -> bool:
    """Heuristic headline detection with numeric/table guards and font-size hints."""
    if not line:
        return False
    s = line.strip()
    if len(s) < 3:
        return False

    # Reject lines dominated by numbers/punctuation (tables, axis ticks)
    alnum_count = sum(1 for ch in s if ch.isalnum())
    digit_count = sum(1 for ch in s if ch.isdigit())
    digit_ratio = (digit_count / alnum_count) if alnum_count else 1.0
    if digit_ratio >= 0.5:
        return False
    # Many numeric tokens separated by spaces -> likely table row/legend/axis
    if len(re.findall(r'\b[\d.,]+\b', s)) >= 3:
        return False

    # Numbered headings like "1", "1.1", "2.3.4 Some Title", "3) Title"
    # Require that the numbering is followed by a letter, not a number
    if re.match(r'^\d+(?:\.\d+)*(?:\)|\.)?\s+[A-Za-zÄÖÜäöü]', s):
        return True

    # ALL CAPS or mostly caps short-ish lines
    letters = [ch for ch in s if ch.isalpha()]
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if upper_ratio >= 0.7 and len(s) <= 120 and digit_ratio < 0.3:
            return True

    # Ends with colon (section headers)
    # if len(s) <= 120 and s.endswith(":") and digit_ratio < 0.3:
    #     return True

    # Font-size-based promotion to headline if clearly larger than body text
    if meta:
        avg_size = meta.get("avg_size") or 0.0
        page_avg_size = meta.get("page_avg_size") or 0.0
        if avg_size and page_avg_size and avg_size >= page_avg_size * 1.2 and letters:
            return True

    return False


def _get_table_bboxes(page):
    """Return list of (x0, top, x1, bottom) for detected tables."""
    try:
        tables = page.find_tables()
    except Exception:
        tables = []
    bboxes = []
    for t in (tables or []):
        bbox = getattr(t, "bbox", None)
        if bbox and len(bbox) == 4:
            bboxes.append(bbox)
    return bboxes

def _intersects_bbox(word, bbox):
    """Check whether a word box intersects a bbox."""
    x0, top, x1, bottom = bbox
    wx0 = word.get("x0", 0.0); wx1 = word.get("x1", 0.0)
    wtop = word.get("top", 0.0); wbot = word.get("bottom", 0.0)
    return not (wx1 <= x0 or wx0 >= x1 or wbot <= top or wtop >= bottom)

def _filter_words_outside_bboxes(page):
    """Extract words and remove those overlapping table regions."""
    try:
        words = page.extract_words(
            use_text_flow=True,
            keep_blank_chars=False,
            extra_attrs=["size", "fontname"],
        ) or []
    except Exception:
        words = []
    bboxes = _get_table_bboxes(page)
    if not bboxes:
        return words
    out = []
    for w in words:
        if not any(_intersects_bbox(w, bb) for bb in bboxes):
            out.append(w)
    return out

def _compute_page_font_stats(words):
    """Compute simple page-level font statistics."""
    sizes = [w.get("size", 0.0) for w in (words or []) if isinstance(w.get("size", None), (int, float))]
    avg = (sum(sizes) / len(sizes)) if sizes else 0.0
    return {"avg_size": avg}

def _group_words_into_lines(words, y_tolerance: float = 3.0):
    """Group words into lines by their vertical position and compute line metrics."""
    if not words:
        return []
    # Sort by vertical center then x
    def y_center(w):
        return (w.get("top", 0.0) + w.get("bottom", 0.0)) / 2.0
    ws = sorted(words, key=lambda w: (y_center(w), w.get("x0", 0.0)))
    lines = []
    current = []
    current_y = None
    for w in ws:
        y = y_center(w)
        if current_y is None or abs(y - current_y) <= y_tolerance:
            current.append(w)
            if current_y is None:
                current_y = y
        else:
            # flush current
            current.sort(key=lambda ww: ww.get("x0", 0.0))
            text = " ".join(ww.get("text", "") for ww in current).strip()
            sizes = [ww.get("size", 0.0) for ww in current if isinstance(ww.get("size", None), (int, float))]
            avg_size = (sum(sizes) / len(sizes)) if sizes else 0.0
            letters = [ch for ch in text if ch.isalpha()]
            upper_ratio = (sum(1 for ch in letters if ch.isupper()) / len(letters)) if letters else 0.0
            alnum_count = sum(1 for ch in text if ch.isalnum())
            digit_count = sum(1 for ch in text if ch.isdigit())
            digit_ratio = (digit_count / alnum_count) if alnum_count else 1.0
            lines.append({
                "text": text,
                "words": current,
                "avg_size": avg_size,
                "upper_ratio": upper_ratio,
                "digit_ratio": digit_ratio,
            })
            current = [w]
            current_y = y
    # flush remaining
    if current:
        current.sort(key=lambda ww: ww.get("x0", 0.0))
        text = " ".join(ww.get("text", "") for ww in current).strip()
        sizes = [ww.get("size", 0.0) for ww in current if isinstance(ww.get("size", None), (int, float))]
        avg_size = (sum(sizes) / len(sizes)) if sizes else 0.0
        letters = [ch for ch in text if ch.isalpha()]
        upper_ratio = (sum(1 for ch in letters if ch.isupper()) / len(letters)) if letters else 0.0
        alnum_count = sum(1 for ch in text if ch.isalnum())
        digit_count = sum(1 for ch in text if ch.isdigit())
        digit_ratio = (digit_count / alnum_count) if alnum_count else 1.0
        lines.append({
            "text": text,
            "words": current,
            "avg_size": avg_size,
            "upper_ratio": upper_ratio,
            "digit_ratio": digit_ratio,
        })
    return lines

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
            # -------- Text: parse lines and detect headlines (exclude table regions) ----------
            words = _filter_words_outside_bboxes(page)
            page_font = _compute_page_font_stats(words)
            lines_info = _group_words_into_lines(words)

            for li in lines_info:
                ln = li["text"]
                if not ln:
                    continue
                # Skip numeric-dominated lines that likely belong to figures/tables
                if li.get("digit_ratio", 0.0) >= 0.5 or len(re.findall(r'\b[\d.,]+\b', ln)) >= 3:
                    continue
                meta_line = {
                    "avg_size": li.get("avg_size", 0.0),
                    "page_avg_size": page_font.get("avg_size", 0.0),
                    "upper_ratio": li.get("upper_ratio", 0.0),
                    "digit_ratio": li.get("digit_ratio", 0.0),
                }
                if is_headline(ln, meta_line):
                    # Commit previous segment if it has any content
                    if current_segment and (current_segment.get("text") or current_segment.get("tables")):
                        structure["segments"].append(current_segment)

                    current_segment = {
                        "segment_id": str(uuid.uuid4()),
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
