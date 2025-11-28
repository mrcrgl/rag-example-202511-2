import os
import json
import pdfplumber
import fitz  # PyMuPDF
from pathlib import Path

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


def main():
    ensure_dir(OUT_DIR)
    extract_text_and_tables(PDF_PATH, OUT_DIR)
    extract_images(PDF_PATH, OUT_DIR)
    print(f"Fertig. Ausgabe in: {OUT_DIR}")


if __name__ == "__main__":
    main()
