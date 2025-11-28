import sys
import pymupdf.layout
import pymupdf4llm

def process_pdf(file_path):
    doc = pymupdf.open(file_path)
    md = pymupdf4llm.to_markdown(doc)
    return md

def save_markdown(content, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_pdf.py <pdf_file_path>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    markdown_content = process_pdf(pdf_file_path)
    save_markdown(markdown_content, pdf_file_path.replace('.pdf', '.md'))

if __name__ == "__main__":
    main()