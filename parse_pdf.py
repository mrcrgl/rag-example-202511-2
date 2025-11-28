from pypdf import PdfReader


FILE_PATH = "data/BMI25028_pks-2024.pdf"

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text(extraction_mode="layout")
    return text

def main():
    text = parse_pdf(FILE_PATH)
    print(text)

if __name__ == "__main__":
    main()
