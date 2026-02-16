import sys
from pypdf import PdfReader

if len(sys.argv) < 3:
    print("Usage: python3 extract_text.py <pdf_file> <output_txt_file>")
    sys.exit(1)

pdf_file = sys.argv[1]
output_file = sys.argv[2]

try:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully extracted text to {output_file}")
except Exception as e:
    print(f"Error extracting text: {e}")
    sys.exit(1)
