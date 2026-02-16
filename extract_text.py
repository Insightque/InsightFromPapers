import sys
import os
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        # Try to install pypdf
        print("Installing pypdf...")
        try:
            install("pypdf")
            from pypdf import PdfReader
        except Exception as e:
            print(f"Error installing pypdf: {e}")
            sys.exit(1)

def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Limit to 20 pages max
        num_pages = len(reader.pages)
        print(f"DEBUG: Processing {num_pages} pages...")
        for i, page in enumerate(reader.pages[:20]): 
            text += f"\n--- Page {i+1} ---\n"
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_text.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        sys.exit(1)
        
    print(extract_text(pdf_path))
