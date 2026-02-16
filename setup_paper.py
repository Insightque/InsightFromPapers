import os
import sys
import json
import requests
from datetime import datetime

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "report_template.md")

def setup_paper(title, pdf_url=None, badges=None):
    if badges is None:
        badges = ["RL"]
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Create folder name: YYYY-MM-DD_Title_Snake_Case
    safe_title = "".join([c if c.isalnum() else "_" for c in title])
    while "__" in safe_title:
        safe_title = safe_title.replace("__", "_")
    folder_name = f"{date_str}_{safe_title.strip('_')}"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, folder_name)
    
    if os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} already exists.")
        return

    os.makedirs(target_dir)
    print(f"Created directory: {target_dir}")

    # Download PDF if URL provided
    paper_filename = ""
    if pdf_url:
        try:
            print(f"Downloading PDF from {pdf_url}...")
            response = requests.get(pdf_url)
            if response.status_code == 200:
                paper_filename = f"{safe_title}_paper.pdf"
                pdf_path = os.path.join(target_dir, paper_filename)
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded PDF to {pdf_path}")
            else:
                print(f"Failed to download PDF. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading PDF: {e}")

    # Create metadata.json
    metadata = {
        "title": title,
        "description": "TODO: Add description",
        "date": date_str,
        "badges": badges,
        "badge_class": "badge-rl",
        "report_file": f"{safe_title}_논문_분석_보고서.html",
        "paper_file": paper_filename
    }
    
    with open(os.path.join(target_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print("Created metadata.json")

    # Create Report Markdown from Template
    report_filename = f"{safe_title}_논문_분석_보고서.md"
    report_path = os.path.join(target_dir, report_filename)
    
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Fill in basic info if template allows
        template_content = template_content.replace("{paper_title}", title)
        template_content = template_content.replace("{date}", date_str)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        print(f"Created report skeleton at {report_path}")
    else:
        print(f"Warning: Template not found at {TEMPLATE_PATH}. Created empty file.")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\nTODO: Analyze paper here.")

    print("\nSetup Complete!")
    print(f"1. Edit {os.path.join(folder_name, 'metadata.json')} to update description and badges.")
    print(f"2. Analyze the paper and write the report in {report_path}.")
    print("3. Run 'python3 generate_web_report.py ...' to generate HTML.")
    print("4. Run 'python3 build_index.py' to update the website.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 setup_paper.py <Paper Title> [PDF URL] [Badges (comma-separated)]")
        sys.exit(1)
    
    title = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else None
    badges_input = sys.argv[3] if len(sys.argv) > 3 else "RL"
    badges = [b.strip() for b in badges_input.split(',')]
    
    setup_paper(title, url, badges)
