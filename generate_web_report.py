#!/usr/bin/env python3
"""
웹 리포트 생성기 (Markdown to HTML)
- MathJax Latex Rendering 지원
- 가독성 최적화 CSS 적용
- 다크 모드 지원 (시스템 테마 연동)

사용법:
python3 generate_web_report.py <input.md> <output.html>
"""
import sys
import os
import re
import markdown

# HTML 템플릿 (CSS/JS 내부의 중괄호는 {{ }}로 이스케이프 처리)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=Pretendard:wght@400;600;700&display=swap" rel="stylesheet">
    
    <!-- MathJax Configuration -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
      }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    
    <style>
        :root {{
            --bg-color: #ffffff;
            --text-color: #333333;
            --heading-color: #111111;
            --link-color: #0066cc;
            --code-bg: #f5f5f5;
            --border-color: #eaeaea;
            --quote-border: #0066cc;
            --table-header-bg: #f8f9fa;
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #1a1a1a;
                --text-color: #e0e0e0;
                --heading-color: #ffffff;
                --link-color: #66b3ff;
                --code-bg: #2d2d2d;
                --border-color: #444444;
                --quote-border: #66b3ff;
                --table-header-bg: #333333;
            }}
        }}

        body {{
            font-family: 'Merriweather', serif; /* 본문은 Serif로 가독성 확보 */
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.8;
            margin: 0;
            padding: 2rem;
            transition: background-color 0.3s, color 0.3s;
        }}

        .container {{
            max-width: 800px; /* 적절한 폭 제한 */
            margin: 0 auto;
            padding-bottom: 5rem;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Pretendard', sans-serif; /* 헤딩은 Sans-serif */
            color: var(--heading-color);
            margin-top: 2rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }}

        h1 {{ font-size: 2.5rem; border-bottom: 2px solid var(--border-color); padding-bottom: 0.5rem; }}
        h2 {{ font-size: 1.8rem; border-bottom: 1px solid var(--border-color); padding-bottom: 0.3rem; }}
        h3 {{ font-size: 1.4rem; }}

        a {{ color: var(--link-color); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}

        code {{
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            background-color: var(--code-bg);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.9em;
        }}

        pre {{
            background-color: var(--code-bg);
            padding: 1rem;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid var(--border-color);
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
        }}

        blockquote {{
            margin: 1.5rem 0;
            padding-left: 1rem;
            border-left: 4px solid var(--quote-border);
            color: var(--text-color);
            font-style: italic;
            opacity: 0.9;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
        }}

        th, td {{
            border: 1px solid var(--border-color);
            padding: 0.75rem;
            text-align: left;
        }}

        th {{
            background-color: var(--table-header-bg);
            font-weight: 600;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 1rem 0;
        }}

        hr {{
            border: 0;
            border-top: 1px solid var(--border-color);
            margin: 2rem 0;
        }}

        /* Print Style */
        @media print {{
            body {{ 
                background-color: white; 
                color: black; 
                font-family: serif;
            }}
            .container {{ 
                max-width: 100%; 
                padding: 0;
            }}
            a {{ text-decoration: none; color: black; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
        <hr>
        <p style="text-align: center; font-size: 0.8rem; color: #888;">
            Generated by <b>Antigravity AI Assistant</b> on {date}
        </p>
    </div>
</body>
</html>
"""

def convert_markdown_to_html(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Pre-processing: Math Blocks Protection
    # Markdown 파서가 $...$ 수식을 오해석하지 않도록 보호
    # 수식 블록 전체를 임시 토큰(PLACEHOLDER)으로 대체하고, 변환 후 복원하는 방식 사용
    
    math_placeholders = {}
    
    def protect_math(text):
        counter = [0]
        
        # $$ ... $$ Block Math
        def replace_block(match):
            token = f"MATH_BLOCK_{counter[0]}"
            math_placeholders[token] = match.group(0)
            counter[0] += 1
            return token
        
        # DOTALL for multiline blocks
        text = re.sub(r'\$\$(.*?)\$\$', replace_block, text, flags=re.DOTALL)
        
        # $ ... $ Inline Math
        def replace_inline(match):
            token = f"MATH_INLINE_{counter[0]}"
            math_placeholders[token] = match.group(0)
            counter[0] += 1
            return token
        
        # Avoid matching empty $$
        text = re.sub(r'\$([^$\n]+)\$', replace_inline, text)
        return text

    md_content = protect_math(md_content)

    # Convert to HTML
    print("Converting Markdown to HTML...")
    html_body = markdown.markdown(md_content, extensions=['fenced_code', 'tables', 'toc'])

    # Post-processing: Restore Math
    # Markdown 파서가 p태그 등으로 감쌀 수 있으므로, 단순 replace로 복원
    # 단, MathJax가 인식할 수 있도록 원래 수식 텍스트 그대로 복원
    for token, original in math_placeholders.items():
        html_body = html_body.replace(token, original)

    # Get Title (First H1)
    title_match = re.search(r'<h1>(.*?)</h1>', html_body)
    title = title_match.group(1) if title_match else "Analysis Report"
    
    # Remove HTML tags from title for <title> tag
    clean_title = re.sub('<[^<]+?>', '', title)

    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    final_html = HTML_TEMPLATE.format(title=clean_title, content=html_body, date=date_str)

    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print("Success!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 generate_web_report.py <input.md> <output.html>")
        sys.exit(1)
    
    convert_markdown_to_html(sys.argv[1], sys.argv[2])
