import os
import json
import glob
from datetime import datetime

# HTML Template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>InsightFromPapers - AI Research Analysis</title>

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link
        href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap"
        rel="stylesheet">

    <!-- Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --text-primary: #1a1a1a;
            --text-secondary: #666666;
            --accent-color: #2563eb;
            --border-color: #e5e7eb;
            --card-bg: #ffffff;
            --card-hover: #f3f4f6;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }

        [data-theme="dark"] {
            --bg-primary: #111827;
            --bg-secondary: #1f2937;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-color: #3b82f6;
            --border-color: #374151;
            --card-bg: #1f2937;
            --card-hover: #374151;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.18);
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 0;
            transition: background-color 0.3s, color 0.3s;
            line-height: 1.6;
        }

        header {
            background-color: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 2rem 0;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(135deg, var(--accent-color), #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }

        .theme-toggle {
            background: none;
            border: 1px solid var(--border-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .theme-toggle:hover {
            background-color: var(--card-hover);
            transform: rotate(15deg);
        }

        main {
            padding: 3rem 0;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 2rem;
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            flex-direction: column;
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border-color: var(--accent-color);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }

        .date {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-family: 'Space Grotesk', monospace;
            background-color: var(--bg-secondary);
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
        }

        .card-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.4rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
            color: var(--text-primary);
        }

        .card-desc {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
            flex-grow: 1;
        }

        .card-actions {
            display: flex;
            gap: 0.75rem;
            margin-top: auto;
        }

        .btn {
            padding: 0.6rem 1rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
        }

        .btn-primary {
            background-color: var(--accent-color);
            color: white;
            border: 1px solid var(--accent-color);
        }

        .btn-primary:hover {
            background-color: #1d4ed8;
            border-color: #1d4ed8;
        }

        .btn-outline {
            background-color: transparent;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-outline:hover {
            border-color: var(--accent-color);
            color: var(--accent-color);
            background-color: var(--bg-secondary);
        }

        footer {
            text-align: center;
            padding: 2rem 0;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            margin-top: 4rem;
        }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 9999px;
            margin-right: 0.5rem;
        }

        .badge-rl { background-color: #dbeafe; color: #1e40af; }
        .badge-marl { background-color: #fce7f3; color: #9d174d; }
        .badge-control { background-color: #d1fae5; color: #065f46; }
    </style>
</head>

<body>
    <header>
        <div class="container header-content">
            <div>
                <h1>InsightFromPapers</h1>
                <div class="subtitle">Deep Analysis of AI Research Papers</div>
            </div>
            <button class="theme-toggle" id="themeToggle" title="Toggle Theme">
                <i class="fas fa-moon"></i>
            </button>
        </div>
    </header>

    <main class="container">
        <div class="grid">
            {cards_html}
        </div>
    </main>

    <footer>
        <div class="container">
            <p>&copy; 2026 Insightque. Analysis powered by Antigravity AI.</p>
        </div>
    </footer>

    <script>
        const themeToggle = document.getElementById('themeToggle');
        const icon = themeToggle.querySelector('i');

        // Check system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.setAttribute('data-theme', 'dark');
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        }

        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            if (currentTheme === 'dark') {
                document.documentElement.removeAttribute('data-theme');
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            } else {
                document.documentElement.setAttribute('data-theme', 'dark');
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            }
        });
    </script>
</body>

</html>
"""

CARD_TEMPLATE = """
            <article class="card">
                <div class="card-header">
                    <div>
                        {badges_html}
                    </div>
                    <span class="date">{date}</span>
                </div>
                <h2 class="card-title">{title}</h2>
                <p class="card-desc">
                    {description}
                </p>
                <div class="card-actions">
                    <a href="{folder}/{report_file}" class="btn btn-primary">
                        <i class="fas fa-file-code"></i> Web Report
                    </a>
                    {paper_link}
                </div>
            </article>
"""

def generate_badges(badges, badge_class):
    html = ""
    for badge in badges:
        html += f'<span class="badge {badge_class}">{badge}</span>'
    return html

def main():
    base_dir = os.getcwd()
    # List directories that start with a year (e.g., 2026)
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d[0].isdigit()]
    
    # Sort folders by date descending (assumes YYYY-MM-DD prefix)
    dirs.sort(key=lambda x: x.split('_')[0], reverse=True)

    cards_html = ""

    print(f"Found {len(dirs)} project directories in {base_dir}")

    for d in dirs:
        meta_path = os.path.join(base_dir, d, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"Skipping {d}: metadata.json not found.")
            continue
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {meta_path}")
                continue

        print(f"Processing {meta['title']}...")

        badge_class = meta.get('badge_class', 'badge-rl')
        # Ensure badge_class is valid if empty
        if not badge_class:
            badge_class = 'badge-rl'
            
        badges_html = generate_badges(meta.get('badges', []), badge_class)
        
        paper_link = ""
        # Check if paper file is specified and exists
        if 'paper_file' in meta and meta['paper_file']:
             paper_path = os.path.join(base_dir, d, meta['paper_file'])
             if os.path.exists(paper_path):
                 paper_link = f"""<a href="{d}/{meta['paper_file']}" class="btn btn-outline" target="_blank">
                        <i class="fas fa-book"></i> Paper
                    </a>"""
        
        card = CARD_TEMPLATE.format(
            badges_html=badges_html,
            date=meta['date'],
            title=meta['title'],
            description=meta['description'],
            folder=d,
            report_file=meta['report_file'],
            paper_link=paper_link
        )
        cards_html += card

    final_html = HTML_TEMPLATE.replace("{cards_html}", cards_html)

    output_path = os.path.join(base_dir, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    
    print(f"Successfully rebuilt index.html at {output_path}")

if __name__ == "__main__":
    main()
