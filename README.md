# InsightFromPapers

InsightFromPapers is a repository dedicated to the technical analysis, organization, and automated report generation for various research papers, with a strong focus on Reinforcement Learning (RL) and AI methodologies.

## Repository Overview

This repository automates the workflow of extracting text from academic papers, analyzing them, and generating stylized HTML reports. It includes an automated system to maintain an updated index page (`index.html`) that links to all paper analyses.

### Key Features
- **Automated Web Page Generation**: `build_index.py` automatically scans paper directories, reads their `metadata.json`, and rebuilds the main `index.html` page to showcase the latest reports.
- **Report Generation**: `generate_web_report.py` creates visually appealing HTML reports based on the technical analysis of the papers.
- **Workflow Setup**: `setup_paper.py` facilitates the quick initialization of a new paper directory structure.
- **Text Extraction**: `extract_text.py` extracts raw text from PDF files for further AI-based analysis.

## Project Structure

- `2026-02-16_*`: Directories containing individual research papers, their extracted data, metadata, and the generated HTML reports.
- `build_index.py`: Script to generate the root `index.html`.
- `generate_web_report.py`: Script to parse markdown analyses and convert them into HTML reports.
- `setup_paper.py`: Utility to scaffold new paper reviews.
- `publish.py`: Script to build and optionally publish the latest updates.

## Automation & Deployment

This project uses Python scripts to automate the management of research papers. Whenever a new paper analysis is completed, running the automation scripts will update the `index.html` to reflect the new addition.
