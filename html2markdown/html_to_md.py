import os
from pathlib import Path
from bs4 import BeautifulSoup
import html2text

SOURCE_DIR = Path("/home/celso/GENIUS/smart_gk/my_grath/html2markdown")
DEST_DIR = Path("/home/celso/GENIUS/smart_gk/my_grath/markdown_files/")

def convert_html_to_markdown(html_content):
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    return h.handle(html_content)

def process_html_files(source_dir, dest_dir):
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)

    for html_file in source_dir.rglob("*.html"):
        print(f"🔄 Converting {html_file.name}")
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            body = soup.body or soup  # fallback if <body> is missing
            markdown = convert_html_to_markdown(str(body))

        md_filename = html_file.stem + ".md"
        md_path = dest_dir / md_filename
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"✅ Saved {md_path}")

if __name__ == "__main__":
    print(f"📂 Converting HTML files from {SOURCE_DIR} to Markdown in {DEST_DIR}")
    process_html_files(SOURCE_DIR, DEST_DIR)
    print("✅ All conversions done.")
