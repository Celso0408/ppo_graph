import argparse
from pathlib import Path
from bs4 import BeautifulSoup
import html2text

# Determine project root two levels above this file
REPO_ROOT = Path(__file__).resolve().parents[1]

# Default directories relative to the repository
DEFAULT_SOURCE = REPO_ROOT / "html2markdown"
DEFAULT_DEST = REPO_ROOT / "markdown_files"

def convert_html_to_markdown(html_content):
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    return h.handle(html_content)

def process_html_files(source_dir: Path, dest_dir: Path):
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HTML files in a directory to Markdown"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Directory containing HTML files (default: html2markdown)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Output directory for Markdown files (default: markdown_files)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        f"📂 Converting HTML files from {args.source} to Markdown in {args.dest}"
    )
    process_html_files(args.source, args.dest)
    print("✅ All conversions done.")
