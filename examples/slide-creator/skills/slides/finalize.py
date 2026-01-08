#!/usr/bin/env python3
"""Finalize a slideshow by generating the HTML viewer."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slide_creator import Slideshow

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


def main():
    parser = argparse.ArgumentParser(description="Finalize slideshow and generate HTML")
    parser.add_argument("--slideshow-id", required=True, help="Slideshow folder name")
    args = parser.parse_args()

    slides_json = OUTPUT_DIR / args.slideshow_id / "slides.json"
    if not slides_json.exists():
        print(json.dumps({"error": f"Slideshow '{args.slideshow_id}' not found"}))
        sys.exit(1)

    slideshow = Slideshow.from_json(slides_json)
    html_path = slideshow.generate_html()

    result = {
        "viewer_path": str(html_path),
        "total_slides": len(slideshow.slides),
        "message": f"Slideshow ready! Open: file://{html_path}",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
