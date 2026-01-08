#!/usr/bin/env python3
"""Add a slide to an existing slideshow."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slide_creator import Slideshow

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


def main():
    parser = argparse.ArgumentParser(description="Add a slide to a slideshow")
    parser.add_argument("--slideshow-id", required=True, help="Slideshow folder name")
    parser.add_argument("--title", required=True, help="Slide title")
    parser.add_argument("--description", required=True, help="What the slide is about")
    parser.add_argument("--scene", required=True, help="Scene description for image generation")
    args = parser.parse_args()

    slides_json = OUTPUT_DIR / args.slideshow_id / "slides.json"
    if not slides_json.exists():
        print(json.dumps({"error": f"Slideshow '{args.slideshow_id}' not found"}))
        sys.exit(1)

    slideshow = Slideshow.from_json(slides_json)

    slide = slideshow.add_slide(
        title=args.title,
        content_description=args.description,
        scene_prompt=args.scene,
    )
    slideshow.save()

    result = {
        "slide_index": slide.index,
        "title": args.title,
        "image_filename": slide.image_filename,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
