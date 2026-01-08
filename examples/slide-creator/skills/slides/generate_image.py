#!/usr/bin/env python3
"""Generate an image for a slide using Nano Banana Pro."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slide_creator import Slideshow, generate_image

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


def main():
    parser = argparse.ArgumentParser(description="Generate image for a slide")
    parser.add_argument("--slideshow-id", required=True, help="Slideshow folder name")
    parser.add_argument("--slide-index", required=True, type=int, help="Slide number (1-based)")
    args = parser.parse_args()

    slides_json = OUTPUT_DIR / args.slideshow_id / "slides.json"
    if not slides_json.exists():
        print(json.dumps({"error": f"Slideshow '{args.slideshow_id}' not found"}))
        sys.exit(1)

    slideshow = Slideshow.from_json(slides_json)
    slide = slideshow.get_slide(args.slide_index)

    if not slide:
        print(json.dumps({"error": f"Slide {args.slide_index} not found"}))
        sys.exit(1)

    output_path = Path(slideshow.output_dir) / slide.image_filename

    print(f"Generating image for slide {args.slide_index}: {slide.title}...", file=sys.stderr)

    try:
        generate_image(
            prompt=slide.full_prompt,
            output_path=output_path,
            aspect_ratio="16:9",
            image_size="2K",
        )
        slide.generated = True
        slideshow.save()

        result = {
            "slide_index": args.slide_index,
            "image_path": str(output_path),
            "message": f"Generated image for slide {args.slide_index}",
        }
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
