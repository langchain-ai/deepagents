#!/usr/bin/env python3
"""Create a new slideshow with a style guide."""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slide_creator import create_slideshow, StyleGuide

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


def main():
    parser = argparse.ArgumentParser(description="Create a new slideshow")
    parser.add_argument("--topic", required=True, help="Slideshow topic")
    parser.add_argument("--art-style", required=True, help="Art style description")
    parser.add_argument("--colors", required=True, help="Comma-separated hex colors")
    parser.add_argument("--mood", required=True, help="Mood/tone")
    parser.add_argument("--lighting", required=True, help="Lighting description")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    style = StyleGuide(
        art_style=args.art_style,
        color_palette=[c.strip() for c in args.colors.split(",")],
        mood=args.mood,
        lighting=args.lighting,
        additional_notes="Clean composition optimized for 16:9 presentation slides",
    )

    slideshow = create_slideshow(
        topic=args.topic,
        style_guide=style,
        base_output_dir=str(OUTPUT_DIR),
    )

    result = {
        "slideshow_id": Path(slideshow.output_dir).name,
        "output_dir": slideshow.output_dir,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
