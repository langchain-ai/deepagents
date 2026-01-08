#!/usr/bin/env python3
"""
Slide Creator CLI - Deep Agents + Skills Demo

Generate AI-powered slideshows from the command line.
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from slide_creator import generate_image, edit_image, Slideshow, StyleGuide, create_slideshow

# Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
SKILLS_DIR = BASE_DIR / "skills"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Tools
# ============================================================================

@tool
def create_new_slideshow(topic: str, art_style: str, color_palette: str, mood: str, lighting: str) -> str:
    """Create a new slideshow with the given topic and style guide.

    Args:
        topic: The slideshow topic (e.g., "Renewable Energy")
        art_style: Visual style (e.g., "flat vector illustration with subtle gradients")
        color_palette: Comma-separated hex colors (e.g., "#1E3A5F, #FF6B6B, #FFF8F0")
        mood: Emotional tone (e.g., "professional yet approachable")
        lighting: Lighting description (e.g., "soft ambient with gentle gradients")

    Returns:
        JSON with slideshow_id and confirmation message
    """
    colors = [c.strip() for c in color_palette.split(",")]

    style = StyleGuide(
        art_style=art_style,
        color_palette=colors,
        mood=mood,
        lighting=lighting,
        additional_notes="Clean composition optimized for 16:9 presentation slides",
    )

    slideshow = create_slideshow(
        topic=topic,
        style_guide=style,
        base_output_dir=str(OUTPUT_DIR),
    )

    return json.dumps({
        "slideshow_id": Path(slideshow.output_dir).name,
        "output_dir": slideshow.output_dir,
        "message": f"Created slideshow '{topic}'",
    })


@tool
def add_slide_to_slideshow(slideshow_id: str, title: str, content_description: str, scene_prompt: str) -> str:
    """Add a slide to an existing slideshow.

    Args:
        slideshow_id: The slideshow folder name (returned from create_new_slideshow)
        title: Slide title (e.g., "Introduction")
        content_description: What the slide is about
        scene_prompt: Detailed description of what should be in the image (the style prefix will be auto-added)

    Returns:
        JSON with slide info
    """
    slides_json = OUTPUT_DIR / slideshow_id / "slides.json"
    if not slides_json.exists():
        return json.dumps({"error": f"Slideshow '{slideshow_id}' not found"})

    slideshow = Slideshow.from_json(slides_json)

    slide = slideshow.add_slide(
        title=title,
        content_description=content_description,
        scene_prompt=scene_prompt,
    )
    slideshow.save()

    return json.dumps({
        "slide_index": slide.index,
        "title": title,
        "image_filename": slide.image_filename,
        "message": f"Added slide {slide.index}: {title}",
    })


@tool
def generate_slide_image(slideshow_id: str, slide_index: int) -> str:
    """Generate the image for a specific slide using AI.

    Args:
        slideshow_id: The slideshow folder name
        slide_index: The slide number (1-based, e.g., 1 for first slide)

    Returns:
        JSON with generation result including image path
    """
    slides_json = OUTPUT_DIR / slideshow_id / "slides.json"
    if not slides_json.exists():
        return json.dumps({"error": f"Slideshow '{slideshow_id}' not found"})

    slideshow = Slideshow.from_json(slides_json)

    slide = slideshow.get_slide(slide_index)
    if not slide:
        return json.dumps({"error": f"Slide {slide_index} not found"})

    output_path = Path(slideshow.output_dir) / slide.image_filename

    print(f"  Generating image for slide {slide_index}: {slide.title}...")

    try:
        generate_image(
            prompt=slide.full_prompt,
            output_path=output_path,
            aspect_ratio="16:9",
            image_size="2K",
        )
        slide.generated = True
        slideshow.save()

        return json.dumps({
            "slide_index": slide_index,
            "image_path": str(output_path),
            "message": f"Generated image for slide {slide_index}: {slide.title}",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def finalize_slideshow(slideshow_id: str) -> str:
    """Generate the HTML viewer for the completed slideshow.

    Args:
        slideshow_id: The slideshow folder name

    Returns:
        JSON with viewer path
    """
    slides_json = OUTPUT_DIR / slideshow_id / "slides.json"
    if not slides_json.exists():
        return json.dumps({"error": f"Slideshow '{slideshow_id}' not found"})

    slideshow = Slideshow.from_json(slides_json)
    html_path = slideshow.generate_html()

    return json.dumps({
        "viewer_path": str(html_path),
        "message": f"Slideshow ready! {len(slideshow.slides)} slides generated.",
    })


TOOLS = [
    create_new_slideshow,
    add_slide_to_slideshow,
    generate_slide_image,
    finalize_slideshow,
]

SYSTEM_PROMPT = """You are an AI slide creator. Create visually consistent slideshows using AI image generation.

When asked to create a slideshow:
1. Create a style guide (art style, colors, mood, lighting)
2. Create the slideshow with that style
3. Add 4-6 slides with descriptive scene prompts
4. Generate images for each slide ONE AT A TIME
5. Finalize the slideshow

Keep it simple and efficient. Don't ask clarifying questions - just create something great.
"""


def create_agent():
    """Create the slide creator agent."""
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

    skills_middleware = SkillsMiddleware(
        backend=FilesystemBackend(),
        sources=[str(SKILLS_DIR)],
    )

    agent = create_deep_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[skills_middleware],
    )

    return agent


def main():
    import sys

    print("\n" + "=" * 50)
    print("  Slide Creator CLI - Deep Agents Demo")
    print("=" * 50 + "\n")

    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("Enter slideshow topic: ").strip()
        if not topic:
            topic = "The Future of AI"

    print(f"\nCreating slideshow: {topic}\n")

    agent = create_agent()

    config = {"recursion_limit": 100}

    result = agent.invoke(
        {"messages": [HumanMessage(content=f"Create a slideshow about: {topic}")]},
        config=config,
    )

    # Print final message
    final_message = result["messages"][-1]
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)

    # Find and print the output path
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and "viewer_path" in str(msg.content):
            try:
                data = json.loads(msg.content)
                if "viewer_path" in data:
                    print(f"\nOpen in browser: file://{data['viewer_path']}")
                    break
            except:
                pass


if __name__ == "__main__":
    main()
