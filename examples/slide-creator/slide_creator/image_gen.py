"""Image generation using Google's Nano Banana Pro (gemini-3-pro-image-preview)."""

import os
from pathlib import Path
from typing import Union

from google import genai
from google.genai import types
from PIL import Image

# Nano Banana Pro - advanced model for professional asset production
MODEL_ID = "gemini-3-pro-image-preview"


def get_client() -> genai.Client:
    """Get configured Gemini client."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


def generate_image(
    prompt: str,
    output_path: Union[str, Path],
    aspect_ratio: str = "16:9",
    image_size: str = "2K",
) -> Path:
    """
    Generate an image from a text prompt using Nano Banana Pro.

    Args:
        prompt: Detailed text prompt for image generation
        output_path: Path to save the generated image
        aspect_ratio: Aspect ratio (default "16:9" for slides)
        image_size: Resolution - "1K", "2K", or "4K"

    Returns:
        Path to the saved image
    """
    client = get_client()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
        ),
    )

    for part in response.parts:
        if part.inline_data is not None:
            image = part.as_image()
            image.save(str(output_path))
            return output_path

    raise RuntimeError("No image was generated in the response")


def edit_image(
    prompt: str,
    reference_image_path: Union[str, Path],
    output_path: Union[str, Path],
    aspect_ratio: str = "16:9",
    image_size: str = "2K",
) -> Path:
    """
    Edit an existing image using Nano Banana Pro.

    Pass a reference image along with an edit prompt to modify the image
    while maintaining style consistency.

    Args:
        prompt: Edit instructions (e.g., "Add dramatic storm clouds to the background")
        reference_image_path: Path to the existing image to edit
        output_path: Path to save the edited image
        aspect_ratio: Aspect ratio (default "16:9" for slides)
        image_size: Resolution - "1K", "2K", or "4K"

    Returns:
        Path to the saved edited image
    """
    client = get_client()
    reference_image_path = Path(reference_image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_image = Image.open(reference_image_path)

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt, reference_image],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
        ),
    )

    for part in response.parts:
        if part.inline_data is not None:
            image = part.as_image()
            image.save(str(output_path))
            return output_path

    raise RuntimeError("No image was generated in the response")
