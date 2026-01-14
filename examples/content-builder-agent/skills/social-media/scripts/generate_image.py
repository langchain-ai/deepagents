"""Generate images for social media posts using Gemini."""

from pathlib import Path

from langchain_core.tools import tool


@tool
def generate_social_image(prompt: str, platform: str, slug: str) -> str:
    """Generate an image for a social media post.

    Args:
        prompt: A detailed description of the image. Should be bold and
                attention-grabbing with simple composition for social media.
        platform: Either "linkedin" or "tweets"
        slug: The post slug (e.g., "ai-agents"). Image will be saved to
              <platform>/<slug>/image.png

    Returns:
        Path to the saved image, or an error message.
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client()

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                output_path = Path(f"{platform}/{slug}/image.png")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(image_data)
                return f"Image saved to {output_path}"

        return "No image was generated"
    except Exception as e:
        return f"Error generating image: {e}"
