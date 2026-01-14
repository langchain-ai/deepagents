"""Generate cover images for blog posts using Gemini."""

from pathlib import Path

from langchain_core.tools import tool


@tool
def generate_cover(prompt: str, slug: str) -> str:
    """Generate a cover image for a blog post.

    Args:
        prompt: A detailed description of the image. Describe the scene, style,
                mood, and composition. Be specific rather than using keywords.
        slug: The blog post slug (e.g., "ai-agents-2025"). Image will be saved
              to blogs/<slug>/hero.png

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
                output_path = Path(f"blogs/{slug}/hero.png")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(image_data)
                return f"Cover image saved to {output_path}"

        return "No image was generated"
    except Exception as e:
        return f"Error generating image: {e}"
