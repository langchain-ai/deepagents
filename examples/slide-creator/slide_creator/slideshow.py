"""Slideshow state management and HTML generation."""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union, List
from jinja2 import Environment, FileSystemLoader


@dataclass
class StyleGuide:
    """Visual style guide for maintaining consistency across slides."""

    art_style: str  # e.g., "flat vector illustration", "3D render", "watercolor"
    color_palette: List[str]  # Hex codes, e.g., ["#1E3A5F", "#FF6B6B", "#FFF8F0"]
    mood: str  # e.g., "professional", "playful", "dramatic"
    lighting: str  # e.g., "soft ambient", "dramatic shadows", "bright and airy"
    additional_notes: str = ""  # Any extra style instructions

    def to_prompt_prefix(self) -> str:
        """Generate a style prefix to prepend to every image prompt."""
        colors = ", ".join(self.color_palette)
        return (
            f"A polished presentation slide in {self.art_style} style. "
            f"Color palette: {colors}. "
            f"Mood: {self.mood}. "
            f"Lighting: {self.lighting}. "
            f"{self.additional_notes} "
            f"Clean composition optimized for a 16:9 presentation slide. "
        )


@dataclass
class Slide:
    """A single slide in the slideshow."""

    index: int
    title: str
    content_description: str  # What this slide is about
    full_prompt: str  # The complete prompt used to generate the image
    image_filename: str  # e.g., "slide_001.png"
    generated: bool = False

    @property
    def image_path(self) -> str:
        return self.image_filename


@dataclass
class Slideshow:
    """Complete slideshow with all slides and metadata."""

    topic: str
    style_guide: StyleGuide
    slides: List[Slide] = field(default_factory=list)
    output_dir: Optional[str] = None

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "Slideshow":
        """Load a slideshow from a slides.json file."""
        json_path = Path(json_path)
        with open(json_path) as f:
            data = json.load(f)

        style_guide = StyleGuide(**data["style_guide"])
        slides = [Slide(**s) for s in data["slides"]]

        return cls(
            topic=data["topic"],
            style_guide=style_guide,
            slides=slides,
            output_dir=str(json_path.parent),
        )

    def to_json(self) -> dict:
        """Convert slideshow to JSON-serializable dict."""
        return {
            "topic": self.topic,
            "style_guide": asdict(self.style_guide),
            "slides": [asdict(s) for s in self.slides],
        }

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Save slideshow state to slides.json."""
        output_dir = Path(output_dir or self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(output_dir)

        json_path = output_dir / "slides.json"
        with open(json_path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

        return json_path

    def add_slide(
        self,
        title: str,
        content_description: str,
        scene_prompt: str,
    ) -> Slide:
        """
        Add a new slide to the slideshow.

        Args:
            title: Slide title
            content_description: What the slide is about
            scene_prompt: The scene-specific part of the prompt (style prefix auto-added)

        Returns:
            The created Slide object
        """
        index = len(self.slides) + 1
        full_prompt = self.style_guide.to_prompt_prefix() + f"Scene: {scene_prompt}"

        slide = Slide(
            index=index,
            title=title,
            content_description=content_description,
            full_prompt=full_prompt,
            image_filename=f"slide_{index:03d}.png",
            generated=False,
        )
        self.slides.append(slide)
        return slide

    def get_slide(self, index: int) -> Optional[Slide]:
        """Get a slide by its 1-based index."""
        for slide in self.slides:
            if slide.index == index:
                return slide
        return None

    def update_slide(self, index: int, **kwargs) -> Optional[Slide]:
        """Update a slide's properties."""
        slide = self.get_slide(index)
        if slide:
            for key, value in kwargs.items():
                if hasattr(slide, key):
                    setattr(slide, key, value)
        return slide

    def generate_html(self, template_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Generate the HTML slideshow viewer.

        Args:
            template_dir: Directory containing slideshow.html template.
                         Defaults to the templates directory in this package.

        Returns:
            Path to the generated index.html
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        else:
            template_dir = Path(template_dir)

        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template("slideshow.html")

        html_content = template.render(
            topic=self.topic,
            slides=self.slides,
            style_guide=self.style_guide,
        )

        output_dir = Path(self.output_dir)
        html_path = output_dir / "index.html"
        with open(html_path, "w") as f:
            f.write(html_content)

        return html_path


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text[:50]  # Limit length


def create_slideshow(
    topic: str,
    style_guide: StyleGuide,
    base_output_dir: Union[str, Path] = "output",
) -> Slideshow:
    """
    Create a new slideshow.

    Args:
        topic: The slideshow topic
        style_guide: Visual style configuration
        base_output_dir: Base directory for output (topic slug appended)

    Returns:
        New Slideshow instance
    """
    base_output_dir = Path(base_output_dir)
    output_dir = base_output_dir / slugify(topic)
    output_dir.mkdir(parents=True, exist_ok=True)

    slideshow = Slideshow(
        topic=topic,
        style_guide=style_guide,
        output_dir=str(output_dir),
    )
    slideshow.save()

    return slideshow
