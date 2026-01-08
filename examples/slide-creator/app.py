"""
Slide Creator Web App - Deep Agents Demo

A polished web interface for AI-powered slideshow generation,
demonstrating the deepagents skills system with Nano Banana Pro image generation.
"""

import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# Deep Agents imports
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from slide_creator import generate_image, edit_image, Slideshow, StyleGuide, create_slideshow

# Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
SKILLS_DIR = BASE_DIR / "skills"

app = FastAPI(title="AI Slide Creator", description="Deep Agents + Skills Demo")

# Serve static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Mount output directory if it exists
OUTPUT_DIR.mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ============================================================================
# Tools for the Agent
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
        JSON with generation result including image_url
    """
    slides_json = OUTPUT_DIR / slideshow_id / "slides.json"
    if not slides_json.exists():
        return json.dumps({"error": f"Slideshow '{slideshow_id}' not found"})

    slideshow = Slideshow.from_json(slides_json)

    slide = slideshow.get_slide(slide_index)
    if not slide:
        return json.dumps({"error": f"Slide {slide_index} not found"})

    output_path = Path(slideshow.output_dir) / slide.image_filename

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
            "image_url": f"/output/{slideshow_id}/{slide.image_filename}",
            "message": f"Generated image for slide {slide_index}: {slide.title}",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def edit_existing_slide(slideshow_id: str, slide_index: int, edit_instructions: str) -> str:
    """Edit an existing slide image based on instructions.

    Args:
        slideshow_id: The slideshow folder name
        slide_index: The slide number (1-based)
        edit_instructions: What to change (e.g., "Add dramatic storm clouds in the background")

    Returns:
        JSON with edit result
    """
    slides_json = OUTPUT_DIR / slideshow_id / "slides.json"
    if not slides_json.exists():
        return json.dumps({"error": f"Slideshow '{slideshow_id}' not found"})

    slideshow = Slideshow.from_json(slides_json)

    slide = slideshow.get_slide(slide_index)
    if not slide:
        return json.dumps({"error": f"Slide {slide_index} not found"})

    image_path = Path(slideshow.output_dir) / slide.image_filename
    if not image_path.exists():
        return json.dumps({"error": f"Slide {slide_index} image not found. Generate it first."})

    # Build edit prompt that preserves style
    edit_prompt = f"""Using this image as a reference, {edit_instructions}.
Maintain the same art style, color palette, and composition.
Keep all elements that aren't specifically being changed."""

    try:
        edit_image(
            prompt=edit_prompt,
            reference_image_path=image_path,
            output_path=image_path,
            aspect_ratio="16:9",
            image_size="2K",
        )

        slideshow.save()
        cache_buster = uuid.uuid4().hex[:8]

        return json.dumps({
            "slide_index": slide_index,
            "image_url": f"/output/{slideshow_id}/{slide.image_filename}?t={cache_buster}",
            "message": f"Edited slide {slide_index}",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def finalize_slideshow(slideshow_id: str) -> str:
    """Generate the HTML viewer for the completed slideshow.

    Args:
        slideshow_id: The slideshow folder name

    Returns:
        JSON with viewer URL
    """
    slides_json = OUTPUT_DIR / slideshow_id / "slides.json"
    if not slides_json.exists():
        return json.dumps({"error": f"Slideshow '{slideshow_id}' not found"})

    slideshow = Slideshow.from_json(slides_json)
    slideshow.generate_html()

    return json.dumps({
        "viewer_url": f"/output/{slideshow_id}/index.html",
        "message": f"Slideshow ready! {len(slideshow.slides)} slides generated.",
    })


@tool
def get_slideshow_info(slideshow_id: str) -> str:
    """Get the current status of a slideshow.

    Args:
        slideshow_id: The slideshow folder name

    Returns:
        JSON with slideshow details
    """
    slides_json = OUTPUT_DIR / slideshow_id / "slides.json"
    if not slides_json.exists():
        return json.dumps({"error": f"Slideshow '{slideshow_id}' not found"})

    slideshow = Slideshow.from_json(slides_json)

    return json.dumps({
        "topic": slideshow.topic,
        "total_slides": len(slideshow.slides),
        "generated_slides": sum(1 for s in slideshow.slides if s.generated),
        "slides": [
            {
                "index": s.index,
                "title": s.title,
                "generated": s.generated,
            }
            for s in slideshow.slides
        ],
    })


# Collect all tools
TOOLS = [
    create_new_slideshow,
    add_slide_to_slideshow,
    generate_slide_image,
    edit_existing_slide,
    finalize_slideshow,
    get_slideshow_info,
]


# ============================================================================
# Agent Factory
# ============================================================================

SYSTEM_PROMPT = """You are an AI slide creator assistant. You help users create beautiful, visually consistent slideshows using AI image generation.

IMPORTANT WORKFLOW:
1. When asked to create a slideshow, first ask clarifying questions about:
   - Target audience (general public, technical, investors, etc.)
   - Mood/tone (professional, playful, dramatic, inspiring)
   - Color preferences (or suggest based on topic)

2. After getting preferences, create a detailed style guide and explain it to the user.

3. Create an outline of 6-10 slides with titles and descriptions.

4. Generate slides ONE AT A TIME, telling the user after each one completes.

5. After all slides are generated, finalize the slideshow.

Be conversational and helpful. Keep responses concise but informative.
"""


def create_agent():
    """Create a deep agent with tools and skills."""
    # Use Claude Sonnet 4.5 for chat, skills loaded from local directory
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

    # Create checkpointer for conversation memory
    checkpointer = MemorySaver()

    # Set up skills middleware to load from local skills directory
    skills_middleware = SkillsMiddleware(
        backend=FilesystemBackend(),
        sources=[str(SKILLS_DIR)],
    )

    # Create the agent using deepagents
    agent = create_deep_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[skills_middleware],
        checkpointer=checkpointer,
    )

    return agent


# Global agent instance
AGENT = None


def get_agent():
    global AGENT
    if AGENT is None:
        AGENT = create_agent()
    return AGENT


# ============================================================================
# WebSocket Chat Handler
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.thread_ids: dict[str, str] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.thread_ids[session_id] = str(uuid.uuid4())

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.thread_ids.pop(session_id, None)

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if not user_message:
                continue

            agent = get_agent()
            thread_id = manager.thread_ids.get(session_id)

            # Send thinking indicator
            await manager.send_message(session_id, {
                "type": "thinking",
                "content": "Thinking...",
            })

            try:
                config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 100,
                }

                full_response = ""

                # Stream the response
                async for event in agent.astream_events(
                    {"messages": [HumanMessage(content=user_message)]},
                    config=config,
                    version="v2",
                ):
                    kind = event["event"]

                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                            # Handle both string and list content (Anthropic returns list of blocks)
                            if isinstance(content, str):
                                text = content
                            elif isinstance(content, list) and len(content) > 0:
                                # Extract text from content blocks
                                text = ""
                                for block in content:
                                    if hasattr(block, "text"):
                                        text += block.text
                                    elif isinstance(block, dict) and "text" in block:
                                        text += block["text"]
                            else:
                                text = ""

                            if text:
                                full_response += text
                                await manager.send_message(session_id, {
                                    "type": "stream",
                                    "content": text,
                                })

                    elif kind == "on_tool_start":
                        tool_name = event["name"]
                        await manager.send_message(session_id, {
                            "type": "tool_start",
                            "tool": tool_name,
                        })

                    elif kind == "on_tool_end":
                        tool_name = event["name"]
                        output = event["data"].get("output", "")
                        # Convert to string if it's a ToolMessage or other object
                        if hasattr(output, "content"):
                            output = output.content
                        elif not isinstance(output, str):
                            output = str(output)

                        await manager.send_message(session_id, {
                            "type": "tool_end",
                            "tool": tool_name,
                            "output": output,
                        })

                        # Parse tool output for UI updates
                        try:
                            output_data = json.loads(output) if isinstance(output, str) else output

                            if "slideshow_id" in output_data and "message" in output_data:
                                if "Created slideshow" in output_data.get("message", ""):
                                    await manager.send_message(session_id, {
                                        "type": "slideshow_created",
                                        "data": output_data,
                                    })

                            if "image_url" in output_data:
                                await manager.send_message(session_id, {
                                    "type": "slide_generated",
                                    "data": output_data,
                                })
                        except (json.JSONDecodeError, TypeError):
                            pass

                await manager.send_message(session_id, {
                    "type": "done",
                    "content": full_response,
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                await manager.send_message(session_id, {
                    "type": "error",
                    "content": f"Error: {str(e)}",
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/slideshows")
async def list_slideshows():
    slideshows = []
    if OUTPUT_DIR.exists():
        for folder in OUTPUT_DIR.iterdir():
            if folder.is_dir() and (folder / "slides.json").exists():
                try:
                    slideshow = Slideshow.from_json(folder / "slides.json")
                    slideshows.append({
                        "id": folder.name,
                        "topic": slideshow.topic,
                        "slides": len(slideshow.slides),
                    })
                except Exception:
                    pass
    return JSONResponse({"slideshows": slideshows})


@app.get("/api/slideshow/{slideshow_id}")
async def get_slideshow(slideshow_id: str):
    slides_json = OUTPUT_DIR / slideshow_id / "slides.json"
    if not slides_json.exists():
        return JSONResponse({"error": "Slideshow not found"}, status_code=404)

    slideshow = Slideshow.from_json(slides_json)
    return JSONResponse({
        "id": slideshow_id,
        "topic": slideshow.topic,
        "slides": [
            {
                "index": s.index,
                "title": s.title,
                "content_description": s.content_description,
                "generated": s.generated,
                "image_url": f"/output/{slideshow_id}/{s.image_filename}" if s.generated else None,
            }
            for s in slideshow.slides
        ],
    })


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print("  AI Slide Creator - Deep Agents Demo")
    print("=" * 50)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Skills: {SKILLS_DIR}")
    print(f"  URL: http://localhost:8000")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
