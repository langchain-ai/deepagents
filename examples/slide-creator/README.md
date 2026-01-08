# AI Slide Creator

A web demo showcasing **Deep Agents + Skills** for AI-powered slideshow generation.

Uses **Claude Sonnet 4.5** (via deepagents) for conversation and **Nano Banana Pro** for image generation.

## What This Demonstrates

1. **Deep Agents Framework**: Uses `create_deep_agent()` with the skills system
2. **Skills System**: The `slides` skill teaches the agent how to create consistent slideshows
3. **Custom Tools**: Agent uses tools to create, generate, and edit slides
4. **Real-time UI**: WebSocket-based chat with live slideshow preview

## Quick Start

```bash
# 1. Install the example (from this directory)
uv pip install -e .

# 2. Set your API keys
export ANTHROPIC_API_KEY=your_anthropic_key
export GOOGLE_API_KEY=your_google_key

# 3. Run the app
uv run python app.py
```

Open http://localhost:8000 in your browser.

## How It Works

### Architecture

```
+------------------------------------------------------------------+
|                     Browser (localhost:8000)                      |
|  +---------------------+    +-------------------------------+     |
|  |     Chat Panel      |    |       Slideshow Preview       |     |
|  |   (WebSocket)       |    |    (Real-time updates)        |     |
|  +----------+----------+    +---------------+---------------+     |
+-------------|---------------------------------|-------------------+
              |                                 |
              v                                 v
+------------------------------------------------------------------+
|                     FastAPI Backend (app.py)                      |
|  +--------------------------------------------------------------+ |
|  |                      Deep Agent                               | |
|  |  +-----------------+  +-----------------+                     | |
|  |  | Skills          |  | Tools           |                     | |
|  |  | Middleware      |  | - create_slide  |                     | |
|  |  |                 |  | - generate_img  |                     | |
|  |  | Loads SKILL.md  |  | - edit_slide    |                     | |
|  |  +--------+--------+  +--------+--------+                     | |
|  |           |                    |                              | |
|  |           v                    v                              | |
|  |  +---------------------------------------------+              | |
|  |  |              Claude Sonnet 4.5              |              | |
|  |  +---------------------------------------------+              | |
|  +--------------------------------------------------------------+ |
|                               |                                   |
|                               v                                   |
|  +--------------------------------------------------------------+ |
|  |        Nano Banana Pro (gemini-3-pro-image-preview)           | |
|  +--------------------------------------------------------------+ |
+------------------------------------------------------------------+
```

### The Skills System

The agent loads the `slides` skill from `skills/slides/SKILL.md`. This skill teaches the agent:

1. **How to create style guides** - Consistent colors, art style, mood
2. **How to write image prompts** - Natural language, specific details
3. **How to maintain consistency** - Same style prefix for every slide
4. **How to handle edits** - Pass reference image + edit instructions

When you ask the agent to create a slideshow:

1. Agent sees "slides" skill via SkillsMiddleware
2. Agent reads the full `SKILL.md` to get detailed instructions
3. Agent follows the workflow: clarify -> style guide -> outline -> generate
4. Agent uses tools to create slides and generate images

### Custom Tools

The agent has access to these tools:

| Tool | Description |
|------|-------------|
| `create_new_slideshow` | Create a new slideshow with topic and style guide |
| `add_slide_to_slideshow` | Add a slide with title and scene prompt |
| `generate_slide_image` | Generate the image for a slide |
| `edit_existing_slide` | Edit an existing slide with reference image |
| `finalize_slideshow` | Generate the HTML viewer |
| `get_slideshow_info` | Check slideshow progress |

## Project Structure

```
examples/slide-creator/
├── app.py                    # FastAPI + Deep Agents backend
├── static/
│   ├── index.html            # Main page
│   ├── style.css             # Dark theme styling
│   └── app.js                # WebSocket chat + slideshow UI
├── skills/
│   └── slides/
│       └── SKILL.md          # Agent instructions for slideshow creation
├── slide_creator/
│   ├── image_gen.py          # Nano Banana Pro image generation
│   ├── slideshow.py          # Slideshow state management
│   └── templates/
│       └── slideshow.html    # Export template
├── output/                   # Generated slideshows
└── pyproject.toml
```

## Usage

### Creating a Slideshow

1. Open http://localhost:8000
2. Type: "Create a slideshow about quantum computing"
3. Agent will ask clarifying questions (audience, mood, colors)
4. Watch as slides are generated in real-time
5. Click "Export" to open the standalone viewer

### Editing Slides

1. Navigate to the slide you want to edit
2. Type in the edit box: "Add more dramatic lighting"
3. Agent will regenerate with your changes

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `<-` `->` | Navigate slides |

## Configuration

### Environment Variables

```bash
ANTHROPIC_API_KEY=your_anthropic_key  # Required - for Claude chat
GOOGLE_API_KEY=your_google_key        # Required - for Nano Banana Pro images
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929  # Optional - override model
```

### Customizing the Skill

Edit `skills/slides/SKILL.md` to change how the agent creates slideshows:

- Modify the default style guide suggestions
- Change the prompting strategy
- Add new workflows (e.g., for specific industries)

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run with auto-reload
uv run uvicorn app:app --reload --port 8000

# Lint
uv run ruff check .
```

## Troubleshooting

**WebSocket disconnects immediately**
- Check that ANTHROPIC_API_KEY is set
- Verify deepagents is installed: `uv pip list | grep deepagents`

**Images not generating**
- Check that GOOGLE_API_KEY is set
- Nano Banana Pro requires a valid Google AI API key

**Slides look inconsistent**
- The skill instructs the agent to use identical style prefixes
- Try being more specific about the style you want
