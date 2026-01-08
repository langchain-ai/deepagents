---
name: slides
description: Create AI-generated slideshows with consistent visual style using Nano Banana Pro
---

# Slide Creator Skill

Create stunning, visually consistent slideshows using AI image generation (Nano Banana Pro).

## Workflow

When a user asks you to create a slideshow, follow these steps in order.

### Step 1: Create the Slideshow

Run this command to create a new slideshow with a style guide:

```bash
python skills/slides/create.py \
  --topic "Your Topic Here" \
  --art-style "sleek futuristic illustration with clean lines" \
  --colors "#0A1128, #1E88E5, #00D4FF, #B388FF, #F5F5F5" \
  --mood "professional and inspiring" \
  --lighting "soft ambient with subtle glows"
```

**Save the `slideshow_id` from the output - you need it for all following steps.**

### Step 2: Add Slides (4-6 slides)

For each slide, run:

```bash
python skills/slides/add_slide.py \
  --slideshow-id "your-topic-slug" \
  --title "Slide Title" \
  --description "What this slide covers" \
  --scene "Detailed visual scene description"
```

**Scene descriptions should be vivid and specific.** Example:
- GOOD: "A futuristic cityscape at sunset with flying vehicles, holographic billboards, and people on elevated walkways"
- BAD: "future city"

### Step 3: Generate Images

Generate each slide image ONE AT A TIME:

```bash
python skills/slides/generate_image.py \
  --slideshow-id "your-topic-slug" \
  --slide-index 1
```

Then slide 2, then 3, etc. Each image takes a few seconds to generate.

### Step 4: Finalize

Create the HTML viewer:

```bash
python skills/slides/finalize.py \
  --slideshow-id "your-topic-slug"
```

**Tell the user the file path to open in their browser.**

---

## Style Guide Reference

### Art Styles
- "sleek futuristic illustration with clean lines"
- "minimalist flat design with bold shapes"
- "professional 3D render with soft lighting"
- "elegant watercolor with digital accents"
- "modern vector art with gradients"

### Color Palettes (use 5 colors)
**Tech/AI:**
```
#0A1128, #1E88E5, #00D4FF, #B388FF, #F5F5F5
```

**Nature/Eco:**
```
#1B4332, #40916C, #95D5B2, #D8F3DC, #FFFFFF
```

**Corporate:**
```
#1A1A2E, #16213E, #0F3460, #E94560, #FFFFFF
```

**Warm/Creative:**
```
#2D3436, #E17055, #FDCB6E, #00B894, #FFEAA7
```

### Moods
- "professional and authoritative"
- "inspiring and optimistic"
- "calm and thoughtful"
- "energetic and dynamic"
- "futuristic and innovative"

### Lighting
- "soft ambient with subtle glows"
- "dramatic directional lighting"
- "bright and airy natural light"
- "moody with deep shadows"
- "golden hour warmth"

---

## Example Session

**User:** Create a slideshow about renewable energy

**You:** I'll create a renewable energy slideshow with an eco-friendly style.

```bash
python skills/slides/create.py \
  --topic "Renewable Energy Revolution" \
  --art-style "modern flat illustration with organic flowing shapes" \
  --colors "#1B4332, #40916C, #95D5B2, #FFD93D, #FFFFFF" \
  --mood "optimistic and hopeful" \
  --lighting "bright natural sunlight"
```

Output: `{"slideshow_id": "renewable-energy-revolution", ...}`

Now adding slides:

```bash
python skills/slides/add_slide.py \
  --slideshow-id "renewable-energy-revolution" \
  --title "The Solar Revolution" \
  --description "How solar power is transforming energy" \
  --scene "A vast solar farm stretching to the horizon at golden hour, panels gleaming with reflected sunlight, a modern control center in the foreground, birds flying overhead in a clear blue sky"
```

*(repeat for 3-4 more slides)*

Generating images:

```bash
python skills/slides/generate_image.py --slideshow-id "renewable-energy-revolution" --slide-index 1
python skills/slides/generate_image.py --slideshow-id "renewable-energy-revolution" --slide-index 2
# ... etc
```

Finalizing:

```bash
python skills/slides/finalize.py --slideshow-id "renewable-energy-revolution"
```

**Your slideshow is ready! Open this file in your browser:**
```
file:///path/to/output/renewable-energy-revolution/index.html
```

---

## Tips

1. **Be specific in scene descriptions** - More detail = better images
2. **Keep colors consistent** - Use the same palette for all slides
3. **One concept per slide** - Don't overcrowd
4. **Generate images one at a time** - Allows you to report progress
5. **Tell user the exact file path** - So they can open it immediately
