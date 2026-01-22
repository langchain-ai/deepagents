---
name: social-media
description: Use this skill when creating short-form social media content for LinkedIn, Twitter/X, or other platforms
---

# Social Media Content Skill

This skill provides guidelines for creating engaging social media content that drives engagement and shares.

## When to Use This Skill

Use this skill when asked to:
- Write a LinkedIn post
- Create a Twitter/X thread
- Draft social media announcements
- Repurpose blog content for social

## Research First (Required)

**Before writing any social media content, you MUST delegate research:**

1. Use the `task` tool with `subagent_type: "researcher"`
2. In the description, specify BOTH the topic AND where to save:

```
task(
    subagent_type="researcher",
    description="Research [TOPIC]. Save findings to research/[slug].md"
)
```

Example:
```
task(
    subagent_type="researcher",
    description="Research renewable energy trends in 2025. Save findings to research/renewable-energy.md"
)
```

3. After research completes, read the findings file before writing

## Output Structure (Required)

**Every social media post MUST have both content AND an image:**

**LinkedIn posts:**
```
linkedin/
└── <slug>/
    ├── post.md        # The post content
    └── image.png      # REQUIRED: Generated visual
```

**Twitter/X threads:**
```
tweets/
└── <slug>/
    ├── thread.md      # The thread content
    └── image.png      # REQUIRED: Generated visual
```

Example: A LinkedIn post about "prompt engineering" → `linkedin/prompt-engineering/`

**You MUST complete both steps:**
1. Write the content to the appropriate path
2. Generate an image using `generate_image` and save alongside the post

**A social media post is NOT complete without its image.**

## Platform Guidelines

### LinkedIn

**Format:**
- 1,300 character limit (show more after ~210 chars)
- First line is crucial - make it hook
- Use line breaks for readability
- 3-5 hashtags at the end

**Tone:**
- Professional but personal
- Share insights and learnings
- Ask questions to drive engagement
- Use "I" and share experiences

**Structure:**
```
[Hook - 1 compelling line]

[Empty line]

[Context - why this matters]

[Empty line]

[Main insight - 2-3 short paragraphs]

[Empty line]

[Call to action or question]

#hashtag1 #hashtag2 #hashtag3
```

### Twitter/X

**Format:**
- 280 character limit per tweet
- Threads for longer content (use 1/🧵 format)
- No more than 2 hashtags per tweet

**Thread Structure:**
```
1/🧵 [Hook - the main insight]

2/ [Supporting point 1]

3/ [Supporting point 2]

4/ [Example or evidence]

5/ [Conclusion + CTA]
```

## Image Generation

Every social media post needs an eye-catching image. Use the `generate_social_image` tool:

```
generate_social_image(prompt="A detailed description...", platform="linkedin", slug="your-post-slug")
```

The tool saves the image to `<platform>/<slug>/image.png`.

### Social Image Best Practices

Social images need to work at small sizes in crowded feeds:
- **Bold, simple compositions** - one clear focal point
- **High contrast** - stands out when scrolling
- **No text in image** - too small to read, platforms add their own
- **Square or 4:5 ratio** - works across platforms

### Writing Effective Prompts

Include these elements:

1. **Single focal point**: One clear subject, not a busy scene
2. **Bold style**: Vibrant colors, strong shapes, high contrast
3. **Simple background**: Solid color, gradient, or subtle texture
4. **Mood/energy**: Match the post tone (inspiring, urgent, thoughtful)

### Example Prompts

**For an insight/tip post:**
```
Single glowing lightbulb floating against a deep purple gradient background, lightbulb made of interconnected golden geometric lines, rays of soft light emanating outward. Minimal, striking, high contrast. Square composition.
```

**For announcements/news:**
```
Abstract rocket ship made of colorful geometric shapes launching upward with a trail of particles. Bright coral and teal color scheme against clean white background. Energetic, celebratory mood. Bold flat illustration style.
```

**For thought-provoking content:**
```
Two overlapping translucent circles, one blue one orange, creating a glowing intersection in the center. Represents collaboration or intersection of ideas. Dark charcoal background, soft ethereal glow. Minimalist and contemplative.
```

## Content Types

### Announcement Posts
- Lead with the news
- Explain the impact
- Include link or next step

### Insight Posts
- Share one specific learning
- Explain the context briefly
- Make it actionable

### Question Posts
- Ask a genuine question
- Provide your take first
- Keep it focused on one topic

## Quality Checklist

Before finishing:
- [ ] Post saved to `linkedin/<slug>/post.md` or `tweets/<slug>/thread.md`
- [ ] Image generated alongside the post
- [ ] First line hooks attention
- [ ] Content fits platform limits
- [ ] Tone matches platform norms
- [ ] Has clear CTA or question
- [ ] Hashtags are relevant (not generic)
