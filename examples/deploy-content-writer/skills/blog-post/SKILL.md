---
name: blog-post
description: Write structured long-form blog posts with research, SEO optimization, and cover image generation.
---

# Blog Post Writing Skill

## Research First (Required)

Before writing any blog post, delegate research:
1. Use the `task` tool with `subagent_type: "researcher"`
2. Specify both the topic AND where to save findings

## Blog Post Structure

Every blog post should follow this structure:

### 1. Hook (Opening)
- Start with a compelling question, statistic, or statement
- Keep it to 2-3 sentences

### 2. Context (The Problem)
- Explain why this topic matters
- Connect to the reader's experience

### 3. Main Content (3-5 sections)
- Each section covers one key point with an H2 header
- Include code examples where helpful
- Use bullet points for lists of 3+ items

### 4. Practical Application
- Show how to apply the concepts
- Include step-by-step instructions or code snippets

### 5. Conclusion & CTA
- Summarize key takeaways (3 bullets max)
- End with a clear call-to-action

## Output

Save the blog post to `blogs/<slug>/post.md`.

## SEO Considerations

- Include the main keyword in the title and first paragraph
- Keep the title under 60 characters
- Write a meta description (150-160 characters)
