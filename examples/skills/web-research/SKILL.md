---
name: web-research
description: Structured web research workflow with planning, parallel delegation, and synthesis
---

# Web Research Skill

This skill provides a structured approach to conducting comprehensive web research using the `task` tool to spawn research subagents. It emphasizes planning, efficient delegation, and systematic synthesis of findings.

## When to Use This Skill

Use this skill when you need to:
- Research complex topics requiring multiple information sources
- Gather and synthesize current information from the web
- Conduct comparative analysis across multiple subjects
- Produce well-sourced research reports with clear citations

## Research Process

### Step 1: Create and Save Research Plan

Before delegating to subagents, you MUST:

1. **Create a research folder** - Organize all research files in a dedicated folder:
   ```
   mkdir /research_[topic_name]/
   ```
   This keeps files organized and prevents clutter in the working directory.

2. **Analyze the research question** - Break it down into distinct, non-overlapping subtopics

3. **Write a research plan file** - Use the `write_file` tool to create `/research_[topic_name]/research_plan.md` containing:
   - The main research question
   - 2-5 specific subtopics to investigate
   - Expected information from each subtopic
   - How results will be synthesized

**Planning Guidelines:**
- **Simple fact-finding**: 1-2 subtopics
- **Comparative analysis**: 1 subtopic per comparison element (max 3)
- **Complex investigations**: 3-5 subtopics

### Step 2: Delegate to Research Subagents

For each subtopic in your plan:

1. **Use the `task` tool** to spawn a research subagent with:
   - Clear, specific research question (no acronyms)
   - Instructions to write findings to a file: `/research_[topic_name]/findings_[subtopic].md`
   - Budget: 3-5 web searches maximum

2. **Run up to 3 subagents in parallel** for efficient research

**Subagent Instructions Template:**
```
Research [SPECIFIC TOPIC]. Use the web_search tool to gather information.
After completing your research, use write_file to save your findings to /research_[topic_name]/findings_[subtopic].md.
Include key facts, relevant quotes, and source URLs.
Use 3-5 web searches maximum.
```

### Step 3: Synthesize Findings

After all subagents complete:

1. **Use `list_files` and `read_file`** to review all findings files in `/research_[topic_name]/`
2. **Synthesize the information** - Create a comprehensive response that:
   - Directly answers the original question
   - Integrates insights from all subtopics
   - Cites specific sources with URLs
   - Identifies any gaps or limitations

3. **Write final report** (optional) - Use `write_file` to create `/research_[topic_name]/research_report.md` if requested

## Available Tools

You have access to:
- **write_file**: Save research plans and findings
- **read_file**: Review findings from subagents
- **list_files**: See what files exist
- **task**: Spawn research subagents with web_search access

## Research Subagent Configuration

Each subagent you spawn will have access to:
- **web_search**: Search the web using Tavily (parameters: query, max_results, topic, include_raw_content)
- **write_file**: Save their findings to the filesystem

## Best Practices

- **Plan before delegating** - Always write research_plan.md first
- **Clear subtopics** - Ensure each subagent has distinct, non-overlapping scope
- **File-based communication** - Have subagents save findings to files, not return them directly
- **Systematic synthesis** - Read all findings files before creating final response
- **Stop appropriately** - Don't over-research; 3-5 searches per subtopic is usually sufficient
