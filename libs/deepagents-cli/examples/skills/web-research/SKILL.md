---
name: web-research
description: Use this skill for requests related to web research; it provides a structured approach to conducting comprehensive web research
---

# Web Research Skill

This skill provides a structured workflow for conducting comprehensive web research using the `task` tool to spawn research subagents. It emphasizes planning, efficient delegation, and systematic synthesis with proper citations.

## When to Use This Skill

Use this skill when you need to:
- Research complex topics requiring multiple information sources
- Gather and synthesize current information from the web
- Conduct comparative analysis across multiple subjects
- Produce well-sourced research reports with clear citations

## Research Workflow

Follow this workflow for all research requests:

1. **Plan**: Use `write_todos` to break down the research into focused tasks
2. **Save the request**: Use `write_file` to save the user's research question to `/research_request.md`
3. **Research**: Delegate research tasks to sub-agents using the `task` tool
4. **Synthesize**: Review all sub-agent findings and consolidate citations
5. **Write Report**: Write a comprehensive final report to `/final_report.md`
6. **Verify**: Read `/research_request.md` and confirm you've addressed all aspects

## Research Planning Guidelines

**DEFAULT: Start with 1 sub-agent** for most queries:
- "What is quantum computing?" → 1 sub-agent (general overview)
- "List the top 10 coffee shops in San Francisco" → 1 sub-agent
- "Summarize the history of the internet" → 1 sub-agent
- "Research context engineering for AI agents" → 1 sub-agent (covers all aspects)

**ONLY parallelize when explicitly needed:**

**Explicit comparisons** → 1 sub-agent per element:
- "Compare OpenAI vs Anthropic vs Google AI approaches" → 3 parallel sub-agents
- "Compare Python vs JavaScript for web development" → 2 parallel sub-agents

**Clearly separated aspects** → 1 sub-agent per aspect (use sparingly):
- "Research renewable energy adoption in Europe, Asia, and North America" → 3 parallel sub-agents

### Key Principles
- **Bias towards single sub-agent**: One comprehensive research task is more token-efficient
- **Avoid premature decomposition**: Don't break "research X" into multiple narrow tasks unnecessarily
- **Parallelize only for clear comparisons**: Use multiple sub-agents when comparing distinct entities
- **Batch similar tasks**: Minimize overhead by grouping related research

### Research Limits
- **Simple queries**: Use 1 sub-agent with 2-3 searches
- **Complex queries**: Use up to 3 parallel sub-agents, each with up to 5 searches
- **Stop when sufficient**: Don't over-research; stop when you can answer comprehensively

## Delegating to Research Subagents

When using the `task` tool to spawn research subagents:

**Subagent Instructions Template:**
```
Research [SPECIFIC TOPIC]. Use web_search to gather information.

Follow these guidelines:
- Start with broader searches, then narrow as you gather information
- Use 2-3 searches for simple queries, up to 5 for complex topics
- Stop when you can answer the question comprehensively
- Cite sources inline using [1], [2], [3] format
- End with ### Sources section listing each numbered source

Structure your findings with clear headings and cite sources as you reference them.
```

**Parallel Execution:**
- Make multiple `task()` calls in a single response to enable parallel execution
- Use at most 3 parallel sub-agents per iteration
- Each sub-agent returns findings independently

## Report Writing Guidelines

When writing the final report to `/final_report.md`, follow these structure patterns:

### For Comparisons:
1. Introduction
2. Overview of topic A
3. Overview of topic B
4. Detailed comparison
5. Conclusion

### For Lists/Rankings:
Simply list items with details - no introduction needed:
1. Item 1 with explanation
2. Item 2 with explanation
3. Item 3 with explanation

### For Summaries/Overviews:
1. Overview of topic
2. Key concept 1
3. Key concept 2
4. Key concept 3
5. Conclusion

### General Guidelines:
- Use clear section headings (## for sections, ### for subsections)
- Write in paragraph form by default - be text-heavy, not just bullet points
- Do NOT use self-referential language ("I found...", "I researched...")
- Write as a professional report without meta-commentary
- Each section should be comprehensive and detailed
- Use bullet points only when listing is more appropriate than prose

## Citation Format

**Consolidate citations across all sub-agent findings:**
- Each unique URL gets ONE citation number across ALL findings
- Cite sources inline using [1], [2], [3] format
- Number sources sequentially without gaps (1,2,3,4...)
- End report with ### Sources section

**Example:**
```markdown
## Key Findings

Context engineering is a critical technique for AI agents [1]. Studies show that proper context management can improve performance by 40% [2].

### Sources
[1] Context Engineering Guide: https://example.com/context-guide
[2] AI Performance Study: https://example.com/study
```

## Synthesis Process

After all subagents complete:

1. **Read all findings**: Review each sub-agent's response
2. **Consolidate citations**: Assign each unique URL a single number across ALL findings
3. **Integrate insights**: Combine information from all sources into cohesive sections
4. **Write final report**: Create `/final_report.md` with proper structure and citations
5. **Verify completeness**: Check against `/research_request.md` to ensure all aspects are covered

## Available Tools

You have access to:
- **write_todos**: Create task list for research planning
- **write_file**: Save research request, plan, and final report
- **read_file**: Read local files (e.g., findings from subagents, research request)
- **task**: Spawn research subagents with web_search access
- **fetch_url**: Fetch content from specific URLs if needed

## Best Practices

- **Plan first**: Always create a todo list before delegating
- **Save the request**: Write `/research_request.md` so you can verify completeness later
- **Default to single agent**: Use 1 sub-agent unless you have explicit comparisons
- **Clear instructions**: Give each sub-agent specific, focused research questions
- **Consolidate citations**: Each URL should have one number across the entire report
- **Stop appropriately**: Don't over-research; sufficient information > exhaustive coverage
- **Verify at end**: Read `/research_request.md` and confirm you've answered everything
