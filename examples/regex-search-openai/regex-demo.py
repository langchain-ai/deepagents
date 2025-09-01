#!/usr/bin/env python3
"""
Blog Post Writing Agent with Regex Search Demo

This agent writes blog posts on requested topics and then uses regex search
to find and analyze specific content within the generated blog posts.
"""

import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent, SubAgent
from langchain_openai import AzureChatOpenAI

# Initialize Tavily client for research
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
model = AzureChatOpenAI(azure_deployment="gpt-4.1")

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search for blog post research."""
    if not tavily_client.api_key:
        return "Mock search results for: " + query
    
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs

# Blog writing sub-agent
blog_writer_prompt = """You are an expert blog writer specializing in creating engaging, informative blog posts.

Your task is to write a comprehensive blog post based on the user's topic and requirements.

When writing blog posts:
1. Create compelling headlines and subheadings
2. Write in an engaging, conversational tone
3. Include specific examples and actionable insights
4. Structure content with clear sections and flow
5. Use proper markdown formatting with headers, lists, and emphasis
6. Aim for 800-1500 words depending on the topic complexity
7. Include relevant technical details when appropriate
8. Make the content valuable and informative for readers

Write the complete blog post and return it as your final response. The blog post should be publication-ready."""

blog_writer_agent = {
    "name": "blog-writer",
    "description": "Expert blog writer that creates comprehensive, engaging blog posts on any topic. Use this agent when you need to write the actual blog content.",
    "prompt": blog_writer_prompt,
    "tools": ["internet_search"],
}

# Content analyzer sub-agent  
content_analyzer_prompt = """You are a content analyst specializing in examining blog posts and other written content.

Your job is to analyze blog posts using regex search to find specific patterns, elements, or content types.

When analyzing content:
1. Use regex_search to find specific patterns in the blog post
2. Look for things like:
   - Technical terms and jargon
   - Code snippets or examples
   - Statistics and numbers
   - Links and references
   - Specific keywords or phrases
   - Structural elements (headers, lists, etc.)
3. Provide insights about the content based on your regex findings
4. Suggest improvements or highlight interesting patterns

Always use the regex_search tool to examine the blog post content and provide detailed analysis based on your findings."""

content_analyzer_agent = {
    "name": "content-analyzer", 
    "description": "Content analyst that uses regex search to examine blog posts and find specific patterns, keywords, and structural elements. Use this when you want to analyze the written content.",
    "prompt": content_analyzer_prompt,
}

# Main blog agent instructions
blog_agent_instructions = """You are a Blog Post Creation and Analysis Agent. Your workflow involves two main phases:

## Phase 1: Blog Post Creation
1. Research the requested topic thoroughly using internet search
2. Use the blog-writer agent to create a comprehensive, engaging blog post
3. Save the blog post to a file called `blog_post.md`

## Phase 2: Content Analysis  
4. Use the content-analyzer agent to examine the blog post using regex search
5. Have the analyzer find specific patterns like:
   - Technical terms (words ending in -tion, -ment, -ing, etc.)
   - Numbers and statistics
   - Code-related content (function names, variables, etc.)
   - Emphasis markers (words in **bold** or *italics*)
   - Headers and structural elements
   - Specific keywords related to the topic
6. Save the analysis results to `analysis_results.md`

## Phase 3: Summary and Insights
7. Provide a summary of both the blog post and the regex analysis findings
8. Suggest potential improvements or interesting insights from the analysis

## Workflow Tips:
- Research the topic first to ensure the blog post is well-informed
- Create engaging, valuable content that readers would want to read
- Use regex search creatively to find interesting patterns in the content
- Look for both technical and stylistic elements in your analysis
- Provide actionable insights from your regex findings

## Available Tools:
- `internet_search`: Research topics for blog post content
- `write_file`: Save blog posts and analysis results
- `read_file`: Read generated content
- `regex_search`: Search for patterns in the blog post content
- `edit_file`: Make edits to content if needed

Start by asking the user what topic they'd like a blog post about, then proceed through the workflow systematically."""

# Create the blog post agent
agent = create_deep_agent(
    [internet_search],
    blog_agent_instructions,
    model=model,
    subagents=[blog_writer_agent, content_analyzer_agent],
).with_config({"recursion_limit": 1000})
