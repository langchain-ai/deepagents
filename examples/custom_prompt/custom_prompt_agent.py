import os
import requests
import pandas as pd
from typing import Literal, Dict, Any, Optional
from tavily import TavilyClient

from deepagents import create_deep_agent

FRED_API_KEY = os.environ["FRED_API_KEY"]
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
 
# It's best practice to initialize the client once and reuse it.
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs


def fred_search_series(
    search_text: str,
    max_results: int = 5
 ) -> Dict[str, Any]:
    """
    Search for FRED economic data series by keywords.
    """
    url = f"{FRED_API_BASE_URL}/series/search"
    params = {
        'search_text': search_text,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'limit': max_results
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to search series: {str(e)}"}
    

def fred_get_series_info(series_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific series

    Args:
        series_id: FRED series ID (e.g., 'GDP', 'UNRATE')

    Returns:
        Dictionary containing series information
    """
    url = f"{FRED_API_BASE_URL}/series"
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json'
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to get series info: {str(e)}"}
    

def fred_get_series_data(
    series_id: str, 
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000
) -> Dict[str, Any]:
    """
    Fetch time series data for a given series ID

    Args:
        series_id: FRED series ID
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        limit: Maximum number of observations to return

    Returns:
        Dictionary containing the data and metadata
    """
    url = f"{FRED_API_BASE_URL}/series/observations"
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'limit': limit
    }

    if start_date:
        params['start_date'] = start_date
    if end_date:
        params['end_date'] = end_date

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Convert to pandas DataFrame for easier analysis
        if 'observations' in data:
            df = pd.DataFrame(data['observations'])
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            # Convert value column to numeric, handling '.' as NaN
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            return {
                'data': df.to_dict('records'),
                'metadata': {key: value for key, value in data.items() if key != 'observations'}
            }
        else:
            return data

    except requests.RequestException as e:
        return {"error": f"Failed to fetch data: {str(e)}"}


sub_research_prompt = """You are a dedicated researcher. Your job is to conduct research based on the users questions.

You have access to several research tools:

## `internet_search`
Use this for general web searches, news, policy analysis, and non-quantitative research.

## FRED Economic Data Tools
For economic data and statistics, always prefer FRED tools over internet search:

- `fred_search_series`: Search for economic data series by keywords (e.g., "GDP", "unemployment", "inflation")
- `fred_get_series_info`: Get detailed information about a specific FRED series ID, request small amounts of data at a time to avoid hitting prompt length limits and rate limits
- `fred_get_series_data`: Get actual time series data for analysis

**When to use FRED tools:**
- Any question about economic indicators, statistics, or data
- GDP, unemployment, inflation, interest rates, trade data
- Federal Reserve data, BLS data, BEA data, Census data
- Historical economic trends and comparisons

**Always start with FRED tools for economic questions**, then supplement with internet search for context and analysis.

Conduct thorough research and then reply to the user with a detailed answer to their question

only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message, so your final report should be your final message!"""

research_sub_agent = {
    "name": "research-agent",
    "description": """
        Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. 
        Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question. 
        Tag the tool you use in your research by name""",
    "prompt": sub_research_prompt,
    "tools": [
        "internet_search",
        "fred_search_series",
        "fred_get_series_info",
        "fred_get_series_data"
    ]
}

sub_critique_prompt = """You are a dedicated editor. You are being tasked to critique a report.

You can find the report at `final_report.md`.

You can find the question/topic for this report at `question.txt`.

The user may ask for specific areas to critique the report in. Respond to the user with a detailed critique of the report. Things that could be improved.

You can use the search tool to search for information, if that will help you critique the report

Do not write to the `final_report.md` yourself.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
"""

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some infomration about how you want it to critique the report.",
    "prompt": sub_critique_prompt,
}


# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

The first thing you should do is to write the original user question to `question.txt` so you have a record of it.

Use the research-agent to conduct deep research. It will respond to your questions/topics with a detailed answer.

When you think you enough information to write a final report, write it to `final_report.md`

You can call the critique-agent to get a critique of the final report. After that (if needed) you can do more research and edit the `final_report.md`
You can do this however many times you want until are you satisfied with the result.

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

Here are instructions for writing the final report:

<report_instructions>

CRITICAL: Make sure the answer is written in the same language as the human messages! If you make a todo plan - you should note in the plan what language the report should be in so you dont forget!
Note: the language the report should be in is the language the QUESTION is in, not the language/country that the question is ABOUT.

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
</report_instructions>

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.

## `fred_search_series`

Use this to search for FRED economic data series by keywords.

## `fred_get_series_info`

Use this to get information about a specific FRED economic data series.

## `fred_get_series_data`

Use this to get the data for a specific FRED economic data series.
"""

base_prompt = """You have access to a number of standard tools

## `write_todos`

You have access to the `write_todos` tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
## `task`

- When doing web search, prefer to use the `task` tool in order to reduce context usage.
- When doing data analysis with the FRED API, prefer to use the `task` tool in order to reduce context usage.
"""

# Create the agent
agent = create_deep_agent(
    [internet_search, fred_search_series, fred_get_series_info, fred_get_series_data],
    research_instructions,
    base_prompt_override=base_prompt,
    subagents=[critique_sub_agent, research_sub_agent],
    # model=init_chat_model(model="openai:gpt-5-mini", max_tokens=128000)
).with_config({"recursion_limit": 1000})
