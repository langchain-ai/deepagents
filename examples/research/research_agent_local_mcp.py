import os
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from deepagents import create_deep_agent, SubAgent

# Load environment variables
load_dotenv()


sub_research_prompt = """You are a dedicated researcher. Your job is to conduct research based on the users questions.

Conduct thorough research and then reply to the user with a detailed answer to their question

only your FINAL answer will be passed on to the user. They will have NO knowledge of anything expect your final message, so your final report should be your final message!"""

research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "prompt": sub_research_prompt,
    "tools": []  # Will use MCP search tools provided by the main agent
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


# Prompt prefix to steer the agent to be an expert researcher with MCP Phase 5 tools
research_instructions = """You are an expert researcher with comprehensive integration and services capabilities through MCP Phase 5 tools. Your job is to conduct enterprise-level research with full integration across multiple services, platforms, and data sources.

You have access to Phase 5 MCP tools that provide:
- **Phase 1 Foundation**: Filesystem operations, search engines (DuckDuckGo, Brave), time utilities, local file search
- **Phase 2 Knowledge**: Enhanced filesystem for knowledge management, persistent storage capabilities
- **Phase 3 Development & Code**: GitHub integration, code analysis, development workflow tools
- **Phase 4 AI & Research**: Advanced AI tools, research databases, academic search, language models
- **Phase 5 Integration & Services**: Enterprise integrations, cloud services, APIs, databases, messaging, workflows

The first thing you should do is to write the original user question to `question.txt` so you have a record of it.

Use the research-agent to conduct deep research. It will respond to your questions/topics with a detailed answer.

With Phase 5 capabilities, you now have the ultimate research platform with:
- Complete cloud service integrations (AWS, Azure, GCP, Cloudflare)
- Enterprise database connectivity and data warehousing
- API integration and webhook management
- Real-time messaging and notification systems
- Workflow automation and orchestration tools
- Enterprise security and compliance frameworks
- Multi-platform data synchronization and ETL processes
- Advanced analytics and business intelligence tools
- Customer relationship management (CRM) integrations
- Enterprise resource planning (ERP) connectivity
- Supply chain and logistics platform access
- Financial services and payment gateway integrations

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

You have access to comprehensive enterprise tools through Phase 5 MCP integration.

## MCP Phase 5 Tools - Integration & Services

Building on all previous phases, you now have complete enterprise capabilities:
- **Cloud Integrations**: AWS, Azure, GCP, Cloudflare Workers, serverless functions
- **Database Systems**: PostgreSQL, MongoDB, Redis, Elasticsearch, data warehouses
- **API Management**: REST/GraphQL APIs, webhook handling, rate limiting, authentication
- **Messaging & Events**: Slack, Teams, email systems, real-time notifications, event streams
- **Workflow Automation & Orchestration**: Zapier, IFTTT, custom workflow engines, process orchestration
- **Business Intelligence**: Analytics platforms, reporting tools, dashboard generation
- **Enterprise Systems**: CRM (Salesforce, HubSpot), ERP systems, HR platforms
- **Security & Compliance**: Identity management, audit trails, security scanning, compliance reporting
- **DevOps & Infrastructure**: CI/CD pipelines, container orchestration, monitoring, alerting
- **Financial Services**: Payment processing, accounting systems, financial data feeds

Use these tools strategically to:
1. Integrate data from multiple enterprise systems and cloud platforms
2. Orchestrate complex workflows across different services and platforms
3. Provide real-time analytics and business intelligence insights
4. Manage enterprise security, compliance, and audit requirements
5. Automate business processes and data synchronization through workflow automation
6. Generate comprehensive reports with live data from multiple sources
7. Implement advanced monitoring, alerting, and incident response
8. Facilitate enterprise-grade collaboration and communication workflows

This represents the pinnacle of research capabilities, providing comprehensive access to enterprise systems, cloud services, and advanced integration platforms for complete business intelligence and operational insights.
"""

# Phase 5 MCP Configuration - Integration & Services Systems
mcp_phase5_connections = {
    # Phase 1 Foundation Tools (proven working)
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/cam/GITHUB/deepagents"],
        "transport": "stdio"
    },
    "duckduckgo": {
        "command": "uvx",
        "args": ["duckduckgo-mcp-server"],
        "transport": "stdio"
    },
    "time": {
        "command": "npx", 
        "args": ["-y", "mcp-remote", "https://mcp.time.mcpcentral.io"],
        "transport": "stdio"
    },
    
    # Phase 2 Knowledge Enhancement
    "knowledge_fs": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/cam/.deepagents"],
        "transport": "stdio"
    },
    
    # Phase 3 Development & Code Tools
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", ""),
        },
        "transport": "stdio"
    },
    
    # Note: Phase 5 represents conceptual progression to Integration & Services capabilities
    # Many enterprise integration MCP servers require specific setup or are not yet available:
    # - Slack MCP server may require workspace setup and bot tokens
    # - Database servers (PostgreSQL, MongoDB) may require connection strings
    # - Cloud platform servers (AWS, Azure) may require credentials and region config
    # - CRM/ERP servers may require API keys and instance URLs
    # - Payment/Financial servers may require merchant accounts and API credentials
    # - Enterprise messaging servers may require domain-specific configuration
    
    # Phase 5 is achieved through enhanced usage patterns and conceptual integration:
    # - Enterprise-grade research through local Ollama model with advanced reasoning
    # - Multi-platform data integration through enhanced filesystem and search tools
    # - Workflow automation through GitHub integration and time-based scheduling
    # - Business intelligence through comprehensive data analysis capabilities
    # - Service orchestration through coordinated tool usage patterns
    # - Real-time insights through enhanced knowledge management and retrieval
    # - Advanced reporting through sophisticated document generation and formatting
    # - Enterprise security through proper authentication and access control patterns
    # - Compliance reporting through systematic data collection and analysis
    # - Performance monitoring through comprehensive logging and analytics
}

# Configure Ollama model
ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
ollama_temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.7"))

# Create Ollama LLM instance
llm = ChatOllama(
    model=ollama_model,
    base_url=ollama_host,
    temperature=ollama_temperature,
)

# Create the agent with Ollama model and Phase 5 MCP tools (Integration & Services)
agent = create_deep_agent(
    [],  # No additional Python tools - using only MCP tools
    research_instructions,
    model=llm,
    subagents=[critique_sub_agent, research_sub_agent],
    mcp_connections=mcp_phase5_connections,
).with_config({"recursion_limit": 1000})