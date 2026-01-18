import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatZhipuAI

# Load environment variables
load_dotenv()

def create_sql_deep_agent():
    """Create and return a text-to-SQL Deep Agent"""

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Connect to Chinook database
    db_path = os.path.join(base_dir, "chinook.db")
    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        sample_rows_in_table_info=3
    )

    # Initialize ZhipuAI for toolkit initialization
    model = ChatZhipuAI(
        model="glm-4.7",
        temperature=0,
    )

    # Create SQL toolkit and get tools
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    # Create the Deep Agent with all parameters
    agent = create_deep_agent(
        model=model,                                  # ZhipuAI model with temperature=0
        memory=["./AGENTS.md"],                       # Agent identity and general instructions
        skills=["./skills/"],                         # Specialized workflows (query-writing, schema-exploration)
        tools=sql_tools,                              # SQL database tools
        subagents=[],                                 # No subagents needed
        backend=FilesystemBackend(root_dir=base_dir)  # Persistent file storage
    )

    return agent


agent = create_sql_deep_agent()
