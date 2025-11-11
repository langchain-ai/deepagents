"""Example demonstrating how to use Skills with deepagents.

This example shows how to:
1. Load skills from a directory
2. Activate skills during agent execution
3. Use auto-activated skills
4. Access skill instructions dynamically
"""

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.middleware import SkillsMiddleware

# Get the path to example-skills directory
skills_dir = Path(__file__).parent.parent / "example-skills"


def basic_skills_example():
    """Basic example of using skills with an agent."""
    # Create an agent with skills middleware
    agent = create_deep_agent(
        model="claude-sonnet-4-5-20250929",
        middleware=[
            SkillsMiddleware(
                skills_dir=skills_dir,
            )
        ],
    )

    # Use the agent - it can activate skills via the use_skill tool
    messages = [
        ("user", "Please activate the python-expert skill and then help me write a function to validate emails")
    ]

    for event in agent.stream({"messages": messages}):
        if "agent" in event:
            print(event["agent"]["messages"][-1].content)


def auto_activate_skills_example():
    """Example of auto-activating skills on agent initialization."""
    # Create an agent that automatically activates certain skills
    agent = create_deep_agent(
        model="claude-sonnet-4-5-20250929",
        middleware=[
            SkillsMiddleware(
                skills_dir=skills_dir,
                auto_activate=["python-expert"],  # Auto-activate python-expert skill
            )
        ],
    )

    # The python-expert skill is already active
    messages = [("user", "Write a function to parse a CSV file with proper type hints and error handling")]

    for event in agent.stream({"messages": messages}):
        if "agent" in event:
            print(event["agent"]["messages"][-1].content)


def code_review_example():
    """Example of using the code-reviewer skill."""
    agent = create_deep_agent(
        model="claude-sonnet-4-5-20250929",
        middleware=[
            SkillsMiddleware(
                skills_dir=skills_dir,
                auto_activate=["code-reviewer"],
            )
        ],
    )

    code_to_review = """
def process_user_input(data):
    user_id = data['user_id']
    result = db.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return result
"""

    messages = [("user", f"Please review this code:\n\n```python\n{code_to_review}\n```")]

    for event in agent.stream({"messages": messages}):
        if "agent" in event:
            print(event["agent"]["messages"][-1].content)


def multiple_skills_example():
    """Example of using multiple skills together."""
    agent = create_deep_agent(
        model="claude-sonnet-4-5-20250929",
        middleware=[
            SkillsMiddleware(
                skills_dir=skills_dir,
            )
        ],
    )

    # Activate multiple skills during the conversation
    messages = [
        (
            "user",
            "Please activate both the python-expert and code-reviewer skills, "
            "then write a function to process JSON data and review your own code",
        )
    ]

    for event in agent.stream({"messages": messages}):
        if "agent" in event:
            print(event["agent"]["messages"][-1].content)


def custom_skills_example():
    """Example of creating and using custom skills."""
    from deepagents.middleware.skills import Skill

    # Create a custom skill programmatically
    custom_skill = Skill(
        name="math-tutor",
        description="Provide clear mathematical explanations with step-by-step solutions",
        instructions="""
# Math Tutor Skill

When explaining mathematical concepts:
1. Break down complex problems into steps
2. Show all work and intermediate calculations
3. Explain the reasoning behind each step
4. Use clear notation and formatting
5. Verify the answer at the end

Always format equations using proper mathematical notation.
""",
    )

    # Create agent with custom skill
    agent = create_deep_agent(
        model="claude-sonnet-4-5-20250929",
        middleware=[
            SkillsMiddleware(
                skills={"math-tutor": custom_skill},
                auto_activate=["math-tutor"],
            )
        ],
    )

    messages = [("user", "Solve the quadratic equation: 2x² + 5x - 3 = 0")]

    for event in agent.stream({"messages": messages}):
        if "agent" in event:
            print(event["agent"]["messages"][-1].content)


if __name__ == "__main__":
    print("=" * 80)
    print("Basic Skills Example")
    print("=" * 80)
    basic_skills_example()

    print("\n" + "=" * 80)
    print("Auto-Activate Skills Example")
    print("=" * 80)
    auto_activate_skills_example()

    print("\n" + "=" * 80)
    print("Code Review Example")
    print("=" * 80)
    code_review_example()

    print("\n" + "=" * 80)
    print("Multiple Skills Example")
    print("=" * 80)
    multiple_skills_example()

    print("\n" + "=" * 80)
    print("Custom Skills Example")
    print("=" * 80)
    custom_skills_example()
