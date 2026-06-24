from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware import SummarizationMiddleware
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(temperature=0.9, model="claude-2")
backend = StateBackend()

# custom_summary = SummarizationMiddleware(
#     model=model,
#     backend=backend,
#     trigger=("tokens", 50_000),
#     keep=("messages", 20),
# )


# agent = create_deep_agent(
#     model=model,
#     middleware=[custom_summary
# ],
# )

# print(agent)


# 1. Inherited, default name
class MySummarizationMiddleware(SummarizationMiddleware):
    pass
#MySummarizationMiddleware

# 2. Inherited, same explicit name
class MySummarizationMiddlewareSameName(SummarizationMiddleware):
    name = "same name"
#Nishitha


# 3. Same middleware type, different params
custom_summary = SummarizationMiddleware(
    model=model,
    backend=backend,
    trigger=("tokens", 50_000),
    keep=("messages", 20),
)
#SummarizationMiddleware


# 4. Same middleware type, different params and different name
custom_summary_diff_name = SummarizationMiddleware(
    model=model,
    backend=backend,
    trigger=("tokens", 50_000),
    keep=("messages", 20),
name = "custom_summary")
#SummarizationMiddleware

custom_summary_diff_namee = SummarizationMiddleware(
    model=model,
    backend=backend,
    trigger=("tokens", 50_000),
    keep=("messages", 20),
name = "SummarizationMiddleware")
#SummarizationMiddleware


# create_deep_agent(
#     model=model,
#     middleware=[MySummarizationMiddleware(    model=model,
#     backend=backend,
#     trigger=("tokens", 50_000),
#     keep=("messages", 20))]
# )

# create_deep_agent(
#     model=model,
#     middleware=[MySummarizationMiddlewareSameName(    model=model,
#     backend=backend,
#     trigger=("tokens", 50_000),
#     keep=("messages", 20))]
# ) - NO


create_deep_agent(
    model=model,
    middleware=[MySummarizationMiddlewareSameName(    model=model,
    backend=backend,
    trigger=("tokens", 50_000),
    keep=("messages", 20))],
)

# create_deep_agent(
#     model=model,
#     middleware=[custom_summary]
# ) - No

create_deep_agent(
    model=model,
    middleware=[custom_summary_diff_name]
)


create_deep_agent(
    model=model,
    middleware=[custom_summary_diff_namee]
)





from deepagents import create_deep_agent, SubAgent
from deepagents.middleware import SummarizationMiddleware

custom_summary = SummarizationMiddleware(
    model=model,
    backend=backend,
)

agent = create_deep_agent(
    model=model,

    # Main agent middleware
    middleware=[
        custom_summary,
    ],

    subagents=[
        SubAgent(
            name="researcher",
            description="Researches topics",
            system_prompt="Find relevant information.",

            # Subagent middleware
            middleware=[
                SummarizationMiddleware(
                    model=model,
                    backend=backend,
                )
            ],
        )
    ],
)
