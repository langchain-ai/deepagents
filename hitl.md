---
title: Human-in-the-loop
---



The Human-in-the-Loop (HITL) middleware lets you add human oversight to agent tool calls.
When a model proposes an action that might require review — for example, writing to a file or executing SQL — the middleware can pause execution and wait for a decision.

It does this by checking each tool call against a configurable policy. If intervention is needed, the middleware issues an @[interrupt] that halts execution. The graph state is saved using LangGraph’s [persistence layer](/oss/langgraph/persistence), so execution can pause safely and resume later.

A human decision then determines what happens next: the action can be approved as-is (`approve`), modified before running (`edit`), or rejected with feedback (`reject`).

## Interrupt decision types

The middleware defines three built-in ways a human can respond to an interrupt:

| Decision Type | Description                                                               | Example Use Case                                    |
|---------------|---------------------------------------------------------------------------|-----------------------------------------------------|
| ✅ `approve`   | The action is approved as-is and executed without changes.                | Send an email draft exactly as written              |
| ✏️ `edit`     | The tool call is executed with modifications.                             | Change the recipient before sending an email        |
| ❌ `reject`    | The tool call is rejected, with an explanation added to the conversation. | Reject an email draft and explain how to rewrite it |

The available decision types for each tool depend on the policy you configure in `interrupt_on`.
When multiple tool calls are paused at the same time, each action requires a separate decision.
Decisions must be provided in the same order as the actions appear in the interrupt request.

<Tip>
When **editing** tool arguments, make changes conservatively. Significant modifications to the original arguments may cause the model to re-evaluate its approach and potentially execute the tool multiple times or take unexpected actions.
</Tip>


## Configuring interrupts

To use HITL, add the middleware to the agent’s `middleware` list when creating the agent.

You configure it with a mapping of tool actions to the decision types that are allowed for each action. The middleware will interrupt execution when a tool call matches an action in the mapping.

:::python
```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware # [!code highlight]
from langgraph.checkpoint.memory import InMemorySaver # [!code highlight]


agent = create_agent(
    model="openai:gpt-4o",
    tools=[write_file_tool, execute_sql_tool, read_data_tool],
    middleware=[
        HumanInTheLoopMiddleware( # [!code highlight]
            interrupt_on={
                "write_file": True,  # All decisions (approve, edit, reject) allowed
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # No editing allowed
                # Safe operation, no approval needed
                "read_data": False,
            },
            # Prefix for interrupt messages - combined with tool name and args to form the full message
            # e.g., "Tool execution pending approval: execute_sql with query='DELETE FROM...'"
            # Individual tools can override this by specifying a "description" in their interrupt config
            description_prefix="Tool execution pending approval",
        ),
    ],
    # Human-in-the-loop requires checkpointing to handle interrupts.
    # In production, use a persistent checkpointer like AsyncPostgresSaver.
    checkpointer=InMemorySaver(),  # [!code highlight]
)
```
:::

:::js
```ts
import { createAgent, humanInTheLoopMiddleware } from "langchain"; // [!code highlight]
import { MemorySaver } from "@langchain/langgraph"; // [!code highlight]

const agent = createAgent({
    model: "openai:gpt-4o",
    tools: [writeFileTool, executeSQLTool, readDataTool],
    middleware: [
        humanInTheLoopMiddleware({
            interruptOn: {
                write_file: true, // All decisions (approve, edit, reject) allowed
                execute_sql: {
                    allowedDecisions: ["approve", "reject"],
                    // No editing allowed
                    description: "🚨 SQL execution requires DBA approval",
                },
                // Safe operation, no approval needed
                read_data: false,
            },
            // Prefix for interrupt messages - combined with tool name and args to form the full message
            // e.g., "Tool execution pending approval: execute_sql with query='DELETE FROM...'"
            // Individual tools can override this by specifying a "description" in their interrupt config
            descriptionPrefix: "Tool execution pending approval",
        }),
    ],
    // Human-in-the-loop requires checkpointing to handle interrupts.
    // In production, use a persistent checkpointer like AsyncPostgresSaver.
    checkpointer: new MemorySaver(), // [!code highlight]
});
```
:::


<Info>
    You must configure a checkpointer to persist the graph state across interrupts.
    In production, use a persistent checkpointer like @[`AsyncPostgresSaver`]. For testing or prototyping, use @[`InMemorySaver`].

    When invoking the agent, pass a `config` that includes the **thread ID** to associate execution with a conversation thread.
    See the [LangGraph interrupts documentation](/oss/langgraph/interrupts) for details.
</Info>


## Responding to interrupts

When you invoke the agent, it runs until it either completes or an interrupt is raised. An interrupt is triggered when a tool call matches the policy you configured in `interrupt_on`. In that case, the invocation result will include an `__interrupt__` field with the actions that require review. You can then present those actions to a reviewer and resume execution once decisions are provided.


:::python
```python
from langgraph.types import Command

# Human-in-the-loop leverages LangGraph's persistence layer.
# You must provide a thread ID to associate the execution with a conversation thread,
# so the conversation can be paused and resumed (as is needed for human review).
config = {"configurable": {"thread_id": "some_id"}} # [!code highlight]
# Run the graph until the interrupt is hit.
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Delete old records from the database",
            }
        ]
    },
    config=config # [!code highlight]
)

# The interrupt contains the full HITL request with action_requests and review_configs
print(result['__interrupt__'])
# > [
# >    Interrupt(
# >       value={
# >          'action_requests': [
# >             {
# >                'name': 'execute_sql',
# >                'arguments': {'query': 'DELETE FROM records WHERE created_at < NOW() - INTERVAL \'30 days\';'},
# >                'description': 'Tool execution pending approval\n\nTool: execute_sql\nArgs: {...}'
# >             }
# >          ],
# >          'review_configs': [
# >             {
# >                'action_name': 'execute_sql',
# >                'allowed_decisions': ['approve', 'reject']
# >             }
# >          ]
# >       }
# >    )
# > ]


# Resume with approval decision
agent.invoke(
    Command( # [!code highlight]
        resume={"decisions": [{"type": "approve"}]}  # or "edit", "reject" [!code highlight]
    ), # [!code highlight]
    config=config # Same thread ID to resume the paused conversation
)
```
:::

:::js
```typescript
import { HumanMessage } from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";

// You must provide a thread ID to associate the execution with a conversation thread,
// so the conversation can be paused and resumed (as is needed for human review).
const config = { configurable: { thread_id: "some_id" } }; // [!code highlight]

// Run the graph until the interrupt is hit.
const result = await agent.invoke(
    {
        messages: [new HumanMessage("Delete old records from the database")],
    },
    config // [!code highlight]
);


// The interrupt contains the full HITL request with action_requests and review_configs
console.log(result.__interrupt__);
// > [
// >    Interrupt(
// >       value: {
// >          action_requests: [
// >             {
// >                name: 'execute_sql',
// >                arguments: { query: 'DELETE FROM records WHERE created_at < NOW() - INTERVAL \'30 days\';' },
// >                description: 'Tool execution pending approval\n\nTool: execute_sql\nArgs: {...}'
// >             }
// >          ],
// >          review_configs: [
// >             {
// >                action_name: 'execute_sql',
// >                allowed_decisions: ['approve', 'reject']
// >             }
// >          ]
// >       }
// >    )
// > ]

// Resume with approval decision
await agent.invoke(
    new Command({ // [!code highlight]
        resume: { decisions: [{ type: "approve" }] }, // or "edit", "reject" [!code highlight]
    }), // [!code highlight]
    config // Same thread ID to resume the paused conversation
);
```
:::

### Decision types

<Tabs>
<Tab title="✅ approve">
Use `approve` to approve the tool call as-is and execute it without changes.

:::python
```python
agent.invoke(
    Command(
        # Decisions are provided as a list, one per action under review.
        # The order of decisions must match the order of actions
        # listed in the `__interrupt__` request.
        resume={
            "decisions": [
                {
                    "type": "approve",
                }
            ]
        }
    ),
    config=config  # Same thread ID to resume the paused conversation
)
```
:::

:::js
```typescript
await agent.invoke(
    new Command({
        // Decisions are provided as a list, one per action under review.
        // The order of decisions must match the order of actions
        // listed in the `__interrupt__` request.
        resume: {
            decisions: [
                {
                    type: "approve",
                }
            ]
        }
    }),
    config  // Same thread ID to resume the paused conversation
);
```
:::
</Tab>
<Tab title="✏️ edit">
    Use `edit` to modify the tool call before execution.
    Provide the edited action with the new tool name and arguments.

    :::python
    ```python
    agent.invoke(
        Command(
            # Decisions are provided as a list, one per action under review.
            # The order of decisions must match the order of actions
            # listed in the `__interrupt__` request.
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        # Edited action with tool name and args
                        "edited_action": {
                            # Tool name to call.
                            # Will usually be the same as the original action.
                            "name": "new_tool_name",
                            # Arguments to pass to the tool.
                            "args": {"key1": "new_value", "key2": "original_value"},
                        }
                    }
                ]
            }
        ),
        config=config  # Same thread ID to resume the paused conversation
    )
    ```
    :::

    :::js
    ```typescript
    await agent.invoke(
        new Command({
            // Decisions are provided as a list, one per action under review.
            // The order of decisions must match the order of actions
            // listed in the `__interrupt__` request.
            resume: {
                decisions: [
                    {
                        type: "edit",
                        // Edited action with tool name and args
                        editedAction: {
                            // Tool name to call.
                            // Will usually be the same as the original action.
                            name: "new_tool_name",
                            // Arguments to pass to the tool.
                            args: { key1: "new_value", key2: "original_value" },
                        }
                    }
                ]
            }
        }),
        config  // Same thread ID to resume the paused conversation
    );
    ```
    :::

    <Tip>
        When **editing** tool arguments, make changes conservatively. Significant modifications to the original arguments may cause the model to re-evaluate its approach and potentially execute the tool multiple times or take unexpected actions.
    </Tip>

</Tab>

<Tab title="❌ reject">
    Use `reject` to reject the tool call and provide feedback instead of execution.

    :::python
    ```python
    agent.invoke(
        Command(
            # Decisions are provided as a list, one per action under review.
            # The order of decisions must match the order of actions
            # listed in the `__interrupt__` request.
            resume={
                "decisions": [
                    {
                        "type": "reject",
                        # An explanation about why the action was rejected
                        "message": "No, this is wrong because ..., instead do this ...",
                    }
                ]
            }
        ),
        config=config  # Same thread ID to resume the paused conversation
    )
    ```
    :::

    :::js
    ```typescript
    await agent.invoke(
        new Command({
            // Decisions are provided as a list, one per action under review.
            // The order of decisions must match the order of actions
            // listed in the `__interrupt__` request.
            resume: {
                decisions: [
                    {
                        type: "reject",
                        // An explanation about why the action was rejected
                        message: "No, this is wrong because ..., instead do this ...",
                    }
                ]
            }
        }),
        config  // Same thread ID to resume the paused conversation
    );
    ```
    :::

The `message` is added to the conversation as feedback to help the agent understand why the action was rejected and what it should do instead.

---

### Multiple decisions

When multiple actions are under review, provide a decision for each action in the same order as they appear in the interrupt:

:::python
```python
{
    "decisions": [
        {"type": "approve"},
        {
            "type": "edit",
            "edited_action": {
                "name": "tool_name",
                "args": {"param": "new_value"}
            }
        },
        {
            "type": "reject",
            "message": "This action is not allowed"
        }
    ]
}
```
:::

:::js
```typescript
{
    decisions: [
        { type: "approve" },
        {
            type: "edit",
            editedAction: {
                name: "tool_name",
                args: { param: "new_value" }
            }
        },
        {
            type: "reject",
            message: "This action is not allowed"
        }
    ]
}
```
:::

</Tab>
</Tabs>


## Execution lifecycle

The middleware defines an `after_model` hook that runs after the model generates a response but before any tool calls are executed:

1. The agent invokes the model to generate a response.
2. The middleware inspects the response for tool calls.
3. If any calls require human input, the middleware builds a `HITLRequest` with `action_requests` and `review_configs` and calls @[interrupt].
4. The agent waits for human decisions.
5. Based on the `HITLResponse` decisions, the middleware executes approved or edited calls, synthesizes @[ToolMessage]'s for rejected calls, and resumes execution.



## Custom HITL logic

For more specialized workflows, you can build custom HITL logic directly using the @[interrupt] primitive and [middleware](/oss/langchain/middleware) abstraction.

Review the [execution lifecycle](#execution-lifecycle) above to understand how to integrate interrupts into the agent's operation.

