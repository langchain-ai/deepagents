# in-memory registry of sub-agents
SUBAGENT_REGISTRY = {}


def register_subagent(agent_type):
    def decorator(factory_fn):
        SUBAGENT_REGISTRY[agent_type] = factory_fn
        return factory_fn

    return decorator
