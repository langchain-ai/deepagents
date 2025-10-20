"""Integration tests for deep agent memory functionality."""
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from deepagents.middleware.filesystem import DEFAULT_MEMORY
import pytest

from openevals.llm import create_llm_as_judge

@pytest.mark.vcr()
def test_remembers_you_are():
    """Test 1: Updates memory when explicitly told role."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "2"}}
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "you will be an internet researcher"}]},
        config=config
    )
    memory = store.get(("2", "filesystem"), "/agent.md")
    assert memory is not None
    content = "\n".join(memory.value['content'])
    assert content != DEFAULT_MEMORY

    MY_CUSTOM_PROMPT = """Is the following memory block representative of an internet researcher.
    
    <memory>
    {content}
    </memory>
    """

    custom_prompt_evaluator = create_llm_as_judge(
        prompt=MY_CUSTOM_PROMPT,
        model="claude-haiku-4-5-20251001",
    )
    eval_result = custom_prompt_evaluator(
        content=content
    )
    assert eval_result['score'] == True


@pytest.mark.vcr()
def test_updates_memory_from_feedback():
    """Test 2: Updates memory when given feedback on previous work."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "feedback_test"}}
    
    # Give feedback
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "That was good, but next time use more descriptive variable names"}]},
        config=config
    )
    
    memory = store.get(("feedback_test", "filesystem"), "/agent.md")
    assert memory is not None
    content = "\n".join(memory.value['content'])
    
    eval_prompt = """Does the following memory contain guidance about using descriptive variable names?
    
    <memory>
    {content}
    </memory>
    """
    
    evaluator = create_llm_as_judge(
        prompt=eval_prompt,
        model="claude-haiku-4-5-20251001",
    )
    eval_result = evaluator(content=content)
    assert eval_result['score'] == True


@pytest.mark.vcr()
def test_explicit_remember_request():
    """Test 3: Updates memory when explicitly asked to remember."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "remember_test"}}
    
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "Remember that I prefer Python 3.11+ syntax"}]},
        config=config
    )
    
    memory = store.get(("remember_test", "filesystem"), "/agent.md")
    assert memory is not None
    content = "\n".join(memory.value['content'])
    
    eval_prompt = """Does the following memory contain information about preferring Python 3.11+ syntax?
    
    <memory>
    {content}
    </memory>
    """
    
    evaluator = create_llm_as_judge(
        prompt=eval_prompt,
        model="claude-haiku-4-5-20251001",
    )
    eval_result = evaluator(content=content)
    assert eval_result['score'] == True


@pytest.mark.vcr()
def test_coding_style_preferences():
    """Test 4: Updates memory when told coding style preferences."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "style_test"}}
    
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "I always want functions to be under 20 lines"}]},
        config=config
    )
    
    memory = store.get(("style_test", "filesystem"), "/agent.md")
    assert memory is not None
    content = "\n".join(memory.value['content'])
    
    eval_prompt = """Does the following memory contain guidance about keeping functions under 20 lines?
    
    <memory>
    {content}
    </memory>
    """
    
    evaluator = create_llm_as_judge(
        prompt=eval_prompt,
        model="claude-haiku-4-5-20251001",
    )
    eval_result = evaluator(content=content)
    assert eval_result['score'] == True


@pytest.mark.vcr()
def test_no_update_for_regular_tasks():
    """Test 5: Does NOT update memory for regular task completion."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "task_test"}}
    
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "Write a function that adds two numbers"}]},
        config=config
    )
    
    memory = store.get(("task_test", "filesystem"), "/agent.md")
    # Memory should be empty or default
    if memory is None:
        # Memory not created at all - that's fine
        assert True
    else:
        content = "\n".join(memory.value['content'])
        # Should be empty string (default)
        assert content == "" or content == DEFAULT_MEMORY


@pytest.mark.vcr()
def test_no_update_for_questions():
    """Test 6: Does NOT update memory for simple questions."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "question_test"}}
    
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "What is Python?"}]},
        config=config
    )
    
    memory = store.get(("question_test", "filesystem"), "/agent.md")
    # Memory should be empty or default
    if memory is None:
        # Memory not created at all - that's fine
        assert True
    else:
        content = "\n".join(memory.value['content'])
        # Should be empty string (default)
        assert content == "" or content == DEFAULT_MEMORY


@pytest.mark.vcr()
def test_updates_from_correction():
    """Test 7: Updates memory when corrected after mistake."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "correction_test"}}
    
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "No, I meant async functions not sync functions. Remember this distinction."}]},
        config=config
    )
    
    memory = store.get(("correction_test", "filesystem"), "/agent.md")
    assert memory is not None
    content = "\n".join(memory.value['content'])
    
    eval_prompt = """Does the following memory contain guidance about using async functions (not sync)?
    
    <memory>
    {content}
    </memory>
    """
    
    evaluator = create_llm_as_judge(
        prompt=eval_prompt,
        model="claude-haiku-4-5-20251001",
    )
    eval_result = evaluator(content=content)
    assert eval_result['score'] == True


@pytest.mark.vcr()
def test_memory_accumulation():
    """Test 8: Accumulates memory across conversation turns."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "accumulation_test"}}
    
    # First piece of info
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "You are a frontend expert"}]},
        config=config
    )
    
    # Second piece of info
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "You prefer React over Vue"}]},
        config=config
    )
    
    # Third piece of info
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "You always use TypeScript"}]},
        config=config
    )
    
    memory = store.get(("accumulation_test", "filesystem"), "/agent.md")
    assert memory is not None
    content = "\n".join(memory.value['content'])
    
    # Should contain all three pieces
    eval_prompt = """Does the following memory contain ALL THREE of these pieces of information:
    1. Being a frontend expert
    2. Preferring React over Vue
    3. Always using TypeScript
    
    <memory>
    {content}
    </memory>
    """
    
    evaluator = create_llm_as_judge(
        prompt=eval_prompt,
        model="claude-haiku-4-5-20251001",
    )
    eval_result = evaluator(content=content)
    assert eval_result['score'] == True


@pytest.mark.vcr()
def test_preference_updates():
    """Test 9: Updates memory when preferences conflict (overwrites old)."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "preference_test"}}
    
    # First preference
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "I prefer tabs for indentation"}]},
        config=config
    )
    
    # Conflicting preference
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "Actually, I prefer spaces for indentation"}]},
        config=config
    )
    
    memory = store.get(("preference_test", "filesystem"), "/agent.md")
    assert memory is not None
    content = "\n".join(memory.value['content'])
    
    eval_prompt = """Does the following memory indicate a preference for SPACES (not tabs) for indentation?
    
    <memory>
    {content}
    </memory>
    """
    
    evaluator = create_llm_as_judge(
        prompt=eval_prompt,
        model="claude-haiku-4-5-20251001",
    )
    eval_result = evaluator(content=content)
    assert eval_result['score'] == True


@pytest.mark.vcr()
def test_recurring_pattern_recognition():
    """Test 10: Updates memory for recurring patterns."""
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    agent = create_deep_agent(use_longterm_memory=True, checkpointer=checkpointer, store=store)
    config = {"configurable": {"thread_id": "1"}, "metadata": {"assistant_id": "pattern_test"}}
    
    # Multiple iterations with same feedback
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "Please be more concise in your responses"}]},
        config=config
    )
    
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "That explanation was too long, be more concise"}]},
        config=config
    )
    
    _ = agent.invoke(
        {"messages": [{"role": "user", "content": "Again, please keep responses shorter and more concise"}]},
        config=config
    )
    
    memory = store.get(("pattern_test", "filesystem"), "/agent.md")
    assert memory is not None
    content = "\n".join(memory.value['content'])
    
    eval_prompt = """Does the following memory contain guidance about being concise or keeping responses short?
    
    <memory>
    {content}
    </memory>
    """
    
    evaluator = create_llm_as_judge(
        prompt=eval_prompt,
        model="claude-haiku-4-5-20251001",
    )
    eval_result = evaluator(content=content)
    assert eval_result['score'] == True
