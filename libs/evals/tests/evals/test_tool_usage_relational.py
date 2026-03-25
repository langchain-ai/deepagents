"""Eval tests for relational data tool usage.

Recreates the relational data environment from langchain-benchmarks: fake
users, locations, and foods connected by IDs.  The agent receives *only* the
lookup / search tools (no filesystem) and must chain them to answer questions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from langchain_core.tools import ToolException, tool
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.utils import (
    TrajectoryScorer,
    final_text_contains,
    run_agent,
    tool_call,
)

pytestmark = [pytest.mark.eval_category("tool_use")]

# ---------------------------------------------------------------------------
# Static relational data
# ---------------------------------------------------------------------------


class UserRecord(TypedDict):
    id: int
    name: str
    email: str
    location: int
    favorite_color: str
    favorite_foods: list[int]


class LocationRecord(TypedDict):
    id: int
    city: str
    current_time: str
    current_weather: str


class FoodRecord(TypedDict):
    id: int
    name: str
    calories: int
    allergic_ingredients: list[str]


class UserSearchResult(TypedDict):
    id: int
    name: str


class LocationSearchResult(TypedDict):
    id: int
    city: str


class FoodSearchResult(TypedDict):
    id: int
    name: str


USER_DATA: list[UserRecord] = [
    {
        "id": 1,
        "name": "Alice",
        "email": "alice@gmail.com",
        "location": 1,
        "favorite_color": "red",
        "favorite_foods": [1, 2, 3],
    },
    {
        "id": 21,
        "name": "Bob",
        "email": "bob@hotmail.com",
        "location": 2,
        "favorite_color": "orange",
        "favorite_foods": [4, 5, 6],
    },
    {
        "id": 35,
        "name": "Charlie",
        "email": "charlie@yahoo.com",
        "location": 3,
        "favorite_color": "yellow",
        "favorite_foods": [3, 7, 2],
    },
    {
        "id": 41,
        "name": "Donna",
        "email": "donna@example.com",
        "location": 4,
        "favorite_color": "green",
        "favorite_foods": [6, 1, 4],
    },
    {
        "id": 42,
        "name": "Eve",
        "email": "eve@example.org",
        "location": 5,
        "favorite_color": "blue",
        "favorite_foods": [5, 7, 4],
    },
    {
        "id": 43,
        "name": "Frank The Cat",
        "email": "frank.the.cat@langchain.dev",
        "location": 5,
        "favorite_color": "yellow",
        "favorite_foods": [3],
    },
]

LOCATION_DATA: list[LocationRecord] = [
    {
        "id": 1,
        "city": "New York",
        "current_time": "2023-11-14 10:30 AM",
        "current_weather": "Partly Cloudy, Temperature: 68\u00b0F",
    },
    {
        "id": 2,
        "city": "Los Angeles",
        "current_time": "2023-11-14 7:45 AM",
        "current_weather": "Sunny, Temperature: 75\u00b0F",
    },
    {
        "id": 3,
        "city": "Chicago",
        "current_time": "2023-11-14 11:15 AM",
        "current_weather": "Mostly Cloudy, Temperature: 60\u00b0F",
    },
    {
        "id": 4,
        "city": "Houston",
        "current_time": "2023-11-14 12:00 PM",
        "current_weather": "Rainy, Temperature: 55\u00b0F",
    },
    {
        "id": 5,
        "city": "Miami",
        "current_time": "2023-11-14 1:20 PM",
        "current_weather": "Partly Cloudy, Temperature: 80\u00b0F",
    },
]

FOOD_DATA: list[FoodRecord] = [
    {
        "id": 1,
        "name": "Pizza",
        "calories": 285,
        "allergic_ingredients": ["Gluten", "Dairy"],
    },
    {
        "id": 2,
        "name": "Chocolate",
        "calories": 50,
        "allergic_ingredients": ["Milk", "Soy"],
    },
    {
        "id": 3,
        "name": "Sushi",
        "calories": 300,
        "allergic_ingredients": ["Fish", "Soy"],
    },
    {
        "id": 4,
        "name": "Burger",
        "calories": 350,
        "allergic_ingredients": ["Gluten", "Dairy"],
    },
    {
        "id": 5,
        "name": "Ice Cream",
        "calories": 200,
        "allergic_ingredients": ["Dairy"],
    },
    {
        "id": 6,
        "name": "Pasta",
        "calories": 180,
        "allergic_ingredients": ["Gluten"],
    },
    {
        "id": 7,
        "name": "Salad",
        "calories": 50,
        "allergic_ingredients": [],
    },
]


# ---------------------------------------------------------------------------
# Internal helpers (not exposed as tools)
# ---------------------------------------------------------------------------


def _similarity_search(
    data: list[UserRecord] | list[LocationRecord] | list[FoodRecord],
    query: str,
    key: str,
) -> list[UserSearchResult] | list[LocationSearchResult] | list[FoodSearchResult]:
    """Jaccard-similarity search over a string field."""

    def _score(x: str) -> float:
        return len(set(x) & set(query)) / len(set(x) | set(query))

    ranked = sorted(data, key=lambda x: _score(x[key]), reverse=True)
    return [{"id": d["id"], key: d[key]} for d in ranked]


def _get_user(user_id: int) -> UserRecord:
    for user in USER_DATA:
        if user["id"] == user_id:
            return user
    msg = f"User ID {user_id} cannot be resolved"
    raise ToolException(msg)


def _get_location(location_id: int) -> LocationRecord:
    for loc in LOCATION_DATA:
        if loc["id"] == location_id:
            return loc
    msg = f"Location ID {location_id} cannot be resolved"
    raise ToolException(msg)


def _get_food(food_id: int) -> FoodRecord:
    for food in FOOD_DATA:
        if food["id"] == food_id:
            return food
    msg = f"Food ID {food_id} cannot be resolved"
    raise ToolException(msg)


# ---------------------------------------------------------------------------
# Tools  (plain functions decorated with @tool)
# ---------------------------------------------------------------------------


@tool
def get_user_name(user_id: int) -> str:
    """Get the name of the user with the given user ID.

    Args:
        user_id: The user's ID.
    """
    return _get_user(user_id)["name"]


@tool
def list_user_ids() -> list[int]:
    """List all the user IDs."""
    return [u["id"] for u in USER_DATA]


@tool
def find_users_by_name(name: str) -> list[UserSearchResult]:
    """Find users with the given name.

    Args:
        name: The name to search for.
    """
    return _similarity_search(USER_DATA, name, "name")


@tool
def find_locations_by_name(city: str) -> list[LocationSearchResult]:
    """Find locations with the given city name.

    Args:
        city: The city name to search for.
    """
    return _similarity_search(LOCATION_DATA, city, "city")


@tool
def find_foods_by_name(food: str) -> list[FoodSearchResult]:
    """Find foods with the given name.

    Args:
        food: The food name to search for.
    """
    return _similarity_search(FOOD_DATA, food, "name")


@tool
def get_user_email(user_id: int) -> str:
    """Get the email of the user with the given user ID.

    Args:
        user_id: The user's ID.
    """
    return _get_user(user_id)["email"]


@tool
def get_user_location(user_id: int) -> int:
    """Get the location ID of the user with the given user ID.

    Args:
        user_id: The user's ID.
    """
    return _get_user(user_id)["location"]


@tool
def get_user_favorite_color(user_id: int) -> str:
    """Get the favorite color of the user with the given user ID.

    Args:
        user_id: The user's ID.
    """
    return _get_user(user_id)["favorite_color"]


@tool
def get_user_favorite_foods(user_id: int) -> list[int]:
    """Get the list of favorite food IDs of the user with the given user ID.

    Args:
        user_id: The user's ID.
    """
    return _get_user(user_id)["favorite_foods"]


@tool
def get_weather_at_location(location_id: int) -> str:
    """Get the current weather at the location with the given location ID.

    Args:
        location_id: The location's ID.
    """
    return _get_location(location_id)["current_weather"]


@tool
def get_city_for_location(location_id: int) -> str:
    """Get the city for the location with the given location ID.

    Args:
        location_id: The location's ID.
    """
    return _get_location(location_id)["city"]


@tool
def get_current_time_for_location(location_id: int) -> str:
    """Get the current time for the location with the given location ID.

    Args:
        location_id: The location's ID.
    """
    return _get_location(location_id)["current_time"]


@tool
def get_food_name(food_id: int) -> str:
    """Get the name of the food with the given food ID.

    Args:
        food_id: The food's ID.
    """
    return _get_food(food_id)["name"]


@tool
def get_food_calories(food_id: int) -> int:
    """Get the calories per serving for the food with the given food ID.

    Args:
        food_id: The food's ID.
    """
    return _get_food(food_id)["calories"]


@tool
def get_food_allergic_ingredients(food_id: int) -> list[str]:
    """Get the list of allergic ingredients for the food with the given food ID.

    Args:
        food_id: The food's ID.
    """
    return _get_food(food_id)["allergic_ingredients"]


@tool
def get_current_user_id() -> int:
    """Get the current user's ID."""
    return 35


# ---------------------------------------------------------------------------
# All relational-data tools collected for easy import
# ---------------------------------------------------------------------------

RELATIONAL_TOOLS = [
    get_user_name,
    list_user_ids,
    find_users_by_name,
    find_locations_by_name,
    find_foods_by_name,
    get_user_email,
    get_user_location,
    get_user_favorite_color,
    get_user_favorite_foods,
    get_weather_at_location,
    get_city_for_location,
    get_current_time_for_location,
    get_food_name,
    get_food_calories,
    get_food_allergic_ingredients,
    get_current_user_id,
]

RELATIONAL_TOOL_NAMES = [tool.name for tool in RELATIONAL_TOOLS]
RELATIONAL_TOOL_IMPLEMENTATIONS = {tool.name: tool for tool in RELATIONAL_TOOLS}

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def _create_agent(model: BaseChatModel):
    """Create agent."""
    return create_deep_agent(
        model=model,
        tools=RELATIONAL_TOOLS,
    )


@pytest.mark.langsmith
def test_single_tool_list_user_ids(model: BaseChatModel) -> None:
    """Agent lists all user IDs with a single tool call."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="What are all the user IDs in the system?",
        # 1st step: call list_user_ids.
        # 2nd step: answer with the IDs.
        # 1 tool call: list_user_ids.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("1"),
            final_text_contains("21"),
            final_text_contains("35"),
            final_text_contains("41"),
            final_text_contains("42"),
            final_text_contains("43"),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[tool_call(name="list_user_ids", step=1)],
        ),
    )


@pytest.mark.langsmith
def test_single_tool_get_user_email(model: BaseChatModel) -> None:
    """Agent retrieves a user's email by ID."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="What is the email address of user 21?",
        # 1st step: call get_user_email with user_id=21.
        # 2nd step: answer with the email.
        # 1 tool call: get_user_email.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("bob@hotmail.com"),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[tool_call(name="get_user_email", step=1, args_contains={"user_id": 21})],
        ),
    )


@pytest.mark.langsmith
def test_single_tool_get_food_calories(model: BaseChatModel) -> None:
    """Agent retrieves calorie info for a food by ID."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="How many calories does food 5 have per serving?",
        # 1st step: call get_food_calories with food_id=5.
        # 2nd step: answer with the calorie count.
        # 1 tool call: get_food_calories.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("200"),
        )
        .expect(
            agent_steps=2,
            tool_call_requests=1,
            tool_calls=[tool_call(name="get_food_calories", step=1, args_contains={"food_id": 5})],
        ),
    )


@pytest.mark.langsmith
def test_two_tools_user_name_from_current_id(model: BaseChatModel) -> None:
    """Agent gets the current user ID, then looks up the name."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="What is the name of the current user?",
        # 1st step: call get_current_user_id -> 35.
        # 2nd step: call get_user_name(user_id=35) -> "Charlie".
        # 3rd step: answer with the name.
        # 2 tool calls: get_current_user_id, get_user_name.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Charlie"),
        )
        .expect(
            agent_steps=3,
            tool_call_requests=2,
            tool_calls=[
                tool_call(name="get_current_user_id", step=1),
                tool_call(name="get_user_name", step=2, args_contains={"user_id": 35}),
            ],
        ),
    )


@pytest.mark.langsmith
def test_two_tools_city_for_user(model: BaseChatModel) -> None:
    """Agent resolves user 1's location ID, then gets the city name."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="What city does user 1 live in?",
        # 1st step: call get_user_location(user_id=1) -> 1.
        # 2nd step: call get_city_for_location(location_id=1) -> "New York".
        # 3rd step: answer with the city.
        # 2 tool calls: get_user_location, get_city_for_location.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("New York"),
        )
        .expect(
            agent_steps=3,
            tool_call_requests=2,
            tool_calls=[
                tool_call(name="get_user_location", step=1, args_contains={"user_id": 1}),
                tool_call(
                    name="get_city_for_location",
                    step=2,
                    args_contains={"location_id": 1},
                ),
            ],
        ),
    )


@pytest.mark.langsmith
def test_two_tools_find_user_then_email(model: BaseChatModel) -> None:
    """Agent searches for a user by name, then gets their email."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="What is Eve's email address?",
        # 1st step: call find_users_by_name(name="Eve") -> [{id: 42, ...}, ...].
        # 2nd step: call get_user_email(user_id=42) -> "eve@example.org".
        # 3rd step: answer with the email.
        # 2 tool calls: find_users_by_name, get_user_email.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("eve@example.org"),
        )
        .expect(
            agent_steps=3,
            tool_call_requests=2,
            tool_calls=[
                tool_call(name="find_users_by_name", step=1, args_contains={"name": "Eve"}),
                tool_call(name="get_user_email", step=2, args_contains={"user_id": 42}),
            ],
        ),
    )


@pytest.mark.langsmith
def test_three_tools_current_user_city(model: BaseChatModel) -> None:
    """Agent resolves current user -> location ID -> city name."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="What city does the current user live in?",
        # 1st step: get_current_user_id -> 35.
        # 2nd step: get_user_location(35) -> 3.
        # 3rd step: get_city_for_location(3) -> "Chicago".
        # 4th step: answer.
        # 3 tool calls.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Chicago"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=3,
            tool_calls=[
                tool_call(name="get_current_user_id", step=1),
                tool_call(name="get_user_location", step=2, args_contains={"user_id": 35}),
                tool_call(
                    name="get_city_for_location",
                    step=3,
                    args_contains={"location_id": 3},
                ),
            ],
        ),
    )


@pytest.mark.langsmith
def test_three_tools_find_user_then_city(model: BaseChatModel) -> None:
    """Agent searches for Alice by name, gets her location ID, then resolves the city."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="What city does Alice live in?",
        # 1st step: find_users_by_name("Alice") -> [{id: 1, ...}, ...].
        # 2nd step: get_user_location(1) -> 1.
        # 3rd step: get_city_for_location(1) -> "New York".
        # 4th step: answer.
        # 3 tool calls.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("New York"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=3,
            tool_calls=[
                tool_call(name="find_users_by_name", step=1, args_contains={"name": "Alice"}),
                tool_call(name="get_user_location", step=2, args_contains={"user_id": 1}),
                tool_call(
                    name="get_city_for_location",
                    step=3,
                    args_contains={"location_id": 1},
                ),
            ],
        ),
    )


@pytest.mark.langsmith
def test_three_tools_current_user_weather(model: BaseChatModel) -> None:
    """Agent resolves current user -> location ID -> weather."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="What is the current weather where the current user lives?",
        # 1st step: get_current_user_id -> 35.
        # 2nd step: get_user_location(35) -> 3.
        # 3rd step: get_weather_at_location(3) -> "Mostly Cloudy, Temperature: 60F".
        # 4th step: answer.
        # 3 tool calls.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("60", case_insensitive=True),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=3,
            tool_calls=[
                tool_call(name="get_current_user_id", step=1),
                tool_call(name="get_user_location", step=2, args_contains={"user_id": 35}),
                tool_call(
                    name="get_weather_at_location",
                    step=3,
                    args_contains={"location_id": 3},
                ),
            ],
        ),
    )


@pytest.mark.langsmith
def test_four_tools_current_user_favorite_food_names(model: BaseChatModel) -> None:
    """Agent resolves current user -> favorite food IDs -> food names (parallel)."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query="What are the names of the current user's favorite foods?",
        # 1st step: get_current_user_id -> 35.
        # 2nd step: get_user_favorite_foods(35) -> [3, 7, 2].
        # 3rd step: get_food_name(3), get_food_name(7), get_food_name(2) in parallel.
        # 4th step: answer.
        # 5 tool call requests: 1 + 1 + 3 parallel.
        # 4 agent steps: the 3 parallel calls count as one step.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Sushi"),
            final_text_contains("Salad"),
            final_text_contains("Chocolate"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=5,
            tool_calls=[
                tool_call(name="get_current_user_id", step=1),
                tool_call(
                    name="get_user_favorite_foods",
                    step=2,
                    args_contains={"user_id": 35},
                ),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 3}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 7}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 2}),
            ],
        ),
    )


@pytest.mark.langsmith
def test_four_tools_find_user_food_name_and_calories(model: BaseChatModel) -> None:
    """Agent finds Frank The Cat -> fav foods -> food name + calories (parallel)."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "How many calories per serving does Frank The Cat's favorite food have? Also tell me the name of the food."
        ),
        # 1st step: find_users_by_name("Frank The Cat") -> [{id: 43, ...}, ...].
        # 2nd step: get_user_favorite_foods(43) -> [3].
        # 3rd step: get_food_name(3) and get_food_calories(3) in parallel.
        # 4th step: answer.
        # 4 tool call requests: 1 + 1 + 2 parallel.
        # 4 agent steps.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Sushi"),
            final_text_contains("300"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(name="find_users_by_name", step=1),
                tool_call(
                    name="get_user_favorite_foods",
                    step=2,
                    args_contains={"user_id": 43},
                ),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 3}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 3}),
            ],
        ),
    )


@pytest.mark.langsmith
def test_four_tools_current_user_location_time_and_weather(
    model: BaseChatModel,
) -> None:
    """Agent resolves current user -> location -> time + weather (parallel)."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query=("What is the current time and weather where the current user lives?"),
        # 1st step: get_current_user_id -> 35.
        # 2nd step: get_user_location(35) -> 3.
        # 3rd step: get_current_time_for_location(3) and
        #           get_weather_at_location(3) in parallel.
        # 4th step: answer.
        # 4 tool call requests: 1 + 1 + 2 parallel.
        # 4 agent steps.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("11:15"),
            final_text_contains("60", case_insensitive=True),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[
                tool_call(name="get_current_user_id", step=1),
                tool_call(name="get_user_location", step=2, args_contains={"user_id": 35}),
                tool_call(
                    name="get_current_time_for_location",
                    step=3,
                    args_contains={"location_id": 3},
                ),
                tool_call(
                    name="get_weather_at_location",
                    step=3,
                    args_contains={"location_id": 3},
                ),
            ],
        ),
    )


@pytest.mark.langsmith
def test_five_steps_current_user_food_names_and_calories(model: BaseChatModel) -> None:
    """Agent resolves current user -> fav foods -> names + calories (all parallel)."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "For each of the current user's favorite foods, tell me the food name and how many calories per serving it has."
        ),
        # 1st step: get_current_user_id -> 35.
        # 2nd step: get_user_favorite_foods(35) -> [3, 7, 2].
        # 3rd step: get_food_name and get_food_calories for each food ID, all 6 in parallel.
        # 4th step: answer.
        # 8 tool call requests: 1 + 1 + 6 parallel.
        # 4 agent steps.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Sushi"),
            final_text_contains("300"),
            final_text_contains("Salad"),
            final_text_contains("50"),
            final_text_contains("Chocolate"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=8,
            tool_calls=[
                tool_call(name="get_current_user_id", step=1),
                tool_call(
                    name="get_user_favorite_foods",
                    step=2,
                    args_contains={"user_id": 35},
                ),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 3}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 7}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 2}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 3}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 7}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 2}),
            ],
        ),
    )


@pytest.mark.langsmith
def test_four_steps_find_user_city_and_weather(model: BaseChatModel) -> None:
    """Agent finds Bob -> location -> city + time + weather (parallel)."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "Find Bob and tell me what city he lives in, the current time there, and the current weather."
        ),
        # 1st step: find_users_by_name("Bob") -> [{id: 21, ...}, ...].
        # 2nd step: get_user_location(21) -> 2.
        # 3rd step: get_city_for_location(2), get_current_time_for_location(2),
        #           get_weather_at_location(2) in parallel.
        # 4th step: answer.
        # 5 tool call requests: 1 + 1 + 3 parallel.
        # 4 agent steps.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Los Angeles"),
            final_text_contains("7:45"),
            final_text_contains("75", case_insensitive=True),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=5,
            tool_calls=[
                tool_call(name="find_users_by_name", step=1, args_contains={"name": "Bob"}),
                tool_call(name="get_user_location", step=2, args_contains={"user_id": 21}),
                tool_call(
                    name="get_city_for_location",
                    step=3,
                    args_contains={"location_id": 2},
                ),
                tool_call(
                    name="get_current_time_for_location",
                    step=3,
                    args_contains={"location_id": 2},
                ),
                tool_call(
                    name="get_weather_at_location",
                    step=3,
                    args_contains={"location_id": 2},
                ),
            ],
        ),
    )


@pytest.mark.langsmith
def test_four_steps_find_user_food_allergies(model: BaseChatModel) -> None:
    """Agent finds Alice -> fav foods -> food names + allergies (all parallel)."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query=("What are the names and allergic ingredients of all of Alice's favorite foods?"),
        # 1st step: find_users_by_name("Alice") -> [{id: 1, ...}, ...].
        # 2nd step: get_user_favorite_foods(1) -> [1, 2, 3].
        # 3rd step: get_food_name and get_food_allergic_ingredients for each, all 6 in parallel.
        # 4th step: answer.
        # 8 tool call requests: 1 + 1 + 6 parallel.
        # 4 agent steps.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Pizza"),
            final_text_contains("Gluten"),
            final_text_contains("Chocolate"),
            final_text_contains("Milk"),
            final_text_contains("Sushi"),
            final_text_contains("Fish"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=8,
            tool_calls=[
                tool_call(name="find_users_by_name", step=1, args_contains={"name": "Alice"}),
                tool_call(name="get_user_favorite_foods", step=2, args_contains={"user_id": 1}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 1}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 2}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 3}),
                tool_call(
                    name="get_food_allergic_ingredients",
                    step=3,
                    args_contains={"food_id": 1},
                ),
                tool_call(
                    name="get_food_allergic_ingredients",
                    step=3,
                    args_contains={"food_id": 2},
                ),
                tool_call(
                    name="get_food_allergic_ingredients",
                    step=3,
                    args_contains={"food_id": 3},
                ),
            ],
        ),
    )


@pytest.mark.langsmith
def test_four_steps_current_user_food_names_calories_and_allergies(
    model: BaseChatModel,
) -> None:
    """Agent resolves current user -> favorite foods -> all requested food details in parallel."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "For each of the current user's favorite foods, tell me the food name, calories per serving, and allergic ingredients."
        ),
        # 1st step: get_current_user_id -> 35.
        # 2nd step: get_user_favorite_foods(35) -> [3, 7, 2].
        # 3rd step: get_food_name, get_food_calories, and get_food_allergic_ingredients
        #           for each food ID, all 9 calls in parallel.
        # 4th step: answer.
        # 11 tool call requests: 1 + 1 + 9 parallel.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Sushi"),
            final_text_contains("300"),
            final_text_contains("Fish"),
            final_text_contains("Salad"),
            final_text_contains("Chocolate"),
            final_text_contains("Milk"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=11,
            tool_calls=[
                tool_call(name="get_current_user_id", step=1),
                tool_call(
                    name="get_user_favorite_foods",
                    step=2,
                    args_contains={"user_id": 35},
                ),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 3}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 7}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 2}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 3}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 7}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 2}),
                tool_call(
                    name="get_food_allergic_ingredients",
                    step=3,
                    args_contains={"food_id": 3},
                ),
                tool_call(
                    name="get_food_allergic_ingredients",
                    step=3,
                    args_contains={"food_id": 7},
                ),
                tool_call(
                    name="get_food_allergic_ingredients",
                    step=3,
                    args_contains={"food_id": 2},
                ),
            ],
        ),
    )


@pytest.mark.langsmith
def test_four_steps_find_user_city_weather_time_and_food_details(
    model: BaseChatModel,
) -> None:
    """Agent finds Donna and gathers location plus detailed favorite-food info."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "Find Donna and tell me which city she lives in, the current weather and time there, and the names "
            "and calories of all her favorite foods."
        ),
        # 1st step: find_users_by_name("Donna") -> [{id: 41, ...}, ...].
        # 2nd step: get_user_location(41) and get_user_favorite_foods(41) in parallel.
        # 3rd step: get_city_for_location(4), get_current_time_for_location(4),
        #           get_weather_at_location(4), get_food_name for each favorite food ID,
        #           and get_food_calories for each favorite food ID, all 9 calls in parallel.
        # 4th step: answer.
        # 12 tool call requests: 1 + 2 + 9 parallel.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("Houston"),
            final_text_contains("12:00"),
            final_text_contains("55", case_insensitive=True),
            final_text_contains("Pasta"),
            final_text_contains("Pizza"),
            final_text_contains("Burger"),
            final_text_contains("180"),
            final_text_contains("285"),
            final_text_contains("350"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=12,
            tool_calls=[
                tool_call(name="find_users_by_name", step=1, args_contains={"name": "Donna"}),
                tool_call(name="get_user_location", step=2, args_contains={"user_id": 41}),
                tool_call(
                    name="get_user_favorite_foods",
                    step=2,
                    args_contains={"user_id": 41},
                ),
                tool_call(
                    name="get_city_for_location",
                    step=3,
                    args_contains={"location_id": 4},
                ),
                tool_call(
                    name="get_current_time_for_location",
                    step=3,
                    args_contains={"location_id": 4},
                ),
                tool_call(
                    name="get_weather_at_location",
                    step=3,
                    args_contains={"location_id": 4},
                ),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 6}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 1}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 4}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 6}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 1}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 4}),
            ],
        ),
    )


@pytest.mark.langsmith
def test_four_steps_find_user_email_city_foods_calories_and_allergies(
    model: BaseChatModel,
) -> None:
    """Agent finds Eve and returns contact, location, and detailed food facts."""
    agent = _create_agent(model)
    run_agent(
        agent,
        model=model,
        query=(
            "Find Eve and tell me her email address, what city she lives in, and for each of her favorite foods, "
            "give the food name, calories per serving, and allergic ingredients."
        ),
        # 1st step: find_users_by_name("Eve") -> [{id: 42, ...}, ...].
        # 2nd step: get_user_email(42), get_user_location(42), and get_user_favorite_foods(42) in parallel.
        # 3rd step: get_city_for_location(5), get_food_name for each favorite food ID,
        #           get_food_calories for each favorite food ID, and get_food_allergic_ingredients
        #           for each favorite food ID, all 10 calls in parallel.
        # 4th step: answer.
        # 14 tool call requests: 1 + 3 + 10 parallel.
        scorer=TrajectoryScorer()
        .success(
            final_text_contains("eve@example.org"),
            final_text_contains("Miami"),
            final_text_contains("Ice Cream"),
            final_text_contains("200"),
            final_text_contains("Dairy"),
            final_text_contains("Salad"),
            final_text_contains("Burger"),
            final_text_contains("350"),
            final_text_contains("Gluten"),
        )
        .expect(
            agent_steps=4,
            tool_call_requests=14,
            tool_calls=[
                tool_call(name="find_users_by_name", step=1, args_contains={"name": "Eve"}),
                tool_call(name="get_user_email", step=2, args_contains={"user_id": 42}),
                tool_call(name="get_user_location", step=2, args_contains={"user_id": 42}),
                tool_call(
                    name="get_user_favorite_foods",
                    step=2,
                    args_contains={"user_id": 42},
                ),
                tool_call(
                    name="get_city_for_location",
                    step=3,
                    args_contains={"location_id": 5},
                ),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 5}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 7}),
                tool_call(name="get_food_name", step=3, args_contains={"food_id": 4}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 5}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 7}),
                tool_call(name="get_food_calories", step=3, args_contains={"food_id": 4}),
                tool_call(
                    name="get_food_allergic_ingredients",
                    step=3,
                    args_contains={"food_id": 5},
                ),
                tool_call(
                    name="get_food_allergic_ingredients",
                    step=3,
                    args_contains={"food_id": 7},
                ),
                tool_call(
                    name="get_food_allergic_ingredients",
                    step=3,
                    args_contains={"food_id": 4},
                ),
            ],
        ),
    )
