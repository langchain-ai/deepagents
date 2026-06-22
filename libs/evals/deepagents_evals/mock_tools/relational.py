"""Mock relational-data tools for the relational tool-usage eval suite.

Recreates the relational data environment from langchain-benchmarks: fake
users, locations, and foods connected by IDs. The agent receives only the
lookup / search tools (no filesystem) and must chain them to answer questions.

Extracted from `tests/evals/test_tool_usage_relational.py` so both the pytest
suite and the Harbor sandbox dispatcher share the same tool definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import ToolException, tool
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Static relational data
# ---------------------------------------------------------------------------


class UserRecord(TypedDict):
    """User record."""

    id: int
    name: str
    email: str
    location: int
    favorite_color: str
    favorite_foods: list[int]


class LocationRecord(TypedDict):
    """Location record."""

    id: int
    city: str
    current_time: str
    current_weather: str


class FoodRecord(TypedDict):
    """Food record."""

    id: int
    name: str
    calories: int
    allergic_ingredients: list[str]


class UserSearchResult(TypedDict):
    """Search result for users."""

    id: int
    name: str


class LocationSearchResult(TypedDict):
    """Search result for locations."""

    id: int
    city: str


class FoodSearchResult(TypedDict):
    """Search result for foods."""

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


def _rank_by_similarity[ItemT](
    data: list[ItemT], query: str, value: Callable[[ItemT], str]
) -> list[ItemT]:
    """Jaccard-similarity search over a string field."""

    def _score(x: str) -> float:
        return len(set(x) & set(query)) / len(set(x) | set(query))

    return sorted(data, key=lambda item: _score(value(item)), reverse=True)


def _search_users_by_name(name: str) -> list[UserSearchResult]:
    return [
        {"id": user["id"], "name": user["name"]}
        for user in _rank_by_similarity(USER_DATA, name, lambda user: user["name"])
    ]


def _search_locations_by_city(city: str) -> list[LocationSearchResult]:
    return [
        {"id": location["id"], "city": location["city"]}
        for location in _rank_by_similarity(LOCATION_DATA, city, lambda location: location["city"])
    ]


def _search_foods_by_name(name: str) -> list[FoodSearchResult]:
    return [
        {"id": food["id"], "name": food["name"]}
        for food in _rank_by_similarity(FOOD_DATA, name, lambda food: food["name"])
    ]


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
# Tools (plain functions decorated with @tool)
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
    return _search_users_by_name(name)


@tool
def find_locations_by_name(city: str) -> list[LocationSearchResult]:
    """Find locations with the given city name.

    Args:
        city: The city name to search for.
    """
    return _search_locations_by_city(city)


@tool
def find_foods_by_name(food: str) -> list[FoodSearchResult]:
    """Find foods with the given name.

    Args:
        food: The food name to search for.
    """
    return _search_foods_by_name(food)


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
