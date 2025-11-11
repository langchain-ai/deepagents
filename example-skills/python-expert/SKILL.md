---
name: python-expert
description: Expert Python development following PEP 8, type hints, and modern best practices
---

# Python Expert Skill

Write high-quality Python code following modern best practices, PEP standards, and Pythonic idioms.

## Coding Standards

### Style Guide (PEP 8)
- Use 4 spaces for indentation
- Maximum line length of 88 characters (Black formatter standard)
- Use snake_case for functions and variables
- Use PascalCase for classes
- Use UPPER_CASE for constants

### Type Hints (PEP 484)
Always include type hints:
```python
def process_data(items: list[str], threshold: int = 10) -> dict[str, int]:
    """Process items and return counts."""
    result: dict[str, int] = {}
    # ... implementation
    return result
```

### Documentation
- Use docstrings for all public modules, functions, classes, and methods
- Follow Google or NumPy docstring format
- Include type information, parameters, returns, and raises sections

### Modern Python Features
- Use f-strings for formatting (Python 3.6+)
- Use type hints with `from __future__ import annotations` (Python 3.7+)
- Use `pathlib.Path` instead of `os.path`
- Use `dataclasses` for simple data containers
- Use `typing` module for complex types

## Best Practices

### Error Handling
```python
from pathlib import Path

def read_config(path: Path) -> dict:
    """Read configuration from file."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in config: {e}")
```

### Context Managers
Always use context managers for resources:
```python
with open("file.txt") as f:
    data = f.read()

# Or create custom context managers
from contextlib import contextmanager

@contextmanager
def temporary_setting(name: str, value: Any):
    """Temporarily change a setting."""
    old_value = get_setting(name)
    set_setting(name, value)
    try:
        yield
    finally:
        set_setting(name, old_value)
```

### List Comprehensions and Generators
Prefer comprehensions and generators for efficiency:
```python
# List comprehension
squared = [x**2 for x in numbers if x > 0]

# Generator expression for large datasets
sum_of_squares = sum(x**2 for x in large_dataset)

# Dict comprehension
name_to_age = {person.name: person.age for person in people}
```

## Testing

Write tests using pytest:
```python
import pytest

def test_process_data():
    result = process_data(["a", "b", "c"])
    assert len(result) == 3
    assert "a" in result

def test_process_data_empty():
    result = process_data([])
    assert result == {}

@pytest.mark.parametrize("input,expected", [
    ([1, 2, 3], 6),
    ([0], 0),
    ([], 0),
])
def test_sum_values(input, expected):
    assert sum_values(input) == expected
```

## Code Organization

Structure Python projects:
```
project/
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── pyproject.toml
├── README.md
└── .gitignore
```

## Common Patterns

### Dataclasses
```python
from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int
    hobbies: list[str] = field(default_factory=list)
```

### Enums
```python
from enum import Enum, auto

class Status(Enum):
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
```

### Abstract Base Classes
```python
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data."""
        pass
```

## Guidelines

1. Write clear, readable code that follows Python conventions
2. Include type hints for all function signatures
3. Use modern Python features (3.10+ where appropriate)
4. Write comprehensive docstrings
5. Handle errors explicitly and provide helpful error messages
6. Write testable code with appropriate test coverage
7. Use `ruff` for linting and `black` for formatting
