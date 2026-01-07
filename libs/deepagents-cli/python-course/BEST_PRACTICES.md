# Python Best Practices and Coding Standards

Guidelines for writing clean, professional, maintainable Python code.

## 📋 Table of Contents
1. [Code Style](#code-style)
2. [Naming Conventions](#naming-conventions)
3. [Documentation](#documentation)
4. [Error Handling](#error-handling)
5. [Testing](#testing)
6. [Performance](#performance)
7. [Security](#security)
8. [Code Organization](#code-organization)
9. [Version Control](#version-control)
10. [Debugging and Logging](#debugging-and-logging)

---

## Code Style

### PEP 8 Compliance

Follow PEP 8 (Python Enhancement Proposal 8) for consistent code style.

#### Line Length
```python
# ✓ Good - Under 79 characters
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

# ❌ Bad - Over 79 characters
def calculate_average_of_all_positive_numbers_excluding_outliers(numbers):
    return sum(numbers) / len(numbers)
```

#### Indentation
```python
# ✓ Use 4 spaces
def my_function():
    if True:
        print("Hello")

# ❌ Don't use tabs
def my_function():
	if True:
		print("Hello")
```

#### Blank Lines
```python
# ✓ Two blank lines between top-level functions
def function_one():
    pass


def function_two():
    pass


# One blank line between methods
class MyClass:
    def method_one(self):
        pass
    
    def method_two(self):
        pass
```

#### Imports
```python
# ✓ Standard order
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from my_module import my_function

# ❌ Don't mix import styles
from os import *
import datetime, sys
```

#### Spacing
```python
# ✓ Good spacing
x = 5
y = x + 3
my_dict = {"key": "value"}
my_list = [1, 2, 3]
my_function(arg1, arg2, kwarg=value)

# ❌ Bad spacing
x=5
y=x+3
my_dict = { "key" : "value" }
my_list = [ 1,2,3 ]
my_function(arg1,arg2,kwarg = value)
```

---

## Naming Conventions

### Variable Names
```python
# ✓ Descriptive, lowercase with underscores
user_age = 25
first_name = "Alice"
max_attempts = 3
is_active = True

# ❌ Avoid single letters (except in loops)
x = 25
fn = "Alice"
m = 3

# ✓ Acceptable single letters in specific contexts
for i in range(10):
    print(i)

for x, y in coordinates:
    print(x, y)
```

### Function Names
```python
# ✓ Descriptive verb, lowercase with underscores
def get_user_data():
    pass

def validate_email(email):
    pass

def calculate_total_price(items):
    pass

# ❌ Unclear names
def process(x):
    pass

def get_stuff():
    pass

def func1():
    pass
```

### Class Names
```python
# ✓ PascalCase (capitalize first letter of each word)
class UserAccount:
    pass

class DatabaseConnection:
    pass

class DataProcessor:
    pass

# ❌ Don't use lowercase
class user_account:
    pass
```

### Constants
```python
# ✓ All caps with underscores
MAX_RETRIES = 3
API_TIMEOUT = 30
DEFAULT_LANGUAGE = "en"

PI = 3.14159
VALID_STATUSES = ["pending", "active", "completed"]

# ❌ Don't use lowercase
max_retries = 3
api_timeout = 30
```

### Private Variables
```python
# ✓ Single underscore prefix for "protected"
class MyClass:
    def __init__(self):
        self._internal_state = None
    
    def _helper_method(self):
        pass

# Double underscore for name mangling
class MyClass:
    def __init__(self):
        self.__private_var = None
    
    def __private_method(self):
        pass
```

---

## Documentation

### Docstrings

#### Module Docstring
```python
"""
This module provides utilities for data processing.

It includes functions for:
- Data validation
- Data transformation
- Data aggregation

Example:
    >>> from data_utils import validate_data
    >>> data = {"name": "Alice", "age": 25}
    >>> is_valid = validate_data(data)
"""
```

#### Function Docstring
```python
def calculate_average(numbers):
    """
    Calculate the average of a list of numbers.
    
    Args:
        numbers (list): List of numeric values
        
    Returns:
        float: The average of the numbers
        
    Raises:
        ValueError: If the list is empty
        TypeError: If any item is not numeric
        
    Example:
        >>> calculate_average([1, 2, 3, 4, 5])
        3.0
    """
    if not numbers:
        raise ValueError("List cannot be empty")
    return sum(numbers) / len(numbers)
```

#### Class Docstring
```python
class DatabaseConnection:
    """
    Manages database connections and queries.
    
    Attributes:
        host (str): Database host address
        port (int): Database port number
        connected (bool): Connection status
        
    Example:
        >>> db = DatabaseConnection("localhost", 5432)
        >>> db.connect()
        >>> results = db.query("SELECT * FROM users")
        >>> db.close()
    """
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connected = False
```

### Comments

```python
# ✓ Use comments to explain WHY, not WHAT
# Retry with exponential backoff to handle temporary failures
for attempt in range(max_retries):
    try:
        result = api_call()
        break
    except TemporaryError:
        time.sleep(2 ** attempt)

# ❌ Don't comment obvious code
# Add 1 to x
x = x + 1

# ❌ Don't over-comment
# Loop through items (obviously you're looping)
for item in items:
    # Process the item (obviously we're processing)
    process(item)
```

### Type Hints

```python
# ✓ Use type hints for clarity
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

def filter_active_users(users: list[dict]) -> list[dict]:
    """Filter users with active status."""
    return [u for u in users if u.get("status") == "active"]

def process_data(
    data: dict[str, int],
    multiplier: float = 1.0
) -> dict[str, float]:
    """Process data by multiplying values."""
    return {k: v * multiplier for k, v in data.items()}
```

---

## Error Handling

### Specific Exceptions

```python
# ✓ Catch specific exceptions
try:
    result = int(user_input)
except ValueError:
    print("Invalid number")
except KeyboardInterrupt:
    print("User cancelled")

# ❌ Don't catch everything
try:
    result = int(user_input)
except:
    print("Something went wrong")
```

### Custom Exceptions

```python
# ✓ Create meaningful custom exceptions
class InvalidUserError(ValueError):
    """Raised when user data is invalid."""
    pass

class DatabaseError(Exception):
    """Base exception for database operations."""
    pass

class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass

# Use them
if not user_data.get("email"):
    raise InvalidUserError("Email is required")
```

### Error Messages

```python
# ✓ Provide helpful error messages
if age < 0:
    raise ValueError(
        f"Age must be non-negative, got {age}"
    )

# ❌ Vague messages
if age < 0:
    raise ValueError("Invalid input")
```

---

## Testing

### Unit Tests

```python
# ✓ Write comprehensive unit tests
def test_calculate_average():
    """Test average calculation."""
    assert calculate_average([1, 2, 3]) == 2.0
    assert calculate_average([5]) == 5.0
    
def test_calculate_average_empty():
    """Test average with empty list."""
    with pytest.raises(ValueError):
        calculate_average([])

def test_calculate_average_negative():
    """Test average with negative numbers."""
    assert calculate_average([-1, -2, -3]) == -2.0
```

### Test Coverage

```python
# ✓ Test multiple scenarios
def test_user_creation():
    # Normal case
    user = User("Alice", 25)
    assert user.name == "Alice"
    
    # Edge case
    user = User("", 0)
    assert user.name == ""
    
    # Error case
    with pytest.raises(ValueError):
        User("Alice", -1)
```

---

## Performance

### List vs Generator

```python
# ❌ Create entire list in memory
def get_squares(n):
    return [i**2 for i in range(n)]  # All in memory

# ✓ Use generator for large datasets
def get_squares(n):
    for i in range(n):
        yield i**2  # Lazy evaluation
```

### String Concatenation

```python
# ❌ Inefficient - creates new string each time
result = ""
for item in items:
    result = result + str(item)

# ✓ Efficient - collect then join
parts = [str(item) for item in items]
result = "".join(parts)
```

### Dictionary Lookup

```python
# ❌ Slow - searches list
if status in ["active", "pending", "completed"]:
    pass

# ✓ Fast - O(1) lookup
VALID_STATUSES = {"active", "pending", "completed"}
if status in VALID_STATUSES:
    pass
```

---

## Security

### Input Validation

```python
# ✓ Validate all user input
def create_user(email, age):
    if not isinstance(email, str):
        raise TypeError("Email must be string")
    if "@" not in email:
        raise ValueError("Invalid email format")
    if not isinstance(age, int) or age < 0:
        raise ValueError("Age must be non-negative integer")
    
    return User(email, age)
```

### SQL Injection Prevention

```python
# ❌ NEVER do string formatting with SQL
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)

# ✓ Use parameterized queries
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

### Sensitive Data

```python
# ✓ Don't log sensitive information
password = "secret123"
# logger.info(f"Login attempt: {password}")  # NEVER!

logger.info("Login attempt")  # Better

# ✓ Use environment variables for secrets
import os
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY environment variable not set")
```

---

## Code Organization

### File Structure

```
my_project/
├── __init__.py              # Package initialization
├── config.py                # Configuration
├── constants.py             # Constants
├── models.py                # Data models
├── utils.py                 # Utility functions
├── handlers.py              # Main logic
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_utils.py
│   └── test_handlers.py
└── README.md
```

### Logical Grouping

```python
# ✓ Group related functionality
class User:
    """Represents a user."""
    
    # Initialization
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    # Validation
    def validate(self):
        if not self.email:
            raise ValueError("Email required")
    
    # Serialization
    def to_dict(self):
        return {"name": self.name, "email": self.email}
    
    # String representation
    def __str__(self):
        return f"User({self.name})"
```

### Avoid Deep Nesting

```python
# ❌ Too many nested levels
if condition1:
    if condition2:
        if condition3:
            if condition4:
                do_something()

# ✓ Use early returns
def process(data):
    if not data:
        return None
    if not validate(data):
        return None
    if is_outdated(data):
        return None
    
    return transform(data)
```

---

## Version Control

### Commit Messages

```
# ✓ Clear, descriptive commits
git commit -m "feat: Add user authentication"
git commit -m "fix: Handle null pointer in data processor"
git commit -m "docs: Update API documentation"

# ❌ Vague messages
git commit -m "fixed stuff"
git commit -m "update"
git commit -m "changes"
```

### Commit Frequency

```
# ✓ Logical, related changes
git commit -m "refactor: Extract validation logic to separate module"

# ❌ Too many unrelated changes
git commit -m "add feature, fix bug, update docs, refactor code"
```

### Branching

```bash
# ✓ Use feature branches
git checkout -b feature/user-authentication
git checkout -b fix/null-pointer-bug
git checkout -b docs/api-documentation

# ❌ Work on main branch
git checkout main
# ... make changes directly
```

---

## Debugging and Logging

### Logging Levels

```python
import logging

logger = logging.getLogger(__name__)

# DEBUG - Detailed information, typically for diagnostics
logger.debug("Variable x = 5")

# INFO - Confirmation things are working as expected
logger.info("User logged in successfully")

# WARNING - Warning, something unexpected happened
logger.warning("Retrying connection, attempt 2 of 3")

# ERROR - Serious problem, functionality impaired
logger.error("Failed to connect to database")

# CRITICAL - Very serious, program may not continue
logger.critical("Out of disk space, cannot save data")
```

### Logging Setup

```python
import logging
import logging.handlers

# ✓ Configure logging properly
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Debug Output

```python
# ✓ Use logging instead of print
logger.debug(f"Processing user: {user_id}")

# ✓ Use assertions for debug checks
assert isinstance(data, dict), "Data must be dictionary"

# ❌ Don't print to console in production code
print(f"Debug: user_id = {user_id}")
```

---

## Code Review Checklist

Before submitting code for review:

- [ ] Code follows PEP 8
- [ ] All variables/functions have clear names
- [ ] Functions have docstrings
- [ ] Complex logic has comments explaining WHY
- [ ] Error handling is comprehensive
- [ ] Unit tests are included
- [ ] No hardcoded values (use constants)
- [ ] No commented-out code
- [ ] No print statements (use logging)
- [ ] Security vulnerabilities checked
- [ ] Performance is acceptable
- [ ] Code is DRY (Don't Repeat Yourself)
- [ ] Functions are focused (single responsibility)
- [ ] Classes follow SOLID principles

---

## Common Patterns

### Singleton Pattern

```python
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Usage
db1 = DatabaseConnection()
db2 = DatabaseConnection()
assert db1 is db2  # Same instance
```

### Context Manager

```python
class FileManager:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Usage
with FileManager("data.txt") as f:
    data = f.read()
```

### Decorator

```python
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} took {duration:.4f}s")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(1)
```

---

## Tools for Code Quality

### Linters
- **pylint** - Comprehensive code analysis
- **flake8** - PEP 8 compliance
- **pycodestyle** - PEP 8 checking

### Formatters
- **black** - Opinionated code formatter
- **autopep8** - PEP 8 formatting

### Type Checking
- **mypy** - Static type checker
- **pyright** - Type checker from Microsoft

### Testing
- **pytest** - Testing framework
- **coverage.py** - Code coverage

---

## Summary

**Key Principles:**
1. **Readability** - Write for humans first, computers second
2. **Consistency** - Follow established patterns and standards
3. **Simplicity** - Keep it as simple as possible
4. **Documentation** - Explain complex decisions
5. **Testing** - Verify code works as expected
6. **Security** - Always validate and sanitize input
7. **Performance** - Optimize when needed, not prematurely

Remember: **"Code is read much more often than it is written."** - Guido van Rossum

---

Last Updated: 2024
Python Version: 3.8+
