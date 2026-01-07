"""
MODULE 3: DATA STRUCTURES
=========================

Master the fundamental data structures in Python.

Topics:
1. Lists - Ordered, mutable sequences
2. Tuples - Ordered, immutable sequences
3. Dictionaries - Key-value pairs
4. Sets - Unordered collections of unique items
5. Working with Multiple Structures
"""

# ============================================================================
# 1. LISTS
# ============================================================================

print("=" * 60)
print("1. LISTS - Ordered, Mutable Sequences")
print("=" * 60)

# Creating lists
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]
mixed = [1, "hello", 3.14, True]
empty = []

print(f"Numbers: {numbers}")
print(f"Names: {names}")
print(f"Mixed: {mixed}")
print(f"Length: {len(names)}")

# Accessing elements (0-indexed)
print(f"\nFirst name: {names[0]}")
print(f"Last name: {names[-1]}")
print(f"Second name: {names[1]}")

# Slicing
print(f"First two names: {names[0:2]}")
print(f"All except first: {names[1:]}")

# Modifying lists
numbers[0] = 10
print(f"\nAfter changing first element: {numbers}")

# Adding elements
numbers.append(6)
print(f"After append(6): {numbers}")

numbers.insert(2, 100)
print(f"After insert(2, 100): {numbers}")

# Removing elements
numbers.remove(100)
print(f"After remove(100): {numbers}")

last = numbers.pop()
print(f"After pop(): {numbers}, removed {last}")

# List methods
letters = ["c", "a", "b", "d", "a"]
print(f"\nList: {letters}")
print(f"Count of 'a': {letters.count('a')}")
print(f"Index of 'b': {letters.index('b')}")

letters_sorted = sorted(letters)
print(f"Sorted: {letters_sorted}")

letters.reverse()
print(f"After reverse(): {letters}")

# Looping through lists
print("\nLooping through names:")
for name in names:
    print(f"  {name}")

# List with enumerate
print("\nLooping with index:")
for i, name in enumerate(names):
    print(f"  {i}: {name}")

# List comprehension (creating lists in one line)
squared = [x**2 for x in range(1, 6)]
print(f"\nSquared numbers [1-5]: {squared}")

evens = [x for x in range(10) if x % 2 == 0]
print(f"Even numbers [0-9]: {evens}")


# ============================================================================
# 2. TUPLES
# ============================================================================

print("\n" + "=" * 60)
print("2. TUPLES - Ordered, Immutable Sequences")
print("=" * 60)

# Creating tuples
colors = ("red", "green", "blue")
coords = (10, 20)
single = (42,)  # Note the comma for single-element tuple
empty = ()

print(f"Colors: {colors}")
print(f"Coordinates: {coords}")
print(f"Single: {single}")

# Accessing elements
print(f"\nFirst color: {colors[0]}")
print(f"Last color: {colors[-1]}")

# Slicing
print(f"First two: {colors[0:2]}")

# Tuples are immutable
try:
    colors[0] = "yellow"  # This will cause an error
except TypeError as e:
    print(f"\nError: Can't modify tuple - {e}")

# But you can create a new tuple
colors = colors + ("yellow",)
print(f"New tuple: {colors}")

# Tuple unpacking
red, green, blue = colors[:3]
print(f"\nUnpacked: red={red}, green={green}, blue={blue}")

# Returning multiple values from functions
def get_user_info():
    return ("Alice", 25, "Engineer")

name, age, job = get_user_info()
print(f"\nUser: {name}, Age: {age}, Job: {job}")


# ============================================================================
# 3. DICTIONARIES
# ============================================================================

print("\n" + "=" * 60)
print("3. DICTIONARIES - Key-Value Pairs")
print("=" * 60)

# Creating dictionaries
person = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

print(f"Person: {person}")
print(f"Length: {len(person)}")

# Accessing values
print(f"\nName: {person['name']}")
print(f"Age: {person['age']}")

# Safe access (returns None if key doesn't exist)
print(f"Country: {person.get('country')}")
print(f"Country (with default): {person.get('country', 'USA')}")

# Adding/modifying
person["age"] = 26
person["email"] = "alice@example.com"
print(f"\nAfter modifications: {person}")

# Removing
del person["email"]
print(f"After delete: {person}")

removed_age = person.pop("age")
print(f"Popped age: {removed_age}")

# Dictionary methods
print(f"\nKeys: {list(person.keys())}")
print(f"Values: {list(person.values())}")
print(f"Items: {list(person.items())}")

# Looping through dictionaries
print("\nLooping through dictionary:")
for key, value in person.items():
    print(f"  {key}: {value}")

# Creating dictionaries from lists
keys = ["a", "b", "c"]
values = [1, 2, 3]
d = dict(zip(keys, values))
print(f"\nDict from zip: {d}")

# Dictionary comprehension
squares = {x: x**2 for x in range(1, 6)}
print(f"Squares dict: {squares}")


# ============================================================================
# 4. SETS
# ============================================================================

print("\n" + "=" * 60)
print("4. SETS - Unordered Collections of Unique Items")
print("=" * 60)

# Creating sets
colors = {"red", "green", "blue"}
numbers = {1, 2, 3, 2, 1}  # Duplicates removed
empty = set()  # Empty set must use set(), not {}

print(f"Colors: {colors}")
print(f"Numbers with duplicates input {1, 2, 3, 2, 1}: {numbers}")

# Checking membership
print(f"\n'red' in colors: {'red' in colors}")
print(f"'yellow' in colors: {'yellow' in colors}")

# Adding and removing
colors.add("yellow")
print(f"After add('yellow'): {colors}")

colors.remove("yellow")
print(f"After remove('yellow'): {colors}")

colors.discard("purple")  # No error if not found
print(f"After discard('purple'): {colors}")

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

print(f"\nSet1: {set1}")
print(f"Set2: {set2}")
print(f"Union (all): {set1 | set2}")
print(f"Intersection (common): {set1 & set2}")
print(f"Difference (in set1 not set2): {set1 - set2}")

# Removing duplicates from a list
numbers_with_duplicates = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique_numbers = set(numbers_with_duplicates)
print(f"\nRemoving duplicates: {numbers_with_duplicates}")
print(f"Unique: {unique_numbers}")


# ============================================================================
# 5. WORKING WITH MULTIPLE STRUCTURES
# ============================================================================

print("\n" + "=" * 60)
print("5. WORKING WITH MULTIPLE STRUCTURES")
print("=" * 60)

# List of dictionaries
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]

print("\nStudents:")
for student in students:
    print(f"  {student['name']}: {student['grade']}")

# Dictionary of lists
class_info = {
    "math": [85, 90, 88],
    "english": [92, 88, 95],
    "science": [78, 82, 80]
}

print("\nGrades by subject:")
for subject, grades in class_info.items():
    average = sum(grades) / len(grades)
    print(f"  {subject}: avg = {average:.1f}")

# Nested structures
company = {
    "name": "TechCorp",
    "employees": [
        {"id": 1, "name": "Alice", "skills": {"python", "javascript"}},
        {"id": 2, "name": "Bob", "skills": {"java", "cpp"}}
    ]
}

print("\nCompany structure:")
print(f"Company: {company['name']}")
for emp in company['employees']:
    print(f"  {emp['name']}: {emp['skills']}")


# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("KEY CONCEPTS SUMMARY")
print("=" * 60)

summary = """
LISTS: [1, 2, 3]
  - Ordered, changeable, allow duplicates
  - Access by index, can add/remove elements
  - Methods: append(), insert(), remove(), pop(), sort()

TUPLES: (1, 2, 3)
  - Ordered, unchangeable, allow duplicates
  - Good for fixed data, function return values
  - Lighter weight than lists

DICTIONARIES: {"key": "value"}
  - Unordered (3.7+: ordered), mutable
  - Access by key, great for structured data
  - Methods: keys(), values(), items(), get()

SETS: {1, 2, 3}
  - Unordered, immutable items, NO duplicates
  - Great for membership testing and unique items
  - Support: union (|), intersection (&), difference (-)

CHOOSING THE RIGHT STRUCTURE:
  - Use LIST when: order matters, need to change items
  - Use TUPLE when: data shouldn't change
  - Use DICT when: mapping keys to values
  - Use SET when: uniqueness matters, need fast lookup
"""

print(summary)

print("=" * 60)
print("Ready for Module 3 Exercises!")
print("=" * 60)
