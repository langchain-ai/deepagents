"""
MODULE 3 SOLUTIONS
==================

Solutions for Module 3 exercises.
"""

# Exercise 1: List Operations
# ============================

print("EXERCISE 1: List Operations")
print("-" * 40)

fruits = ["apple", "banana", "orange", "grape", "kiwi"]
print(f"Original: {fruits}")

fruits.append("mango")
print(f"After append: {fruits}")

fruits.remove("apple")
print(f"After remove: {fruits}")

print()


# Exercise 2: List Comprehension
# ===============================

print("EXERCISE 2: List Comprehension")
print("-" * 40)

squares = [x**2 for x in range(1, 11)]
print(f"Squares 1-10: {squares}")
print()


# Exercise 3: Tuple Unpacking
# ============================

print("EXERCISE 3: Tuple Unpacking")
print("-" * 40)

person = ("Alice", 25, "New York")
name, age, city = person
print(f"Name: {name}")
print(f"Age: {age}")
print(f"City: {city}")
print()


# Exercise 4: Dictionary - Student Grades
# ========================================

print("EXERCISE 4: Dictionary - Student Grades")
print("-" * 40)

grades = {
    "Alice": 85,
    "Bob": 92,
    "Charlie": 78
}
print(f"Initial: {grades}")

grades["David"] = 88
print(f"After adding David: {grades}")

grades["Alice"] = 90
print(f"After updating Alice: {grades}")

for student, grade in grades.items():
    print(f"  {student}: {grade}")

print()


# Exercise 5: Dictionary - Methods
# =================================

print("EXERCISE 5: Dictionary Methods")
print("-" * 40)

cities = {
    "New York": 8336817,
    "Los Angeles": 3979576,
    "Chicago": 2693976
}

print(f"Keys: {list(cities.keys())}")
print(f"Values: {list(cities.values())}")
print(f"Items: {list(cities.items())}")
print()


# Exercise 6: Set Operations
# ===========================

print("EXERCISE 6: Set Operations")
print("-" * 40)

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

print(f"Set1: {set1}")
print(f"Set2: {set2}")
print(f"Union: {set1 | set2}")
print(f"Intersection: {set1 & set2}")
print(f"Difference: {set1 - set2}")
print()


# Exercise 7: Removing Duplicates
# ================================

print("EXERCISE 7: Removing Duplicates")
print("-" * 40)

numbers_with_dupes = [1, 2, 2, 3, 3, 3, 4, 5, 5]
print(f"With duplicates: {numbers_with_dupes}")

unique_numbers = set(numbers_with_dupes)
print(f"Unique (as set): {unique_numbers}")
print()


# Exercise 8: List of Dictionaries
# =================================

print("EXERCISE 8: List of Dictionaries")
print("-" * 40)

products = [
    {"name": "Laptop", "price": 999, "in_stock": True},
    {"name": "Mouse", "price": 25, "in_stock": True},
    {"name": "Keyboard", "price": 75, "in_stock": False}
]

for product in products:
    print(f"{product['name']}: ${product['price']}, In Stock: {product['in_stock']}")

print()


# Exercise 9: Dictionary Iteration
# =================================

print("EXERCISE 9: Dictionary Iteration")
print("-" * 40)

capitals = {
    "France": "Paris",
    "Italy": "Rome",
    "Japan": "Tokyo"
}

for country, capital in capitals.items():
    print(f"The capital of {country} is {capital}")

print()


# Exercise 10: Combine Structures
# ===============================

print("EXERCISE 10: Combine Structures")
print("-" * 40)

student = {
    "name": "Alice",
    "courses": ["Math", "Physics", "Chemistry"],
    "scores": {
        "Math": 85,
        "Physics": 92,
        "Chemistry": 88
    }
}

print(f"Student: {student['name']}")
print(f"Courses: {', '.join(student['courses'])}")
print("Scores:")
for course, score in student['scores'].items():
    print(f"  {course}: {score}")

print()


# CHALLENGE EXERCISES
# ===================

print("=" * 40)
print("CHALLENGE SOLUTIONS")
print("=" * 40 + "\n")

# Challenge 1: Word Frequency Counter
# ====================================

print("CHALLENGE 1: Word Frequency Counter")
print("-" * 40)

def count_words(sentence):
    words = sentence.lower().split()
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count

result = count_words("hello world hello python hello")
print(f"Word counts: {result}")
print()


# Challenge 2: Find Common Elements
# ==================================

print("CHALLENGE 2: Find Common Elements")
print("-" * 40)

list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

common = set(list1) & set(list2)
print(f"List1: {list1}")
print(f"List2: {list2}")
print(f"Common elements: {common}")
print()


# Challenge 3: Sort Dictionary by Value
# ======================================

print("CHALLENGE 3: Sort Dictionary by Value")
print("-" * 40)

ages = {
    "Alice": 25,
    "Bob": 30,
    "Charlie": 22,
    "David": 28
}

print(f"Original: {ages}")

sorted_ages = dict(sorted(ages.items(), key=lambda x: x[1], reverse=True))
print(f"Sorted by age (descending): {sorted_ages}")

for name, age in sorted_ages.items():
    print(f"  {name}: {age}")

print()
print("=" * 40)
print("Congratulations! You completed all modules!")
print("=" * 40)
