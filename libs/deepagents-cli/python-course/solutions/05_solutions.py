"""
MODULE 5 SOLUTIONS
==================

Solutions for Module 5 exercises.
"""

# Exercise 1: Create a Simple Class
# ==================================

print("EXERCISE 1: Create a Simple Class")
print("-" * 40)

class Book:
    """A book with title, author, and year."""
    
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year
    
    def info(self):
        return f"'{self.title}' by {self.author} ({self.year})"

book = Book("1984", "George Orwell", 1949)
print(book.info())
print()


# Exercise 2: Create a BankAccount Class
# =======================================

print("EXERCISE 2: Create a BankAccount Class")
print("-" * 40)

class BankAccount:
    """Bank account with balance operations."""
    
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"Deposited ${amount}")
        else:
            print("Deposit amount must be positive")
    
    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew ${amount}")
        else:
            print("Insufficient balance")
    
    def show_balance(self):
        print(f"Account: {self.owner}")
        print(f"Balance: ${self.balance}")

account = BankAccount("Alice", 1000)
account.deposit(200)
account.withdraw(300)
account.show_balance()
print()


# Exercise 3: Multiple Objects
# =============================

print("EXERCISE 3: Multiple Objects")
print("-" * 40)

books = [
    Book("To Kill a Mockingbird", "Harper Lee", 1960),
    Book("The Great Gatsby", "F. Scott Fitzgerald", 1925),
    Book("Pride and Prejudice", "Jane Austen", 1813)
]

for book in books:
    print(book.info())

print()


# Exercise 4: Modify Object Attributes
# =====================================

print("EXERCISE 4: Modify Object Attributes")
print("-" * 40)

class Student:
    """Student with grade."""
    
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def show(self):
        print(f"{self.name}: {self.grade}%")

student = Student("Bob", 85)
print("Initial grade:")
student.show()

student.grade = 92
print("Updated grade:")
student.show()

print()


# Exercise 5: Method with Return Value
# =====================================

print("EXERCISE 5: Method with Return Value")
print("-" * 40)

import math

class Circle:
    """Circle with radius."""
    
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def circumference(self):
        return 2 * math.pi * self.radius

circle = Circle(5)
print(f"Radius: {circle.radius}")
print(f"Area: {circle.area():.2f}")
print(f"Circumference: {circle.circumference():.2f}")

print()


# Exercise 6: Constructor with Defaults
# ======================================

print("EXERCISE 6: Constructor with Defaults")
print("-" * 40)

class Point:
    """2D point with x, y coordinates."""
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def show(self):
        print(f"({self.x}, {self.y})")

p1 = Point()
p2 = Point(3, 4)
p3 = Point(5)

print("Point 1 (default):", end=" ")
p1.show()
print("Point 2 (3, 4):", end=" ")
p2.show()
print("Point 3 (5, 0):", end=" ")
p3.show()

print()


# Exercise 7: String Representation
# ==================================

print("EXERCISE 7: String Representation")
print("-" * 40)

class Person:
    """Person with name and age."""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.name} ({self.age} years old)"

person = Person("Charlie", 28)
print(person)  # Uses __str__ method

print()


# Exercise 8: Simple Inheritance
# ===============================

print("EXERCISE 8: Simple Inheritance")
print("-" * 40)

class Vehicle:
    """Base vehicle class."""
    
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def describe(self):
        return f"{self.brand} {self.model}"

class Car(Vehicle):
    """Car inherits from Vehicle."""
    
    def __init__(self, brand, model, doors):
        super().__init__(brand, model)
        self.doors = doors
    
    def describe(self):
        return f"{super().describe()} ({self.doors} doors)"

car = Car("Toyota", "Camry", 4)
print(f"Car: {car.describe()}")

print()


# Exercise 9: Override Parent Method
# ===================================

print("EXERCISE 9: Override Parent Method")
print("-" * 40)

class Animal:
    """Base animal."""
    
    def speak(self):
        return "Some sound"

class Dog(Animal):
    """Dog overrides speak method."""
    
    def speak(self):
        return "Woof!"

class Cat(Animal):
    """Cat overrides speak method."""
    
    def speak(self):
        return "Meow!"

animals = [Dog(), Cat(), Animal()]
for animal in animals:
    print(f"{animal.__class__.__name__}: {animal.speak()}")

print()


# Exercise 10: Store Objects in List
# ===================================

print("EXERCISE 10: Store Objects in List")
print("-" * 40)

books_list = [
    Book("1984", "George Orwell", 1949),
    Book("Brave New World", "Aldous Huxley", 1932),
    Book("Fahrenheit 451", "Ray Bradbury", 1953)
]

print("Books in library:")
for i, book in enumerate(books_list, 1):
    print(f"{i}. {book.info()}")

print()


# CHALLENGE EXERCISES
# ===================

print("=" * 40)
print("CHALLENGE SOLUTIONS")
print("=" * 40 + "\n")

# Challenge 1: Todo List
# =====================

print("CHALLENGE 1: Todo List")
print("-" * 40)

class Task:
    """A task with description, status, and priority."""
    
    def __init__(self, description, priority=1, done=False):
        self.description = description
        self.priority = priority
        self.done = done
    
    def mark_done(self):
        self.done = True
    
    def __str__(self):
        status = "[✓]" if self.done else "[ ]"
        return f"{status} {self.description} (Priority: {self.priority})"

tasks = [
    Task("Learn Python", 1, False),
    Task("Build project", 1, False),
    Task("Read documentation", 2, True),
    Task("Practice exercises", 1, False)
]

print("Todo List:")
for task in tasks:
    print(f"  {task}")

print()


# Challenge 2: Student Grade Tracker
# ===================================

print("CHALLENGE 2: Student Grade Tracker")
print("-" * 40)

class StudentTracker:
    """Track student with multiple grades."""
    
    def __init__(self, name):
        self.name = name
        self.grades = []
    
    def add_grade(self, grade):
        self.grades.append(grade)
    
    def average(self):
        if not self.grades:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def status(self):
        avg = self.average()
        if avg >= 70:
            return "Passing"
        else:
            return "Failing"
    
    def __str__(self):
        return f"{self.name}: Avg={self.average():.1f}, Status={self.status()}"

student = StudentTracker("David")
student.add_grade(85)
student.add_grade(90)
student.add_grade(78)
print(student)

print()


# Challenge 3: Library System
# ============================

print("CHALLENGE 3: Library System")
print("-" * 40)

class Library:
    """Library that manages books."""
    
    def __init__(self, name):
        self.name = name
        self.books = []
    
    def add_book(self, book):
        self.books.append(book)
        print(f"Added: {book.info()}")
    
    def remove_book(self, title):
        self.books = [b for b in self.books if b.title != title]
        print(f"Removed: {title}")
    
    def search(self, author):
        results = [b for b in self.books if b.author == author]
        return results
    
    def list_all(self):
        print(f"\n{self.name} Catalog:")
        for book in self.books:
            print(f"  {book.info()}")

library = Library("City Library")
library.add_book(Book("1984", "George Orwell", 1949))
library.add_book(Book("Animal Farm", "George Orwell", 1945))
library.add_book(Book("To Kill a Mockingbird", "Harper Lee", 1960))

library.list_all()

print("\nBooks by George Orwell:")
for book in library.search("George Orwell"):
    print(f"  {book.info()}")

print()
print("=" * 40)
print("Fantastic! You've completed Module 5!")
print("=" * 40)
