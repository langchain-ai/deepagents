"""
MODULE 5: OBJECT-ORIENTED PROGRAMMING BASICS
==============================================

Introduction to OOP - organizing code with classes and objects.

Topics:
1. Classes and Objects
2. Attributes and Methods
3. __init__ Constructor
4. The self Parameter
5. Inheritance
6. Encapsulation (Properties)
"""

# ============================================================================
# 1. CLASSES AND OBJECTS
# ============================================================================

print("=" * 60)
print("1. CLASSES AND OBJECTS")
print("=" * 60)

# Defining a simple class
class Dog:
    """A simple Dog class."""
    
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    # Constructor - called when creating an object
    def __init__(self, name, age):
        """Initialize a dog with name and age."""
        self.name = name
        self.age = age
    
    # Methods - functions inside a class
    def bark(self):
        """Make the dog bark."""
        return f"{self.name} says: Woof!"
    
    def get_info(self):
        """Return dog information."""
        return f"{self.name} is {self.age} years old"

# Creating objects (instances)
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(f"Dog 1: {dog1.name}")
print(f"Dog 2: {dog2.name}")

# Calling methods
print(f"\n{dog1.bark()}")
print(f"{dog2.bark()}")

# Accessing attributes
print(f"\nSpecies: {Dog.species}")
print(f"{dog1.get_info()}")
print(f"{dog2.get_info()}")


# ============================================================================
# 2. ATTRIBUTES AND METHODS
# ============================================================================

print("\n" + "=" * 60)
print("2. ATTRIBUTES AND METHODS")
print("=" * 60)

class Car:
    """A Car class with attributes and methods."""
    
    def __init__(self, brand, model, year):
        # Instance attributes
        self.brand = brand
        self.model = model
        self.year = year
        self.speed = 0
    
    def accelerate(self, amount):
        """Increase speed."""
        self.speed += amount
        return f"Accelerating... Speed: {self.speed} km/h"
    
    def brake(self, amount):
        """Decrease speed."""
        self.speed = max(0, self.speed - amount)
        return f"Braking... Speed: {self.speed} km/h"
    
    def honk(self):
        """Honk the horn."""
        return f"{self.brand} {self.model}: Beep beep!"
    
    def info(self):
        """Return car information."""
        return f"{self.year} {self.brand} {self.model}"

# Create car instance
car = Car("Toyota", "Camry", 2023)
print(f"\nCar: {car.info()}")

# Call methods that modify state
print(f"{car.accelerate(50)}")
print(f"{car.accelerate(30)}")
print(f"{car.brake(20)}")
print(f"{car.honk()}")


# ============================================================================
# 3. CONSTRUCTOR (__init__)
# ============================================================================

print("\n" + "=" * 60)
print("3. THE __init__ CONSTRUCTOR")
print("=" * 60)

class Person:
    """Person class with constructor."""
    
    def __init__(self, name, age, email):
        """Initialize a person."""
        self.name = name
        self.age = age
        self.email = email
    
    def introduce(self):
        return f"Hi, I'm {self.name}, {self.age} years old"

# Constructor is called automatically
person1 = Person("Alice", 25, "alice@example.com")
person2 = Person("Bob", 30, "bob@example.com")

print(f"Person 1: {person1.introduce()}")
print(f"Email: {person1.email}")

print(f"Person 2: {person2.introduce()}")
print(f"Email: {person2.email}")

# Each instance has its own attributes
print(f"\nAlice's age: {person1.age}")
print(f"Bob's age: {person2.age}")


# ============================================================================
# 4. THE SELF PARAMETER
# ============================================================================

print("\n" + "=" * 60)
print("4. THE SELF PARAMETER")
print("=" * 60)

class BankAccount:
    """Bank account with balance."""
    
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        """Add money to account."""
        self.balance += amount
        print(f"Deposited ${amount}")
        self._show_balance()
    
    def withdraw(self, amount):
        """Remove money from account."""
        if amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew ${amount}")
        else:
            print(f"Insufficient funds!")
        self._show_balance()
    
    def _show_balance(self):
        """Private method to show balance."""
        print(f"Current balance: ${self.balance}")

# Using self
account = BankAccount("Alice", 1000)
print(f"Account owner: {account.owner}")

account.deposit(200)
account.withdraw(300)
account.withdraw(2000)


# ============================================================================
# 5. INHERITANCE
# ============================================================================

print("\n" + "=" * 60)
print("5. INHERITANCE")
print("=" * 60)

# Parent class
class Animal:
    """Base animal class."""
    
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return f"{self.name} makes a sound"
    
    def move(self):
        return f"{self.name} moves"

# Child classes inherit from parent
class Cat(Animal):
    """Cat class inheriting from Animal."""
    
    def speak(self):
        """Override parent method."""
        return f"{self.name} says: Meow!"

class Bird(Animal):
    """Bird class inheriting from Animal."""
    
    def speak(self):
        return f"{self.name} says: Tweet!"
    
    def move(self):
        """Override parent method."""
        return f"{self.name} flies"

# Using inheritance
cat = Cat("Whiskers")
bird = Bird("Tweety")
dog = Dog("Buddy", 3)  # Using Dog class from earlier

print(f"\n{cat.speak()}")
print(f"{cat.move()}")

print(f"\n{bird.speak()}")
print(f"{bird.move()}")

# Inheritance allows reusing code
print(f"\nDog inherits from Animal pattern:")
print(f"Dog name: {dog.name}")


# ============================================================================
# 6. ENCAPSULATION (PROPERTIES)
# ============================================================================

print("\n" + "=" * 60)
print("6. ENCAPSULATION (PROPERTIES)")
print("=" * 60)

class Student:
    """Student class with properties."""
    
    def __init__(self, name, grade):
        self.name = name
        self._grade = grade  # Convention: _ means "private"
    
    # Property for reading
    @property
    def grade(self):
        """Get the grade."""
        return self._grade
    
    # Property for setting with validation
    @grade.setter
    def grade(self, value):
        """Set the grade with validation."""
        if 0 <= value <= 100:
            self._grade = value
        else:
            print(f"Error: Grade must be 0-100, got {value}")
    
    def get_status(self):
        if self._grade >= 70:
            return "Passing"
        else:
            return "Failing"

# Using properties
student = Student("Charlie", 85)
print(f"Student: {student.name}")
print(f"Grade: {student.grade}")
print(f"Status: {student.get_status()}")

# Setting with validation
student.grade = 95
print(f"New grade: {student.grade}")

# Invalid grade is rejected
student.grade = 150
print(f"Grade after invalid attempt: {student.grade}")


# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

print("\n" + "=" * 60)
print("PRACTICAL EXAMPLES")
print("=" * 60)

# Example 1: Rectangle class
class Rectangle:
    """Rectangle with area and perimeter."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
    
    def describe(self):
        return f"Rectangle {self.width}x{self.height}: Area={self.area()}, Perimeter={self.perimeter()}"

rect = Rectangle(5, 3)
print(f"\n{rect.describe()}")

# Example 2: Counter class
class Counter:
    """Simple counter with increment/decrement."""
    
    def __init__(self, start=0):
        self.count = start
    
    def increment(self):
        self.count += 1
        return self.count
    
    def decrement(self):
        self.count -= 1
        return self.count
    
    def reset(self):
        self.count = 0
    
    def __str__(self):
        return f"Counter: {self.count}"

counter = Counter(10)
print(f"\n{counter}")
print(f"After increment: {counter.increment()}")
print(f"After decrement: {counter.decrement()}")


# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("KEY CONCEPTS SUMMARY")
print("=" * 60)

summary = """
CLASSES:
  class ClassName:
      def __init__(self, ...):
          self.attribute = value
      def method(self):
          return something

OBJECTS:
  obj = ClassName(arguments)
  obj.attribute
  obj.method()

SELF:
  - Refers to the current instance
  - Always first parameter in methods
  - Access instance attributes with self.name

CONSTRUCTOR:
  __init__(self, ...): Called when creating object
  Initialize attributes here

INHERITANCE:
  class Child(Parent):
      def method(self):
          return something
  
  Child inherits all Parent methods
  Can override methods

ENCAPSULATION:
  _attribute: Convention for "private"
  @property: Create getters
  @property.setter: Create setters with validation

BENEFITS OF OOP:
  - Organize related code together
  - Reuse code through inheritance
  - Protect data with encapsulation
  - Model real-world entities
"""

print(summary)

print("=" * 60)
print("Ready for Module 5 Exercises!")
print("=" * 60)
