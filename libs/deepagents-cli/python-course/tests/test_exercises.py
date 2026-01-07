"""
Test Suite for Python Course Exercises

Run with: python -m pytest tests/test_exercises.py -v
"""

import pytest
import sys
import os
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModule1Basics:
    """Tests for Module 1: Basic Python concepts."""
    
    def test_variable_assignment(self):
        """Test variable assignment and types."""
        name = "Alice"
        age = 25
        height = 1.75
        is_student = True
        
        assert name == "Alice"
        assert age == 25
        assert height == 1.75
        assert is_student is True
    
    def test_arithmetic_operations(self):
        """Test basic arithmetic."""
        assert 10 + 3 == 13
        assert 10 - 3 == 7
        assert 10 * 3 == 30
        assert 10 / 3 == pytest.approx(3.333, rel=0.01)
        assert 10 // 3 == 3
        assert 10 % 3 == 1
        assert 10 ** 2 == 100
    
    def test_string_operations(self):
        """Test string manipulation."""
        text = "Python"
        assert len(text) == 6
        assert text[0] == "P"
        assert text[-1] == "n"
        assert text[0:3] == "Pyt"
        assert text.upper() == "PYTHON"
        assert "Python" + " is fun" == "Python is fun"
    
    def test_comparison_operations(self):
        """Test comparison operators."""
        assert 10 > 5
        assert 5 < 10
        assert 10 >= 10
        assert 5 <= 10
        assert 10 == 10
        assert 10 != 5
    
    def test_boolean_logic(self):
        """Test logical operations."""
        assert (True and True) is True
        assert (True and False) is False
        assert (True or False) is True
        assert (False or False) is False
        assert (not True) is False
        assert (not False) is True


class TestModule2ControlFlow:
    """Tests for Module 2: Control flow and functions."""
    
    def test_if_statement(self):
        """Test if statement."""
        age = 18
        status = "adult" if age >= 18 else "minor"
        assert status == "adult"
    
    def test_if_elif_else(self):
        """Test if/elif/else."""
        score = 85
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        else:
            grade = "F"
        
        assert grade == "B"
    
    def test_for_loop(self):
        """Test for loop."""
        total = 0
        for i in range(1, 6):
            total += i
        assert total == 15
    
    def test_while_loop(self):
        """Test while loop."""
        count = 0
        while count < 5:
            count += 1
        assert count == 5
    
    def test_function_definition(self):
        """Test function creation and calling."""
        def add(a, b):
            return a + b
        
        assert add(3, 5) == 8
    
    def test_function_with_default(self):
        """Test function with default parameters."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        assert greet("Alice") == "Hello, Alice!"
        assert greet("Bob", "Hi") == "Hi, Bob!"
    
    def test_function_return_multiple(self):
        """Test function returning multiple values."""
        def get_coords():
            return 10, 20
        
        x, y = get_coords()
        assert x == 10
        assert y == 20


class TestModule3DataStructures:
    """Tests for Module 3: Data structures."""
    
    def test_list_operations(self):
        """Test list creation and manipulation."""
        fruits = ["apple", "banana", "orange"]
        assert len(fruits) == 3
        assert fruits[0] == "apple"
        
        fruits.append("grape")
        assert len(fruits) == 4
        
        fruits.remove("banana")
        assert "banana" not in fruits
    
    def test_list_slicing(self):
        """Test list slicing."""
        numbers = [1, 2, 3, 4, 5]
        assert numbers[0:3] == [1, 2, 3]
        assert numbers[1:] == [2, 3, 4, 5]
        assert numbers[-2:] == [4, 5]
    
    def test_tuple_operations(self):
        """Test tuple creation and unpacking."""
        coords = (10, 20, 30)
        assert coords[0] == 10
        x, y, z = coords
        assert x == 10 and y == 20 and z == 30
    
    def test_dictionary_operations(self):
        """Test dictionary creation and access."""
        person = {"name": "Alice", "age": 25, "city": "NYC"}
        assert person["name"] == "Alice"
        assert person.get("age") == 25
        assert person.get("country", "USA") == "USA"
        
        person["email"] = "alice@example.com"
        assert "email" in person
    
    def test_set_operations(self):
        """Test set creation and operations."""
        set1 = {1, 2, 3, 4}
        set2 = {3, 4, 5, 6}
        
        assert set1 | set2 == {1, 2, 3, 4, 5, 6}
        assert set1 & set2 == {3, 4}
        assert set1 - set2 == {1, 2}
    
    def test_list_comprehension(self):
        """Test list comprehension."""
        squares = [x**2 for x in range(5)]
        assert squares == [0, 1, 4, 9, 16]
        
        evens = [x for x in range(10) if x % 2 == 0]
        assert evens == [0, 2, 4, 6, 8]
    
    def test_dictionary_comprehension(self):
        """Test dictionary comprehension."""
        squares_dict = {x: x**2 for x in range(4)}
        assert squares_dict == {0: 0, 1: 1, 2: 4, 3: 9}


class TestModule4FileIOExceptions:
    """Tests for Module 4: File I/O and exceptions."""
    
    def test_exception_handling(self):
        """Test try/except."""
        result = None
        try:
            result = int("not a number")
        except ValueError:
            result = 0
        
        assert result == 0
    
    def test_exception_with_else(self):
        """Test try/except/else."""
        result = None
        try:
            result = int("42")
        except ValueError:
            result = 0
        else:
            result = result * 2
        
        assert result == 84
    
    def test_custom_exception(self):
        """Test custom exception."""
        class CustomError(Exception):
            pass
        
        with pytest.raises(CustomError):
            raise CustomError("Test error")
    
    def test_multiple_exceptions(self):
        """Test catching multiple exceptions."""
        result = None
        try:
            x = [1, 2, 3]
            result = x[10]
        except IndexError:
            result = "index error"
        except ValueError:
            result = "value error"
        
        assert result == "index error"


class TestModule5OOP:
    """Tests for Module 5: Object-Oriented Programming."""
    
    def test_class_creation(self):
        """Test basic class."""
        class Dog:
            def __init__(self, name):
                self.name = name
        
        dog = Dog("Buddy")
        assert dog.name == "Buddy"
    
    def test_class_methods(self):
        """Test class with methods."""
        class Calculator:
            def __init__(self, value=0):
                self.value = value
            
            def add(self, x):
                self.value += x
                return self.value
        
        calc = Calculator(10)
        assert calc.add(5) == 15
    
    def test_inheritance(self):
        """Test inheritance."""
        class Animal:
            def __init__(self, name):
                self.name = name
            
            def speak(self):
                return "Sound"
        
        class Dog(Animal):
            def speak(self):
                return "Woof"
        
        dog = Dog("Rex")
        assert dog.name == "Rex"
        assert dog.speak() == "Woof"
    
    def test_multiple_instances(self):
        """Test multiple instances are independent."""
        class Counter:
            def __init__(self):
                self.count = 0
            
            def increment(self):
                self.count += 1
        
        c1 = Counter()
        c2 = Counter()
        
        c1.increment()
        c1.increment()
        c2.increment()
        
        assert c1.count == 2
        assert c2.count == 1


# Integration Tests
class TestIntegration:
    """Integration tests combining multiple concepts."""
    
    def test_data_processing_pipeline(self):
        """Test processing data through multiple operations."""
        # Read data
        numbers = [1, 2, 3, 4, 5]
        
        # Transform with list comprehension
        squared = [x**2 for x in numbers]
        
        # Filter
        large = [x for x in squared if x > 10]
        
        # Aggregate
        total = sum(large)
        
        assert squared == [1, 4, 9, 16, 25]
        assert large == [16, 25]
        assert total == 41
    
    def test_class_with_data_structures(self):
        """Test class containing data structures."""
        class Classroom:
            def __init__(self, name):
                self.name = name
                self.students = []
            
            def add_student(self, name, grade):
                self.students.append({"name": name, "grade": grade})
            
            def average_grade(self):
                if not self.students:
                    return 0
                total = sum(s["grade"] for s in self.students)
                return total / len(self.students)
        
        room = Classroom("101")
        room.add_student("Alice", 85)
        room.add_student("Bob", 90)
        
        assert len(room.students) == 2
        assert room.average_grade() == 87.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
