"""
FINAL PROJECT 3: ADVANCED CALCULATOR
====================================

Build a sophisticated calculator with expression parsing and history.

Features:
- Basic arithmetic operations
- Parentheses and order of operations
- Variable support (x = 5, then use x in expressions)
- Calculation history with results
- Error handling for invalid expressions
- Statistics (sum, average of calculations)
- Expression validation

This project demonstrates:
- Expression parsing and evaluation
- String manipulation and validation
- Data structures (lists, dicts)
- Exception handling
- OOP design
- Encapsulation
"""

import re
from enum import Enum


class Operation(Enum):
    """Supported operations."""
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    POWER = "**"
    MODULO = "%"


class Calculator:
    """Advanced calculator with expression parsing and history."""
    
    def __init__(self):
        """Initialize calculator."""
        self.history = []
        self.variables = {}
        self.decimal_places = 2
    
    def set_variable(self, name, value):
        """
        Set a variable for use in expressions.
        
        Args:
            name (str): Variable name
            value (float): Variable value
        """
        if not re.match(r"^[a-zA-Z_]\w*$", name):
            raise ValueError(f"Invalid variable name: {name}")
        
        try:
            self.variables[name] = float(value)
        except ValueError:
            raise ValueError(f"Invalid value: {value}")
    
    def get_variable(self, name):
        """Get variable value."""
        if name not in self.variables:
            raise ValueError(f"Undefined variable: {name}")
        return self.variables[name]
    
    def validate_expression(self, expression):
        """
        Validate expression syntax.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        # Check for unmatched parentheses
        if expression.count("(") != expression.count(")"):
            return False, "Unmatched parentheses"
        
        # Check for consecutive operators
        if re.search(r"(\+|-|\*|/)\s*(\+|-|\*|/)", expression):
            return False, "Consecutive operators"
        
        # Check for invalid characters
        allowed = set("0123456789+\-*/.() ")
        allowed.update(set(self.variables.keys()))
        allowed.update(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"))
        
        for char in expression:
            if char not in allowed:
                return False, f"Invalid character: {char}"
        
        return True, ""
    
    def replace_variables(self, expression):
        """Replace variable names with their values."""
        for var_name, value in self.variables.items():
            # Use word boundary to avoid partial replacements
            pattern = rf"\b{var_name}\b"
            expression = re.sub(pattern, str(value), expression)
        return expression
    
    def calculate(self, expression):
        """
        Calculate expression result.
        
        Args:
            expression (str): Mathematical expression
            
        Returns:
            float: Result of calculation
        """
        # Clean up expression
        expression = expression.strip()
        
        # Validate
        is_valid, error = self.validate_expression(expression)
        if not is_valid:
            raise ValueError(error)
        
        # Replace variables
        expression = self.replace_variables(expression)
        
        try:
            # Use eval for safety (in production, use proper parser)
            # For this educational project, eval is acceptable with validation
            result = eval(expression)
            
            if isinstance(result, (int, float)):
                result = float(result)
            else:
                raise ValueError("Invalid expression result")
            
            # Store in history
            self.history.append({
                "expression": expression,
                "result": result
            })
            
            return result
        
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(f"Calculation error: {str(e)}")
    
    def format_result(self, value):
        """Format result for display."""
        if isinstance(value, float):
            if value == int(value):
                return str(int(value))
            else:
                return f"{value:.{self.decimal_places}f}"
        return str(value)
    
    def get_history(self, limit=None):
        """
        Get calculation history.
        
        Args:
            limit (int): Maximum number of entries to return
            
        Returns:
            list: History entries
        """
        if limit:
            return self.history[-limit:]
        return self.history
    
    def clear_history(self):
        """Clear calculation history."""
        self.history = []
    
    def get_statistics(self):
        """
        Get statistics about calculations.
        
        Returns:
            dict: Statistics
        """
        if not self.history:
            return {"count": 0}
        
        results = [h["result"] for h in self.history]
        return {
            "count": len(results),
            "sum": sum(results),
            "average": sum(results) / len(results),
            "min": min(results),
            "max": max(results)
        }
    
    def display_history(self, limit=10):
        """Display calculation history."""
        history = self.get_history(limit)
        
        if not history:
            print("No calculation history")
            return
        
        print("\n" + "=" * 60)
        print("CALCULATION HISTORY")
        print("=" * 60)
        
        for i, entry in enumerate(history, 1):
            result_str = self.format_result(entry["result"])
            print(f"{i:2}. {entry['expression']:30} = {result_str}")
        
        print("=" * 60 + "\n")


class CalculatorApp:
    """Calculator application with user interface."""
    
    def __init__(self):
        """Initialize app."""
        self.calculator = Calculator()
    
    def display_menu(self):
        """Display main menu."""
        print("\n" + "=" * 60)
        print("ADVANCED CALCULATOR")
        print("=" * 60)
        print("\nCommands:")
        print("  calc <expr>      - Calculate expression")
        print("  var <name>=<val> - Set variable")
        print("  vars             - Show all variables")
        print("  history          - Show calculation history")
        print("  stats            - Show statistics")
        print("  clear            - Clear history")
        print("  precision <n>    - Set decimal precision")
        print("  help             - Show help")
        print("  quit             - Exit")
        print("=" * 60 + "\n")
    
    def process_command(self, command):
        """Process user command."""
        parts = command.split(maxsplit=1)
        if not parts:
            return
        
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd == "calc":
                if not arg:
                    print("Usage: calc <expression>")
                    return
                result = self.calculator.calculate(arg)
                print(f"Result: {self.calculator.format_result(result)}")
            
            elif cmd == "var":
                if "=" not in arg:
                    print("Usage: var <name>=<value>")
                    return
                name, value = arg.split("=")
                self.calculator.set_variable(name.strip(), value.strip())
                print(f"Set {name.strip()} = {value.strip()}")
            
            elif cmd == "vars":
                if not self.calculator.variables:
                    print("No variables defined")
                else:
                    print("\nVariables:")
                    for name, value in self.calculator.variables.items():
                        print(f"  {name} = {value}")
            
            elif cmd == "history":
                self.calculator.display_history()
            
            elif cmd == "stats":
                stats = self.calculator.get_statistics()
                if stats["count"] == 0:
                    print("No calculations yet")
                else:
                    print("\nStatistics:")
                    print(f"  Count: {stats['count']}")
                    print(f"  Sum: {self.calculator.format_result(stats['sum'])}")
                    print(f"  Average: {self.calculator.format_result(stats['average'])}")
                    print(f"  Min: {self.calculator.format_result(stats['min'])}")
                    print(f"  Max: {self.calculator.format_result(stats['max'])}")
            
            elif cmd == "clear":
                self.calculator.clear_history()
                print("History cleared")
            
            elif cmd == "precision":
                try:
                    precision = int(arg)
                    self.calculator.decimal_places = precision
                    print(f"Precision set to {precision} decimal places")
                except ValueError:
                    print("Usage: precision <number>")
            
            elif cmd == "help":
                self.display_menu()
            
            else:
                print("Unknown command. Try 'help'")
        
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    def run_interactive(self):
        """Run interactive mode."""
        self.display_menu()
        
        while True:
            try:
                command = input("> ").strip()
                if command.lower() == "quit":
                    print("Goodbye!")
                    break
                if command:
                    self.process_command(command)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    def run_demo(self):
        """Run demonstration."""
        print("\n" + "=" * 60)
        print("CALCULATOR DEMO")
        print("=" * 60 + "\n")
        
        # Basic calculations
        print("Basic Calculations:")
        calculations = [
            "2 + 3",
            "10 - 4",
            "3 * 7",
            "20 / 4",
            "2 ** 3",
            "(5 + 3) * 2"
        ]
        
        for calc in calculations:
            result = self.calculator.calculate(calc)
            print(f"  {calc} = {self.calculator.format_result(result)}")
        
        # Variables
        print("\nUsing Variables:")
        self.calculator.set_variable("x", 5)
        self.calculator.set_variable("y", 3)
        
        result = self.calculator.calculate("x ** 2 + y ** 2")
        print(f"  x=5, y=3")
        print(f"  x² + y² = {self.calculator.format_result(result)}")
        
        # History
        self.calculator.display_history()
        
        # Statistics
        stats = self.calculator.get_statistics()
        print("Statistics:")
        print(f"  Total calculations: {stats['count']}")
        print(f"  Average: {self.calculator.format_result(stats['average'])}")
        print(f"  Min result: {self.calculator.format_result(stats['min'])}")
        print(f"  Max result: {self.calculator.format_result(stats['max'])}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    app = CalculatorApp()
    app.run_demo()
    
    print("\nTo use interactive mode, uncomment line below:")
    print("# app.run_interactive()")
    
    # Uncomment to run interactive:
    # app = CalculatorApp()
    # app.run_interactive()
