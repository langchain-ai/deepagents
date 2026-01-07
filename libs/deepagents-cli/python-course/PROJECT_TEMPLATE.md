# Python Project Template

A comprehensive template and guide for building your own Python projects.

## 📋 Project Structure

```
my_python_project/
├── README.md                 # Project documentation
├── REQUIREMENTS.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── .env.example             # Environment variables template
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── main.py             # Entry point
│   ├── config.py           # Configuration
│   ├── constants.py        # Constants
│   ├── models.py           # Data models
│   ├── utils.py            # Utility functions
│   └── handlers.py         # Business logic
│
├── tests/                  # Tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_utils.py
│   └── test_handlers.py
│
├── docs/                   # Documentation
│   ├── api.md
│   ├── architecture.md
│   └── setup.md
│
├── scripts/                # Utility scripts
│   ├── setup.py           # Setup script
│   └── migrate.py         # Migration script
│
└── data/                   # Data files (if needed)
    └── sample_data.json
```

---

## 🚀 Getting Started

### Step 1: Create Project Directory

```bash
mkdir my_python_project
cd my_python_project

# Initialize git
git init

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows
```

### Step 2: Create Main Files

```bash
# Create directories
mkdir src tests docs data scripts

# Create init files
touch src/__init__.py tests/__init__.py

# Create main files
touch src/main.py src/config.py src/constants.py
touch tests/test_main.py

# Create documentation
touch README.md REQUIREMENTS.txt .gitignore
```

### Step 3: Install Dependencies

```bash
# Create requirements file with your packages
pip install <package_names>

# Save requirements
pip freeze > requirements.txt
```

---

## 📄 File Templates

### README.md Template

```markdown
# Project Name

Brief one-liner description.

## Features
- Feature 1
- Feature 2
- Feature 3

## Installation

1. Clone the repository
```bash
git clone <repo-url>
cd my_python_project
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

```python
from src.main import MyClass

obj = MyClass()
result = obj.do_something()
```

## Testing

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Write tests
5. Submit pull request

## License

MIT License
```

### .gitignore Template

```
# Virtual environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
data/raw/
output/
```

### REQUIREMENTS.txt Template

```
# Core dependencies
requests==2.28.1
python-dotenv==0.20.0

# Data processing
pandas==1.4.3
numpy==1.23.2

# Testing
pytest==7.1.3
pytest-cov==3.0.0

# Code quality
black==22.8.0
flake8==4.0.1
mypy==0.981

# Documentation
sphinx==5.1.1
```

---

## 🏗️ Code Organization

### Main Module (src/main.py)

```python
"""
Main module - entry point for the application.
"""

import logging
from src.config import config
from src.models import DataModel
from src.handlers import DataHandler

logger = logging.getLogger(__name__)


def main():
    """Main application function."""
    try:
        # Initialize
        config.load()
        logger.info("Application started")
        
        # Process
        handler = DataHandler()
        result = handler.process()
        
        # Output
        logger.info(f"Result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
```

### Config Module (src/config.py)

```python
"""
Configuration management.
"""

import os
from dotenv import load_dotenv


class Config:
    """Application configuration."""
    
    def __init__(self):
        self.debug = False
        self.timeout = 30
        self.api_key = None
    
    def load(self):
        """Load configuration from environment."""
        load_dotenv()
        self.debug = os.getenv("DEBUG", "False") == "True"
        self.timeout = int(os.getenv("TIMEOUT", 30))
        self.api_key = os.getenv("API_KEY")
        
        if not self.api_key:
            raise ValueError("API_KEY not configured")


config = Config()
```

### Models Module (src/models.py)

```python
"""
Data models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataModel:
    """Represents data structure."""
    id: int
    name: str
    value: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value
        }
```

### Utils Module (src/utils.py)

```python
"""
Utility functions.
"""

import logging

logger = logging.getLogger(__name__)


def validate_input(data):
    """Validate input data."""
    if not data:
        raise ValueError("Data cannot be empty")
    return True


def format_output(result):
    """Format output data."""
    return {
        "status": "success",
        "data": result
    }
```

### Handlers Module (src/handlers.py)

```python
"""
Business logic handlers.
"""

import logging
from src.models import DataModel
from src.utils import validate_input, format_output

logger = logging.getLogger(__name__)


class DataHandler:
    """Handle data processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process(self, data=None):
        """Process data."""
        try:
            validate_input(data)
            result = self._transform(data)
            return format_output(result)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def _transform(self, data):
        """Transform data."""
        # Your logic here
        return data
```

---

## 🧪 Testing Setup

### Test Module (tests/test_main.py)

```python
"""
Tests for main module.
"""

import pytest
from src.main import main
from src.models import DataModel
from src.handlers import DataHandler


class TestDataHandler:
    """Test DataHandler class."""
    
    def setup_method(self):
        """Setup for each test."""
        self.handler = DataHandler()
    
    def test_process_valid_data(self):
        """Test processing valid data."""
        result = self.handler.process("test")
        assert result is not None
    
    def test_process_invalid_data(self):
        """Test processing invalid data."""
        with pytest.raises(ValueError):
            self.handler.process(None)


class TestDataModel:
    """Test DataModel class."""
    
    def test_model_creation(self):
        """Test creating a model."""
        model = DataModel(id=1, name="Test", value=42.0)
        assert model.id == 1
        assert model.name == "Test"
    
    def test_to_dict(self):
        """Test converting to dict."""
        model = DataModel(id=1, name="Test", value=42.0)
        d = model.to_dict()
        assert d["id"] == 1
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_main.py

# Run with verbose output
pytest -v tests/
```

---

## 📚 Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes, write tests
# Commit frequently
git add .
git commit -m "feat: Add new feature"

# Push to remote
git push origin feature/new-feature

# Create pull request
```

### 2. Testing

```bash
# Run tests before committing
pytest

# Run with coverage
pytest --cov=src

# Check code quality
flake8 src/

# Format code
black src/
```

### 3. Documentation

```bash
# Add docstrings
# Update README.md
# Document new features
```

### 4. Releasing

```bash
# Update version in setup.py
# Update CHANGELOG
# Create release tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

---

## 🔧 Configuration Examples

### Environment Variables (.env)

```
DEBUG=True
TIMEOUT=30
API_KEY=your-api-key-here
DATABASE_URL=postgresql://user:password@localhost/dbname
LOG_LEVEL=INFO
```

### Logging Setup (src/config.py)

```python
import logging
import logging.handlers

def setup_logging():
    """Configure logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.handlers.RotatingFileHandler(
        'app.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    
    # Console handler
    ch = logging.StreamHandler()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
```

---

## 🚀 Deployment

### Create setup.py

```python
"""
Setup configuration for package.
"""

from setuptools import setup, find_packages

setup(
    name="my-project",
    version="1.0.0",
    description="Project description",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "python-dotenv>=0.20.0",
    ],
    entry_points={
        "console_scripts": [
            "my-project=src.main:main",
        ],
    },
)
```

### Docker Setup (optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "src.main"]
```

---

## 📋 Project Checklist

Before considering your project complete:

### Code Quality
- [ ] All functions have docstrings
- [ ] Code follows PEP 8 style
- [ ] No hardcoded values (use constants)
- [ ] Error handling throughout
- [ ] Logging implemented

### Testing
- [ ] Unit tests written
- [ ] Integration tests written
- [ ] Test coverage > 80%
- [ ] All tests passing

### Documentation
- [ ] README complete
- [ ] API documented
- [ ] Installation instructions clear
- [ ] Usage examples provided
- [ ] Contributing guidelines included

### Version Control
- [ ] Initial commit
- [ ] Feature commits logical
- [ ] Commit messages clear
- [ ] .gitignore complete
- [ ] No secrets in repository

### Deployment
- [ ] requirements.txt updated
- [ ] Virtual environment working
- [ ] No global dependencies
- [ ] Error messages user-friendly

---

## Tips for Project Success

1. **Start Simple** - Don't over-engineer
2. **Write Tests First** - TDD approach helps
3. **Document as You Go** - Don't leave for later
4. **Commit Often** - Small, logical commits
5. **Review Your Code** - Before committing
6. **Use Type Hints** - Makes code clearer
7. **Handle Errors** - Don't ignore exceptions
8. **Plan Architecture** - Before coding
9. **Keep It DRY** - Don't repeat yourself
10. **Refactor Regularly** - Improve as you go

---

## Common Mistakes to Avoid

- ✗ No version control from start
- ✗ Skipping tests
- ✗ No error handling
- ✗ Storing secrets in code
- ✗ No documentation
- ✗ Monolithic design
- ✗ Ignoring logging
- ✗ No config management
- ✗ Hardcoded values
- ✗ Incomplete .gitignore

---

## Resources

### Project Management
- GitHub Projects - project management
- Trello - task tracking
- Notion - documentation

### Code Quality
- Black - code formatter
- Flake8 - linter
- MyPy - type checker
- Pytest - testing framework

### Documentation
- Sphinx - documentation generation
- MkDocs - markdown documentation
- ReadTheDocs - hosting

### Deployment
- Heroku - PaaS platform
- AWS - cloud services
- DigitalOcean - VPS hosting

---

## Next Steps

1. Choose your project idea
2. Set up directory structure
3. Initialize git repository
4. Create virtual environment
5. Start building!

Remember: Every expert was once a beginner. Start small, build progressively, and learn from the process! 🚀

---

Last Updated: 2024
