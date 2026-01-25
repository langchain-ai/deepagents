# Local Development Setup Guide

## Prerequisites
- Ensure you have Python 3.8+ installed.
- Install Node.js (for frontend development).

## Setting Up the Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/langchain-ai/deepagents.git
   cd deepagents
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. For frontend development, navigate to the frontend directory and install dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Running the Application
- To run the application, use:
```bash
python app.py
```

## Running Tests
- To run tests, execute:
```bash
pytest
```
``