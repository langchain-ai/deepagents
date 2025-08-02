#!/usr/bin/env python
"""Runner for the local Ollama research agent with MCP Phase 5 integration."""

import os
import sys
import subprocess

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from examples.research.research_agent_local_mcp import agent


def check_ollama_running():
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get(f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api/tags")
        return response.status_code == 200
    except:
        return False


def main():
    # Check Ollama configuration
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    print(f"🧠 Using Ollama model: {ollama_model}")
    print(f"🔗 Ollama host: {ollama_host}")
    print("🔧 MCP Phase 5: Integration & Services")
    print("   Phase 1 Foundation:")
    print("   • Filesystem operations")
    print("   • DuckDuckGo search")  
    print("   • Time/date utilities")
    print("   Phase 2 Knowledge & Memory:")
    print("   • Enhanced filesystem for knowledge storage")
    print("   • Knowledge persistence and retrieval")
    print("   Phase 3 Development & Code:")
    print("   • GitHub integration and repository analysis")
    print("   • Code analysis and development workflows")
    print("   Phase 4 AI & Research:")
    print("   • Advanced search capabilities and AI tools")
    print("   • AI-powered research and analysis")
    print("   Phase 5 Integration & Services:")
    print("   • Enterprise cloud service integrations")
    print("   • Database and API connectivity")
    print("   • Workflow automation and orchestration")
    print("   • Business intelligence and analytics")
    print("   • CRM/ERP system integrations")
    print("   • Real-time messaging and notifications")
    print()
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("\nError: Ollama doesn't appear to be running.")
        print("\nPlease start Ollama with:")
        print("  ollama serve")
        print(f"\nAnd make sure you have pulled the model:")
        print(f"  ollama pull {ollama_model}")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("\nUsage: python run_research_agent_local_mcp.py 'Your research question here'")
        print("\nExample: python run_research_agent_local_mcp.py 'Analyze the enterprise integration landscape for AI platforms in 2024'")
        print("Example: python run_research_agent_local_mcp.py 'What are the best practices for implementing multi-cloud service orchestration?'")
        print("Example: python run_research_agent_local_mcp.py 'Compare enterprise CRM platforms and their API integration capabilities'")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    print(f"🔍 Researching: {question}\n")
    print("This may take a few minutes...\n")
    
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        
        # Collect the output
        output_content = ""
        
        # Print and collect the final response
        if "messages" in result:
            for message in result["messages"]:
                # Handle both dict and object message types
                if hasattr(message, 'type') and message.type == "ai":
                    print(message.content)
                    output_content = message.content
                elif isinstance(message, dict) and message.get("role") == "assistant":
                    content = message.get("content", "")
                    print(content)
                    output_content = content
        else:
            print(result)
            output_content = str(result)
        
        # Create output-examples directory if it doesn't exist
        os.makedirs("output-examples", exist_ok=True)
        
        # Save the output as markdown
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_question = "".join(c for c in question[:50] if c.isalnum() or c in " -_").strip()
        safe_question = safe_question.replace(" ", "_")
        output_filename = f"output-examples/{timestamp}_MCP_Phase5_{safe_question}.md"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Research Output: {question}\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model:** {ollama_model} (Local Ollama)\n")
            f.write(f"**MCP Integration:** Phase 5 - Integration & Services\n\n")
            f.write("---\n\n")
            f.write(output_content)
        
        print(f"\n\n📄 Output saved to: {output_filename}")
        
        # Save virtual files to disk if they exist
        if "files" in result and result["files"]:
            print("\n\n📁 Virtual files created during research:")
            print("-" * 40)
            for filename, content in result["files"].items():
                # Save to output-examples directory
                file_path = os.path.join("output-examples", f"{timestamp}_MCP_Phase5_{filename}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✓ Saved: {file_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()