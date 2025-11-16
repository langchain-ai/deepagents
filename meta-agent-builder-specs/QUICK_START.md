# 🚀 QUICK START - Meta-Agent Builder

Get up and running in 15 minutes!

---

## ✅ Prerequisites Checklist

- [ ] Python 3.10 or higher installed
- [ ] Anthropic API key
- [ ] Tavily API key (for web search)
- [ ] Basic understanding of Deep Agents

---

## 📦 Step 1: Installation (2 minutes)

```bash
# Install required packages
pip install deepagents>=0.2.9 \
    langchain>=0.3.0 \
    langchain-anthropic>=0.3.0 \
    langgraph>=0.2.0 \
    tavily-python>=0.5.0

# Set environment variables
export ANTHROPIC_API_KEY='your-anthropic-key-here'
export TAVILY_API_KEY='your-tavily-key-here'
```

---

## 🎯 Step 2: Run the Example (3 minutes)

```bash
# Download the example
curl -O https://raw.githubusercontent.com/.../COMPLETE_EXAMPLE.py

# Run it
python COMPLETE_EXAMPLE.py
```

**What it does:**
- Creates a Meta-Orchestrator with 4 specialists
- Generates complete project specs for a research agent
- Demonstrates the full workflow

**Expected output:**
```
🧠 META-AGENT BUILDER - Example

📝 User Request:
Create a research agent system that...

🚀 Generating Specifications...

[Agent creates project brief...]
[Invokes documentation specialist...]
[Invokes architecture specialist...]
[Invokes PRD specialist...]
[Invokes implementation specialist...]
[Creates executive summary...]

✅ Specification generation complete!
```

---

## 📖 Step 3: Understand the Output (5 minutes)

The example generates specifications in a virtual filesystem:

```
/project_specs/
├── project_brief.md              # Project overview
├── architecture/
│   └── architecture.md           # System architecture
├── prd.md                        # Product requirements
└── implementation/
    └── implementation_guide.md   # How to build it
```

**Read through these files to see what the system generates!**

---

## 🔧 Step 4: Customize for Your Project (5 minutes)

### Modify the User Request

```python
user_request = """
YOUR PROJECT DESCRIPTION HERE

Example:
- Create a coding assistant with file analysis
- Build a data processing pipeline
- Design a customer support agent
"""

orchestrator.generate_specs(user_request)
```

### Add More Specialists

```python
specialists.append({
    "name": "your-specialist",
    "description": "What it does",
    "system_prompt": "Detailed instructions...",
    "tools": [your_tools],
})
```

---

## 📚 Next Steps

### Learn the Architecture
- Read: [Technical Specification](./00-TECHNICAL_SPECIFICATION.md)
- Understand: [Meta-Orchestrator](./architecture/META_ORCHESTRATOR_SPECIFICATION.md)
- Explore: [Specialists](./specialists/)

### Implement Full System
- Follow: [Implementation Guide](./implementation/IMPLEMENTATION_GUIDE.md)
- Use: Code templates
- Test: Validation pipeline

### Advanced Features
1. **Add Persistent Storage**
   ```python
   from langgraph.store.postgres import PostgresStore

   store = PostgresStore(connection_string)
   orchestrator = MetaOrchestrator(use_persistent=True, store=store)
   ```

2. **Enable Code Execution**
   ```python
   from deepagents.backends.sandbox import SandboxBackend

   backend = CompositeBackend(
       default=SandboxBackend(),  # Enables execute()
       routes={...}
   )
   ```

3. **Add More Specialists**
   - Context Engineering Specialist
   - Middleware Specialist
   - Orchestration Specialist
   - Validation Specialist

4. **Build Template Library**
   ```python
   # Save successful patterns
   save_as_template(project_type, specs)

   # Reuse for similar projects
   template = load_template(project_type)
   ```

---

## 🐛 Troubleshooting

### Error: "API key not found"
```bash
# Check environment variables
echo $ANTHROPIC_API_KEY
echo $TAVILY_API_KEY

# Set them if missing
export ANTHROPIC_API_KEY='sk-ant-...'
export TAVILY_API_KEY='tvly-...'
```

### Error: "Module not found"
```bash
# Reinstall dependencies
pip install --upgrade deepagents langchain langchain-anthropic langgraph
```

### Slow execution
- Normal for first run (research phase)
- Use prompt caching (automatic with Anthropic)
- Enable template reuse for similar projects

---

## 💡 Tips for Success

1. **Be Specific in Requests**
   - ❌ "Create an agent"
   - ✅ "Create a research agent with web search, document analysis, and report generation"

2. **Check Generated Specs**
   - Review architecture for completeness
   - Validate PRD against requirements
   - Test implementation guide

3. **Iterate**
   - First run: May need refinement
   - Provide feedback
   - System learns and improves

4. **Start Simple**
   - Begin with 2-3 specialists
   - Add more as needed
   - Build complexity gradually

---

## 📊 What to Expect

### First Run
- **Time:** 20-30 minutes
- **Cost:** ~$2-3 in API calls
- **Output:** Complete project specs

### Subsequent Runs (Similar Projects)
- **Time:** 8-12 minutes (60% faster)
- **Cost:** ~$1-2 (caching helps)
- **Output:** Higher quality (learned patterns)

---

## 🎓 Learning Resources

### Deep Agents Documentation
- [Overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [Quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart)
- [Customization](https://docs.langchain.com/oss/python/deepagents/customization)

### Meta-Agent Builder Specs
- [Technical Specification](./00-TECHNICAL_SPECIFICATION.md)
- [Implementation Guide](./implementation/IMPLEMENTATION_GUIDE.md)
- [Specialist Specs](./specialists/)

---

## ✨ Success Criteria

You're ready to move forward when you:
- [x] Ran the example successfully
- [x] Understand the generated specifications
- [x] Can customize the user request
- [x] Know where to find detailed documentation

---

## 🚀 Ready to Build?

**Option 1:** Use the example as-is
- Good for understanding the system
- Quick way to generate specs
- Limited to 4 specialists

**Option 2:** Follow the implementation guide
- Build the complete system
- All 7 specialists
- Full validation pipeline
- Template library
- Production-ready

**Recommended:** Start with Option 1, move to Option 2

---

## 📞 Need Help?

- **Documentation Issues:** Check [Technical Specification](./00-TECHNICAL_SPECIFICATION.md)
- **Implementation Questions:** See [Implementation Guide](./implementation/IMPLEMENTATION_GUIDE.md)
- **Bugs or Errors:** Review troubleshooting section above

---

**Happy building! 🎉**

The Meta-Agent Builder team
