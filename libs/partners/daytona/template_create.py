from langsmith.sandbox import SandboxClient

client = SandboxClient()

client.create_template(
    name="python-sandbox",
    image="python:3.12-slim",
)
