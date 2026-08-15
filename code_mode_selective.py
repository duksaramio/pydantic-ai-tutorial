import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness import CodeMode

# 1. Load environment variables from .env
load_dotenv()

# 2. Define Agent using Ollama model
ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)


def search(query: str) -> str:
    """Search the web for information."""
    return f"results for query: '{query}' - found 3 matching articles"


def fetch(url: str) -> str:
    """Fetch the contents of a webpage given its URL."""
    return f"contents of {url}: summary of pydantic ai harness features"


# Sandboxed toolset: tagged with metadata code_mode=True
search_tools = FunctionToolset(tools=[search, fetch]).with_metadata(code_mode=True)

# Create Agent where only tools with code_mode=True are wrapped into run_code
agent = Agent(
    model=model,
    toolsets=[search_tools],
    capabilities=[CodeMode(tools={"code_mode": True})],
)


@agent.tool_plain
def log_action(action: str) -> str:
    """Native tool that stays outside run_code for host logging."""
    return f"Action logged natively: {action}"


def main():
    print("--- Running Code Mode Selective Tool Sandboxing Example ---")
    result = agent.run_sync(
        "Search for 'pydantic-ai harness' and fetch 'https://docs.pydantic.dev/ai/harness'"
    )
    print("\nResult Output:")
    print(result.output)


if __name__ == "__main__":
    main()
