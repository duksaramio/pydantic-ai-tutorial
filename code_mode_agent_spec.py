import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai_harness import CodeMode

# 1. Load environment variables from .env
load_dotenv()

# 2. Define Agent using Ollama model
ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

yaml_path = Path(__file__).parent / "code_mode_agent.yaml"

# Load agent from yaml specification with CodeMode as custom capability
agent = Agent.from_file(
    yaml_path,
    model=model,
    custom_capability_types=[CodeMode],
)


@agent.tool_plain
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width


def main():
    print(f"--- Running Agent Loaded from YAML Spec ({yaml_path.name}) ---")
    result = agent.run_sync(
        "Calculate the total area of two rooms: room A is 12 by 15, room B is 10 by 20. Return the sum."
    )
    print("\nResult Output:")
    print(result.output)


if __name__ == "__main__":
    main()
