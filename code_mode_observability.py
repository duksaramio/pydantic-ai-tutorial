import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturnPart
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

agent = Agent(
    model=model,
    capabilities=[CodeMode()],
)


@agent.tool_plain
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp_f": 72, "condition": "sunny"}


def main():
    print("--- Running Code Mode Observability / Metadata Example ---")
    result = agent.run_sync("What's the weather in Paris?")
    print("\nFinal Output:")
    print(result.output)

    print("\nInspecting Message Parts for nested tool execution metadata:")
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == "run_code":
                metadata = part.metadata or {}
                tool_calls = metadata.get("tool_calls", {})
                tool_returns = metadata.get("tool_returns", {})
                print(f"-> Found run_code ToolReturnPart:")
                print(f"   Nested tool_calls count: {len(tool_calls)}")
                print(f"   Nested tool_returns count: {len(tool_returns)}")
                for call_id, call_part in tool_calls.items():
                    print(f"   - Call ID {call_id}: {call_part.tool_name} with args: {call_part.args}")


if __name__ == "__main__":
    main()
