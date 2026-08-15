import os
from dotenv import load_dotenv
from langfuse import get_client
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai_harness import CodeMode

# 1. Load environment variables from .env
load_dotenv()

# 2. Initialize Langfuse client & Pydantic AI instrumentation
langfuse = get_client()
Agent.instrument_all()

# 3. Define Agent using Ollama model
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
    weather_data = {
        "Paris": {"temp_f": 72, "condition": "sunny"},
        "Tokyo": {"temp_f": 64, "condition": "rainy"},
        "London": {"temp_f": 59, "condition": "cloudy"},
    }
    return weather_data.get(city, {"temp_f": 70, "condition": "unknown"})


def main():
    print("--- Running Code Mode Basic Example (Weather lookup with Langfuse) ---")
    result = agent.run_sync("What's the weather in Paris and Tokyo, in Celsius?")
    print("\nResult Output:")
    print(result.output)
    print("\nUsage:")
    print(result.usage)

    # Flush Langfuse traces before exiting
    langfuse.flush()


if __name__ == "__main__":
    main()
