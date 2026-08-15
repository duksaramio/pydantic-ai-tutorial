import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai_harness import CodeMode
from pydantic_monty import NOT_HANDLED, OSAccess

# 1. Load environment variables from .env
load_dotenv()

# 2. Define Agent using Ollama model
ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

# Example A: Static OSAccess with environment variables
agent_static_env = Agent(
    model=model,
    capabilities=[
        CodeMode(
            os_access=OSAccess(
                environ={"API_BASE": "https://api.example.com", "APP_ENV": "development"}
            )
        )
    ],
)

# Example B: Dynamic Callback Handler for granular OS control
allowed_env = {
    "ALLOWED_KEY": "secret-xyz-123",
    "SERVICE_NAME": "analytics-core",
}


def custom_os_handler(fn: str, args: tuple, kwargs: dict):
    if fn == "os.getenv":
        key = args[0] if args else None
        # Return value if allowed; returns None for other keys (standard unset variable)
        return allowed_env.get(key)
    # Refuse all other OS calls by returning NOT_HANDLED
    return NOT_HANDLED


agent_custom_handler = Agent(
    model=model,
    capabilities=[CodeMode(os_access=custom_os_handler)],
)


def main():
    print("--- Running Code Mode OS Access Examples ---")

    print("\n1. Testing Static OSAccess:")
    res_a = agent_static_env.run_sync(
        "Using python os.getenv, check the value of 'API_BASE' and 'APP_ENV' and return them."
    )
    print("Static Env Output:", res_a.output)

    print("\n2. Testing Custom OS Callback Handler:")
    res_b = agent_custom_handler.run_sync(
        "Using python os.getenv, check the value of 'ALLOWED_KEY' and 'UNAUTHORIZED_KEY'."
    )
    print("Custom Callback Output:", res_b.output)


if __name__ == "__main__":
    main()
