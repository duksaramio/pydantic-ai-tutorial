import os
import asyncio
from dotenv import load_dotenv
from langfuse import get_client
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# 1. Load environment variables from .env
load_dotenv()

# 2. Initialize Langfuse client
langfuse = get_client()

# 3. Initialize Pydantic AI instrumentation
Agent.instrument_all()

# 4. Define Agent using Ollama model
ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)
agent = Agent(
    model=model,
    system_prompt="You are a helpful, concise AI assistant running locally.",
)


# 5. Simple Chatbot Loop
async def main():
    message_history: list[ModelMessage] = []
    print("--- Muse Glimmer Chatbot (Langfuse Tracing Enabled) ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = await agent.run(user_input, message_history=message_history)
        print(f"\nAssistant: {result.output}\n")

        message_history.extend(result.new_messages())
        langfuse.flush()


if __name__ == "__main__":
    asyncio.run(main())
