import os
import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# Load environment variables from .env
load_dotenv()

# 1. Instantiate the OllamaModel
ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)
# 2. Define the Agent
agent = Agent(
    model=model,
    system_prompt="You are a helpful, concise AI assistant running locally.",
)

async def main():
    # Maintain message history across turns
    message_history: list[ModelMessage] = []

    print("--- Muse Glimmer Chatbot (Type 'exit' or 'quit' to stop) ---\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Pass message_history to maintain context in multi-turn conversation
        result = await agent.run(
            user_input,
            message_history=message_history,
        )

        # Print model response
        print(f"\nAssistant: {result}\n")

        # Update history with new messages from this run
        message_history.extend(result.new_messages())


if __name__ == "__main__":
    asyncio.run(main())