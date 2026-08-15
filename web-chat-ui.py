import os
from dotenv import load_dotenv
from pydantic_ai import Agent
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

@agent.tool_plain
def get_weather(city: str) -> str:
    return f'The weather in {city} is sunny'
app = agent.to_web()