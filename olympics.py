import os
from dotenv import load_dotenv
from langfuse import get_client
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# 1. Load environment variables from .env
load_dotenv()

# 2. Initialize Langfuse client
langfuse = get_client()


class CityLocation(BaseModel):
    city: str
    country: str


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
    output_type=CityLocation,
)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage)
#> RunUsage(cost=Decimal('0.0000525'), input_tokens=57, output_tokens=8, requests=1)

# Ensure all spans/traces are sent to local Langfuse server before script termination
langfuse.flush()