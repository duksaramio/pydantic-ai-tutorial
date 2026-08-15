import os
from dotenv import load_dotenv
from langfuse import get_client
from pydantic_ai import Agent, RunContext
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
roulette_agent = Agent(
    model=model,
    deps_type=int,
    output_type=bool,
    system_prompt=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)

@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
  """check if the square is a winner"""
  return 'winner' if square == ctx.deps else 'loser'
# Run the agent
success_number = 18
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.output)
#> True
result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.output)
#> False