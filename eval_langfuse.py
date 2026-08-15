import os
from dotenv import load_dotenv
from langfuse import get_client

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, IsInstance

# 1. Load environment variables (.env contains LANGFUSE_* and OLLAMA_* settings)
load_dotenv()

# 2. Initialize Langfuse Client & Instrument Pydantic AI
langfuse = get_client()
Agent.instrument_all()

# 3. Configure Ollama Model and Agent
ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

agent = Agent(
    model=model,
    system_prompt="You are a geographic knowledge assistant. Answer with only the capital city name.",
)


# 4. Define Task Function wrapped in a Langfuse Observation Trace
async def get_capital(country: str) -> str:
    # Creating a named observation groups the prompt, agent execution, and tool calls into a Langfuse trace
    with langfuse.start_as_current_observation(name=f"eval_capital_{country.lower()}"):
        res = await agent.run(f"What is the capital of {country}? Output only the city name.")
        return res.output.strip()


# 5. Define Pydantic Evals Dataset
dataset = Dataset(
    name="capitals_langfuse_eval",
    cases=[
        Case(
            name="france_capital",
            inputs="France",
            expected_output="Paris",
            evaluators=[Contains(value="Paris", case_sensitive=False)],
        ),
        Case(
            name="japan_capital",
            inputs="Japan",
            expected_output="Tokyo",
            evaluators=[Contains(value="Tokyo", case_sensitive=False)],
        ),
    ],
    evaluators=[
        IsInstance(type_name="str"),
    ],
)


# 6. Run Evaluation and Flush to Langfuse
def main():
    print(f"Running Pydantic Evals with Langfuse tracing enabled (Model: {ollama_model})...\n")
    report = dataset.evaluate_sync(get_capital, max_concurrency=1)

    print("\n--- Evaluation Summary Table ---")
    report.print(include_input=True, include_output=True, include_reasons=True)

    # 7. CRITICAL: Flush Langfuse client to deliver all buffered traces before script exits
    print("\nFlushing pending traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Done! Visit http://localhost:3000 to see all traces logged in real time.")


if __name__ == "__main__":
    main()
