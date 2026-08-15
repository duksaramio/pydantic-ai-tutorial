import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langfuse import get_client

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, Evaluator, EvaluatorContext, EvaluationReason, IsInstance

# 1. Load environment variables & initialize Langfuse
load_dotenv()
langfuse = get_client()
Agent.instrument_all()

ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# 2. Configure Ollama Model and Agent
model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

agent = Agent(
    model=model,
    system_prompt="You are a concise geography expert. Answer questions directly in a single short sentence or phrase.",
)


# 3. Define a Custom Evaluator
@dataclass
class MaxWordCount(Evaluator):
    """Asserts that output length in words does not exceed a maximum threshold."""

    max_words: int = 15

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        word_count = len(str(ctx.output).split())
        passed = word_count <= self.max_words
        return EvaluationReason(
            value=passed,
            reason=f"Output has {word_count} words (max allowed: {self.max_words})",
        )


# 4. Define the task function to evaluate
async def get_capital(query: str) -> str:
    result = await agent.run(query)
    return result.output.strip()


# 5. Build Dataset with Test Cases and Evaluators
dataset = Dataset(
    name="world_capitals_eval",
    cases=[
        Case(
            name="france_capital",
            inputs="What is the capital of France?",
            expected_output="Paris",
            metadata={"difficulty": "easy"},
            evaluators=[Contains(value="Paris", case_sensitive=False)],
        ),
        Case(
            name="japan_capital",
            inputs="What is the capital of Japan?",
            expected_output="Tokyo",
            metadata={"difficulty": "easy"},
            evaluators=[Contains(value="Tokyo", case_sensitive=False)],
        ),
        Case(
            name="australia_capital",
            inputs="What is the capital of Australia?",
            expected_output="Canberra",
            metadata={"difficulty": "medium"},
            evaluators=[Contains(value="Canberra", case_sensitive=False)],
        ),
    ],
    evaluators=[
        IsInstance(type_name="str"),
        MaxWordCount(max_words=15),
    ],
)


# 6. Execute Experiment, Print Results, and Flush to Langfuse
def main():
    print(f"Running evaluation against Ollama model '{ollama_model}' with Langfuse logging...\n")
    report = dataset.evaluate_sync(get_capital, max_concurrency=1)

    print("\n--- Evaluation Summary Table ---")
    report.print(include_input=True, include_output=True, include_reasons=True)

    avg = report.averages()
    if avg:
        print(f"\nOverall Pass Rate: {avg.assertions * 100:.1f}%")
        print(f"Average Duration: {avg.task_duration * 1000:.1f}ms per case")

    # Flush all traces to local Langfuse server
    print("\nFlushing traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Flushed successfully!")


if __name__ == "__main__":
    main()
