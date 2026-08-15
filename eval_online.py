import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from dotenv import load_dotenv
from langfuse import get_client

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals.evaluators import (
    EvaluationReason,
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    EvaluatorFailure,
)
from pydantic_evals.online import (
    OnlineEvalConfig,
    OnlineEvaluator,
    wait_for_evaluations,
)

# 1. Load environment variables & initialize Langfuse
load_dotenv()
langfuse = get_client()
Agent.instrument_all()

ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# 2. Configure Model and Agent
model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

assistant = Agent(
    model=model,
    system_prompt="You are a helpful software assistant. Be concise and precise.",
)


# 3. Define Custom Evaluators for Online Monitoring
@dataclass
class QualitySafetyCheck(Evaluator):
    """Checks that live outputs are non-empty and avoid forbidden phrases."""

    forbidden_phrases: tuple[str, ...] = ("I cannot help", "error occurred")

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        text = str(ctx.output).lower()
        if not text.strip():
            return EvaluationReason(value=False, reason="Output was empty")

        for phrase in self.forbidden_phrases:
            if phrase.lower() in text:
                return EvaluationReason(
                    value=False, reason=f"Found forbidden phrase '{phrase}' in output"
                )

        return EvaluationReason(value=True, reason="Passed safety and quality checks")


# 4. Define an In-Memory Telemetry Sink for Online Events
evaluation_events: list[str] = []


async def custom_telemetry_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    """Receives background evaluation results as they complete."""
    for res in results:
        status_symbol = "✔ PASS" if res.value else "✗ FAIL"
        evaluation_events.append(f"[{res.name}] {status_symbol} (Reason: {res.reason})")
    for fail in failures:
        evaluation_events.append(f"[{fail.name}] ERROR: {fail.error_message}")


# 5. Configure Online Evaluation
online_config = OnlineEvalConfig(
    default_sink=custom_telemetry_sink,
    default_sample_rate=1.0,  # 100% of live calls evaluated
    metadata={"environment": "production", "service": "assistant-service"},
)


# 6. Decorate Live Function with Online Evaluation
@online_config.evaluate(
    OnlineEvaluator(
        evaluator=QualitySafetyCheck(),
        sample_rate=1.0,
        max_concurrency=5,
    ),
    target="live_assistant",
)
async def ask_live_assistant(prompt: str) -> str:
    """Production endpoint: caller receives immediate response while evaluation runs in background."""
    res = await assistant.run(prompt)
    return res.output.strip()


async def main():
    print(f"Starting online evaluation demo with Langfuse logging (Model: {ollama_model})...\n")

    test_queries = [
        "What is the time complexity of binary search?",
        "Explain Python list comprehension in one sentence.",
    ]

    for q in test_queries:
        print(f"[Live Call] User: {q}")
        # The user receives the output immediately
        response = await ask_live_assistant(q)
        print(f"[Live Response] Assistant: {response}\n")

    print("Waiting for background evaluations to finish...")
    await wait_for_evaluations()

    print("\n--- Telemetry Sink Recorded Events ---")
    for event in evaluation_events:
        print(" ", event)

    # Flush all traces to local Langfuse server
    print("\nFlushing traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Flushed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
