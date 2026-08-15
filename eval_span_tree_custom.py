import os
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
from langfuse import get_client

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.otel import SpanTreeRecordingError

# 1. Load environment & setup OpenTelemetry tracing
load_dotenv()
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
langfuse = get_client()
Agent.instrument_all()

ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

# 2. Agent with Multiple Internal Spans
multi_step_agent = Agent(
    model=model,
    system_prompt=(
        "You are an assistant. Call `fetch_external_api` to get real-time info."
    ),
)


@multi_step_agent.tool
def fetch_external_api(ctx: RunContext, service_name: str) -> str:
    """Simulates an external API call wrapped in sub-spans."""
    with logfire.span("http.client.request", method="GET", endpoint="/api/v1/status"):
        with logfire.span("http.dns.resolve", host="api.service.internal"):
            pass
        return "Service status: 200 OK. All systems operational."


async def run_monitored_task(query: str) -> str:
    with logfire.span("user_transaction.root", transaction_type="status_query"):
        res = await multi_step_agent.run(query)
        return res.output


# 3. Custom Evaluator Inspecting SpanTree and SpanNodes
@dataclass
class DeepSpanInspector(Evaluator):
    """Custom Evaluator inspecting SpanTree hierarchy, depths, attributes, and durations."""

    max_allowed_dns_duration_sec: float = 2.0

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float | EvaluationReason]:
        try:
            tree = ctx.span_tree
        except SpanTreeRecordingError:
            return {"span_tree_available": EvaluationReason(value=False, reason="Span tree recording not available")}

        # 1. Find specific spans using tree.find()
        http_spans = tree.find(lambda node: "http.client.request" in node.name)
        dns_spans = tree.find(lambda node: "http.dns.resolve" in node.name)

        # 2. Calculate total HTTP duration across matching nodes
        total_http_time = sum(span.duration.total_seconds() for span in http_spans)

        # 3. Check parent-child span relationships
        has_proper_parent = False
        for dns_span in dns_spans:
            if dns_span.parent and "http.client.request" in dns_span.parent.name:
                has_proper_parent = True
                break

        # 4. Check span attributes
        has_correct_endpoint = any(
            span.attributes.get("endpoint") == "/api/v1/status" for span in http_spans
        )

        return {
            "http_calls_executed": len(http_spans) > 0,
            "dns_nested_under_http": has_proper_parent,
            "endpoint_attribute_valid": has_correct_endpoint,
            "total_http_time_sec": total_http_time,
            "dns_performance_ok": EvaluationReason(
                value=total_http_time < self.max_allowed_dns_duration_sec,
                reason=f"HTTP/DNS time was {total_http_time:.4f}s (budget: {self.max_allowed_dns_duration_sec}s)",
            ),
        }


# 4. Build Dataset
dataset = Dataset(
    name="custom_spantree_eval",
    cases=[
        Case(
            name="deep_trace_inspection",
            inputs="Check the status of our payment service.",
            evaluators=[DeepSpanInspector()],
        )
    ],
)


def main():
    print(f"Running Custom SpanTree Inspection Evaluation (Model: {ollama_model})...\n")
    report = dataset.evaluate_sync(run_monitored_task, max_concurrency=1)

    print("\n--- Span Tree Analysis Report ---")
    report.print(include_input=True, include_output=True, include_reasons=True)

    print("\n--- Programmatic Access to Case Results ---")
    for case in report.cases:
        print(f"Case '{case.name}':")
        print(f"  Assertions: {case.assertions}")
        print(f"  Scores / Metrics: {case.scores}")

    print("\nFlushing traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Flushed successfully!")


if __name__ == "__main__":
    main()
