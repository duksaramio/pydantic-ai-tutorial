import os
from dotenv import load_dotenv
import logfire
from langfuse import get_client

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import HasMatchingSpan

# 1. Load environment variables & configure OpenTelemetry tracing
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

# 2. Define Agent with Internal Telemetry Instrumentation
rag_agent = Agent(
    model=model,
    system_prompt=(
        "You are a RAG assistant. When asked a knowledge question, you MUST:\n"
        "1. Call `retrieve_docs` to query the vector database.\n"
        "2. Call `rerank_context` to filter the most relevant passages.\n"
        "3. Provide a concise answer based on the retrieved documents."
    ),
)


@rag_agent.tool
def retrieve_docs(ctx: RunContext, query: str) -> list[str]:
    """Search knowledge base and emit a custom OpenTelemetry span with metadata."""
    with logfire.span("vector_db.search", index_name="knowledge_base", top_k=5):
        return [
            "Doc 1: Pydantic Evals uses OpenTelemetry spans for behavioral evaluation.",
            "Doc 2: Span-based evaluation verifies execution flow and tool latency.",
        ]


@rag_agent.tool
def rerank_context(ctx: RunContext, docs: list[str]) -> str:
    """Rerank retrieved passages and emit a custom span with attributes."""
    with logfire.span("reranker.process", reranker_model="cross-encoder-v1", doc_count=len(docs)):
        return "\n".join(docs[:1])


async def answer_rag_query(user_query: str) -> str:
    with logfire.span("rag_pipeline.execution"):
        res = await rag_agent.run(user_query)
        return res.output


# 3. Define Dataset with Span-Based Assertions
dataset = Dataset(
    name="rag_span_eval",
    cases=[
        Case(
            name="rag_pipeline_trace_check",
            inputs="How does Pydantic Evals test agent behavior?",
            evaluators=[
                # A. Name Conditions: verify retrieval and reranking spans executed
                HasMatchingSpan(
                    query={"name_equals": "vector_db.search"},
                    evaluation_name="vector_search_span_exists",
                ),
                HasMatchingSpan(
                    query={"name_equals": "reranker.process"},
                    evaluation_name="rerank_span_exists",
                ),
                # B. Attribute Conditions: verify telemetry metadata on spans
                HasMatchingSpan(
                    query={
                        "has_attributes": {
                            "index_name": "knowledge_base",
                            "top_k": 5,
                        }
                    },
                    evaluation_name="search_attributes_correct",
                ),
                HasMatchingSpan(
                    query={"has_attribute_keys": ["reranker_model", "doc_count"]},
                    evaluation_name="rerank_attributes_present",
                ),
                # C. Logical Operators: ensure no error spans occurred during execution
                HasMatchingSpan(
                    query={
                        "name_equals": "rag_pipeline.execution",
                        "not_": {"has_status": "error"},
                        "no_descendant_has": {"has_status": "error"},
                    },
                    evaluation_name="clean_execution_no_errors",
                ),
                # D. Duration Conditions: verify database query completed within SLA (e.g. 5.0s)
                HasMatchingSpan(
                    query={
                        "name_equals": "vector_db.search",
                        "max_duration": 5.0,
                    },
                    evaluation_name="db_query_within_sla",
                ),
                # E. Hierarchy Conditions: ensure pipeline span has child spans
                HasMatchingSpan(
                    query={
                        "name_equals": "rag_pipeline.execution",
                        "min_descendant_count": 2,
                    },
                    evaluation_name="pipeline_has_subspans",
                ),
            ],
        )
    ]
)


def main():
    print(f"Running Span-Based Evaluation against Ollama model '{ollama_model}'...\n")
    report = dataset.evaluate_sync(answer_rag_query, max_concurrency=1)

    print("\n--- Span-Based Evaluation Summary Table ---")
    report.print(include_input=True, include_output=True, include_reasons=True)

    print("\nFlushing traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Flushed successfully!")


if __name__ == "__main__":
    main()
