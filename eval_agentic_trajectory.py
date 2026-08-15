import os
from dotenv import load_dotenv
import logfire
from langfuse import get_client

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ArgumentCorrectness,
    MaxModelRequests,
    MaxToolCalls,
    ToolCorrectness,
    TrajectoryMatch,
)

# 1. Load environment variables
load_dotenv()

# 2. Configure Observability
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
langfuse = get_client()
Agent.instrument_all()

# 3. Model Configuration
ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

# 4. Multi-Step Research & Publication Agent
research_agent = Agent(
    model=model,
    system_prompt=(
        "You are an academic researcher. Follow this exact workflow:\n"
        "1. Call `search_papers` to find literature.\n"
        "2. Call `extract_key_findings` with the paper_id.\n"
        "3. Call `generate_summary` to finalize the summary."
    ),
)


@research_agent.tool
def search_papers(ctx: RunContext, topic: str) -> str:
    """Search academic databases for papers on a topic."""
    return "Found Paper #P100: 'Advances in Agentic AI Evaluation'"


@research_agent.tool
def extract_key_findings(ctx: RunContext, paper_id: str) -> str:
    """Extract key insights from a specific paper ID."""
    return f"Key findings from {paper_id}: Deterministic trajectory checks ensure tool reliability."


@research_agent.tool
def generate_summary(ctx: RunContext, findings: str) -> str:
    """Compile extracted findings into final review format."""
    return f"Summary: {findings}"


# 5. Define Task
async def research_topic(query: str) -> str:
    res = await research_agent.run(query)
    return res.output


# 6. Dataset Testing Trajectory Matching Modes (Exact, In-Order LCS, and Any-Order)
dataset = Dataset(
    name="research_pipeline_trajectory_eval",
    cases=[
        Case(
            name="complete_pipeline",
            inputs="Research topic 'Agentic AI' starting from paper search to key findings and summary.",
            evaluators=[
                # Test Multiset Coverage
                ToolCorrectness(
                    expected_tools=["search_papers", "extract_key_findings", "generate_summary"],
                    allow_extra=True,
                    evaluation_name="tools_present",
                ),
                # Test Sequential Order via Longest Common Subsequence (LCS F1 score)
                TrajectoryMatch(
                    expected_trajectory=["search_papers", "extract_key_findings", "generate_summary"],
                    order="in_order",
                    evaluation_name="in_order_lcs_match",
                ),
                # Test Strict Exact Order (1.0 only if exact match, 0.0 otherwise)
                TrajectoryMatch(
                    expected_trajectory=["search_papers", "extract_key_findings", "generate_summary"],
                    order="exact",
                    evaluation_name="exact_trajectory_match",
                ),
                # Test First Tool's Argument Value
                ArgumentCorrectness(
                    tool_name="search_papers",
                    expected_arguments={"topic": "Agentic AI"},
                    match_mode="subset",
                    occurrence="first",
                    evaluation_name="topic_arg_check",
                ),
            ],
        ),
    ],
    evaluators=[
        MaxToolCalls(max_calls=5, evaluation_name="tool_budget"),
        MaxModelRequests(max_requests=5, evaluation_name="request_budget"),
    ],
)


def main():
    print(f"Running Trajectory Comparison Evaluation (Model: {ollama_model})...\n")
    report = dataset.evaluate_sync(research_topic, max_concurrency=1)

    print("\n--- Summary Table ---")
    report.print(include_input=True, include_output=True, include_reasons=True)

    print("\nFlushing traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Flushed successfully!")


if __name__ == "__main__":
    main()
