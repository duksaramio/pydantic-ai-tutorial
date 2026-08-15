import os
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
from langfuse import get_client

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ArgumentCorrectness,
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    MaxModelRequests,
    MaxToolCalls,
    ToolCorrectness,
    TrajectoryMatch,
)
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output
from pydantic_evals.otel import SpanTreeRecordingError

# 1. Load environment variables (.env)
load_dotenv()

# 2. Configure Observability & Tracing (Logfire + Langfuse)
# Logfire captures OpenTelemetry spans that Agentic Evaluators inspect
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

# Langfuse records traces in http://localhost:3000
langfuse = get_client()
Agent.instrument_all()

# 3. Configure Ollama Model
ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

# 4. Define Agent with Multiple Tools
support_agent = Agent(
    model=model,
    system_prompt=(
        "You are an e-commerce support agent. When a user asks about order status or refund eligibility:\n"
        "1. First call `lookup_order` with the given order_id.\n"
        "2. If the user wants a refund or return, call `check_refund_policy` with the order_id.\n"
        "3. Provide a clear, polite summary based on the tool outputs."
    ),
)


@support_agent.tool
def lookup_order(ctx: RunContext, order_id: str) -> str:
    """Retrieve details and item status for a given order ID."""
    return f"Order #{order_id}: Total=$120, Item='Wireless Headphones', DeliveredDate='2026-08-01', Status='Delivered'"


@support_agent.tool
def check_refund_policy(ctx: RunContext, order_id: str) -> str:
    """Check refund eligibility and warranty status for an order."""
    return f"Order #{order_id} is within the 30-day return window. Status: Eligible for full refund."


# 5. Define Custom Trajectory Judge Evaluator (LLM Judgement on Execution Path)
@dataclass
class TrajectoryJudge(Evaluator):
    """LLM Judge that inspects both the user prompt and the tool execution sequence."""

    rubric: str = "The agent completed the task correctly and the tool sequence is appropriate."

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return EvaluationReason(value=False, reason="No OpenTelemetry span tree available.")

        # Extract tool names from OpenTelemetry execution spans
        tool_names = [
            node.attributes["gen_ai.tool.name"]
            for node in span_tree
            if "gen_ai.tool.name" in node.attributes
            and "pydantic_ai.tool.deferral.name" not in node.attributes
            and node.status != "error"
            and (node.name == "running tool" or node.name.startswith("execute_tool "))
            and not str(node.attributes.get("logfire.msg", "")).startswith("running output function:")
        ]
        trajectory_str = ", ".join(str(n) for n in tool_names) or "(none)"

        grading = await judge_input_output(
            {"user_query": ctx.inputs, "tool_trajectory": trajectory_str},
            ctx.output,
            self.rubric,
            model=model,
        )
        return EvaluationReason(
            value=grading.pass_,
            reason=f"Trajectory: [{trajectory_str}] - Judge Reason: {grading.reason}",
        )


# 6. Task Function to Evaluate
async def run_support_task(query: str) -> str:
    res = await support_agent.run(query)
    return res.output


# 7. Define Dataset with Agentic Evaluators
dataset = Dataset(
    name="agentic_trajectory_eval",
    cases=[
        Case(
            name="order_and_refund_check",
            inputs="Can I get a refund for my order ORD-9988? Please check the order details first.",
            evaluators=[
                # 1. Tool Coverage: Agent must call both tools
                ToolCorrectness(
                    expected_tools=["lookup_order", "check_refund_policy"],
                    allow_extra=False,
                    evaluation_name="tool_coverage",
                ),
                # 2. Trajectory Shape: Must call lookup_order before check_refund_policy
                TrajectoryMatch(
                    expected_trajectory=["lookup_order", "check_refund_policy"],
                    order="in_order",
                    evaluation_name="trajectory_order_f1",
                ),
                # 3. Argument Quality: lookup_order must receive correct order_id
                ArgumentCorrectness(
                    tool_name="lookup_order",
                    expected_arguments={"order_id": "ORD-9988"},
                    match_mode="subset",
                    evaluation_name="order_id_arg_check",
                ),
            ],
        ),
        Case(
            name="simple_order_status",
            inputs="What is the current status of order ORD-5544?",
            evaluators=[
                # Only needs order lookup, not refund policy
                ToolCorrectness(
                    expected_tools=["lookup_order"],
                    allow_extra=False,
                    evaluation_name="tool_coverage",
                ),
                ArgumentCorrectness(
                    tool_name="lookup_order",
                    expected_arguments={"order_id": "ORD-5544"},
                    match_mode="subset",
                    evaluation_name="order_id_arg_check",
                ),
            ],
        ),
    ],
    evaluators=[
        # 4. Budget Discipline: Global limits on tool calls and model requests
        MaxToolCalls(max_calls=4, evaluation_name="tool_call_budget"),
        MaxModelRequests(max_requests=4, evaluation_name="model_request_budget"),
        # 5. Qualitative Trajectory LLM Judge
        TrajectoryJudge(
            rubric=(
                "Verify that the response accurately answers the customer's question, "
                "and that the tool_trajectory appropriately consulted necessary tools."
            )
        ),
    ],
)


# 8. Run Experiment and Print Detailed Trajectory Results
def main():
    print(f"Running Agentic Evaluation against Ollama model '{ollama_model}'...\n")
    report = dataset.evaluate_sync(run_support_task, max_concurrency=1)

    print("\n--- Agentic Evaluation Summary Table ---")
    report.print(include_input=True, include_output=True, include_reasons=True)

    print("\n--- Detailed Trajectory & Argument Assertions ---")
    for case in report.cases:
        print(f"\n[Case: {case.name}]")
        print(f"Input: {case.inputs}")
        print(f"Output: {case.output}")
        print("Scores:")
        for name, score in case.scores.items():
            print(f"  - {name}: {score.value:.3f} (Reason: {score.reason})")
        print("Assertions:")
        for name, assertion in case.assertions.items():
            status = "✔ PASS" if assertion.value else "✗ FAIL"
            print(f"  - {name}: {status}")
            if assertion.reason:
                print(f"    Reason: {assertion.reason}")

    # Flush all traces to Langfuse
    print("\nFlushing traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Flushed successfully!")


if __name__ == "__main__":
    main()
