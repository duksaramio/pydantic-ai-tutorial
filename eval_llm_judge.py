import os
from dotenv import load_dotenv
from langfuse import get_client

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

# 1. Load environment variables & initialize Langfuse
load_dotenv()
langfuse = get_client()
Agent.instrument_all()

ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# 2. Configure Model for Task Agent & Judge Evaluator
model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

customer_support_agent = Agent(
    model=model,
    system_prompt=(
        "You are an empathetic, professional customer support agent for Acme Cloud. "
        "Acknowledge the user's issue, offer clear steps, and keep a helpful and warm tone."
    ),
)


# 3. Define Task Function
async def handle_customer_inquiry(query: str) -> str:
    res = await customer_support_agent.run(query)
    return res.output


# 4. Build Dataset with Global and Case-Specific LLM Judges
dataset = Dataset(
    name="customer_support_llm_judge_eval",
    cases=[
        Case(
            name="refund_request",
            inputs="I was double billed for my subscription last month and I need a refund immediately.",
            expected_output=None,
            metadata={"category": "billing"},
            evaluators=[
                # Case-specific judge: checks billing policy empathy and process
                LLMJudge(
                    rubric=(
                        "Response must: 1. Empathetically apologize for the billing issue, "
                        "2. Assure the customer that the charge will be investigated/refunded, "
                        "3. Ask for or confirm billing details."
                    ),
                    include_input=True,
                    model=model,
                    assertion={"evaluation_name": "billing_empathy_check", "include_reason": True},
                )
            ],
        ),
        Case(
            name="outage_inquiry",
            inputs="Is your API down? My requests are returning 502 Bad Gateway errors right now.",
            expected_output=None,
            metadata={"category": "technical"},
            evaluators=[
                # Case-specific judge: checks technical status and troubleshooting steps
                LLMJudge(
                    rubric=(
                        "Response must: 1. Direct the user to the status page or acknowledge investigating the outage, "
                        "2. Provide concrete next steps (e.g. check status page or share request IDs)."
                    ),
                    include_input=True,
                    model=model,
                    assertion={"evaluation_name": "outage_guidance_check", "include_reason": True},
                )
            ],
        ),
    ],
    evaluators=[
        # Dataset-level LLM Judge: checks tone & helpfulness across all responses
        LLMJudge(
            rubric="Response maintains a polite, professional, and respectful tone without defensive language.",
            include_input=True,
            model=model,
            assertion={"evaluation_name": "tone_compliance", "include_reason": True},
        ),
    ],
)


def main():
    print(f"Running LLM-as-a-Judge evaluation with Langfuse logging (Model: {ollama_model})...\n")
    report = dataset.evaluate_sync(handle_customer_inquiry, max_concurrency=1)

    print("\n--- Evaluation Summary Table ---")
    report.print(include_input=True, include_output=True, include_reasons=True)

    print("\n--- Detailed Case Results & Judge Reasons ---")
    for case in report.cases:
        print(f"\n[Case: {case.name}]")
        print(f"Prompt: {case.inputs}")
        print(f"Response: {case.output}\n")
        for assert_name, result in case.assertions.items():
            status = "PASSED" if result.value else "FAILED"
            print(f"  Assertion '{assert_name}': {status}")
            if result.reason:
                print(f"    Reason: {result.reason}")

    # Flush all traces (agent runs & LLM judge runs) to Langfuse
    print("\nFlushing traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Flushed successfully!")


if __name__ == "__main__":
    main()
