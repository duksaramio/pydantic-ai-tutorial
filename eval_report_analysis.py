import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Literal
from langfuse import get_client

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ConfusionMatrixEvaluator,
    ReportEvaluator,
    ReportEvaluatorContext,
)
from pydantic_evals.reporting.analyses import ReportAnalysis, ScalarResult, TableResult

# 1. Load environment variables & initialize Langfuse
load_dotenv()
langfuse = get_client()
Agent.instrument_all()

ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# 2. Classifier Agent Configuration
Category = Literal["billing", "technical", "general"]


class TicketClassification(BaseModel):
    category: Category


model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

classifier_agent = Agent(
    model=model,
    output_type=TicketClassification,
    system_prompt=(
        "You are an incoming support ticket router. "
        "Classify the ticket into exactly one of: 'billing', 'technical', or 'general'."
    ),
)


async def classify_ticket(ticket_text: str) -> str:
    res = await classifier_agent.run(ticket_text)
    return res.output.category


# 3. Custom Report Evaluator producing Scalar and Table Analyses
@dataclass
class ClassificationSummary(ReportEvaluator):
    """Computes overall accuracy as a ScalarResult and per-class precision/recall as a TableResult."""

    def evaluate(self, ctx: ReportEvaluatorContext) -> list[ReportAnalysis]:
        cases = ctx.report.cases
        if not cases:
            return []

        labels = ["billing", "technical", "general"]

        # 1. Scalar Analysis: Overall Accuracy
        correct = sum(1 for c in cases if str(c.output) == str(c.expected_output))
        accuracy_val = (correct / len(cases)) * 100
        accuracy = ScalarResult(
            title="Classification Accuracy",
            value=accuracy_val,
            unit="%",
            description="Percentage of tickets correctly categorized.",
        )

        # 2. Table Analysis: Per-Class Precision & Recall
        rows = []
        for label in labels:
            tp = sum(1 for c in cases if str(c.output) == label and str(c.expected_output) == label)
            fp = sum(1 for c in cases if str(c.output) == label and str(c.expected_output) != label)
            fn = sum(1 for c in cases if str(c.output) != label and str(c.expected_output) == label)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            rows.append([label, f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}"])

        table = TableResult(
            title="Per-Class Performance",
            columns=["Category", "Precision", "Recall", "F1 Score"],
            rows=rows,
            description="Detailed precision, recall, and F1 per category.",
        )

        return [accuracy, table]


# 4. Build Dataset with Report Evaluators
dataset = Dataset(
    name="ticket_classification_eval",
    cases=[
        Case(name="case_1", inputs="Where can I download my latest VAT invoice?", expected_output="billing"),
        Case(name="case_2", inputs="I want to update my credit card on file.", expected_output="billing"),
        Case(name="case_3", inputs="WebSocket connection disconnects after 30 seconds with 1006.", expected_output="technical"),
        Case(name="case_4", inputs="The SDK throws an SSL certificate verification error.", expected_output="technical"),
        Case(name="case_5", inputs="What are your office hours and timezone?", expected_output="general"),
        Case(name="case_6", inputs="Who is the CEO of the company?", expected_output="general"),
    ],
    report_evaluators=[
        ConfusionMatrixEvaluator(
            predicted_from="output",
            expected_from="expected_output",
            title="Ticket Category Confusion Matrix",
        ),
        ClassificationSummary(),
    ],
)


def main():
    print(f"Running report-level evaluation with Langfuse logging (Model: {ollama_model})...\n")
    report = dataset.evaluate_sync(classify_ticket, max_concurrency=1)

    print("\n--- Evaluation Summary Table ---")
    report.print(include_input=True, include_output=True)

    print("\n--- Experiment-Wide Analyses ---")
    for analysis in report.analyses:
        print(f"\n[Analysis: {analysis.title} ({analysis.type})]")
        if isinstance(analysis, ScalarResult):
            print(f"  Value: {analysis.value:.2f}{analysis.unit or ''}")
            if analysis.description:
                print(f"  Description: {analysis.description}")
        elif isinstance(analysis, TableResult):
            headers = " | ".join(analysis.columns)
            print(f"  {headers}")
            print("  " + "-" * len(headers))
            for row in analysis.rows:
                print("  " + " | ".join(str(cell) for cell in row))

    # Flush all traces to local Langfuse server
    print("\nFlushing traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Flushed successfully!")


if __name__ == "__main__":
    main()
