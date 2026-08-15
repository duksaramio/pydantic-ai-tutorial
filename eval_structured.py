import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from langfuse import get_client
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason, IsInstance

# 1. Load environment variables & initialize Langfuse
load_dotenv()
langfuse = get_client()
Agent.instrument_all()

ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# 2. Define Structured Output Schema
class CityDetails(BaseModel):
    city: str = Field(description="Name of the city")
    country: str = Field(description="Name of the country")
    continent: str = Field(description="Continent where the city is located")
    is_capital: bool = Field(description="Whether the city is a country capital")


# 3. Define Ollama Agent with Structured Output
model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

agent = Agent(
    model=model,
    output_type=CityDetails,
    system_prompt="You are a geographic knowledge system. Extract accurate structured city details.",
)


# 4. Custom Evaluator to Validate Field Logic
@dataclass
class ValidCityInfo(Evaluator):
    """Validates that city details contain non-empty strings and valid continents."""

    allowed_continents: tuple[str, ...] = (
        "Asia", "Africa", "North America", "South America", "Europe", "Australia", "Antarctica", "Oceania"
    )

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        if not isinstance(ctx.output, CityDetails):
            return EvaluationReason(value=False, reason="Output is not CityDetails instance")

        details: CityDetails = ctx.output
        if not details.city or not details.country:
            return EvaluationReason(value=False, reason="City or Country is empty")

        if details.continent not in self.allowed_continents:
            return EvaluationReason(
                value=False,
                reason=f"Unknown continent '{details.continent}'. Must be one of {self.allowed_continents}",
            )

        return EvaluationReason(
            value=True,
            reason=f"Valid CityDetails: {details.city}, {details.country} ({details.continent})",
        )


# 5. Define Task
async def extract_city(prompt: str) -> CityDetails:
    res = await agent.run(prompt)
    return res.output


# 6. Define and Serialize Dataset
def build_and_run_dataset():
    dataset_file = Path("city_eval_dataset.yaml")

    dataset = Dataset(
        name="city_extraction_eval",
        cases=[
            Case(
                name="tokyo_query",
                inputs="Tell me about Tokyo.",
                expected_output=CityDetails(
                    city="Tokyo", country="Japan", continent="Asia", is_capital=True
                ),
            ),
            Case(
                name="sydney_query",
                inputs="Tell me about Sydney.",
                expected_output=CityDetails(
                    city="Sydney", country="Australia", continent="Oceania", is_capital=False
                ),
            ),
            Case(
                name="cairo_query",
                inputs="Tell me about Cairo.",
                expected_output=CityDetails(
                    city="Cairo", country="Egypt", continent="Africa", is_capital=True
                ),
            ),
        ],
        evaluators=[
            IsInstance(type_name="CityDetails"),
            ValidCityInfo(),
        ],
    )

    # Save dataset to YAML (generates city_eval_dataset.yaml and city_eval_dataset_schema.json)
    print(f"Saving dataset to {dataset_file}...")
    dataset.to_file(dataset_file, custom_evaluator_types=[ValidCityInfo])

    # Reload from file with custom evaluator type registered
    print("Loading dataset from YAML...")
    loaded_dataset = Dataset.from_file(
        dataset_file, custom_evaluator_types=[ValidCityInfo]
    )

    # Run evaluation
    print(f"Running evaluation on loaded dataset with Langfuse logging (Model: {ollama_model})...\n")
    report = loaded_dataset.evaluate_sync(extract_city, max_concurrency=1)
    report.print(include_input=True, include_output=True, include_reasons=True)

    # Flush all traces to local Langfuse server
    print("\nFlushing traces to Langfuse (http://localhost:3000)...")
    langfuse.flush()
    print("Flushed successfully!")


if __name__ == "__main__":
    build_and_run_dataset()
