import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai_harness import CodeMode
from pydantic_monty import MountDir

# 1. Load environment variables from .env
load_dotenv()

# 2. Define Agent using Ollama model
ollama_model = os.getenv("OLLAMA_MODEL", "muse-glimmer")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

model = OllamaModel(
    ollama_model, provider=OllamaProvider(base_url=ollama_base_url)
)

# Prepare a local workspace directory on the host
host_workspace = Path("/tmp/agent-workspace")
host_workspace.mkdir(parents=True, exist_ok=True)
sample_file = host_workspace / "data.txt"
sample_file.write_text("item,qty,price\napple,10,1.5\nbanana,20,0.8\norange,15,1.2\n")

# Configure agent with MountDir: sandbox maps /work to host_workspace
agent = Agent(
    model=model,
    capabilities=[
        CodeMode(
            mount=MountDir(
                virtual_path="/work",
                host_path=str(host_workspace),
                mode="read-write",
            )
        )
    ],
)


def main():
    print("--- Running Code Mode Filesystem Mount Example ---")
    print(f"Host directory: {host_workspace}")
    print(f"Existing files: {[f.name for f in host_workspace.iterdir()]}")

    prompt = (
        "Read the file '/work/data.txt' and compute total revenue (qty * price). "
        "Write the summary report to '/work/summary.txt' and return the result."
    )
    result = agent.run_sync(prompt)
    print("\nResult Output:")
    print(result.output)

    summary_file = host_workspace / "summary.txt"
    if summary_file.exists():
        print(f"\nWritten Host File Content ({summary_file}):\n{summary_file.read_text()}")


if __name__ == "__main__":
    main()
