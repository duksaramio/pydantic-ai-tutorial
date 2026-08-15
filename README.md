# Pydantic AI Tutorial

A collection of practical examples and patterns for building AI agents with [Pydantic AI](https://ai.pydantic.dev/), powered by local LLMs via [Ollama](https://ollama.com/) and instrumented with [Langfuse](https://langfuse.com/) for tracing and observability.

---

## Features

- **Local LLM Execution**: Runs local models (e.g. `muse-glimmer`, `llama3.2`, `qwen2.5`) via Ollama.
- **Environment-based Configuration**: Easily configure model names, base URLs, and API credentials via `.env`.
- **Full Observability**: Integrated with Langfuse for automatic tracing of agent runs, tool calls, and LLM requests.
- **Structured Outputs**: Validate responses into typed Pydantic models.
- **Tool Calling & Dependency Injection**: Dynamic tool execution with type-safe runtime context (`RunContext`).
- **Built-in Web Chat UI**: Launch an instant browser-based chat interface.

---

## Project Structure

```text
pydantic-ai-tutorial/
├── .env.example            # Environment variables template
├── pyproject.toml          # Project metadata and dependencies
├── main.py                 # Interactive CLI chatbot with message history
├── langfuse-integration.py # Multi-turn chatbot with Langfuse tracing
├── olympics.py             # Structured output example (Pydantic model)
├── roulette_wheel.py       # Tool calling with RunContext dependencies
└── web-chat-ui.py          # Browser-based web chat UI
```

---

## Getting Started

### 1. Prerequisites

- Python `>= 3.14` (or managed via `uv`)
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- [Ollama](https://ollama.com/) installed and running locally
- *(Optional)* [Langfuse](https://langfuse.com/docs/deployment/local) running locally (e.g. via Docker on port 3000)

### 2. Installation

Clone the repository and install dependencies using `uv`:

```bash
git clone https://github.com/duksaramio/pydantic-ai-tutorial.git
cd pydantic-ai-tutorial
uv sync
```

### 3. Environment Configuration

Copy the example environment file and customize it as needed:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Ollama Configuration
OLLAMA_MODEL=muse-glimmer
OLLAMA_BASE_URL=http://localhost:11434/v1

# Langfuse Configuration (Optional / Local)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=http://localhost:3000
LANGFUSE_HOST=http://localhost:3000
```

---

## Examples

### 1. Interactive CLI Chatbot (`main.py`)
Multi-turn conversational assistant running in the terminal with continuous history.

```bash
uv run python main.py
```

### 2. Langfuse Tracing Integration (`langfuse-integration.py`)
Chatbot with automatic Langfuse instrumentation enabled via `Agent.instrument_all()`.

```bash
uv run python langfuse-integration.py
```

### 3. Structured Output (`olympics.py`)
Enforces typed structured output conforming to a Pydantic `BaseModel`.

```bash
uv run python olympics.py
```

### 4. Tool Calling & Dependencies (`roulette_wheel.py`)
Demonstrates registering custom tools and injecting runtime dependencies via `RunContext`.

```bash
uv run python roulette_wheel.py
```

### 5. Web Chat UI (`web-chat-ui.py`)
Spins up a lightweight web interface with custom agent tools using `agent.to_web()`.

```bash
uv run uvicorn web-chat-ui:app --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## License

MIT License.
