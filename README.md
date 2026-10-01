# LangGraph Agent with Web Search

A simple LangGraph agent that can search the web using DuckDuckGo, exposed via a FastAPI server.

## Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- OpenAI API key

## Setup

1. Install dependencies:

```bash
poetry install
```

2. Create a `.env` file from the example:

```bash
cp .env.example .env
```

3. Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_actual_api_key
```

## Running the API

Start the FastAPI server:

```bash
poetry run uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### POST /chat

Send a message to the agent and receive a response.

**Request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the latest news about AI?"}'
```

**Response:**

```json
{
  "response": "Based on my search, here are the latest developments in AI..."
}
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```


## Project Structure

```
se-interview/
├── pyproject.toml   # Poetry dependencies
├── .env.example     # Environment variable template
├── README.md        # This file
├── agent.py         # LangGraph agent implementation
└── api.py           # FastAPI server
```

## How It Works

1. The agent receives a user message via the `/chat` endpoint
2. It calls GPT-4o with the message and available tools (DuckDuckGo search)
3. If the LLM decides to search, it executes the search and feeds results back
4. The loop continues until the LLM provides a final response
5. The response is returned to the user

## Trip Budget Estimator

The added `estimate_trip_budget` tool performs simple Python arithmetic. GPT-4o
chooses when to call it and extracts the arguments. It uses the same LangGraph
loop as DuckDuckGo, with no new dependencies or API keys.

### Inputs

All costs are supplied estimates in **USD for the entire group**. All inputs are
required; explicitly use zero when a cost does not apply.

| Input | Meaning |
| --- | --- |
| `days` | Trip days for food and activities |
| `nights` | Hotel nights, supplied separately from days |
| `travelers` | People splitting the total equally |
| `hotel_per_night` | Group hotel cost per night |
| `food_per_day` | Group food budget per day |
| `activities_per_day` | Group budget for all activities combined per day |
| `transportation_total` | Group transportation cost for the whole trip |

### Design decisions

Hotel cost is nights times the nightly rate. Food and activity costs are days
times their daily budgets. These amounts plus transportation give the total;
dividing by travelers or days gives the per-person or per-day estimate.

The tool returns JSON text containing `currency`, `breakdown`, `total`,
`per_person`, and `per_day`, with amounts rounded to two decimal places. JSON
preserves structured fields while fitting the starter's existing tool-message
handling. The model then explains the estimate in its final response.

All original functions and the original system prompt are unchanged. The tool's
description explains when to use it and what inputs it requires. Integration adds
only the tool definition, imports, and its entry in the existing tools list.

This intentionally small demonstration assumes valid inputs: positive days and
travelers, and nonnegative nights and costs. Typed inputs provide basic schema
validation, but there are no custom range checks or error recovery. For example,
zero travelers would cause division by zero. The tool description asks the model to clarify
missing or ambiguous information; this is guidance, not guaranteed validation.

The tool does not fetch prices or itemize individual activities. Only supplied
costs are included. Python handles arithmetic so we can separately inspect the
model's tool choice, arguments, and explanation. No conversation memory is added;
after clarification, resubmit the complete request with the missing information.

### Manual demo scenarios

These are expected behaviors to verify with live requests and, later, traces:

1. **Budget:** "Estimate a 4-day, 3-night San Diego trip for two people. All costs
   are in USD for both people combined: hotel $300/night, food $150/day,
   activities $100/day, and transportation $250 total."
   Expected: budget tool; total $2,150, per person $1,075, per day $537.50.
2. **Search:** "Find the current opening hours of the San Diego Zoo."
   Expected: DuckDuckGo.
3. **Neither:** "Give me three general tips for packing light."
   Expected: direct answer.
4. **Incomplete:** "Estimate a 4-day San Diego trip for two people. Hotel is
   $300/night and activities are $100/day."
   Expected: ask about nights, cost basis, food, and transportation.

## Phoenix tracing

Start Phoenix locally in a separate terminal:

```bash
uvx arize-phoenix serve
```

The API registers the `se-interview` project with Phoenix during startup.
`arize-phoenix-otel` configures OpenTelemetry and exports spans to the local
Phoenix collector automatically. With the current setup, traces use the local
gRPC collector on port 4317 and the Phoenix UI remains at `http://localhost:6006`.
`openinference-instrumentation-langchain` automatically captures the LangGraph
workflow, LangChain LLM calls, and tool calls using OpenInference conventions.

No Phoenix API key is needed for a local instance. Set
`PHOENIX_COLLECTOR_ENDPOINT` only when sending traces to a different Phoenix
server. Restart the API after installing the tracing dependencies, then make
requests through `/chat` and inspect the `se-interview` project in Phoenix.
