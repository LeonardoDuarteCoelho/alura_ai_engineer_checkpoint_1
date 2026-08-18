---
title: Alura AI Engineer Checkpoint 1
sdk: static
---

# Agentic Comment Moderation

An educational project for learning how to build agentic AI workflows with
LangGraph, shared state, external tools, and human-in-the-loop decisions.

## Project purpose

The project simulates comment moderation for an online course platform. A high
volume of student comments makes it difficult for human moderators to review
everything manually, so the system uses several specialized agents to assist
with the process while keeping the final decision under human control.

The intended workflow is:

1. An Analyzer classifies a student comment as positive, neutral, or potentially
   problematic.
2. A Policy Checker uses Tavily to find community-policy information relevant to
   a potentially problematic comment.
3. A Reviewer combines the analysis and policy evidence to recommend whether the
   comment should be approved, removed, or edited.
4. A human moderator reviews the recommendation before any critical action is
   executed.

The project focuses on understanding:

- Agent orchestration with LangGraph
- Shared state using `TypedDict`
- Prompt design and structured LLM output
- Local LLM inference through LM Studio
- External tool use through Tavily
- Conditional graph routing
- Checkpointing and human-in-the-loop interruptions

## Current implementation

The current code in `src/main.py` contains:

- The shared `AgentState` definition
- An Analyzer using the local LM Studio model
- A Tavily-based Policy Checker
- A Reviewer that produces a moderation recommendation
- JSON parsing and basic output validation

The LangGraph construction and human-in-the-loop execution are the next stages
of the project.

## Technologies

- Python
- LangChain
- LangGraph
- LM Studio with an OpenAI-compatible local API
- Tavily Search API
- SQLite checkpointing (planned for the graph stage)

## Installation

Create or activate a virtual environment, then install the dependencies:

```powershell
pip install langchain langgraph langchain-openai tavily-python python-dotenv aiosqlite
```

## Environment variables

Create a `.env` file in the project root. Do not commit this file to Git.

```env
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
TAVILY_API_KEY=tvly-your-key-here
```

`OPENAI_BASE_URL` points the LangChain client to LM Studio rather than a cloud
endpoint. `TAVILY_API_KEY` is a real Tavily credential because Tavily is an
external search service.

## Running the current prototype

Start the LM Studio server, make sure a model is available, and run:

```powershell
.\venv\Scripts\python.exe .\src\main.py
```

The current test case sends a problematic comment through the Analyzer, Policy
Checker, and Reviewer, then prints the state updates returned by each agent.

## Project structure

```text
checkpoint_1/
|-- src/
|   `-- main.py
|-- .env                 # Local secrets; not committed
|-- instructions.txt     # Overall project brief
|-- step_1.txt           # Initial agent workflow requirements
|-- step_2.txt           # Human-in-the-loop requirements
`-- README.md
```

This project is intended as a learning exercise. The agents provide
recommendations, while a human moderator remains responsible for the final
moderation decision.
