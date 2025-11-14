# Cortex‑R: Multi-MCP Agent App 🤖🧠

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://www.python.org/) [![LLM](https://img.shields.io/badge/LLM-Ready-brightgreen?logo=openai)](https://platform.openai.com/) [![RAG](https://img.shields.io/badge/RAG-Enabled-orange)]() [![Heuristics](https://img.shields.io/badge/Heuristics-Input%2FOutput%20Validation-yellow)]() [![MCP](https://img.shields.io/badge/MCP-Multi--Server-purple)]()

## Overview

Cortex‑R is a reasoning-driven, multi-tool AI agent. It blends structured prompting, heuristic validation, and memory to dispatch tasks to MCP-enabled tool servers (math, documents, web search). The agent runs from the command line (`uv run agent.py`), orchestrating perception → planning → execution loops until it reaches a final answer.

**Key Features**

- **Multi-MCP Server Orchestration**: Math, document, web, and memory servers
- **Multi-step, tool-aware reasoning** through `core/loop.AgentLoop`
- **Strategic prompt engineering**: Conservative, Exploratory, Fallback, etc.
- **Heuristic guardrails**: For safe inputs, prompts, LLM calls and tool usage
- **Historical memory ingestion**: So past sessions influence new plans
- **Memory and context management**: For stepwise, context-aware problem solving
- **Flexible config**: via `config/profiles.yaml` and `config/models.json`
- **RAG & Document Search**: FAISS-based semantic search over local and web docs
- **Extensible Tooling**: Add new tools and servers with minimal code changes
---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or pip for dependency management
- MCP-compatible tool servers (already present under `mcp_server_*.py`)
- Optional: [Ollama](https://ollama.ai) for local text/embedding models (expects `http://localhost:11434`)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Session-9

# Using uv (recommended)
uv sync

# or with pip / venv
python -m venv .venv
. .venv/Scripts/activate            # or source .venv/bin/activate
pip install -r requirements.txt     # if provided
```

### Directory Structure

```
Session-9/
├── agent.py                    # Main entry point
├── models.py                   # Pydantic models for MCP tools
├── pyproject.toml              # Project dependencies (uv)
├── uv.lock                     # Dependency lock file
│
├── config/                     # Configuration files
│   ├── profiles.yaml          # Agent strategy, memory, MCP server configs
│   └── models.json            # LLM model mappings (Gemini, Ollama, etc.)
│
├── core/                       # Core agent orchestration
│   ├── context.py             # AgentContext, AgentProfile
│   ├── loop.py                # Main AgentLoop (perception → planning → execution)
│   ├── session.py             # MCP client/dispatcher (MultiMCP)
│   └── strategy.py            # Prompt path selection, decision logic
│
├── modules/                    # Agent modules
│   ├── action.py              # Sandbox execution (run_python_sandbox)
│   ├── decision.py            # Plan generation (generate_plan)
│   ├── memory.py              # MemoryManager (session + historical)
│   ├── model_manager.py       # LLM wrapper (Gemini, Ollama)
│   ├── perception.py          # Intent extraction (extract_perception)
│   ├── tools.py               # Tool summarization utilities
│   └── mcp_server_memory.py   # MCP server for memory queries
│
├── prompts/                    # LLM prompt templates
│   ├── perception_prompt.txt
│   ├── decision_prompt_conservative.txt
│   ├── decision_prompt_conservative_opt2.txt
│   ├── decision_prompt_exploratory_parallel.txt
│   └── decision_prompt_exploratory_sequential.txt
│
├── mcp_server_*.py            # MCP tool servers (math, documents, web)
│   ├── mcp_server_1.py        # Math tools (add, subtract, fibonacci, etc.)
│   ├── mcp_server_2.py        # Document tools (search, extract PDF, markdown)
│   └── mcp_server_3.py        # Web search tools (DuckDuckGo)
│
├── memory/                     # Session memory storage (date-based)
│   └── YYYY/MM/DD/session/    # Individual session JSON files
│
├── history/                    # Historical transcripts (if enabled)
│   └── transcripts.jsonl     # Shared transcript log
│
├── faiss_index/               # FAISS vector index for document search
│   ├── index.bin
│   ├── metadata.json
│   └── doc_index_cache.json
│
├── documents/                  # Source documents for indexing/search
│   ├── *.pdf, *.md, *.txt, *.docx
│   └── images/
│
├── heuristics_lib/            # Input/output validation heuristics
│   ├── heuristics_validators.py
│   └── test_heuristics.py
│
├── tests/                      # Unit tests
│   ├── test_prompt_performance.py
│   └── prompt_benchmark_samples.json
│
└── images/                     # Documentation assets
    └── classdiag.png           # High level class diagram
    └── sequencediag.png        # High level sequence diagram
    └── fibonacciquerylog.txt   # Sample log 1
    └── fibonacciquerylog.txt   # Sample log 1
    └── fibonacciquerylog.txt   # Sample log 1
    
```

### Running the Agent

```bash
uv run agent.py
```

You’ll be greeted with `🧠 Cortex-R Agent Ready`. Interact via the CLI prompt:

- Type a task (e.g., `Summarize this page: https://theschoolof.ai/`).
- Type `new` to reset the session but keep the process running.
- Type `exit` to quit.

---

## Configuration

### Model & Tool Profiles

- `config/profiles.yaml` holds strategy, memory, and MCP server definitions.
- `config/models.json` maps logical model keys (`gemini`, `llama3:8b`, `nomic`, …) to actual providers (Gemini API, Ollama, etc.).

Switch to a local model by editing:

```yaml
llm:
  text_generation: llama3:8b
  embedding: nomic  # uses Ollama nomic-embed-text when enabled below
```

### Memory & Historical Index

```yaml
memory:
  memory_service: true
  summarize_tool_results: true
  tag_interactions: true
  storage:
    base_dir: "memory"
  historical_index:
    enabled: true
    path: "faiss_index/history"
    max_items: 50
    file_extensions: [".json", ".jsonl", ".txt"]
    transcripts:
      enabled: true
      filename: "transcripts.jsonl"
```

- Set `enabled: true` to ingest prior transcripts stored under `faiss_index/history`.
- Flip `transcripts.enabled` to `true` so each session appends to the shared JSONL log.
- At startup the agent logs `Historical index active: loaded N item(s) from …` so you can verify ingestion.

---

## Enhancements

### 1. Heuristics Library
- Implementation: `heuristics_lib/heuristics_validators.py`
- Unit tests: `heuristics_lib/test_heuristics.py`

Run:

```bash
python -m unittest heuristics_lib.test_heuristics
```

### 2. Prompt Optimization & Test Harness
- Optimized prompt: `prompts/decision_prompt_conservative_opt2.txt`
- Evaluation suite: `tests/test_prompt_performance.py`
- Judge shim with optional real LLM checks: `tools/llm_judge.py`

```bash
python -m unittest tests.test_prompt_performance
python tests/test_prompt_performance.py --prompt both \
    --llm-judge-command "python tools/llm_judge.py --pretty"
```

Add `--use-real-llm` to the judge command to route comparisons through Ollama.

### 3. Historical Conversation Indexing
- Historical transcript ingestion controlled via `memory.historical_index` in `profiles.yaml`.
- Each run can append to `faiss_index/history/transcripts.jsonl` when `transcripts.enabled` is true.
- JSON/JSONL entries are normalized into `MemoryItem`s tagged `historical` and become part of the planning context.

---

## Architecture & Design

### Class Diagram
<a href="images/classdiag.png" target="_blank">
  <img src="images/classdiag.png" alt="Class diagram" width="80%" />
</a>

> 🔍 **Note:** Click the diagram to open the full-scale version.

### Important Classes

```
================================================================================
CORTEX-R AGENT - CODE EXPLANATION
================================================================================

This section provides brief explanations of classes and functions in the
core/ and modules/ directories.

================================================================================
CORE PACKAGE
================================================================================

Classes: core/context.py

1. StrategyProfile (BaseModel)
   - Pydantic model representing agent strategy configuration
   - Used to validate and store strategy settings from profiles.yaml

2. AgentProfile
   - Loads and stores agent configuration from profiles.yaml
   
3. AgentContext
   - Main context class holding all session state, user input, memory, and strategies
     sets up memory manager with historical config, logs session start

--------------------------------------------------------------------------------
Classes: core/loop.py

1. AgentLoop
   - Main orchestration loop for perception → planning → execution cycle
     * Iterates through max_steps
     * For each step: runs perception, selects tools, generates plan, executes in sandbox
     * Handles FINAL_ANSWER, FURTHER_PROCESSING_REQUIRED, and error cases
     * Manages lifelines (retries) per step
     * Persists transcript on completion
     * Returns dict with status and result

--------------------------------------------------------------------------------
Classes: core/session.py

1. MCP
   - Lightweight wrapper for one-time MCP tool calls using stdio transport
   - Each call spins up a new subprocess and terminates cleanly

2. MultiMCP
   - Stateless dispatcher for multiple MCP servers
   - Discovers tools from multiple servers, reconnects per tool call

--------------------------------------------------------------------------------
Classes: core/strategy.py

1. Key functions 
   This class has these key functions which are for selecting the right decision prompt, deciding the next action, plan generation

================================================================================
MODULES PACKAGE
================================================================================

Classes: modules/action.py
--------------------------------------------------------------------------------

1. ToolCallResult (BaseModel)
   - Pydantic model for tool call results

1. run_python_sandbox(code, dispatcher)
   - Executes dynamically generated solve() function in isolated sandbox
   - Creates fresh module scope with limited built-ins (json, re)
   - Injects SandboxMCP wrapper to limit tool calls (MAX_TOOL_CALLS_PER_PLAN = 5)
   - Compiles and executes code, extracts solve() function
   - Handles both sync and async solve() functions
   - Returns formatted result string or error message
   - Catches exceptions and returns formatted error

--------------------------------------------------------------------------------
Classes: modules/decision.py

1. generate_plan(user_input, perception, memory_items, tool_descriptions, 
                 prompt_path, step_num, max_steps)
   - Generates full solve() function plan for the agent
   - Formats memory items as text for prompt context
   - Loads prompt template and formats with tool descriptions and user input
   - Adds delay to avoid rate limiting
   - Calls LLM to generate plan
   - Extracts code from markdown code blocks if present
   - Validates that result contains solve() function
   - Returns plan string or error message

--------------------------------------------------------------------------------
Classes: modules/memory.py

1. MemoryItem (BaseModel)
   - Pydantic model representing a single memory entry for a session

2. MemoryManager
   - Manages session memory (read/write/append) and optional historical records
   - load(): Loads memory items from JSON file if exists
   - save(): Saves current memory items to JSON file
   - _load_historical_items(directory, max_items, allowed_extensions):
     * Recursively walks directory for historical memory files
     * Loads JSON, JSONL, and text files
     * Limits to max_items if specified
     * Returns list of MemoryItem objects
   - _normalize_json_entries(data, source_file, max_items):
     * Normalizes various JSON structures into MemoryItem objects
     * Handles nested items arrays, session_id fields
     * Creates MemoryItem with appropriate metadata
     * Returns list of normalized MemoryItem objects
   - add(item): Appends MemoryItem and saves to file
   - add_tool_call(tool_name, tool_args, tags): Creates and adds tool_call MemoryItem
   - add_tool_output(tool_name, tool_args, tool_result, success, tags): 
     Creates and adds tool_output MemoryItem
   - add_final_answer(text): Creates and adds final_answer MemoryItem
   - find_recent_successes(limit): Finds tool names that succeeded recently
   - add_tool_success(tool_name, success): Updates success status of last matching tool
   - get_session_items(): Returns combined current and historical memory items
   - get_historical_items(): Returns only historical memory items
   - persist_transcript(): Appends session transcript to JSONL file if enabled

--------------------------------------------------------------------------------
Classes: modules/model_manager.py
--------------------------------------------------------------------------------

1. ModelManager
   - Manages LLM model interactions (Gemini, Ollama)
     * Loads models.json and profiles.yaml
     * Determines model type (gemini or ollama)
     * Initializes Gemini client if needed
   - generate_text(prompt): Async method to generate text from LLM
     * Routes to _gemini_generate() or _ollama_generate() based on model_type
   - _gemini_generate(prompt): Generates text using Google Gemini API
     * Safely extracts response text from various response formats
   - _ollama_generate(prompt): Generates text using Ollama local API
     * Sends POST request to Ollama endpoint
     * Returns response text

--------------------------------------------------------------------------------
Classes: modules/perception.py

1. PerceptionResult (BaseModel)
   - Pydantic model for perception extraction results
   - Represents LLM's understanding of user intent and tool selection

Functions:
1. extract_perception(user_input, mcp_server_descriptions)
   - Extracts perception details and selects relevant MCP servers
   - Formats server descriptions for prompt
   - Calls LLM with perception prompt template
   - Parses JSON response into PerceptionResult
   - Falls back to selecting all servers if parsing fails
   - Returns PerceptionResult object

2. run_perception(context, user_input)
   - Clean wrapper to call perception from AgentContext
   - Extracts user_input and mcp_server_descriptions from context
   - Calls extract_perception() and returns result

--------------------------------------------------------------------------------
Classes: modules/tools.py
--------------------------------------------------------------------------------

1. extract_json_block(text)
   - Extracts JSON block from markdown code fence (```json ... ```)
   - Returns extracted JSON string or original text if no block found

2. summarize_tools(tools)
   - Generates string summary of tools for LLM prompt injection
   - Format: "- tool_name: description" (one per line)
   - Returns formatted string

3. filter_tools_by_hint(tools, hint)
   - Filters tools based on tool_hint from perception
   - Performs case-insensitive substring matching on tool names
   - Returns filtered list or all tools if no hint or no matches

4. get_tool_map(tools)
   - Returns dictionary mapping tool_name → tool object for fast lookup
   - Useful for finding tools by name

5. tool_expects_input(tool_name)
   - Checks if a tool expects input wrapped in 'input' parameter
   - Examines tool parameters structure
   - Returns True if top-level parameter is just 'input'

6. load_prompt(path)
   - Loads prompt template from file
   - Returns file contents as string

--------------------------------------------------------------------------------
Classes: modules/mcp_server_memory.py
--------------------------------------------------------------------------------

1. SearchInput (BaseModel)
   - Pydantic model for search input
   - Field: query (string)

2. MemoryStore
   - Manages memory storage and retrieval for MCP server
   - __init__(): Initializes with base memory directory from config
   - load_session(session_id): Sets current session ID
   - _list_all_memories(): 
     * Recursively loads all memory files from date-based directory structure
     * Returns list of all memory entries
   - _get_conversation_flow(conversation_id):
     * Gets sequence of interactions in a conversation
     * Returns structured conversation flow with queries, intents, tool calls,
       final answers, and timestamps

Functions:

1. handle_shutdown(signum, frame)
   - Global shutdown handler for signal processing
   - Exits cleanly on SIGINT/SIGTERM

2. get_current_conversations(input)
   - MCP tool: Gets current session interactions
   - Finds most recent session file for today
   - Returns session_id and interactions (excluding run_metadata)

3. search_historical_conversations(input)
   - MCP tool: Searches conversation memory
   - Searches all historical memories for query terms
   - Filters results to stay within word limit (10000 words)
   - Returns matching conversations with user_query, final_answer, timestamp, intent


```


### Runtime Sequence

<a href="images/sequencediag.png" target="_blank">
  <img src="images/sequencediag.png" alt="Sequence diagram flow" width="80%" />
</a>

> 🔍 **Note:** Click the diagram to open the full-scale version.

### Key Modules

- `core/context.py`: wraps `AgentProfile`, `MemoryManager`, and session metadata.
- `core/loop.py`: orchestrates perception → planning → execution with heuristics and error handling.
- `modules/perception.py`: turns user input into intent + server/tool hints.
- `modules/decision.py`: prompts the LLM to output an executable `solve()` plan.
- `modules/action.py`: sandbox with tool call limits and dispatcher injection.
- `modules/memory.py`: stores live memory, loads historical transcripts, persists new sessions.

---

## Testing & Quality

- Prompt parity: `python -m unittest tests.test_prompt_performance`
- Heuristics: `python -m unittest heuristics_lib.test_heuristics`
- Manual CLI runs (`uv run agent.py`) are encouraged; logs highlight follow-up heuristics and tool errors.

---

## Application Output Logs

The agent produces detailed console logs showing the perception → planning → execution flow. Below is a sample log snippet from a typical session:

### Log Snippet 1 (Product of six fibonacci numbers starting from 2nd fibonacci number)

```
🧠 Cortex-R Agent Ready
🧑 What do you want to solve today? → Find the product of six fibonacci numbers starting from second fibonacci number
🔁 Step 1/10 starting...
Waiting for 3 seconds before generating perception...
[15:41:55] [perception] Raw output: """json
{
  "intent": "Calculate the product of six Fibonacci numbers",
  "entities": ["six", "Fibonacci numbers", "product"],
  "tool_hint": "python sandbox",
  "selected_servers": ["math"]
}
"""
result {'intent': 'Calculate the product of six Fibonacci numbers', 'entities': ['six', 'Fibonacci numbers', 'product'], 'tool_hint': 'python sandbox', 'selected_servers': ['math']}
[perception] intent='Calculate the product of six Fibonacci numbers' entities=['six', 'Fibonacci numbers', 'product'] tool_hint='python sandbox' tags=[] selected_servers=['math']
Using conservative prompt file:  prompts/decision_prompt_conservative_opt2.txt
Waiting for 3 seconds before generating decision...
[15:42:00] [plan] LLM output: ```python
import json
async def solve():
    """fibonacci_numbers: Generate first n Fibonacci numbers. Usage: input={"input": {"n": 10}}"""
    input_payload = {"input": {"n": 7}}
    result = await mcp.call_tool('fibonacci_numbers', input_payload)
    fibonacci_sequence = json.loads(result.content[0].text)["result"]

    product = 1
    for i in range(1, 7):
        product *= fibonacci_sequence[i]

    return f"FINAL_ANSWER: {product}"

[plan] import json
async def solve():
    """fibonacci_numbers: Generate first n Fibonacci numbers. Usage: input={"input": {"n": 10}}"""
    input_payload = {"input": {"n": 7}}
    result = await mcp.call_tool('fibonacci_numbers', input_payload)
    fibonacci_sequence = json.loads(result.content[0].text)["result"]

    product = 1
    for i in range(1, 7):
        product *= fibonacci_sequence[i]

    return f"FINAL_ANSWER: {product}"
[loop] Detected solve() plan — running sandboxed...
[action] 🔍 Entered run_python_sandbox()

💡 Final Answer: 240

```

**Full log file**: Click [`images\fibonacciquerylog.txt`](images/fibonacciquerylog.txt) for a complete example session log.

### Log Snippet 2 (News Summarization from CNN website)

```
🧠 Cortex-R Agent Ready
🧑 What do you want to solve today? → Summarize this news article for me in 100 words https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote

[11:26:04] [memory] ✅ Transcript enabled: will write to history/transcripts.jsonl
[11:26:04] [memory] Historical index active: loaded 10 item(s) from history

🔁 Step 1/10 starting...
Waiting for 3 seconds before generating perception...
[15:31:20] [perception] Raw output: """json
{
  "intent": "Summarize a news article",
  "entities": ["news article", "https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote", "100 words"],
  "tool_hint": "Summarization tool",
  "selected_servers": ["websearch", "documents"]
}
"""
result {'intent': 'Summarize a news article', 'entities': ['news article', 'https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote', '100 words'], 'tool_hint': 'Summarization tool', 'selected_servers': ['websearch', 'documents']}
[perception] intent='Summarize a news article' entities=['news article', 'https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote', '100 words'] tool_hint='Summarization tool' tags=[] selected_servers=['websearch', 'documents']
Using conservative prompt file:  prompts/decision_prompt_conservative_opt2.txt
Waiting for 3 seconds before generating decision...
[15:31:24] [plan] LLM output: """python
import json
async def solve():
    """Convert Webpage Usage: input={{"input": {{"url": "https://example.com"}}}}"""
    input_payload = {"input": {"url": "https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote"}}
    result = await mcp.call_tool('convert_webpage_url_into_markdown', input_payload)
    return f"FURTHER_PROCESSING_REQUIRED: {result}"
"""
[plan] import json
async def solve():
    """Convert Webpage Usage: input={{"input": {{"url": "https://example.com"}}}}"""
    input_payload = {"input": {"url": "https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote"}}
    result = await mcp.call_tool('convert_webpage_url_into_markdown', input_payload)
    return f"FURTHER_PROCESSING_REQUIRED: {result}"
[loop] Detected solve() plan — running sandboxed...
[action] 🔍 Entered run_python_sandbox()
[15:31:26] [loop] 📨 Forwarding intermediate result to next step:

.
.
.

[15:31:26] [loop] 🔁 Continuing based on FURTHER_PROCESSING_REQUIRED — Step 1 continues...
🔁 Step 2/10 starting...
Waiting for 3 seconds before generating perception...
[15:31:31] [perception] Raw output: """json
{
  "intent": "Summarize a news article.",
  "entities": [
    "news article",
    "https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote",
    "100 words"
  ],
  "tool_hint": "Summarization tool",
  "selected_servers": ["documents"]
}
"""
result {'intent': 'Summarize a news article.', 'entities': ['news article', 'https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote', '100 words'], 'tool_hint': 'Summarization tool', 'selected_servers': ['documents']}
[perception] intent='Summarize a news article.' entities=['news article', 'https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote', '100 words'] tool_hint='Summarization tool' tags=[] selected_servers=['documents']
Using conservative prompt file:  prompts/decision_prompt_conservative_opt2.txt
Waiting for 3 seconds before generating decision...
[15:31:37] [plan] LLM output: """python
import json
async def solve():
    """Convert webpage to markdown Usage: input={{"input": {{"url": "https://example.com"}}}}"""
    input_payload = {"input": {"url": "https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote"}}
    result = await mcp.call_tool('convert_webpage_url_into_markdown', input_payload)
    content = result.content[0].text
    markdown = json.loads(content)["markdown"]
    summary = "President Trump signed a funding package to end the longest government shutdown in history. The House voted 222-209 to pass a deal that keeps the government running through January and funds key agencies for the remainder of fiscal year 2026. The agreement reverses mass federal layoffs and resumes paychecks and critical services. Trump hailed it as a victory over Democrats. The deal guarantees a Senate vote on Obamacare subsidies, but its success is unlikely. Speaker Johnson vowed to fix a contentious provision allowing senators to retroactively sue the Department of Justice."
    return f"FINAL_ANSWER: {summary}"
"""
[plan] import json
async def solve():
    """Convert webpage to markdown Usage: input={{"input": {{"url": "https://example.com"}}}}"""
    input_payload = {"input": {"url": "https://edition.cnn.com/2025/11/12/politics/government-shutdown-funding-bill-house-vote"}}
    result = await mcp.call_tool('convert_webpage_url_into_markdown', input_payload)
    content = result.content[0].text
    markdown = json.loads(content)["markdown"]
    summary = "President Trump signed a funding package to end the longest government shutdown in history. The House voted 222-209 to pass a deal that keeps the government running through January and funds key agencies for the remainder of fiscal year 2026. The agreement reverses mass federal layoffs and resumes paychecks and critical services. Trump hailed it as a victory over Democrats. The deal guarantees a Senate vote on Obamacare subsidies, but its success is unlikely. Speaker Johnson vowed to fix a contentious provision allowing senators to retroactively sue the Department of Justice."
    return f"FINAL_ANSWER: {summary}"
[loop] Detected solve() plan — running sandboxed...
[action] 🔍 Entered run_python_sandbox()

💡 Final Answer: President Trump signed a funding package to end the longest government shutdown in history. The House voted 222-209 to pass a deal that keeps the government running through January and funds key agencies for the remainder of fiscal year 2026. The agreement reverses mass federal layoffs and resumes paychecks and critical services. Trump hailed it as a victory over Democrats. The deal guarantees a Senate vote on Obamacare subsidies, but its success is unlikely. Speaker Johnson vowed to fix a contentious provision allowing senators to retroactively sue the Department of Justice.
```

**Full log file**: Click [`images/newssummarizelog.txt`](images/newssummarizelog.txt) for a complete example session log.

### Log Snippet 3 (Query on Tesla based on the uploaded documents)

```
🧠 Cortex-R Agent Ready
🧑 What do you want to solve today? → What are Elon Musk's diverse array of interests with respect to Tesla? Use local documents and summarize.

🔁 Step 1/10 starting...
Waiting for 3 seconds before generating perception...
[15:25:03] [perception] Raw output: """json
{
  "intent": "Summarize Elon Musk's interests related to Tesla using local documents.",
  "entities": ["Elon Musk", "Tesla", "interests", "local documents"],
  "tool_hint": "document summarization",
  "selected_servers": ["documents"]
}
"""
result {'intent': "Summarize Elon Musk's interests related to Tesla using local documents.", 'entities': ['Elon Musk', 'Tesla', 'interests', 'local documents'], 'tool_hint': 'document summarization', 'selected_servers': ['documents']}
[perception] intent="Summarize Elon Musk's interests related to Tesla using local documents." entities=['Elon Musk', 'Tesla', 'interests', 'local documents'] tool_hint='document summarization' tags=[] selected_servers=['documents']
Using conservative prompt file:  prompts/decision_prompt_conservative_opt2.txt
Waiting for 3 seconds before generating decision...
[15:25:09] [plan] LLM output: """python
import json
async def solve():
    """Search documents to get relevant extracts. Usage: input={"input": {"query": "your query"}} result = await mcp.call_tool('search_stored_documents', input)"""
    input_payload = {"input": {"query": "Elon Musk's interests related to Tesla"}}
    result = await mcp.call_tool('search_stored_documents', input_payload)
    return f"FURTHER_PROCESSING_REQUIRED: {result}"
"""
[plan] import json
async def solve():
    """Search documents to get relevant extracts. Usage: input={"input": {"query": "your query"}} result = await mcp.call_tool('search_stored_documents', input)"""
    input_payload = {"input": {"query": "Elon Musk's interests related to Tesla"}}
    result = await mcp.call_tool('search_stored_documents', input_payload)
    return f"FURTHER_PROCESSING_REQUIRED: {result}"
[loop] Detected solve() plan — running sandboxed...
[action] 🔍 Entered run_python_sandbox()
[15:25:13] [loop] 📨 Forwarding intermediate result to next step:
Original user task: What are Elon Musk's diverse array of interests with respect to Tesla? Use local documents and summarize.

Your last tool produced this result:

meta=None content=[TextContent(type='text', text='## **AUGUST 2014 ** # **TESLA MOTORS: ** **INTELLECTUAL PROPERTY, OPEN INNOVATION, ** **AND THE CARBON CRISIS ** **Image:** A red Tesla Roadster convertible drives along a paved road with a driver visible. The car is positioned slightly angled, showcasing its sleek design. In the background, a vast landscape of wind turbines stretches across a desert environment under a clear blue sky. ## **DR MATTHEW RIMMER** AUSTRALIAN RESEARCH COUNCIL FUTURE FELLOW ASSOCIATE PROFESSOR THE AUSTRALIAN NATIONAL UNIVERSITY COLLEGE OF LAW The Australian National University College of Law, Canberra, ACT, 0200 Work Telephone Number: (02) 61254164 E-Mail Address: drmatthewrimmer@gmail.com 1 ----- ## **Introduction ** Tesla Motors is an innovative United States manufacturer of electric vehicles. In its annual report for 2012, the company summarizes its business operations: We design, develop, manufacture and sell high-performance fully electric vehicles and advanced electric vehicle powertrain components. We own our sales and service network and have operationally structured our business in a manner that we believe will enable us to rapidly develop and launch advanced electric vehicles and technologies. We believe our vehicles, electric vehicle engineering expertise, and operational structure differentiates us from incumbent automobile manufacturers. We are the first company to commercially produce a federally-compliant electric vehicle, the Tesla Roadster, which achieves a market-leading range on a single charge combined with attractive design, driving performance and zero tailpipe emissions. As of December 31, 2012, we had delivered approximately 2,450 Tesla Roadsters to customers in over 30 countries. While we have concluded the production run of the Tesla Roadster, its proprietary electric vehicle powertrain system is the foundation of our business. We modified this system for our Model S sedan and plan to continue to enhance it for use in our future 

.
.
.

result {'intent': "Summarize Elon Musk's interests related to Tesla using local documents.", 'entities': ['Elon Musk', 'Tesla', 'interests', 'local documents'], 'tool_hint': 'The document content seems relevant, potentially needs further summarization.', 'selected_servers': ['documents']}
[perception] intent="Summarize Elon Musk's interests related to Tesla using local documents." entities=['Elon Musk', 'Tesla', 'interests', 'local documents'] tool_hint='The document content seems relevant, potentially needs further summarization.' tags=[] selected_servers=['documents']
Using conservative prompt file:  prompts/decision_prompt_conservative_opt2.txt
Waiting for 3 seconds before generating decision...
[15:27:21] [plan] LLM output: """python
import json
async def solve():
    """search_stored_documents: Search documents to get relevant extracts. Usage: input={"input": {"query": "your query"}} result = await mcp.call_tool('search_stored_documents', input)
- convert_webpage_url_into_markdown: Return clean webpage content without Ads, and clutter. Usage: input={{"input": {{"url": "https://example.com"}}}} result = await mcp.call_tool('convert_webpage_url_into_markdown', input)
- extract_pdf: Convert PDF to markdown. Usage: input={"input": {"file_path": "documents/sample.pdf"} } result = await mcp.call_tool('extract_pdf', input)"""
    query = "Elon Musk's interests with respect to Tesla"
    input_payload = {"input": {"query": query}}
    result = await mcp.call_tool('search_stored_documents', input_payload)

    # Extracting information from the provided document excerpts.
    interests = [
        "product development and design (Tesla Roadster, Model S, Model X)",
        "impact of disruptive technologies upon established business models",
        "sustainable energy economy and electric vehicles",
        "clean technologies to address climate change",
        "spacecraft development (SpaceX)",
        "solar power systems (SolarCity)",
    ]
    summary = f"Elon Musk's interests related to Tesla are diverse and include: {', '.join(interests)}."
    return f"FINAL_ANSWER: {summary}"
"""
[plan] import json
async def solve():
    """search_stored_documents: Search documents to get relevant extracts. Usage: input={"input": {"query": "your query"}} result = await mcp.call_tool('search_stored_documents', input)
- convert_webpage_url_into_markdown: Return clean webpage content without Ads, and clutter. Usage: input={{"input": {{"url": "https://example.com"}}}} result = await mcp.call_tool('convert_webpage_url_into_markdown', input)
- extract_pdf: Convert PDF to markdown. Usage: input={"input": {"file_path": "documents/sample.pdf"} } result = await mcp.call_tool('extract_pdf', input)"""
    query = "Elon Musk's interests with respect to Tesla"
    input_payload = {"input": {"query": query}}
    result = await mcp.call_tool('search_stored_documents', input_payload)

    # Extracting information from the provided document excerpts.
    interests = [
        "product development and design (Tesla Roadster, Model S, Model X)",
        "impact of disruptive technologies upon established business models",
        "sustainable energy economy and electric vehicles",
        "clean technologies to address climate change",
        "spacecraft development (SpaceX)",
        "solar power systems (SolarCity)",
    ]
    summary = f"Elon Musk's interests related to Tesla are diverse and include: {', '.join(interests)}."
    return f"FINAL_ANSWER: {summary}"
[loop] Detected solve() plan — running sandboxed...
[action] 🔍 Entered run_python_sandbox()

💡 Final Answer: Elon Musk's interests related to Tesla are diverse and include: product development and design (Tesla Roadster, Model S, Model X), impact of disruptive technologies upon established business models, sustainable energy economy and electric vehicles, clean technologies to address climate change, spacecraft development (SpaceX), solar power systems (SolarCity).
```

**Full log file**: Click [`images\teslaquerylog.txt`](images\teslaquerylog.txt) for a complete example session log.

---

## Troubleshooting

| Issue | Suggested Fix |
| ----- | -------------- |
| Ollama not reachable | Ensure `ollama serve` is running; pull `nomic-embed-text` / `llama3:8b` |
| `404 Client Error` from embeddings | Verify embeddings endpoint or switch `embedding` model |
| `429 RESOURCE_EXHAUSTED` from Gemini | Increase quota, throttle requests, or switch to local Ollama |
| Redirect pages returned as answers | Built-in heuristics re-plan; consider alternative tools if it persists |


---

## Acknowledgements

- MCP specification and sample servers
- Ollama (local LLM inference)
- Pyreverse / Graphviz tooling
