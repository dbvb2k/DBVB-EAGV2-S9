
--------------------------------------------------------------------------------
CORTEX-R AGENT - CODE EXPLANATION

This section provides brief explanations of classes and functions in the
core/ and modules/ directories.

--------------------------------------------------------------------------------
CORE PACKAGE
--------------------------------------------------------------------------------

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

--------------------------------------------------------------------------------
MODULES PACKAGE
--------------------------------------------------------------------------------

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


