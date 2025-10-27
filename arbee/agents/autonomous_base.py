"""
Autonomous Agent Base Class with ReAct Pattern
Implements reasoning + acting loops with tool use and memory integration
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Literal, TypedDict, Annotated
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import asyncio
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import AnyMessage as GraphAnyMessage, add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

from config.settings import Settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ThoughtStep(BaseModel):
    """Single step in agent's reasoning trace"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    thought: str = Field(description="What the agent is thinking")
    reasoning: str = Field(description="Why the agent is taking this action")
    action_plan: Optional[str] = Field(default=None, description="What action to take next")


class ToolCallRecord(BaseModel):
    """Record of a tool invocation"""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None


class MemoryItem(BaseModel):
    """Memory retrieved from long-term storage"""
    key: str
    content: Any
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    """
    State schema for autonomous agents with ReAct pattern

    This state is passed through the agent's reasoning loop and
    accumulates context as the agent thinks, acts, and observes.
    """
    # Conversation with tools (LangChain message format)
    messages: Annotated[List[GraphAnyMessage], add_messages]

    # Agent reasoning trace (what the agent is thinking)
    reasoning_trace: Annotated[List[ThoughtStep], "Chain of thought steps"]

    # Tool usage tracking
    tool_calls: Annotated[List[ToolCallRecord], "Tools used during reasoning"]

    # Memory access tracking
    memory_accessed: Annotated[List[MemoryItem], "Memories retrieved from long-term storage"]

    # Intermediate working data
    intermediate_results: Annotated[Dict[str, Any], "Temporary data during reasoning"]

    # Final structured output (if agent has completed task)
    final_output: Annotated[Optional[BaseModel], "Final structured result"]

    # Control flow
    next_action: Annotated[Literal["continue", "end", "escalate"], "What to do next"]

    # Iteration tracking
    iteration_count: Annotated[int, "Number of reasoning loops completed"]
    max_iterations: Annotated[int, "Maximum iterations before forcing termination"]

    # Task context (set at start)
    task_description: Annotated[str, "High-level description of what agent should accomplish"]
    task_input: Annotated[Dict[str, Any], "Input data for the task"]


class AutonomousReActAgent(ABC):
    """
    Abstract base class for autonomous agents using ReAct pattern

    ReAct Pattern: Reason + Act
    - Agent observes current state
    - Agent thinks about what to do next (reasoning)
    - Agent takes action (tool use)
    - Agent observes results
    - Loop continues until task complete

    Features:
    - Autonomous reasoning loops (task-completion driven)
    - Tool integration (agents select and use tools)
    - Memory access (short-term + long-term)
    - Self-monitoring (decides when done)
    - Full transparency (reasoning trace, tool calls visible)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_iterations: int = 20,
        store: Optional[BaseStore] = None,
        *,
        auto_extend_iterations: bool = True,
        iteration_extension: int = 5,
        max_iteration_cap: int = 50,
        recursion_limit: Optional[int] = None,
        llm_timeout: float = 60.0,
        agent_timeout: float = 600.0,
        # Feature flags for rollback (NEW)
        enable_memory_tracking: bool = True,
        enable_query_deduplication: bool = True,
        enable_url_blocking: bool = True,
        enable_circuit_breakers: bool = True,
        enable_auto_memory_query: bool = True
    ) -> None:
        """
        Initialize autonomous agent

        Args:
            settings: Configuration settings
            model_name: LLM model to use
            temperature: Sampling temperature
            max_iterations: Max reasoning loops before forced termination
            store: LangGraph Store for cross-thread memory
            auto_extend_iterations: Increase iteration budget automatically if needed
            iteration_extension: Number of loops to extend when auto-extending
            max_iteration_cap: Hard cap on total iterations
            recursion_limit: Optional LangGraph recursion limit override
            llm_timeout: Timeout in seconds for each LLM API call (default: 60s)
            agent_timeout: Timeout in seconds for entire agent execution (default: 600s/10min)
            enable_memory_tracking: Enable memory tracking (query history, URL attempts)
            enable_query_deduplication: Enable query deduplication
            enable_url_blocking: Enable URL blocking after failures
            enable_circuit_breakers: Enable enhanced circuit breakers
            enable_auto_memory_query: Enable automatic memory query at agent start
        """
        self.settings = settings or Settings()
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.store = store
        self.auto_extend_iterations = auto_extend_iterations
        self.iteration_extension = iteration_extension
        self.max_iteration_cap = max_iteration_cap
        self.recursion_limit = recursion_limit
        self.llm_timeout = llm_timeout
        self.agent_timeout = agent_timeout

        # Feature flags for rollback (NEW)
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_query_deduplication = enable_query_deduplication
        self.enable_url_blocking = enable_url_blocking
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_auto_memory_query = enable_auto_memory_query

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize LLM with tool binding and timeout
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=temperature,
            openai_api_key=self.settings.OPENAI_API_KEY,
            request_timeout=self.llm_timeout  # Configurable timeout per API call
        )

        # Get tools and bind to LLM
        self.tools = self.get_tools()
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

        # Create tool executor node
        self.tool_node = ToolNode(self.tools) if self.tools else None

        # Statistics
        self.stats = {
            'total_invocations': 0,
            'successful_completions': 0,
            'max_iterations_reached': 0,
            'total_tool_calls': 0,
            'total_memory_accesses': 0,
            'average_iterations': 0.0
        }

        self.logger.info(
            f"{self.__class__.__name__} initialized with {len(self.tools)} tools, "
            f"model={model_name}, max_iterations={max_iterations}, "
            f"auto_extend={self.auto_extend_iterations}"
        )

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent

        This prompt should:
        1. Describe the agent's role and responsibilities
        2. Explain what tools are available and when to use them
        3. Define what constitutes task completion
        4. Provide reasoning guidelines

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """
        Return the list of tools this agent can use

        Tools are LangChain Tool objects that the agent can invoke.
        Each tool should have:
        - Clear name
        - Descriptive docstring
        - Type-annotated parameters
        - Async implementation if needed

        Returns:
            List of BaseTool objects
        """
        pass

    def _message_text(self, message: BaseMessage) -> str:
        """Normalize message content to a plain string for logging and storage."""
        content = message.content
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif block.get("type") == "json":
                        parts.append(str(block.get("json", "")))
            if parts:
                return " ".join(parts).strip()

        return str(content)

    def _preview(self, message: BaseMessage, limit: int = 240) -> str:
        """Return a short preview string safe for logs."""
        text = self._message_text(message)
        return text if len(text) <= limit else f"{text[:limit]}…"

    def _format_tool_args(self, args: Any, limit: int = 140) -> str:
        """Serialize tool arguments for logging without overwhelming output."""
        try:
            serialized = json.dumps(args, default=str, ensure_ascii=False)
        except Exception:
            serialized = repr(args)
        return serialized if len(serialized) <= limit else f"{serialized[:limit]}…"

    def _parse_tool_result(self, tool_message: ToolMessage) -> Any:
        """
        Parse tool result from ToolMessage into structured data.

        Handles multiple formats:
        - LangGraph tool artifacts
        - JSON string content
        - Dict content
        - Plain text content

        Returns:
            Parsed result (dict, list, str, etc.) or None if parsing fails
        """
        # Try artifact first (preferred format)
        artifact = getattr(tool_message, "artifact", None)
        if artifact is not None:
            if isinstance(artifact, (dict, list, str)):
                return artifact
            # Try to extract from pydantic model
            if hasattr(artifact, "model_dump"):
                try:
                    return artifact.model_dump()
                except Exception:
                    pass
            # Try dict() for dict-like objects
            if hasattr(artifact, "__dict__"):
                try:
                    return vars(artifact)
                except Exception:
                    pass

        # Try content as JSON
        content = self._message_text(tool_message)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Content is plain text
                return content

        return None

    def _sanitize_message_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Validate tool message pairing but keep all messages unless severely malformed.

        LangGraph's tool_node creates proper message sequences, so we should trust it.
        Only drop messages if they would cause an immediate OpenAI API error.

        Strategy: Track tool_call_ids but be permissive - allow messages through
        unless they're truly orphaned (no AIMessage with tool_calls in recent history).
        """
        sanitized: List[BaseMessage] = []
        pending_tool_ids: List[str] = []

        for idx, message in enumerate(messages):
            if isinstance(message, AIMessage):
                sanitized.append(message)
                # Track all tool_call_ids from this AIMessage
                pending_tool_ids = [
                    call.get("id")
                    for call in (message.tool_calls or [])
                    if isinstance(call, dict) and call.get("id")
                ]
            elif isinstance(message, ToolMessage):
                tool_call_id = getattr(message, "tool_call_id", None)

                if pending_tool_ids:
                    # We have pending tool calls - match this ToolMessage
                    if tool_call_id and tool_call_id in pending_tool_ids:
                        pending_tool_ids.remove(tool_call_id)
                        sanitized.append(message)
                    elif not tool_call_id:
                        # No ID, assume FIFO order
                        if pending_tool_ids:
                            pending_tool_ids.pop(0)
                        sanitized.append(message)
                    else:
                        # ID doesn't match - might be from a previous iteration that was already sanitized
                        # Keep the message but log a warning
                        self.logger.debug(
                            f"Tool message id {tool_call_id} not in current pending list {pending_tool_ids}; "
                            "this may be from a previous call group - keeping message"
                        )
                        sanitized.append(message)
                else:
                    # No pending tool IDs - check if there's ANY recent AIMessage with tool_calls
                    # Look back up to 5 messages
                    has_recent_tool_call = False
                    for prev_msg in reversed(sanitized[-5:]):
                        if isinstance(prev_msg, AIMessage) and prev_msg.tool_calls:
                            has_recent_tool_call = True
                            break

                    if has_recent_tool_call:
                        # There IS a recent AIMessage with tool_calls, so keep this ToolMessage
                        # (it's probably a multi-tool scenario we already processed)
                        self.logger.debug(
                            f"ToolMessage at index {idx} (id={tool_call_id}) has no pending IDs "
                            "but found recent AIMessage with tool_calls - keeping"
                        )
                        sanitized.append(message)
                    else:
                        # Truly orphaned - no recent AIMessage with tool_calls
                        self.logger.warning(
                            f"Dropping truly orphaned tool message at index {idx} (id={tool_call_id}) - "
                            "no AIMessage with tool_calls in recent history"
                        )
            else:
                # HumanMessage, SystemMessage, etc - always keep
                sanitized.append(message)
                # Clear pending on HumanMessage (new conversation turn)
                if isinstance(message, HumanMessage):
                    pending_tool_ids = []

        return sanitized

    def _build_memory_context(self, state: AgentState) -> str:
        """
        Build a dynamic memory context message listing recent actions.

        This is injected before each LLM call to remind agents of:
        - Recent queries they've already tried
        - URLs that have been blocked
        - Other relevant memory items

        This enforces memory awareness at the framework level, not just in prompts.

        Args:
            state: Current agent state

        Returns:
            Memory context string, or empty string if no memory to report
        """
        intermediate = state.get('intermediate_results', {})
        context_parts = []

        # List recent queries (to prevent duplicates)
        last_queries = intermediate.get('last_N_queries', [])
        if last_queries:
            recent_5 = last_queries[-5:]  # Last 5 queries
            context_parts.append(
                "🧠 MEMORY: You have already tried these queries:\n"
                + "\n".join(f"  • {q}" for q in recent_5)
                + "\n\n⚠️ DO NOT repeat these exact queries. Try different search terms or angles."
            )

        # List blocked URLs (to prevent extraction attempts)
        blocked = intermediate.get('blocked_urls', set())
        if blocked:
            blocked_list = sorted(list(blocked))[:3]  # Show max 3
            context_parts.append(
                "🚫 BLOCKED URLs (extraction failed 2+ times):\n"
                + "\n".join(f"  • {url[:80]}..." if len(url) > 80 else f"  • {url}" for url in blocked_list)
                + "\n\n⚠️ DO NOT attempt to extract from these URLs again."
            )

        # Additional memory items for specific agent types
        # (Subclasses can override this method to add more context)

        return "\n\n".join(context_parts) if context_parts else ""

    def handle_tool_message(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_message: ToolMessage
    ) -> None:
        """
        Hook for subclasses to react to tool outputs.

        Default implementation is a no-op. Specific agents can override this
        to capture tool artifacts or update their intermediate state whenever
        a tool returns data (e.g., storing evidence after extraction).
        """
        return

    @abstractmethod
    async def is_task_complete(self, state: AgentState) -> bool:
        """
        Determine if the agent has completed its task

        This is called after each reasoning iteration to decide
        whether to continue or terminate.

        Args:
            state: Current agent state

        Returns:
            True if task is complete, False to continue reasoning
        """
        pass

    @abstractmethod
    async def extract_final_output(self, state: AgentState) -> BaseModel:
        """
        Extract final structured output from agent state

        Called when task is complete. Should parse the agent's work
        and return a structured Pydantic model.

        Args:
            state: Final agent state

        Returns:
            Structured output (Pydantic model)
        """
        pass

    async def query_memory(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5
    ) -> List[MemoryItem]:
        """
        Query long-term memory for relevant information

        Uses LangGraph Store API to search across past interactions.

        Args:
            query: Search query
            namespace: Memory namespace to search
            limit: Maximum results to return

        Returns:
            List of relevant memory items
        """
        if not self.store:
            self.logger.warning("No store configured, memory query skipped")
            return []

        try:
            # Use store to search for relevant memories
            # Note: Implementation depends on store backend (Redis, PostgreSQL, etc.)
            # This is a placeholder for the actual store API call
            results = await self.store.asearch(
                query=query,
                namespace=namespace or self.__class__.__name__,
                limit=limit
            )

            memory_items = [
                MemoryItem(
                    key=result.key,
                    content=result.value,
                    relevance_score=result.score if hasattr(result, 'score') else 1.0,
                    metadata=result.metadata if hasattr(result, 'metadata') else {}
                )
                for result in results
            ]

            self.stats['total_memory_accesses'] += 1
            self.logger.info(f"Memory query '{query}' returned {len(memory_items)} results")

            return memory_items

        except Exception as e:
            self.logger.error(f"Memory query failed: {e}")
            return []

    async def store_memory(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Store information in long-term memory

        Args:
            key: Memory key
            value: Value to store
            metadata: Optional metadata
            namespace: Memory namespace

        Returns:
            True if stored successfully
        """
        if not self.store:
            self.logger.warning("No store configured, memory storage skipped")
            return False

        try:
            await self.store.aput(
                namespace=namespace or self.__class__.__name__,
                key=key,
                value=value,
                metadata=metadata or {}
            )

            self.logger.info(f"Stored memory: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Memory storage failed: {e}")
            return False

    async def _query_memory_at_start(
        self,
        task_description: str,
        task_input: Dict[str, Any],
        state: AgentState
    ) -> str:
        """
        Query memory systems at agent start to find relevant past knowledge.

        This method automatically searches for:
        1. Similar market questions analyzed in the past
        2. Historical evidence on related topics
        3. Successful strategies from past analyses

        Args:
            task_description: High-level task description
            task_input: Task input data
            state: Agent state to update with memory findings

        Returns:
            Formatted string summarizing memory findings for injection into system prompt
        """
        if not self.enable_memory_tracking:
            self.logger.debug("Memory tracking disabled, skipping memory query")
            return ""

        from config.system_constants import (
            AUTO_QUERY_SIMILAR_MARKETS_LIMIT,
            AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT,
            AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT,
            MEMORY_QUERY_TIMEOUT_SECONDS
        )

        memory_context_parts = []
        memory_items = []

        try:
            # Extract market question or main topic from task_input
            market_question = (
                task_input.get('market_question') or
                task_input.get('question') or
                task_description
            )

            self.logger.info(f"🧠 Querying memory for: '{market_question[:60]}'...")

            # Query 1: Search for similar past markets
            try:
                from arbee.tools.memory_search import search_similar_markets_tool

                similar_markets = await asyncio.wait_for(
                    search_similar_markets_tool.ainvoke({
                        'market_question': market_question,
                        'limit': AUTO_QUERY_SIMILAR_MARKETS_LIMIT
                    }),
                    timeout=MEMORY_QUERY_TIMEOUT_SECONDS
                )

                if similar_markets:
                    memory_context_parts.append(
                        f"📚 SIMILAR PAST MARKETS ({len(similar_markets)} found):\n" +
                        "\n".join([
                            f"  • {m.get('question', 'Unknown')[:80]}"
                            f" (prior={m.get('prior', 'N/A')}, outcome={m.get('outcome', 'N/A')})"
                            for m in similar_markets[:3]
                        ])
                    )

                    # Store in memory_accessed
                    for market in similar_markets:
                        memory_items.append(MemoryItem(
                            key=market.get('id', 'unknown'),
                            content=market,
                            relevance_score=market.get('score', 0.8),
                            metadata={'type': 'similar_market'}
                        ))

                    self.logger.info(f"  ✓ Found {len(similar_markets)} similar markets")
                else:
                    self.logger.info(f"  ℹ️ No similar markets found in memory")

            except asyncio.TimeoutError:
                self.logger.warning("  ⏰ Similar markets search timed out")
            except Exception as e:
                self.logger.warning(f"  ⚠️ Similar markets search failed: {e}")

            # Query 2: Search for historical evidence
            try:
                from arbee.tools.memory_search import search_historical_evidence_tool

                # Extract topic keywords from market question
                topic = market_question[:100]  # Use first 100 chars as topic

                historical_evidence = await asyncio.wait_for(
                    search_historical_evidence_tool.ainvoke({
                        'topic': topic,
                        'limit': AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT
                    }),
                    timeout=MEMORY_QUERY_TIMEOUT_SECONDS
                )

                if historical_evidence:
                    memory_context_parts.append(
                        f"\n📊 HISTORICAL EVIDENCE ({len(historical_evidence)} items):\n" +
                        "\n".join([
                            f"  • {e.get('title', 'Unknown')[:60]} (LLR={e.get('llr', 0):.2f}, support={e.get('support', 'unknown')})"
                            for e in historical_evidence[:3]
                        ])
                    )

                    # Store in memory_accessed
                    for evidence in historical_evidence:
                        memory_items.append(MemoryItem(
                            key=evidence.get('id', 'unknown'),
                            content=evidence,
                            relevance_score=evidence.get('relevance_score', 0.7),
                            metadata={'type': 'historical_evidence'}
                        ))

                    self.logger.info(f"  ✓ Found {len(historical_evidence)} historical evidence items")
                else:
                    self.logger.info(f"  ℹ️ No historical evidence found")

            except asyncio.TimeoutError:
                self.logger.warning("  ⏰ Historical evidence search timed out")
            except Exception as e:
                self.logger.warning(f"  ⚠️ Historical evidence search failed: {e}")

            # Query 3: Search for successful strategies
            try:
                if self.store:
                    strategy_results = await asyncio.wait_for(
                        self.store.asearch(
                            ("strategies",),
                            query=market_question,
                            limit=AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT
                        ),
                        timeout=MEMORY_QUERY_TIMEOUT_SECONDS
                    )

                    if strategy_results:
                        strategies = []
                        for item in strategy_results:
                            value = item.value or {}
                            if isinstance(value, dict):
                                effectiveness = value.get('effectiveness', 0)
                                if effectiveness >= 0.7:  # Only show highly effective strategies
                                    strategies.append({
                                        'description': value.get('description', 'Unknown'),
                                        'effectiveness': effectiveness,
                                        'type': value.get('strategy_type', 'unknown')
                                    })

                        if strategies:
                            memory_context_parts.append(
                                f"\n💡 SUCCESSFUL STRATEGIES ({len(strategies)} found):\n" +
                                "\n".join([
                                    f"  • [{s['type']}] {s['description'][:70]} (effectiveness={s['effectiveness']:.1%})"
                                    for s in strategies[:3]
                                ])
                            )

                            for strategy in strategies:
                                memory_items.append(MemoryItem(
                                    key=f"strategy_{hash(strategy['description'])}",
                                    content=strategy,
                                    relevance_score=strategy['effectiveness'],
                                    metadata={'type': 'successful_strategy'}
                                ))

                            self.logger.info(f"  ✓ Found {len(strategies)} successful strategies")
                        else:
                            self.logger.info(f"  ℹ️ No high-effectiveness strategies found")

            except asyncio.TimeoutError:
                self.logger.warning("  ⏰ Strategy search timed out")
            except Exception as e:
                self.logger.warning(f"  ⚠️ Strategy search failed: {e}")

            # Update state with memory findings
            if memory_items:
                state['memory_accessed'] = memory_items
                self.logger.info(f"✅ Memory query complete: {len(memory_items)} items retrieved")

                # Build formatted context for system message
                memory_context = (
                    "═══════════════════════════════════════════════════════\n"
                    "🧠 MEMORY CONTEXT (from past analyses)\n"
                    "═══════════════════════════════════════════════════════\n"
                    + "\n".join(memory_context_parts) +
                    "\n\n💡 Use these insights to inform your analysis strategy.\n"
                    "═══════════════════════════════════════════════════════"
                )

                return memory_context
            else:
                self.logger.info("ℹ️ No relevant memory found for this task")
                return ""

        except Exception as e:
            self.logger.error(f"❌ Memory query failed: {e}")
            return ""

    async def agent_node(self, state: AgentState) -> AgentState:
        """
        Main reasoning node - agent decides what to do next

        This is where the "Reason" part of ReAct happens:
        1. Observe current state (messages, intermediate results)
        2. Think about what to do next (invoke LLM with tools)
        3. LLM either calls a tool or provides final answer

        Args:
            state: Current agent state

        Returns:
            Updated state with agent's decision (tool call or completion)
        """
        # Increment iteration
        state['iteration_count'] = state.get('iteration_count', 0) + 1

        # Log iteration header with context
        self.logger.info("\n" + "=" * 80)
        self.logger.info(
            f"🤔 ITERATION {state['iteration_count']}/{state['max_iterations']} - Agent Reasoning"
        )
        self.logger.info("=" * 80)

        # PRE-ITERATION STALL CHECK - detect repetitive tool calls before LLM invocation
        # This prevents timeouts by catching stalls before the expensive LLM call
        state_messages = state.get('messages', [])
        if state_messages and state['iteration_count'] >= 5:
            recent_tools = []
            for msg in state_messages[-10:]:  # Look at last 5 pairs (AI + ToolMessage)
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        recent_tools.append(call.get("name", "unknown"))

            if len(recent_tools) >= 5:
                # Check if same tool is being called repeatedly
                last_5_tools = recent_tools[-5:]
                if len(set(last_5_tools)) == 1:  # All last 5 are the same tool
                    self.logger.error(
                        f"🛑 PRE-ITERATION STALL DETECTED: Tool '{last_5_tools[0]}' called 5+ times"
                    )
                    self.logger.warning("   → Forcing completion to prevent timeout")
                    state['_forced_stop'] = True
                    # Return early without calling LLM
                    return state

        # Log current progress in task
        if state.get('intermediate_results'):
            results = state['intermediate_results']
            result_keys = list(results.keys())[:5]  # First 5 keys
            self.logger.info(f"📊 Progress: {len(results)} keys in intermediate_results: {result_keys}")

        # Surface the most recent observation (typically tool output)
        if state.get('messages'):
            last_msg = state['messages'][-1]
            if isinstance(last_msg, ToolMessage):
                self.logger.info(
                    f"🛠️ Previous tool '{last_msg.name}' returned: {self._preview(last_msg, limit=220)}"
                )

        # Build context for LLM
        system_message = SystemMessage(content=self.get_system_prompt())

        # Add task context
        task_message = HumanMessage(
            content=f"Task: {state['task_description']}\n\nInput: {state['task_input']}"
        )

        # Prepare messages (system + task + conversation history)
        state_messages = state.get('messages', [])
        if state_messages:
            sanitized_messages = self._sanitize_message_history(state_messages)
            if len(sanitized_messages) != len(state_messages):
                state['messages'] = sanitized_messages
                state_messages = sanitized_messages

        # INJECT MEMORY CONTEXT to prevent duplicate queries/actions
        memory_context = self._build_memory_context(state)
        if memory_context:
            # Add memory as a system message before LLM call
            memory_message = SystemMessage(content=memory_context)
            messages = [system_message, task_message, memory_message] + state_messages
            self.logger.debug(f"💾 Injected memory context ({len(memory_context)} chars)")
        else:
            messages = [system_message, task_message] + state_messages

        # Invoke LLM with tools
        try:
            response = await self.llm_with_tools.ainvoke(messages)

            # Add response to messages
            state['messages'] = state.get('messages', []) + [response]

            # Extract full reasoning text (not just preview)
            response_text = self._message_text(response)

            # Log full reasoning with better formatting
            self.logger.info(f"💬 Agent response: {self._preview(response, limit=220)}")

            if response_text:
                # Log FULL reasoning text (not truncated) with clear formatting
                self.logger.info("=" * 60)
                self.logger.info(f"🧠 FULL REASONING (Iteration {state['iteration_count']}):")
                self.logger.info("-" * 60)
                # Split into lines for better readability
                for line in response_text.split('\n'):
                    if line.strip():
                        self.logger.info(f"   {line}")
                self.logger.info("=" * 60)
            elif response.tool_calls:
                # When LLM makes tool calls without reasoning text
                self.logger.info("💭 Agent made tool calls without explicit reasoning text (common with tool-use models)")
            else:
                # No reasoning and no tool calls - might be completion
                self.logger.info("📝 Agent response contains no reasoning text or tool calls")

            # Extract structured reasoning if present
            if hasattr(response, 'content') and response.content:
                # Extract condensed thought for trace
                thought_text = response_text[:500] if len(response_text) > 500 else response_text

                thought = ThoughtStep(
                    thought=thought_text,
                    reasoning="See full reasoning above",
                    action_plan="See tool calls" if response.tool_calls else "Task complete"
                )
                state['reasoning_trace'] = state.get('reasoning_trace', []) + [thought]

            # Log tool calls with better formatting
            if response.tool_calls:
                self.logger.info(f"\n📎 TOOL CALLS ({len(response.tool_calls)}):")
                for idx, call in enumerate(response.tool_calls, 1):
                    tool_name = call.get("name", "unknown_tool")
                    args_preview = self._format_tool_args(call.get("args", {}))
                    self.logger.info(f"   [{idx}] {tool_name}")
                    self.logger.info(f"       Args: {args_preview}")
            else:
                self.logger.info("✓ No tool calls - agent may be completing task")

            return state

        except Exception as e:
            self.logger.error(f"Agent node failed: {e}")
            state['error'] = str(e)
            raise

    async def should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """
        Decide whether to continue reasoning or terminate

        This is called after agent_node to determine next step:
        - If agent made tool calls → route to "tools" node
        - If no tool calls and task complete → route to "end"
        - If max iterations reached → force "end"

        Args:
            state: Current agent state

        Returns:
            "tools" to continue reasoning, "end" to terminate
        """
        # Check if forced stop was triggered in agent_node (pre-iteration stall detection)
        if state.get('_forced_stop'):
            self.logger.warning("🛑 Forced stop detected - routing to end")
            return "end"

        # EARLY LOOP DETECTION - Detect query/action loops before they waste iterations
        if state['iteration_count'] >= 5:
            intermediate = state.get('intermediate_results', {})

            # Check for query loops (researcher agents)
            last_queries = intermediate.get('last_N_queries', [])
            if len(last_queries) >= 5:
                recent_5 = last_queries[-5:]
                unique_queries = len(set(recent_5))

                if unique_queries <= 2:
                    self.logger.error(
                        f"🔁 QUERY LOOP DETECTED: Only {unique_queries} unique queries in last 5 attempts"
                    )
                    self.logger.warning("   → Forcing termination to prevent infinite loop")
                    self.logger.info(f"   → Queries: {recent_5[:3]}...")
                    return "end"

            # Check for tool repetition loops (all agents)
            recent_tools = []
            for msg in state['messages'][-10:]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        recent_tools.append(call.get("name", "unknown"))

            if len(recent_tools) >= 6:
                last_6_tools = recent_tools[-6:]
                unique_tools = len(set(last_6_tools))

                # If same 1-2 tools repeated 6 times, we're stuck
                if unique_tools <= 2:
                    from collections import Counter
                    tool_counts = Counter(last_6_tools)
                    self.logger.error(
                        f"🔁 TOOL LOOP DETECTED: Only {unique_tools} unique tools in last 6 calls"
                    )
                    self.logger.warning(f"   → Tool distribution: {dict(tool_counts)}")
                    self.logger.warning("   → Forcing termination to prevent infinite loop")
                    return "end"

        # Check if agent made tool calls
        last_message = state['messages'][-1] if state['messages'] else None

        if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            self.logger.info(f"→ Routing to tools ({len(last_message.tool_calls)} calls)")
            return "tools"

        # DIAGNOSTIC: Log tool call history for debugging
        recent_tools = []
        for msg in state['messages'][-10:]:  # Look at last 5 pairs (AI + ToolMessage)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    recent_tools.append(call.get("name", "unknown"))

        if recent_tools:
            # Log tool usage summary
            from collections import Counter
            tool_counts = Counter(recent_tools)
            self.logger.info(
                f"📊 Recent tool usage (last {len(recent_tools)} calls): "
                f"{dict(tool_counts)}"
            )

        # ENHANCED: Detect stalled states and validation loops
        # This prevents infinite loops where agent keeps repeating actions

        # 1. Detect same tool called repeatedly (original check)

        if len(recent_tools) >= 5:
            # Check if same tool is being called repeatedly
            last_5_tools = recent_tools[-5:]
            if len(set(last_5_tools)) == 1:  # All last 5 are the same tool
                self.logger.warning(
                    f"⚠️  STALL DETECTED: Tool '{last_5_tools[0]}' called 5+ times consecutively"
                )
                self.logger.warning("   → Forcing task completion to prevent infinite loop")
                return "end"

        # 2. Detect validation loops (validate_prior_tool specifically)
        if len(recent_tools) >= 4:
            validation_calls = [t for t in recent_tools[-6:] if 'validate' in t.lower()]
            if len(validation_calls) >= 4:
                self.logger.warning(
                    f"⚠️  VALIDATION LOOP DETECTED: {len(validation_calls)} validation calls in last 6 actions"
                )
                self.logger.warning("   → Forcing completion to prevent infinite loop")

                # Try to extract any prior from tool results in message history
                prior_found = False
                for msg in reversed(state['messages'][-20:]):
                    if isinstance(msg, ToolMessage) and 'validate_prior_tool' in str(msg.name):
                        try:
                            import json
                            tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                            if isinstance(tool_result, dict):
                                # Extract prior from validation result
                                if 'prior_p' in tool_result:
                                    state['intermediate_results'] = state.get('intermediate_results', {})
                                    state['intermediate_results']['p0_prior'] = tool_result['prior_p']
                                    state['intermediate_results']['prior_validated'] = tool_result.get('is_valid', False)
                                    prior_found = True
                                    self.logger.info(f"   ✓ Extracted prior {tool_result['prior_p']:.2%} from validation loop")
                                    break
                        except Exception as e:
                            self.logger.debug(f"Could not extract prior from tool message: {e}")

                if prior_found or state.get('intermediate_results', {}).get('p0_prior'):
                    self.logger.info("   ✓ Prior available, forcing task completion")
                    return "end"
                else:
                    self.logger.warning("   ⚠️  No prior found in intermediate_results, but forcing end anyway to break loop")
                    # Force end even without prior - let extract_final_output handle missing data
                    return "end"

        # 3. Detect progress stall (intermediate_results not changing)
        if state['iteration_count'] >= 5:  # Only check after 5 iterations
            # Store hash of intermediate_results to detect changes
            import json
            current_results = state.get('intermediate_results', {})
            current_hash = hash(json.dumps(current_results, sort_keys=True, default=str))

            # Get previous hash from state (if stored)
            prev_hash = state.get('_results_hash')
            state['_results_hash'] = current_hash

            # Track how many iterations with no progress
            if prev_hash and prev_hash == current_hash:
                no_progress_count = state.get('_no_progress_count', 0) + 1
                state['_no_progress_count'] = no_progress_count

                if no_progress_count >= 3:
                    self.logger.warning(
                        f"⚠️  PROGRESS STALL: No change in intermediate_results for {no_progress_count} iterations"
                    )
                    self.logger.warning("   → Forcing task completion")
                    return "end"
            else:
                state['_no_progress_count'] = 0  # Reset counter on progress

        # Check iteration budget (after handling tool routing)
        # Warn when approaching limit (at 80% threshold)
        if state['iteration_count'] >= state['max_iterations'] * 0.8:
            if state['iteration_count'] < state['max_iterations']:
                remaining = state['max_iterations'] - state['iteration_count']
                self.logger.warning(
                    f"⚠️  APPROACHING ITERATION LIMIT: {remaining} iterations remaining "
                    f"({state['iteration_count']}/{state['max_iterations']})"
                )

        if state['iteration_count'] >= state['max_iterations']:
            if self.auto_extend_iterations and state['max_iterations'] < self.max_iteration_cap:
                new_limit = min(
                    state['max_iterations'] + self.iteration_extension,
                    self.max_iteration_cap
                )
                self.logger.info(
                    f"➕ Extending iteration budget from {state['max_iterations']} to {new_limit}"
                )
                state['max_iterations'] = new_limit
            else:
                self.logger.warning(
                    f"⚠️  MAX ITERATIONS REACHED ({state['max_iterations']}), forcing termination"
                )
                self.stats['max_iterations_reached'] += 1
                return "end"

        # Check if task is complete
        task_complete = await self.is_task_complete(state)

        if task_complete:
            self.logger.info("✅ Task complete, terminating")
            return "end"

        # Task not complete but no tool calls - this shouldn't happen in well-designed agents
        self.logger.warning("⚠️  No tool calls but task incomplete - terminating anyway")
        return "end"

    async def create_reasoning_graph(self) -> StateGraph:
        """
        Create LangGraph workflow for ReAct reasoning loop

        Graph structure:
        START → agent_node → should_continue (conditional)
                                ↓               ↓
                                ← tool_node ← "tools"
                                ↓
                               "end" → END

        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self.agent_node)
        if self.tool_node:
            async def tools_wrapper(state: AgentState) -> AgentState:
                prior_messages = state.get('messages', [])
                if not prior_messages or not isinstance(prior_messages[-1], AIMessage):
                    self.logger.warning("Tool node invoked without preceding AI message; skipping tool execution.")
                    return state

                tool_result = await self.tool_node.ainvoke(state)
                tool_messages: List[BaseMessage] = tool_result.get("messages", []) if tool_result else []

                if not tool_messages:
                    self.logger.warning("Tool node returned no messages.")
                    return state

                # Map tool_call_id -> call metadata so we can log inputs alongside outputs
                last_ai = prior_messages[-1]
                call_lookup = {}
                if isinstance(last_ai, AIMessage) and last_ai.tool_calls:
                    for call in last_ai.tool_calls:
                        call_lookup[call.get("id")] = call

                state['messages'] = prior_messages + tool_messages

                for msg in tool_messages:
                    if isinstance(msg, ToolMessage):
                        call_meta = call_lookup.get(msg.tool_call_id, {})
                        tool_name = getattr(msg, "name", call_meta.get("name", "unknown_tool"))
                        tool_args = call_meta.get("args", {})
                        preview = self._preview(msg, limit=220)

                        self.logger.info(
                            f"🔧 Tool '{tool_name}' output: {preview}"
                        )

                        tool_record = ToolCallRecord(
                            tool_name=tool_name,
                            tool_input=tool_args,
                            tool_output=self._message_text(msg)
                        )
                        state['tool_calls'] = state.get('tool_calls', []) + [tool_record]
                        self.stats['total_tool_calls'] += 1

                        # AUTO-STORAGE: Automatically store key tool results in intermediate_results
                        # This ensures results are available for is_task_complete() checks
                        if tool_name in ['bayesian_calculate_tool', 'sensitivity_analysis_tool',
                                        'validate_prior_tool', 'estimate_prior_with_base_rates_tool']:
                            try:
                                # Parse tool result from message
                                tool_result = self._parse_tool_result(msg)

                                if tool_result:
                                    if tool_name == 'estimate_prior_with_base_rates_tool':
                                        # Store prior reasoning (Phase 1 enhancement)
                                        self.logger.info(f"📦 Auto-storing prior reasoning from estimate_prior_with_base_rates_tool")
                                        state['intermediate_results']['prior_reasoning'] = tool_result

                                    elif tool_name == 'bayesian_calculate_tool':
                                        # Store Bayesian calculation results
                                        self.logger.info(f"📦 Auto-storing bayesian_calculate_tool results")
                                        state['intermediate_results'].update({
                                            'p0': tool_result.get('p0'),
                                            'p_bayesian': tool_result.get('p_bayesian'),
                                            'log_odds_prior': tool_result.get('log_odds_prior'),
                                            'log_odds_posterior': tool_result.get('log_odds_posterior'),
                                            'p_neutral': tool_result.get('p_neutral', 0.5),
                                            'evidence_summary': tool_result.get('evidence_summary', []),
                                            'correlation_adjustments': tool_result.get('correlation_adjustments', {})
                                        })

                                    elif tool_name == 'sensitivity_analysis_tool':
                                        # Store sensitivity analysis results
                                        self.logger.info(f"📦 Auto-storing sensitivity_analysis_tool results")
                                        state['intermediate_results']['sensitivity_analysis'] = tool_result

                                    elif tool_name == 'validate_prior_tool':
                                        # Store prior validation results
                                        self.logger.info(f"📦 Auto-storing prior validation results")
                                        if isinstance(tool_result, dict):
                                            state['intermediate_results'].update({
                                                'p0_prior': tool_result.get('prior_p', tool_result.get('p0')),
                                                'prior_validated': tool_result.get('is_valid', False),
                                                'prior_justification': tool_result.get('justification', '')
                                            })

                            except Exception as auto_store_exc:
                                self.logger.warning(
                                    f"Auto-storage failed for '{tool_name}': {auto_store_exc}"
                                )

                        # Allow subclasses to post-process tool outputs
                        try:
                            self.handle_tool_message(state, tool_name, tool_args, msg)
                        except Exception as exc:  # Defensive: tool post-processing should not crash agent
                            self.logger.warning(
                                f"Tool message handler failed for '{tool_name}': {exc}"
                            )
                    else:
                        self.logger.info(f"🔧 Tool output message: {self._preview(msg, limit=220)}")

                return state

            workflow.add_node("tools", tools_wrapper)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edge from agent
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools" if self.tool_node else END,
                "end": END
            }
        )

        # Add edge back from tools to agent (creates the loop)
        if self.tool_node:
            workflow.add_edge("tools", "agent")

        # Compile with checkpointing
        checkpointer = MemorySaver()
        app = workflow.compile(checkpointer=checkpointer)

        return app

    async def run(
        self,
        task_description: str,
        task_input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        *,
        max_iterations: Optional[int] = None
    ) -> BaseModel:
        """
        Run the autonomous agent on a task

        This is the main entry point. It:
        1. Initializes agent state
        2. Runs ReAct reasoning loop until completion
        3. Extracts and returns structured output

        Args:
            task_description: High-level description of the task
            task_input: Input data for the task
            config: Optional LangGraph config (thread_id, etc.)

        Returns:
            Structured output (Pydantic model)

        Raises:
            Exception: If agent fails to complete task
        """
        self.stats['total_invocations'] += 1

        self.logger.info(f"🚀 Starting autonomous agent: {task_description}")

        # Determine iteration budget for this run
        effective_max_iterations = max_iterations or self.max_iterations
        if self.auto_extend_iterations:
            effective_max_iterations = min(effective_max_iterations, self.max_iteration_cap)

        # Initialize state
        initial_state: AgentState = {
            'messages': [],
            'reasoning_trace': [],
            'tool_calls': [],
            'memory_accessed': [],
            'intermediate_results': {},
            'final_output': None,
            'next_action': 'continue',
            'iteration_count': 0,
            'max_iterations': effective_max_iterations,
            'task_description': task_description,
            'task_input': task_input
        }

        # Query memory at start to find relevant past knowledge
        from config.system_constants import ENABLE_AUTO_MEMORY_QUERY_DEFAULT

        enable_auto_query = getattr(self, 'enable_auto_memory_query', ENABLE_AUTO_MEMORY_QUERY_DEFAULT)

        if enable_auto_query:
            self.logger.info("═" * 60)
            self.logger.info("🧠 AUTO-QUERYING MEMORY FOR SIMILAR PAST ANALYSES")
            self.logger.info("═" * 60)

            memory_context = await self._query_memory_at_start(
                task_description=task_description,
                task_input=task_input,
                state=initial_state
            )

            # If memory was found, inject it into the first system message
            if memory_context:
                # Add memory context as a HumanMessage so it's visible to agent
                initial_state['messages'].append(
                    HumanMessage(content=memory_context)
                )
                self.logger.info("✅ Memory context added to agent's initial knowledge")
            else:
                self.logger.info("ℹ️ No relevant memory found, starting fresh")

            self.logger.info("═" * 60)
        else:
            self.logger.debug("Auto-memory query disabled")

        # Create reasoning graph
        app = await self.create_reasoning_graph()

        # Run graph
        try:
            default_config = {
                "configurable": {
                    "thread_id": f"{self.__class__.__name__}-{datetime.utcnow().timestamp()}"
                }
            }

            if config:
                merged_config = {**default_config, **config}
                if "configurable" in config:
                    merged_config["configurable"] = {
                        **default_config.get("configurable", {}),
                        **config["configurable"],
                    }
            else:
                merged_config = default_config

            # Set recursion limit with more headroom for complex reasoning
            if "recursion_limit" not in merged_config and (self.recursion_limit or effective_max_iterations):
                # Formula: max_iterations * 5 (gives more headroom for tool calls + reasoning)
                # Each iteration can be: agent → tools → agent, so 3 steps per iteration minimum
                # Adding 2x buffer for safety = 5x multiplier
                calculated_limit = max(60, effective_max_iterations * 5)
                merged_config["recursion_limit"] = self.recursion_limit or calculated_limit
                self.logger.info(f"🔧 Set recursion_limit = {merged_config['recursion_limit']} "
                               f"(max_iterations={effective_max_iterations} × 5)")

            # Add timeout to prevent hanging
            try:
                final_state = await asyncio.wait_for(
                    app.ainvoke(initial_state, merged_config),
                    timeout=self.agent_timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(
                    f"⏰ Agent execution timed out after {self.agent_timeout}s. "
                    "This usually means the LLM is stuck or API is unresponsive."
                )
                raise RuntimeError(
                    f"Agent timed out after {self.agent_timeout}s. "
                    "Check LLM API status or increase timeout."
                )

            # Ensure task completion criteria satisfied before extracting output
            if final_state.get('error'):
                raise RuntimeError(
                    f"Agent terminated with error: {final_state['error']}"
                )

            task_complete = await self.is_task_complete(final_state)
            if not task_complete:
                raise RuntimeError(
                    "Agent terminated without satisfying completion criteria. "
                    "Check intermediate_results for partial progress."
                )

            # Extract final output
            output = await self.extract_final_output(final_state)

            # Update stats
            self.stats['successful_completions'] += 1
            iterations = final_state['iteration_count']
            self.stats['average_iterations'] = (
                (self.stats['average_iterations'] * (self.stats['successful_completions'] - 1) + iterations)
                / self.stats['successful_completions']
            )

            self.logger.info(
                f"✅ Agent completed successfully in {iterations} iterations"
            )

            return output

        except Exception as e:
            self.logger.error(f"❌ Agent execution failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics"""
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_completions'] / self.stats['total_invocations']
                if self.stats['total_invocations'] > 0 else 0.0
            )
        }
