"""Autonomous ReAct base agent with deterministic memory- and tool-aware control flow."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage as GraphAnyMessage, add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from config.system_constants import (
    AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT,
    AUTO_QUERY_SIMILAR_MARKETS_LIMIT,
    AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT,
    MEMORY_QUERY_TIMEOUT_SECONDS,
)
from config.settings import Settings
from utils.rich_logging import setup_rich_logging, RichAgentLogger


logger = logging.getLogger(__name__)


class ThoughtStep(BaseModel):
    """Single reasoning step captured for transparency and auditing.
    
    Attributes:
        timestamp: When the thought occurred
        thought: Agent inner monologue excerpt (max 1000 chars)
        reasoning: Expanded reasoning (max 2000 chars)
        action_plan: Intended next action or None
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    thought: str = Field(description="Agent inner monologue excerpt", max_length=1000)
    reasoning: str = Field(description="Expanded reasoning for the thought", max_length=2000)
    action_plan: Optional[str] = Field(default=None, description="Intended next action")


class ToolCallRecord(BaseModel):
    """Structured record of a tool invocation.
    
    Attributes:
        tool_name: Name of the tool invoked
        tool_input: Input parameters passed to tool
        tool_output: Result returned by tool
        timestamp: When invocation occurred
        success: Whether invocation succeeded
        error_message: Error details if success=False
    """

    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = True
    error_message: Optional[str] = None


class MemoryItem(BaseModel):
    """Item retrieved from long-term memory search.
    
    Attributes:
        key: Unique identifier for memory item
        content: Actual memory content (any type)
        relevance_score: Relevance score [0.0, 1.0]
        metadata: Additional metadata dictionary
    """

    key: str
    content: Any
    relevance_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(TypedDict):
    """Shared state for autonomous agents in the ReAct workflow."""

    messages: Annotated[List[GraphAnyMessage], add_messages]
    reasoning_trace: Annotated[List[ThoughtStep], "Chain of thought steps"]
    tool_calls: Annotated[List[ToolCallRecord], "Tools used during reasoning"]
    memory_accessed: Annotated[List[MemoryItem], "Memories retrieved from long-term storage"]
    intermediate_results: Annotated[Dict[str, Any], "Temporary data during reasoning"]
    final_output: Annotated[Optional[BaseModel], "Final structured result"]
    next_action: Annotated[Literal["continue", "end", "escalate"], "What to do next"]
    iteration_count: Annotated[int, "Number of reasoning loops completed"]
    max_iterations: Annotated[int, "Maximum iterations before forcing termination"]
    task_description: Annotated[str, "High-level description of what agent should accomplish"]
    task_input: Annotated[Dict[str, Any], "Input data for the task"]


class AutonomousReActAgent(ABC):
    """Reusable autonomous agent base implementing the ReAct pattern with strict safeguards.
    
    Provides core functionality for autonomous agents including:
    - ReAct reasoning loop with tool orchestration
    - Memory integration and querying
    - Stall detection and circuit breakers
    - Iteration budget management
    - Rich logging and diagnostics
    
    Subclasses must implement:
    - get_system_prompt(): Return agent-specific system prompt
    - get_tools(): Return list of available tools
    - is_task_complete(): Determine when task is finished
    - extract_final_output(): Produce final structured output
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
        enable_memory_tracking: bool = True,
        enable_query_deduplication: bool = True,
        enable_url_blocking: bool = True,
        enable_circuit_breakers: bool = True,
        enable_auto_memory_query: bool = True,
    ) -> None:
        assert 0.0 <= temperature <= 2.0, "temperature must be in [0, 2]"
        assert max_iterations > 0, "max_iterations must be positive"
        assert iteration_extension > 0, "iteration_extension must be positive"
        assert max_iteration_cap >= max_iterations, "max_iteration_cap must be >= max_iterations"
        assert llm_timeout > 0, "llm_timeout must be positive"
        assert agent_timeout > 0, "agent_timeout must be positive"
        assert agent_timeout >= llm_timeout, "agent_timeout must be >= llm_timeout"
        
        self.settings = settings or Settings()
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        # Auto-create store if not provided and memory persistence is enabled
        if store is None:
            from config.settings import settings as global_settings
            # Check if memory persistence is enabled
            enable_persistence = getattr(global_settings, "ENABLE_MEMORY_PERSISTENCE", True)
            if enable_persistence:
                try:
                    from utils.memory import create_store_from_config
                    store = create_store_from_config()
                    if store is not None:
                        logger.info(f"Auto-created memory store: {type(store).__name__}")
                    else:
                        logger.warning("Memory persistence enabled but store creation failed - memory features disabled")
                except Exception as e:
                    logger.warning(f"Failed to auto-create memory store: {e} - memory features disabled")
                    store = None
            else:
                logger.debug("Memory persistence disabled - store not created")
        
        self.store = store
        self.auto_extend_iterations = auto_extend_iterations
        self.iteration_extension = iteration_extension
        self.max_iteration_cap = max_iteration_cap
        self.recursion_limit = recursion_limit
        self.llm_timeout = llm_timeout
        self.agent_timeout = agent_timeout
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_query_deduplication = enable_query_deduplication
        self.enable_url_blocking = enable_url_blocking
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_auto_memory_query = enable_auto_memory_query

        # Adjust enable_auto_memory_query based on store availability
        if self.enable_auto_memory_query and self.store is None:
            logger.warning("enable_auto_memory_query=True but no store available - disabling auto-query")
            self.enable_auto_memory_query = False

        self.logger = logging.getLogger(self.__class__.__name__)
        self.rich_logger = setup_rich_logging(self.__class__.__name__)
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.settings.OPENAI_API_KEY,
            request_timeout=self.llm_timeout,
        )

        self.tools = self.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm
        self.tool_node = ToolNode(self.tools) if self.tools else None

        self.stats: Dict[str, Any] = {
            "total_invocations": 0,
            "successful_completions": 0,
            "max_iterations_reached": 0,
            "total_tool_calls": 0,
            "total_memory_accesses": 0,
            "average_iterations": 0.0,
        }

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt guiding the agent."""

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Return available tools for the agent."""

    def _record_event(self, state: AgentState, category: str, detail: str) -> None:
        """Record diagnostic event in state.
        
        Args:
            state: Current agent state
            category: Event category (e.g., 'stall_detected', 'memory_context')
            detail: Event description
        """
        diagnostics = state.setdefault("diagnostics", [])
        diagnostics.append({
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": state.get("iteration_count", 0),
            "category": category,
            "detail": detail,
        })

    def _recent_tool_names(self, messages: List[BaseMessage], window: int) -> List[str]:
        """Extract tool names from recent messages.
        
        Args:
            messages: Message history
            window: Number of recent messages to check
            
        Returns:
            List of tool names called in recent window
        """
        assert window > 0, "window must be positive"
        return [
            call.get("name", "unknown")
            for message in messages[-window:]
            if isinstance(message, AIMessage) and message.tool_calls
            for call in message.tool_calls
        ]

    def _recent_queries(self, intermediate: Dict[str, Any], window: int) -> List[str]:
        """Extract recent queries from intermediate results.
        
        Args:
            intermediate: Intermediate results dictionary
            window: Number of recent queries to return
            
        Returns:
            List of recent query strings
        """
        assert window > 0, "window must be positive"
        queries = intermediate.get("last_N_queries", []) if intermediate else []
        return list(queries[-window:]) if queries else []

    def _message_text(self, message: BaseMessage) -> str:
        """Extract text content from message, handling various content formats.
        
        Args:
            message: LangChain message object
            
        Returns:
            Extracted text content as string
        """
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(block.get("text", "")) if block.get("type") == "text"
                else json.dumps(block.get("json", {})) if block.get("type") == "json"
                else ""
                for block in content
                if isinstance(block, dict)
            ]
            return " ".join(parts).strip() if parts else str(content)
        return str(content)

    def _preview(self, message: BaseMessage, limit: int = 240) -> str:
        """Generate preview of message text with length limit.
        
        Args:
            message: Message to preview
            limit: Maximum characters to return
            
        Returns:
            Truncated text with ellipsis if needed
        """
        assert limit > 0, "limit must be positive"
        text = self._message_text(message)
        return text[:limit] + "..." if len(text) > limit else text

    def _format_tool_args(self, args: Any, limit: int = 140) -> str:
        """Serialize tool arguments to JSON string with length limit.
        
        Args:
            args: Tool arguments to serialize
            limit: Maximum characters to return
            
        Returns:
            JSON string, truncated if needed
        """
        assert limit > 0, "limit must be positive"
        serialized = json.dumps(args, default=str)
        return serialized[:limit] + "..." if len(serialized) > limit else serialized

    def _parse_tool_result(self, tool_message: ToolMessage) -> Any:
        """Parse tool message result into structured data.
        
        Args:
            tool_message: ToolMessage containing result
            
        Returns:
            Parsed result (dict/list/str) or None if empty
        """
        artifact = getattr(tool_message, "artifact", None)
        if artifact is not None:
            if isinstance(artifact, (dict, list, str)):
                return artifact
            if hasattr(artifact, "model_dump"):
                return artifact.model_dump()
            if hasattr(artifact, "__dict__"):
                return vars(artifact)
        content = self._message_text(tool_message)
        if not content:
            return None
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            assert len(stripped) > 2, "JSON content too short"
            return json.loads(stripped)
        return content

    def _sanitize_message_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Remove orphaned tool messages and ensure proper message pairing.
        
        Args:
            messages: Raw message history
            
        Returns:
            Sanitized message list with proper AI/Tool message pairs
        """
        sanitized: List[BaseMessage] = []
        pending_tool_ids: List[str] = []
        for message in messages:
            if isinstance(message, AIMessage):
                sanitized.append(message)
                pending_tool_ids = [
                    call.get("id")
                    for call in (message.tool_calls or [])
                    if isinstance(call, dict) and call.get("id")
                ]
            elif isinstance(message, ToolMessage):
                tool_call_id = getattr(message, "tool_call_id", None)
                if pending_tool_ids:
                    if tool_call_id and tool_call_id in pending_tool_ids:
                        pending_tool_ids.remove(tool_call_id)
                    elif not tool_call_id:
                        pending_tool_ids.pop(0)
                    sanitized.append(message)
                else:
                    has_recent_tool_call = any(
                        isinstance(prev_msg, AIMessage) and prev_msg.tool_calls
                        for prev_msg in sanitized[-5:]
                    )
                    if has_recent_tool_call:
                        sanitized.append(message)
            elif isinstance(message, HumanMessage):
                pending_tool_ids = []
                sanitized.append(message)
        return sanitized

    def _build_memory_context(self, state: AgentState) -> str:
        """Build context string from recent queries and blocked URLs.
        
        Args:
            state: Current agent state
            
        Returns:
            Formatted context string or empty string
        """
        intermediate = state.get("intermediate_results", {})
        context_parts: List[str] = []
        
        last_queries = self._recent_queries(intermediate, 5)
        if last_queries:
            context_parts.append(
                "MEMORY REMINDER: recent queries already issued:\n"
                + "\n".join(f"- {query}" for query in last_queries)
            )
        
        blocked = intermediate.get("blocked_urls", set())
        if blocked:
            blocked_list = sorted(list(blocked))[:3]
            context_parts.append(
                "BLOCKED URLS: further extraction should be skipped:\n"
                + "\n".join(f"- {url[:80]}" for url in blocked_list)
            )
        
        return "\n\n".join(context_parts)

    def _process_tool_result(
        self,
        tool_name: str,
        tool_result_payload: Any,
        intermediate: Dict[str, Any],
    ) -> None:
        """Process tool result and update intermediate state.
        
        Args:
            tool_name: Name of tool that was called
            tool_result_payload: Parsed tool result
            intermediate: Intermediate results dictionary (modified in-place)
        """
        if tool_name == "estimate_prior_with_base_rates_tool" and isinstance(tool_result_payload, dict):
            intermediate["prior_reasoning"] = tool_result_payload
        elif tool_name == "bayesian_calculate_tool" and isinstance(tool_result_payload, dict):
            intermediate.update({
                "p0": tool_result_payload.get("p0"),
                "p_bayesian": tool_result_payload.get("p_bayesian"),
                "log_odds_prior": tool_result_payload.get("log_odds_prior"),
                "log_odds_posterior": tool_result_payload.get("log_odds_posterior"),
                "p_neutral": tool_result_payload.get("p_neutral", 0.5),
                "evidence_summary": tool_result_payload.get("evidence_summary", []),
                "correlation_adjustments": tool_result_payload.get("correlation_adjustments", {}),
            })
        elif tool_name == "sensitivity_analysis_tool":
            if isinstance(tool_result_payload, list):
                intermediate["sensitivity_analysis"] = tool_result_payload
            elif isinstance(tool_result_payload, dict):
                intermediate["sensitivity_analysis"] = (
                    tool_result_payload.get("sensitivity_analysis", [tool_result_payload])
                )
        elif tool_name == "validate_prior_tool" and isinstance(tool_result_payload, dict):
            intermediate["p0_prior"] = tool_result_payload.get("prior_p", tool_result_payload.get("p0"))
            intermediate["prior_validated"] = tool_result_payload.get("is_valid", False)
            intermediate["prior_justification"] = tool_result_payload.get("justification", "")
        elif tool_name in [
            "information_asymmetry_tool",
            "market_inefficiency_tool",
            "sentiment_edge_tool",
            "base_rate_violation_tool",
            "analyze_mentions_market_tool",
        ] and isinstance(tool_result_payload, dict):
            if "edge_signals" not in intermediate:
                intermediate["edge_signals"] = []
            intermediate["edge_signals"].append(tool_result_payload)
        elif tool_name == "composite_edge_score_tool" and isinstance(tool_result_payload, dict):
            intermediate["composite_edge_score"] = tool_result_payload

    async def handle_tool_message(
        self,
        state: AgentState,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_message: ToolMessage,
    ) -> None:
        """Handle tool message result (override in subclasses for custom processing).
        
        Args:
            state: Current agent state
            tool_name: Name of tool that was called
            tool_args: Arguments passed to tool
            tool_message: ToolMessage containing result
        """
        return

    @abstractmethod
    async def is_task_complete(self, state: AgentState) -> bool:
        """Return True when the agent has satisfied its completion criteria."""

    @abstractmethod
    async def extract_final_output(self, state: AgentState) -> BaseModel:
        """Produce final structured output from the terminal agent state."""

    async def query_memory(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """Query long-term memory store for relevant items.
        
        Args:
            query: Search query string
            namespace: Optional namespace override (defaults to class name)
            limit: Maximum results to return
            
        Returns:
            List of MemoryItem objects sorted by relevance
            
        Raises:
            AssertionError: If store not configured or invalid parameters
        """
        assert self.store is not None, "Memory store is not configured"
        assert isinstance(query, str) and query.strip(), "query must be non-empty string"
        assert limit > 0, "limit must be positive"
        
        results = await self.store.asearch(
            query=query,
            namespace=namespace or self.__class__.__name__,
            limit=limit,
        )
        memory_items = [
            MemoryItem(
                key=result.key,
                content=result.value,
                relevance_score=getattr(result, "score", 1.0),
                metadata=getattr(result, "metadata", {}),
            )
            for result in results
        ]
        self.stats["total_memory_accesses"] += 1
        return memory_items

    async def store_memory(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> bool:
        """Store item in long-term memory.
        
        Args:
            key: Unique identifier for memory item
            value: Value to store (any serializable type)
            metadata: Optional metadata dictionary
            namespace: Optional namespace override (defaults to class name)
            
        Returns:
            True if successful
            
        Raises:
            AssertionError: If store not configured or invalid key
        """
        assert self.store is not None, "Memory store is not configured"
        assert isinstance(key, str) and key.strip(), "key must be non-empty string"
        await self.store.aput(namespace or self.__class__.__name__, key, value, metadata or {})
        return True

    async def _fetch_similar_markets(
        self,
        market_question: str,
        limit: int,
        timeout: float,
    ) -> List[Dict[str, Any]]:
        """Fetch similar markets from memory store.
        
        Args:
            market_question: Market question to find similar markets for
            limit: Maximum results to return
            timeout: Timeout in seconds
            
        Returns:
            List of similar market dictionaries
            
        Raises:
            AssertionError: If invalid parameters
            asyncio.TimeoutError: If operation exceeds timeout
        """
        assert isinstance(market_question, str) and market_question.strip(), \
            "market_question must be non-empty string"
        assert limit > 0, "limit must be positive"
        assert timeout > 0, "timeout must be positive"
        
        from tools.memory import search_similar_markets_tool
        payload = {"market_question": market_question, "limit": limit}
        return await asyncio.wait_for(search_similar_markets_tool.ainvoke(payload), timeout=timeout)

    async def _fetch_historical_evidence(
        self,
        topic: str,
        limit: int,
        timeout: float,
    ) -> List[Dict[str, Any]]:
        """Fetch historical evidence from memory store.
        
        Args:
            topic: Topic to search for evidence
            limit: Maximum results to return
            timeout: Timeout in seconds
            
        Returns:
            List of historical evidence dictionaries
            
        Raises:
            AssertionError: If invalid parameters
            asyncio.TimeoutError: If operation exceeds timeout
        """
        assert isinstance(topic, str) and topic.strip(), "topic must be non-empty string"
        assert limit > 0, "limit must be positive"
        assert timeout > 0, "timeout must be positive"
        
        from tools.memory import search_historical_evidence_tool
        payload = {"topic": topic, "limit": limit}
        return await asyncio.wait_for(
            search_historical_evidence_tool.ainvoke(payload), timeout=timeout
        )

    async def _fetch_successful_strategies(
        self,
        query: str,
        limit: int,
        timeout: float,
    ) -> List[Dict[str, Any]]:
        """Fetch successful strategies from memory store (effectiveness >= 0.7).
        
        Args:
            query: Search query string
            limit: Maximum results to return
            timeout: Timeout in seconds
            
        Returns:
            List of strategy dictionaries with effectiveness >= 0.7
            
        Raises:
            AssertionError: If store not configured or invalid parameters
            asyncio.TimeoutError: If operation exceeds timeout
        """
        assert self.store is not None, "Memory store is not configured"
        assert isinstance(query, str) and query.strip(), "query must be non-empty string"
        assert limit > 0, "limit must be positive"
        assert timeout > 0, "timeout must be positive"
        
        search_results = await asyncio.wait_for(
            self.store.asearch(("strategies",), query=query, limit=limit), timeout=timeout
        )
        strategies: List[Dict[str, Any]] = []
        for item in search_results:
            payload = item.value or {}
            if isinstance(payload, dict):
                effectiveness = payload.get("effectiveness", 0.0)
                assert isinstance(effectiveness, (int, float)), "effectiveness must be numeric"
                # Use config constant for effectiveness threshold
                from config.system_constants import SENSITIVITY_ROBUST_THRESHOLD
                effectiveness_threshold = max(0.7, SENSITIVITY_ROBUST_THRESHOLD)  # Use 0.7 or robust threshold, whichever is higher
                if effectiveness >= effectiveness_threshold:
                    strategies.append({
                        "description": payload.get("description", ""),
                        "effectiveness": effectiveness,
                        "strategy_type": payload.get("strategy_type", "unknown"),
                    })
        return strategies

    async def _query_memory_at_start(
        self,
        task_description: str,
        task_input: Dict[str, Any],
        state: AgentState,
    ) -> str:
        """Query memory store at task start and build context string.
        
        Args:
            task_description: High-level task description
            task_input: Task input dictionary
            state: Agent state (modified in-place with memory items)
            
        Returns:
            Formatted memory context string or empty string
        """
        if not self.enable_memory_tracking or self.store is None:
            return ""

        market_question = (
            task_input.get("market_question") or task_input.get("question") or task_description
        )
        
        # Use constants from system_constants for limits
        from config.system_constants import (
            AUTO_QUERY_SIMILAR_MARKETS_LIMIT,
            AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT,
            AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT,
            MEMORY_QUERY_TIMEOUT_SECONDS,
        )
        
        similar_markets = await self._fetch_similar_markets(
            market_question, AUTO_QUERY_SIMILAR_MARKETS_LIMIT, MEMORY_QUERY_TIMEOUT_SECONDS
        )
        historical_evidence = await self._fetch_historical_evidence(
            market_question[:100],  # Truncate to reasonable length - this is for query, not storage
            AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT,
            MEMORY_QUERY_TIMEOUT_SECONDS,
        )
        strategies = await self._fetch_successful_strategies(
            market_question, AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT, MEMORY_QUERY_TIMEOUT_SECONDS
        )

        memory_items: List[MemoryItem] = []
        context_lines: List[str] = []

        if similar_markets:
            # Limit display to top 3, but use config constant
            from config.system_constants import AUTO_QUERY_SIMILAR_MARKETS_LIMIT
            display_limit = min(3, AUTO_QUERY_SIMILAR_MARKETS_LIMIT)
            context_lines.append(
                "SIMILAR MARKETS:\n"
                + "\n".join(
                    f"- {item.get('question', 'Unknown')} (prior={item.get('prior', 'N/A')}, outcome={item.get('outcome', 'N/A')})"
                    for item in similar_markets[:display_limit]
                )
            )
            memory_items.extend(
                MemoryItem(
                    key=item.get("id", "unknown"),
                    content=item,
                    relevance_score=item.get("score", 0.8),
                    metadata={"type": "similar_market"},
                )
                for item in similar_markets
            )

        if historical_evidence:
            # Limit display to top 3, but use config constant
            from config.system_constants import AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT
            display_limit = min(3, AUTO_QUERY_HISTORICAL_EVIDENCE_LIMIT)
            context_lines.append(
                "HISTORICAL EVIDENCE:\n"
                + "\n".join(
                    f"- {item.get('title', 'Unknown')} (LLR={item.get('llr', 0.0):.2f}, support={item.get('support', 'unknown')})"
                    for item in historical_evidence[:display_limit]
                )
            )
            memory_items.extend(
                MemoryItem(
                    key=item.get("id", "unknown"),
                    content=item,
                    relevance_score=item.get("relevance_score", 0.7),
                    metadata={"type": "historical_evidence"},
                )
                for item in historical_evidence
            )

        if strategies:
            # Limit display to top 3, but use config constant
            from config.system_constants import AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT
            display_limit = min(3, AUTO_QUERY_SUCCESSFUL_STRATEGIES_LIMIT)
            context_lines.append(
                "SUCCESSFUL STRATEGIES:\n"
                + "\n".join(
                    f"- [{item['strategy_type']}] {item['description']} (effectiveness={item['effectiveness']:.1%})"
                    for item in strategies[:display_limit]
                )
            )
            memory_items.extend(
                MemoryItem(
                    key=f"strategy_{hash(item['description'])}",
                    content=item,
                    relevance_score=item["effectiveness"],
                    metadata={"type": "successful_strategy"},
                )
                for item in strategies
            )

        if memory_items:
            state["memory_accessed"] = memory_items
            self._record_event(state, "memory_context", f"retrieved {len(memory_items)} items")
            
            # Use config constant for display limit
            from config.system_constants import MEMORY_CONTEXT_QUERY_HISTORY_SIZE
            display_limit = min(5, MEMORY_CONTEXT_QUERY_HISTORY_SIZE)
            for item in memory_items[:display_limit]:
                self.rich_logger.log_memory_access(
                    operation="search",
                    key=item.key,
                    found=True,
                    data=item.content if isinstance(item.content, dict) else str(item.content)[:100],
                )
            
            header = "=" * 59
            return f"{header}\nMEMORY CONTEXT\n{header}\n" + "\n\n".join(context_lines) + f"\n{header}"

    def _detect_pre_iteration_stall(self, state: AgentState) -> bool:
        """Detect if agent is stuck repeating the same tool call.
        
        Args:
            state: Current agent state
            
        Returns:
            True if stall detected (same tool called 5+ times consecutively)
        """
        if state["iteration_count"] < 5:
            return False
        recent_tools = self._recent_tool_names(state.get("messages", []), 10)
        if len(recent_tools) >= 5:
            last_five = recent_tools[-5:]
            if len(set(last_five)) == 1:
                self._record_event(state, "stall_detected", f"Repeat tool {last_five[0]}")
                state["_forced_stop"] = True
                return True
        return False

    async def agent_node(self, state: AgentState) -> AgentState:
        """Execute single agent reasoning step in ReAct loop.
        
        Invokes LLM with system prompt, task context, memory context, and message history.
        Records reasoning trace and tool calls for diagnostics.
        
        Args:
            state: Current agent state (modified in-place)
            
        Returns:
            Updated agent state with new LLM response
        """
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        iteration = state["iteration_count"]
        
        # Log agent start on first iteration
        if iteration == 1:
            market_question = state.get("task_input", {}).get("market_question", "")
            self.rich_logger.log_agent_start(
                task_description=state.get("task_description", ""),
                input_info=state.get("task_input", {}),
                market_question=market_question,
            )
        
        if self._detect_pre_iteration_stall(state):
            return state

        system_message = SystemMessage(content=self.get_system_prompt())
        task_message = HumanMessage(
            content=f"Task: {state['task_description']}\n\nInput: {state['task_input']}"
        )
        state_messages = state.get("messages", [])
        memory_context = self._build_memory_context(state)
        messages: List[BaseMessage] = [system_message, task_message]
        if memory_context:
            messages.append(SystemMessage(content=memory_context))
        # For tool-using agents, only include the most recent conversation turn
        # to avoid message pairing issues
        if self.tools:
            # Find the last AI message with tool calls
            last_ai_with_tools = None
            for msg in reversed(state_messages):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    last_ai_with_tools = msg
                    break
            if last_ai_with_tools:
                # Include the AI message and any subsequent tool messages
                ai_index = state_messages.index(last_ai_with_tools)
                relevant_messages = state_messages[ai_index:]
                messages.extend(relevant_messages)
            else:
                # No tool calls in history, just keep recent messages
                recent_messages = state_messages[-10:] if len(state_messages) > 10 else state_messages
                messages.extend(recent_messages)
        else:
            # For non-tool agents, keep more history
            recent_messages = state_messages[-20:] if len(state_messages) > 20 else state_messages
            messages.extend(recent_messages)

        response = await self.llm_with_tools.ainvoke(messages)
        state["messages"] = state.get("messages", []) + [response]

        response_text = self._message_text(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        
        if response_text or tool_calls:
            self.rich_logger.log_reasoning_step(
                iteration=iteration,
                thought=response_text[:500] if response_text else f"Processing {len(tool_calls)} tool calls",
                reasoning=response_text[500:1000] if len(response_text) > 500 else "",
            )
            
            state.setdefault("reasoning_trace", []).append(
                ThoughtStep(
                    thought=response_text[:1000],
                    reasoning=response_text[1000:2000],
                    action_plan="Tool calls scheduled" if response.tool_calls else "Task complete",
                )
            )
            if response.tool_calls:
                call_names = [call.get("name", "unknown") for call in response.tool_calls]
                state.setdefault("diagnostics", []).append({
                    "iteration": state["iteration_count"],
                    "tool_calls": call_names,
                    "response_text": response_text[1000:2000],
                    "memory_context": memory_context[:100] if memory_context else None,
                })
        return state

    def _detect_query_loop(self, state: AgentState) -> bool:
        """Detect if agent is stuck repeating the same queries.
        
        Args:
            state: Current agent state
            
        Returns:
            True if query loop detected (same 2 or fewer queries repeated 5 times)
        """
        if state["iteration_count"] < 5:
            return False
        intermediate = state.get("intermediate_results", {})
        recent = self._recent_queries(intermediate, 5)
        if len(recent) == 5 and len(set(recent)) <= 2:
            self._record_event(state, "query_loop", str(recent))
            return True
        return False

    def _detect_tool_loop(self, state: AgentState, threshold: int) -> bool:
        """Detect if agent is stuck repeating the same tools.
        
        Args:
            state: Current agent state
            threshold: Minimum number of recent tool calls to check
            
        Returns:
            True if tool loop detected (2 or fewer unique tools in recent threshold calls)
        """
        assert threshold > 0, "threshold must be positive"
        recent_tools = self._recent_tool_names(state.get("messages", []), threshold)
        if len(recent_tools) >= threshold:
            unique_tools = len(set(recent_tools[-threshold:]))
            if unique_tools <= 2:
                counts = Counter(recent_tools[-threshold:])
                self._record_event(state, "tool_loop", json.dumps(counts))
                return True
        return False

    def _detect_validation_loop(self, state: AgentState) -> bool:
        """Detect if agent is stuck in validation loop (4+ validate calls in recent 6 tools).
        
        Args:
            state: Current agent state
            
        Returns:
            True if validation loop detected
        """
        recent_tools = self._recent_tool_names(state.get("messages", []), 6)
        if len(recent_tools) < 4:
            return False
        validation_calls = [name for name in recent_tools[-6:] if "validate" in name.lower()]
        if len(validation_calls) < 4:
            return False
        
        prior_found = False
        for message in reversed(state.get("messages", [])[-20:]):
            if isinstance(message, ToolMessage) and "validate_prior_tool" in str(
                getattr(message, "name", "")
            ):
                tool_result = self._parse_tool_result(message)
                if isinstance(tool_result, dict) and "prior_p" in tool_result:
                    intermediate = state.setdefault("intermediate_results", {})
                    intermediate["p0_prior"] = tool_result["prior_p"]
                    intermediate["prior_validated"] = tool_result.get("is_valid", False)
                    intermediate["prior_justification"] = tool_result.get("justification", "")
                    prior_found = True
                    break
        
        self._record_event(state, "validation_loop", "prior captured" if prior_found else "prior missing")
        return True

    def _detect_progress_stall(self, state: AgentState) -> bool:
        """Detect if agent is making no progress (same intermediate_results for 3+ iterations).
        
        Args:
            state: Current agent state
            
        Returns:
            True if progress stall detected
        """
        if state["iteration_count"] < 5:
            return False
        current_results = state.get("intermediate_results", {})
        current_hash = hash(json.dumps(current_results, sort_keys=True, default=str))
        prev_hash = state.get("_results_hash")
        state["_results_hash"] = current_hash
        if prev_hash == current_hash:
            state["_no_progress_count"] = state.get("_no_progress_count", 0) + 1
            if state["_no_progress_count"] >= 3:
                self._record_event(state, "progress_stall", "no change in intermediate_results")
                return True
        else:
            state["_no_progress_count"] = 0
        return False

    def _manage_iteration_budget(self, state: AgentState) -> None:
        """Manage iteration budget, extending if enabled and under cap.
        
        Args:
            state: Current agent state (modified in-place)
        """
        if state["iteration_count"] < state["max_iterations"]:
            return
        if self.auto_extend_iterations and state["max_iterations"] < self.max_iteration_cap:
            new_limit = min(
                state["max_iterations"] + self.iteration_extension, self.max_iteration_cap
            )
            state["max_iterations"] = new_limit
            self._record_event(state, "iteration_extended", str(new_limit))
        else:
            state["_forced_stop_reason"] = "max_iterations"
            self.stats["max_iterations_reached"] += 1

    async def should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """Determine next action: continue to tools or end execution.
        
        Args:
            state: Current agent state
            
        Returns:
            "tools" to continue tool execution, "end" to terminate
        """
        if state.get("_forced_stop"):
            self._record_event(state, "forced_stop", "pre-iteration stall")
            return "end"
        if self._detect_query_loop(state):
            return "end"
        if self._detect_tool_loop(state, 6):
            return "end"
        last_message = state["messages"][-1] if state.get("messages") else None
        if last_message and getattr(last_message, "tool_calls", None):
            return "tools"
        
        if last_message and isinstance(last_message, AIMessage):
            response_text = self._message_text(last_message).lower()
            tool_description_patterns = [
                "processing", "will call", "going to call", "need to call",
                "should call", "must call", "calling",
            ]
            tool_mentioned = any(pattern in response_text for pattern in tool_description_patterns)
            has_tool_calls = bool(getattr(last_message, "tool_calls", None))
            
            if tool_mentioned and not has_tool_calls and self.tools:
                iteration = state.get("iteration_count", 0)
                if iteration <= 3:
                    return "tools"
        
        if self.tools and state["iteration_count"] <= 3:
            if len(state.get("tool_calls", [])) == 0:
                return "tools"
        
        if self._detect_tool_loop(state, 5):
            return "end"
        if self._detect_validation_loop(state):
            return "end"
        if self._detect_progress_stall(state):
            return "end"
        self._manage_iteration_budget(state)
        if state.get("_forced_stop_reason") == "max_iterations":
            return "end"
        if state["iteration_count"] >= state["max_iterations"] and not self.auto_extend_iterations:
            return "end"
        
        iteration = state.get("iteration_count", 0)
        tool_calls_made = len(state.get("tool_calls", []))
        if hasattr(self, '_inject_required_tool_calls') and iteration >= 1 and tool_calls_made == 0:
            results = state.get("intermediate_results", {})
            if not results or len(results) < 3:
                await self._inject_required_tool_calls(state)
        
        if await self.is_task_complete(state):
            return "end"
        return "tools"

    async def create_reasoning_graph(self) -> StateGraph:
        """Create LangGraph StateGraph for ReAct workflow.
        
        Returns:
            Compiled StateGraph with agent and tools nodes
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.agent_node)

        if self.tool_node:

            async def tools_wrapper(state: AgentState) -> AgentState:
                prior_messages = state.get("messages", [])
                if not prior_messages or not isinstance(prior_messages[-1], AIMessage):
                    self._record_event(state, "tool_execution_skipped", "missing AI message")
                    return state
                
                last_ai = prior_messages[-1]
                if isinstance(last_ai, AIMessage) and not last_ai.tool_calls:
                    response_text = self._message_text(last_ai).lower()
                    tool_description_patterns = [
                        "processing", "will call", "going to call", "need to call",
                        "should call", "must call", "calling",
                    ]
                    described_tools = any(pattern in response_text for pattern in tool_description_patterns)
                    
                    if described_tools:
                        force_content = (
                            "CRITICAL ERROR: You described tool calls but did NOT generate tool_calls. "
                            "You MUST generate actual tool_calls NOW, not describe them. "
                            "Stop describing and immediately generate tool_calls using the tool calling format."
                        )
                    else:
                        force_content = (
                            "ERROR: You are in the tools node but did not generate tool_calls. "
                            "You MUST use the available tools by generating actual tool_calls. "
                            "Review the task requirements and call the required tools immediately."
                        )
                    state["messages"] = prior_messages + [SystemMessage(content=force_content)]
                    return state
                
                tool_result = await self.tool_node.ainvoke(state)
                tool_messages: List[BaseMessage] = (
                    tool_result.get("messages", []) if tool_result else []
                )
                if not tool_messages:
                    self._record_event(state, "tool_execution_skipped", "tool returned no messages")
                    return state
                call_lookup = {}
                if isinstance(last_ai, AIMessage) and last_ai.tool_calls:
                    for call in last_ai.tool_calls:
                        call_lookup[call.get("id")] = call
                state["messages"] = prior_messages + tool_messages
                for message in tool_messages:
                    if isinstance(message, ToolMessage):
                        call_meta = call_lookup.get(message.tool_call_id, {})
                        tool_name = getattr(message, "name", call_meta.get("name", "unknown_tool"))
                        tool_args = call_meta.get("args", {})
                        record = ToolCallRecord(
                            tool_name=tool_name,
                            tool_input=tool_args,
                            tool_output=self._message_text(message),
                        )
                        state.setdefault("tool_calls", []).append(record)
                        self.stats["total_tool_calls"] += 1
                        tool_result_payload = self._parse_tool_result(message)
                        self.rich_logger.log_tool_call(
                            tool_name=tool_name,
                            tool_input=tool_args,
                            result=tool_result_payload,
                        )
                        
                        intermediate = state.setdefault("intermediate_results", {})
                        self._process_tool_result(tool_name, tool_result_payload, intermediate)
                        await self.handle_tool_message(state, tool_name, tool_args, message)
                return state

            workflow.add_node("tools", tools_wrapper)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools" if self.tool_node else END,
                "end": END,
            },
        )
        if self.tool_node:
            workflow.add_edge("tools", "agent")
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    async def run(
        self,
        task_description: str,
        task_input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        *,
        max_iterations: Optional[int] = None,
    ) -> BaseModel:
        """Execute agent task with full ReAct workflow.
        
        Args:
            task_description: High-level description of task
            task_input: Input data dictionary
            config: Optional LangGraph configuration
            max_iterations: Optional override for max iterations
            
        Returns:
            Final structured output (BaseModel)
            
        Raises:
            AssertionError: If invalid parameters or task incomplete
            RuntimeError: If agent terminates with error
            asyncio.TimeoutError: If execution exceeds agent_timeout
        """
        assert isinstance(task_description, str) and task_description.strip(), \
            "task_description must be non-empty string"
        assert isinstance(task_input, dict), "task_input must be dict"
        if max_iterations is not None:
            assert max_iterations > 0, "max_iterations must be positive"
        
        self.stats["total_invocations"] += 1
        effective_max_iterations = max_iterations or self.max_iterations
        if self.auto_extend_iterations:
            effective_max_iterations = min(effective_max_iterations, self.max_iteration_cap)
        initial_state: AgentState = {
            "messages": [],
            "reasoning_trace": [],
            "tool_calls": [],
            "memory_accessed": [],
            "intermediate_results": {},
            "final_output": None,
            "next_action": "continue",
            "iteration_count": 0,
            "max_iterations": effective_max_iterations,
            "task_description": task_description,
            "task_input": task_input,
        }
        from config.system_constants import ENABLE_AUTO_MEMORY_QUERY_DEFAULT

        enable_auto_query = getattr(
            self, "enable_auto_memory_query", ENABLE_AUTO_MEMORY_QUERY_DEFAULT
        )
        if enable_auto_query and self.store is not None:
            memory_context = await self._query_memory_at_start(
                task_description, task_input, initial_state
            )
            if memory_context:
                initial_state.setdefault("messages", []).append(
                    HumanMessage(content=memory_context)
                )

        app = await self.create_reasoning_graph()
        default_config = {
            "configurable": {
                "thread_id": f"{self.__class__.__name__}-{datetime.utcnow().timestamp()}",
            }
        }
        merged_config = default_config
        if config:
            merged_config = {**default_config, **config}
            if "configurable" in config:
                merged_config["configurable"] = {
                    **default_config.get("configurable", {}),
                    **config["configurable"],
                }
        if "recursion_limit" not in merged_config:
            merged_config["recursion_limit"] = self.recursion_limit or max(
                60, effective_max_iterations * 5
            )

        final_state = await asyncio.wait_for(
            app.ainvoke(initial_state, merged_config), timeout=self.agent_timeout
        )
        assert isinstance(final_state, dict), "Final state must be dict"
        if final_state.get("error"):
            raise RuntimeError(f"Agent terminated with error: {final_state['error']}")
        assert await self.is_task_complete(final_state), "Agent terminated without satisfying completion criteria"

        output = await self.extract_final_output(final_state)
        self.stats["successful_completions"] += 1
        iterations = final_state["iteration_count"]
        completions = self.stats["successful_completions"]
        prev_avg = self.stats["average_iterations"]
        self.stats["average_iterations"] = ((prev_avg * (completions - 1)) + iterations) / completions
        
        self.rich_logger.log_completion_status(
            complete=True, reason=f"Completed in {iterations} iterations"
        )
        output_dict = output.model_dump() if hasattr(output, 'model_dump') else {"output": str(output)}
        self.rich_logger.log_final_output(
            output=output_dict,
            execution_stats={
                "iterations": iterations,
                "tool_calls": self.stats["total_tool_calls"],
                "memory_accesses": self.stats["total_memory_accesses"],
            }
        )
        return output

    def get_stats(self) -> Dict[str, Any]:
        """Get agent execution statistics.
        
        Returns:
            Dictionary containing stats including success_rate, iterations, tool calls, etc.
        """
        success_rate = (
            self.stats["successful_completions"] / self.stats["total_invocations"]
            if self.stats["total_invocations"] > 0
            else 0.0
        )
        return {**self.stats, "success_rate": success_rate}
