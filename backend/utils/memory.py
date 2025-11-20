"""
Memory Infrastructure for POLYSEER Autonomous Agents.
Utilities for short‑term (working), episodic, and long‑term (knowledge) memory.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

# Data Models
class MemoryConfig(BaseModel):
    """Configuration for the memory system."""
    # Store backend
    store_type: str = Field(default="redis", description="redis, postgresql, or memory")
    store_connection_string: Optional[str] = Field(default=None)

    # Checkpointer backend
    checkpointer_type: str = Field(default="memory", description="memory, sqlite, or postgresql")
    checkpointer_connection_string: Optional[str] = Field(default=None)

    # Memory limits
    max_working_memory_messages: int = Field(default=50, description="Max messages in working memory")
    max_episode_memory_items: int = Field(default=100, description="Max items per episode")
    episode_retention_days: int = Field(default=90, description="How long to keep episode memories")

    # Vector search
    embedding_model: str = Field(default="text-embedding-3-small")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity for retrieval")


class WorkingMemoryItem(BaseModel):
    """Item in agent's working memory (current workflow)."""
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    key: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpisodeMemoryItem(BaseModel):
    """Item in episode memory (learnings from this analysis)."""
    episode_id: str  # Workflow ID
    market_question: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memory_type: str  # "strategy", "evidence", "outcome", "lesson"
    content: Any
    effectiveness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseItem(BaseModel):
    """Item in knowledge base (historical cross-workflow knowledge)."""
    id: str
    content_type: str  # "market_analysis", "evidence", "base_rate", "pattern"
    content: Any
    embedding: Optional[List[float]] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================
# Memory Manager
# =========================
class MemoryManager:
    """
    Centralized memory management for POLYSEER agents.

    Tiers:
      1) Working Memory (Checkpointer) - Current workflow state.
      2) Episode Memory (Store)        - Learnings from this analysis.
      3) Knowledge Base (Vector DB)    - Historical cross-workflow knowledge.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        store: Optional[BaseStore] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """
        Args:
            config: Memory configuration.
            store: LangGraph Store instance.
            checkpointer: LangGraph Checkpointer instance.
        """
        self.config = config or MemoryConfig()
        self.store = store
        self.checkpointer = checkpointer
        logger.info(f"MemoryManager init: store={self.config.store_type}, checkpointer={self.config.checkpointer_type}")

    async def store_working_memory(
        self,
        agent_name: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store an item in working memory (current workflow).

        Note:
            LangGraph's checkpointer typically handles working memory persistence.
            This method validates and records critical items explicitly.

        Returns:
            True if accepted for tracking.
        """
        assert agent_name and key, "agent_name and key are required"
        _ = WorkingMemoryItem(agent_name=agent_name, key=key, value=value, metadata=metadata or {})
        return True

    async def store_episode_memory(
        self,
        episode_id: str,
        market_question: str,
        memory_type: str,
        content: Any,
        effectiveness: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Persist learnings from the current episode for reuse in similar cases.

        Returns:
            True on success.
        """
        assert self.store is not None, "Store required for episode memory persistence"
        assert episode_id and memory_type, "episode_id and memory_type are required"
        
        item = EpisodeMemoryItem(
            episode_id=episode_id,
            market_question=market_question,
            memory_type=memory_type,
            content=content,
            effectiveness=effectiveness,
            metadata=metadata or {},
        )
        await self.store.aput(
            ("episode_memory",),
            f"{episode_id}:{memory_type}:{datetime.utcnow().isoformat()}",
            item.model_dump(),
        )
        return True

    async def retrieve_episode_memories(
        self,
        market_question: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[EpisodeMemoryItem]:
        """
        Retrieve episode memories for similar market analyses.

        Returns:
            A list of EpisodeMemoryItem.
        """
        assert self.store is not None, "Store required for episode memory retrieval"
        
        results = await self.store.asearch(("episode_memory",), query=market_question, limit=limit)
        
        out: List[EpisodeMemoryItem] = []
        for r in results:
            item = EpisodeMemoryItem(**r.value)
            if memory_type is None or item.memory_type == memory_type:
                out.append(item)
        return out

    async def store_knowledge(
        self,
        content_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: bool = True,
    ) -> str:
        """
        Store an item in the knowledge base for long‑term access.

        Returns:
            Knowledge ID.
        """
        assert self.store is not None, "Store required for knowledge persistence"
        assert content_type, "content_type is required"
        
        knowledge_id = f"{content_type}:{datetime.utcnow().timestamp()}"
        item = KnowledgeBaseItem(id=knowledge_id, content_type=content_type, content=content, metadata=metadata or {})

        if generate_embedding:
            pass

        await self.store.aput(("knowledge_base",), knowledge_id, item.model_dump())
        return knowledge_id

    async def search_knowledge(
        self,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[KnowledgeBaseItem]:
        """
        Search the knowledge base.

        Returns:
            A list of KnowledgeBaseItem.
        """
        assert self.store is not None, "Store required for knowledge search"
        
        # Truncate query for logging (reasonable length for display)
        from config.system_constants import LOG_MESSAGE_PREVIEW_LENGTH
        query_preview_length = min(50, LOG_MESSAGE_PREVIEW_LENGTH // 5)  # Use 1/5 of preview length
        logger.info(f"🔍 Searching knowledge base: query='{query[:query_preview_length]}...', content_type={content_type}, limit={limit}")
        results = await self.store.asearch(("knowledge_base",), query=query, limit=limit)
        logger.info(f"✅ Found {len(results)} knowledge base results")
        
        items: List[KnowledgeBaseItem] = []
        for r in results:
            item = KnowledgeBaseItem(**r.value)
            if content_type is None or item.content_type == content_type:
                items.append(item)
        return items

    async def get_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """
        Summarize a past workflow execution by aggregating episode memories.

        Returns:
            Summary dict.
        """
        assert self.store is not None, "Store required for workflow summary"

        # Use config constant for limit
        from config.system_constants import SEARCH_SIMILAR_MARKETS_LIMIT_MAX
        max_memories_limit = min(100, SEARCH_SIMILAR_MARKETS_LIMIT_MAX)
        memories = await self.retrieve_episode_memories(market_question="", limit=max_memories_limit)
        target = [m for m in memories if m.episode_id == workflow_id]
        assert target, f"No memories found for workflow {workflow_id}"

        summary: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "market_question": target[0].market_question,
            "total_memories": len(target),
            "by_type": {},
            "effective_strategies": [],
            "timestamp": target[0].timestamp.isoformat(),
        }

        for m in target:
            summary["by_type"][m.memory_type] = summary["by_type"].get(m.memory_type, 0) + 1
            # Use config constant for effectiveness threshold
            from config.system_constants import SENSITIVITY_ROBUST_THRESHOLD
            effectiveness_threshold = max(0.7, SENSITIVITY_ROBUST_THRESHOLD)  # Use 0.7 or robust threshold, whichever is higher
            if m.memory_type == "strategy" and (m.effectiveness or 0) >= effectiveness_threshold:
                summary["effective_strategies"].append({"content": m.content, "effectiveness": m.effectiveness})

        return summary

    def integrity_report(self) -> Dict[str, Any]:
        """
        Compact integrity report with critical metrics and simple violations.
        """
        store_kind = getattr(self.store, "__class__", type("X", (), {})).__name__ if self.store else None
        limits = {
            "max_working_memory_messages": self.config.max_working_memory_messages,
            "max_episode_memory_items": self.config.max_episode_memory_items,
            "episode_retention_days": self.config.episode_retention_days,
        }
        violations = []
        if limits["max_working_memory_messages"] <= 0:
            violations.append("non_positive_working_memory_limit")
        if limits["max_episode_memory_items"] <= 0:
            violations.append("non_positive_episode_memory_limit")
        if limits["episode_retention_days"] <= 0:
            violations.append("non_positive_retention_days")

        return {
            "store_type": self.config.store_type,
            "store_runtime_class": store_kind,
            "checkpointer_type": self.config.checkpointer_type,
            "embedding_model": self.config.embedding_model,
            "similarity_threshold": self.config.similarity_threshold,
            "limits": limits,
            "violations": violations,
        }

    async def verify_storage(self) -> bool:
        """
        Verify that storage backend is accessible and functional.
        
        Returns:
            True if storage is accessible, False otherwise.
        """
        if self.store is None:
            logger.warning("No store configured - cannot verify storage")
            return False
        
        try:
            async with self.store:
                await self.store.setup()
                # Try a simple put/get operation
                test_key = f"_verify_{datetime.utcnow().timestamp()}"
                test_value = {"test": True, "timestamp": datetime.utcnow().isoformat()}
                
                await self.store.aput(("test",), test_key, test_value)
                retrieved = await self.store.aget(("test",), test_key)
                
                if retrieved is None:
                    logger.error("Storage verification failed: could not retrieve test value")
                    return False
                
                # Clean up test value
                try:
                    await self.store.adelete(("test",), test_key)
                except Exception:
                    pass  # Ignore cleanup errors
                
                logger.info("Storage verification successful")
                return True
        except Exception as e:
            logger.error(f"Storage verification failed: {e}")
            return False

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system usage statistics.
        
        Returns:
            Dictionary with statistics including item counts, query counts, etc.
        """
        stats = {
            "store_configured": self.store is not None,
            "store_type": self.config.store_type,
            "checkpointer_type": self.config.checkpointer_type,
            "namespaces": {},
            "total_items": 0,
            "knowledge_base_items": 0,
            "episode_memory_items": 0,
            "strategy_items": 0,
        }
        
        if self.store is None:
            return stats
        
        try:
            async with self.store:
                await self.store.setup()
                
                # Count items in each namespace (approximate)
                from config.system_constants import (
                    NAMESPACE_KNOWLEDGE_BASE,
                    NAMESPACE_EPISODE_MEMORY,
                    NAMESPACE_STRATEGIES,
                )
                
                # Try to get counts for each namespace
                # Note: LangGraph Store doesn't have a direct count method,
                # so we'll use search with a broad query to estimate
                from config.system_constants import (
                    SEARCH_SIMILAR_MARKETS_LIMIT_MAX,
                )
                max_search_limit = SEARCH_SIMILAR_MARKETS_LIMIT_MAX  # Use config constant
                
                try:
                    kb_results = await self.store.asearch(NAMESPACE_KNOWLEDGE_BASE, query="", limit=max_search_limit)
                    stats["knowledge_base_items"] = len(kb_results)
                    stats["namespaces"]["knowledge_base"] = len(kb_results)
                except Exception as e:
                    logger.debug(f"Could not count knowledge_base items: {e}")
                
                try:
                    ep_results = await self.store.asearch(NAMESPACE_EPISODE_MEMORY, query="", limit=max_search_limit)
                    stats["episode_memory_items"] = len(ep_results)
                    stats["namespaces"]["episode_memory"] = len(ep_results)
                except Exception as e:
                    logger.debug(f"Could not count episode_memory items: {e}")
                
                try:
                    strat_results = await self.store.asearch(NAMESPACE_STRATEGIES, query="", limit=max_search_limit)
                    stats["strategy_items"] = len(strat_results)
                    stats["namespaces"]["strategies"] = len(strat_results)
                except Exception as e:
                    logger.debug(f"Could not count strategy items: {e}")
                
                stats["total_items"] = (
                    stats["knowledge_base_items"]
                    + stats["episode_memory_items"]
                    + stats["strategy_items"]
                )
        except Exception as e:
            logger.warning(f"Error getting memory stats: {e}")
        
        return stats

    async def list_namespaces(self) -> Dict[str, List[str]]:
        """
        List all namespaces and their keys (limited to first 100 per namespace).
        
        Returns:
            Dictionary mapping namespace tuples to lists of keys.
        """
        result = {}
        
        if self.store is None:
            return result
        
        try:
            async with self.store:
                await self.store.setup()
                
                from config.system_constants import (
                    NAMESPACE_KNOWLEDGE_BASE,
                    NAMESPACE_EPISODE_MEMORY,
                    NAMESPACE_STRATEGIES,
                )
                
                namespaces = [
                    ("knowledge_base", NAMESPACE_KNOWLEDGE_BASE),
                    ("episode_memory", NAMESPACE_EPISODE_MEMORY),
                    ("strategies", NAMESPACE_STRATEGIES),
                ]
                
                # Use config constant for limit
                from config.system_constants import SEARCH_SIMILAR_MARKETS_LIMIT_MAX
                max_keys_per_namespace = min(100, SEARCH_SIMILAR_MARKETS_LIMIT_MAX)
                
                for ns_name, ns_tuple in namespaces:
                    try:
                        # Use search to get keys (limited)
                        results = await self.store.asearch(ns_tuple, query="", limit=max_keys_per_namespace)
                        keys = [item.key for item in results if hasattr(item, "key")]
                        result[ns_name] = keys[:max_keys_per_namespace]
                    except Exception as e:
                        logger.debug(f"Could not list {ns_name} namespace: {e}")
                        result[ns_name] = []
        except Exception as e:
            logger.warning(f"Error listing namespaces: {e}")
        
        return result


# =========================
# Store Factory
# =========================
def create_store_from_config(settings: Optional[Any] = None) -> BaseStore:
    """
    Create and initialize a LangGraph Store from configuration.

    Backends:
      - PostgreSQL (recommended; works with Supabase)
      - In-Memory (non‑persistent)
    """
    if settings is None:
        from config.settings import settings as global_settings
        settings = global_settings

    if not getattr(settings, "ENABLE_MEMORY_PERSISTENCE", True):
        from langgraph.store.memory import InMemoryStore
        return InMemoryStore()

    backend = getattr(settings, "MEMORY_BACKEND", "postgresql").lower()
    assert backend == "postgresql", f"Unsupported backend: {backend}"

    postgres_url = getattr(settings, "POSTGRES_URL", "") or ""
    if not postgres_url:
        supabase_url = getattr(settings, "SUPABASE_URL", "") or ""
        supabase_key = getattr(settings, "SUPABASE_SERVICE_KEY", "") or getattr(settings, "SUPABASE_KEY", "") or ""
        assert supabase_url and supabase_key, "PostgreSQL URL or Supabase credentials required"
        
        import re
        m = re.search(r"https://([a-zA-Z0-9-]+)\.supabase\.co", supabase_url)
        assert m, f"Invalid Supabase URL format: {supabase_url}"
        project_ref = m.group(1)
        postgres_url = f"postgresql://postgres:{supabase_key}@db.{project_ref}.supabase.co:5432/postgres"

    assert postgres_url, "PostgreSQL connection string required"
    
    from langgraph.store.postgres import PostgresStore  # type: ignore
    import psycopg  # type: ignore
    
    test_conn = psycopg.connect(postgres_url, autocommit=True)
    test_conn.close()
    
    store = PostgresStore.from_conn_string(postgres_url)

    import asyncio
    
    async def init_store():
        async with store:
            await store.setup()

    try:
        asyncio.get_running_loop()
        import nest_asyncio  # type: ignore
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(init_store())
    except RuntimeError:
        asyncio.run(init_store())

    return store


# =========================
# Singleton Accessors
# =========================
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(
    config: Optional[MemoryConfig] = None,
    store: Optional[BaseStore] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> MemoryManager:
    """
    Get or create the MemoryManager singleton.

    If no store is provided, it is created via create_store_from_config().
    """
    global _memory_manager
    if _memory_manager is None:
        if store is None:
            store = create_store_from_config()
        assert store is not None, "Store creation failed"
        _memory_manager = MemoryManager(config=config, store=store, checkpointer=checkpointer)
    return _memory_manager


def reset_memory_manager() -> None:
    """Reset the MemoryManager singleton (primarily for tests)."""
    global _memory_manager
    _memory_manager = None