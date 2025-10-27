"""
Memory Infrastructure for POLYSEER Autonomous Agents
Provides utilities for short-term, episodic, and long-term memory
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import logging

from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.postgres import PostgresStore
import nest_asyncio

logger = logging.getLogger(__name__)


class MemoryConfig(BaseModel):
    """Configuration for memory system"""
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
    """Item in agent's working memory (current workflow)"""
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    key: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpisodeMemoryItem(BaseModel):
    """Item in episode memory (learnings from this analysis)"""
    episode_id: str  # Workflow ID
    market_question: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memory_type: str  # "strategy", "evidence", "outcome", "lesson"
    content: Any
    effectiveness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseItem(BaseModel):
    """Item in knowledge base (historical cross-workflow knowledge)"""
    id: str
    content_type: str  # "market_analysis", "evidence", "base_rate", "pattern"
    content: Any
    embedding: Optional[List[float]] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryManager:
    """
    Centralized memory management for POLYSEER agents

    Manages three tiers of memory:
    1. Working Memory (Checkpointer) - Current workflow state
    2. Episode Memory (Store) - Learnings from this analysis
    3. Knowledge Base (Vector DB) - Historical cross-workflow knowledge
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        store: Optional[BaseStore] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None
    ):
        """
        Initialize memory manager

        Args:
            config: Memory configuration
            store: LangGraph Store instance
            checkpointer: LangGraph Checkpointer instance
        """
        self.config = config or MemoryConfig()
        self.store = store
        self.checkpointer = checkpointer

        logger.info(
            f"MemoryManager initialized: store={self.config.store_type}, "
            f"checkpointer={self.config.checkpointer_type}"
        )

    async def store_working_memory(
        self,
        agent_name: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store item in working memory (current workflow)

        This is handled by LangGraph's checkpointer automatically,
        but this method provides explicit storage for important items.

        Args:
            agent_name: Name of agent storing memory
            key: Memory key
            value: Value to store
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        item = WorkingMemoryItem(
            agent_name=agent_name,
            key=key,
            value=value,
            metadata=metadata or {}
        )

        logger.info(f"💾 Working memory: {agent_name}/{key}")

        # Working memory is handled by checkpointer automatically
        # This is just for logging/tracking
        return True

    async def store_episode_memory(
        self,
        episode_id: str,
        market_question: str,
        memory_type: str,
        content: Any,
        effectiveness: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store learnings from current episode for future similar cases

        Args:
            episode_id: Workflow ID
            market_question: Market question being analyzed
            memory_type: Type of memory (strategy, evidence, outcome, lesson)
            content: Memory content
            effectiveness: How effective this was (0-1)
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        if not self.store:
            logger.warning("No store configured, episode memory not saved")
            return False

        try:
            item = EpisodeMemoryItem(
                episode_id=episode_id,
                market_question=market_question,
                memory_type=memory_type,
                content=content,
                effectiveness=effectiveness,
                metadata=metadata or {}
            )

            # Store in LangGraph Store
            await self.store.aput(
                ("episode_memory",),
                f"{episode_id}:{memory_type}:{datetime.utcnow().isoformat()}",
                item.model_dump()
            )

            logger.info(f"💾 Episode memory: {memory_type} for {market_question[:50]}")
            return True

        except Exception as e:
            logger.error(f"Failed to store episode memory: {e}")
            return False

    async def retrieve_episode_memories(
        self,
        market_question: str,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[EpisodeMemoryItem]:
        """
        Retrieve episode memories from similar market analyses

        Args:
            market_question: Query to find similar markets
            memory_type: Optional filter by memory type
            limit: Maximum results

        Returns:
            List of relevant episode memories
        """
        if not self.store:
            logger.warning("No store configured, cannot retrieve episode memories")
            return []

        try:
            # Search for similar episodes
            # Note: This is a simplified implementation
            # In production, use vector search on market questions

            results = await self.store.asearch(
                ("episode_memory",),
                query=market_question,
                limit=limit
            )

            memories = []
            for result in results:
                try:
                    item = EpisodeMemoryItem(**result.value)
                    if memory_type is None or item.memory_type == memory_type:
                        memories.append(item)
                except Exception as e:
                    logger.warning(f"Failed to parse episode memory: {e}")

            logger.info(f"🔍 Retrieved {len(memories)} episode memories")
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve episode memories: {e}")
            return []

    async def store_knowledge(
        self,
        content_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: bool = True
    ) -> Optional[str]:
        """
        Store item in knowledge base for long-term cross-workflow access

        Args:
            content_type: Type of knowledge (market_analysis, evidence, etc.)
            content: Knowledge content
            metadata: Additional metadata
            generate_embedding: Whether to generate embedding for vector search

        Returns:
            Knowledge ID if stored successfully
        """
        if not self.store:
            logger.warning("No store configured, knowledge not saved")
            return None

        try:
            knowledge_id = f"{content_type}:{datetime.utcnow().timestamp()}"

            item = KnowledgeBaseItem(
                id=knowledge_id,
                content_type=content_type,
                content=content,
                metadata=metadata or {}
            )

            # TODO: Generate embedding if requested
            # if generate_embedding:
            #     item.embedding = await self.generate_embedding(str(content))

            await self.store.aput(
                ("knowledge_base",),
                knowledge_id,
                item.model_dump()
            )

            logger.info(f"💾 Knowledge: {content_type}")
            return knowledge_id

        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            return None

    async def search_knowledge(
        self,
        query: str,
        content_type: Optional[str] = None,
        limit: int = 10
    ) -> List[KnowledgeBaseItem]:
        """
        Search knowledge base for relevant information

        Args:
            query: Search query
            content_type: Optional filter by content type
            limit: Maximum results

        Returns:
            List of relevant knowledge items
        """
        if not self.store:
            logger.warning("No store configured, cannot search knowledge")
            return []

        try:
            # TODO: Use vector search when embeddings are implemented
            results = await self.store.asearch(
                ("knowledge_base",),
                query=query,
                limit=limit
            )

            items = []
            for result in results:
                try:
                    item = KnowledgeBaseItem(**result.value)
                    if content_type is None or item.content_type == content_type:
                        items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to parse knowledge item: {e}")

            logger.info(f"🔍 Found {len(items)} knowledge items")
            return items

        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []

    async def get_workflow_summary(
        self,
        workflow_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get summary of a past workflow execution

        Useful for learning from similar past analyses

        Args:
            workflow_id: Workflow ID to summarize

        Returns:
            Workflow summary dict
        """
        if not self.store:
            return None

        try:
            # Get all episode memories for this workflow
            memories = await self.retrieve_episode_memories(
                market_question="",  # Get all for this episode
                limit=100
            )

            # Filter to this workflow
            workflow_memories = [
                m for m in memories if m.episode_id == workflow_id
            ]

            if not workflow_memories:
                return None

            # Summarize
            summary = {
                "workflow_id": workflow_id,
                "market_question": workflow_memories[0].market_question,
                "total_memories": len(workflow_memories),
                "by_type": {},
                "effective_strategies": [],
                "timestamp": workflow_memories[0].timestamp.isoformat()
            }

            # Group by type
            for memory in workflow_memories:
                memory_type = memory.memory_type
                summary["by_type"][memory_type] = summary["by_type"].get(memory_type, 0) + 1

                # Collect effective strategies
                if memory.memory_type == "strategy" and memory.effectiveness and memory.effectiveness > 0.7:
                    summary["effective_strategies"].append({
                        "content": memory.content,
                        "effectiveness": memory.effectiveness
                    })

            return summary

        except Exception as e:
            logger.error(f"Failed to get workflow summary: {e}")
            return None


def create_store_from_config(
    settings: Optional[Any] = None
) -> Optional[BaseStore]:
    """
    Create and initialize LangGraph Store from configuration

    Supports multiple backends:
    - PostgreSQL (recommended for production, works with Supabase)
    - Redis (fast, requires Redis server)
    - In-Memory (testing only, not persistent)

    Args:
        settings: Optional Settings instance (defaults to global settings)

    Returns:
        Initialized BaseStore instance, or None if persistence disabled
    """
    # Import settings if not provided
    if settings is None:
        try:
            from config.settings import settings as global_settings
            settings = global_settings
        except ImportError:
            logger.warning("Could not import settings, using in-memory store")
            from langgraph.store.memory import InMemoryStore
            return InMemoryStore()

    # Check if persistence is enabled
    if not getattr(settings, 'ENABLE_MEMORY_PERSISTENCE', True):
        logger.info("Memory persistence disabled, using InMemoryStore")
        from langgraph.store.memory import InMemoryStore
        return InMemoryStore()

    # Get backend type
    backend = getattr(settings, 'MEMORY_BACKEND', 'postgresql').lower()

    # PostgreSQL / Supabase Backend
    if backend == 'postgresql':
        try:
            # Try POSTGRES_URL first, then construct from Supabase
            postgres_url = getattr(settings, 'POSTGRES_URL', '')

            if not postgres_url:
                # Construct from Supabase settings
                supabase_url = getattr(settings, 'SUPABASE_URL', '')
                supabase_key = getattr(settings, 'SUPABASE_SERVICE_KEY', '') or getattr(settings, 'SUPABASE_KEY', '')

                if supabase_url and supabase_key:
                    # Convert Supabase URL to PostgreSQL connection string
                    # Format: postgresql://postgres:[PASSWORD]@[HOST]/postgres
                    # Supabase URL format: https://[PROJECT_REF].supabase.co
                    import re
                    project_ref_match = re.search(r'https://([a-zA-Z0-9-]+)\.supabase\.co', supabase_url)

                    if project_ref_match:
                        project_ref = project_ref_match.group(1)
                        # Supabase PostgreSQL host format
                        postgres_host = f"db.{project_ref}.supabase.co"
                        postgres_url = f"postgresql://postgres:{supabase_key}@{postgres_host}:5432/postgres"
                        logger.info(f"Constructed PostgreSQL URL from Supabase settings")
                    else:
                        logger.warning("Could not parse Supabase URL format")

            if postgres_url:
                logger.info(f"Initializing PostgresStore with connection string")
                logger.info(f"Connection string format: postgresql://postgres:***@{postgres_url.split('@')[1] if '@' in postgres_url else 'unknown'}")

                # PostgresStore needs proper async initialization
                try:
                    import asyncio

                    # Test connection first
                    try:
                        import psycopg
                        test_conn = psycopg.connect(postgres_url, autocommit=True)
                        test_conn.close()
                        logger.info("✅ PostgreSQL connection test successful")
                    except Exception as conn_err:
                        logger.error(f"❌ PostgreSQL connection test failed: {conn_err}")
                        logger.error(f"   Check your SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")
                        logger.error(f"   Expected format: SUPABASE_URL=https://xxxxx.supabase.co")
                        raise

                    # Create store using connection string directly
                    # LangGraph's PostgresStore.from_conn_string() handles the pool internally
                    store = PostgresStore.from_conn_string(postgres_url)

                    # Setup schema (create tables if they don't exist)
                    async def init_store():
                        async with store:
                            await store.setup()

                    try:
                        asyncio.run(init_store())
                        logger.info("✅ PostgresStore initialized successfully with schema")
                    except RuntimeError as runtime_err:
                        # Already in an event loop, try nested approach
                        logger.info("Already in event loop, using nest_asyncio...")
                        nest_asyncio.apply()
                        asyncio.run(init_store())
                        logger.info("✅ PostgresStore initialized (nested event loop)")

                    return store

                except ImportError as import_err:
                    logger.error(f"Missing dependencies: {import_err}")
                    logger.error("Install with: pip install 'langgraph-checkpoint-postgres[pool]' psycopg[binary]")
                except Exception as setup_err:
                    logger.error(f"PostgresStore initialization failed: {setup_err}")
                    logger.warning("Falling back to InMemoryStore")
                    import traceback
                    traceback.print_exc()

        except ImportError as e:
            logger.warning(f"PostgresStore not available: {e}. Install with: pip install langgraph-checkpoint-postgres")
        except Exception as e:
            logger.error(f"Failed to initialize PostgresStore: {e}")

    # In-Memory Backend (default fallback)
    logger.info("Using InMemoryStore (not persistent across restarts)")
    from langgraph.store.memory import InMemoryStore
    return InMemoryStore()


# Singleton memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(
    config: Optional[MemoryConfig] = None,
    store: Optional[BaseStore] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None
) -> MemoryManager:
    """
    Get or create memory manager singleton

    If no store is provided, automatically creates one from settings using
    create_store_from_config(). This ensures memory persistence works by default.

    Args:
        config: Memory configuration (only used on first call)
        store: LangGraph Store instance (only used on first call, auto-created if None)
        checkpointer: Checkpointer instance (only used on first call)

    Returns:
        MemoryManager instance with initialized store
    """
    global _memory_manager

    if _memory_manager is None:
        # Auto-create store from config if not provided
        if store is None:
            logger.info("No store provided, auto-initializing from settings...")
            store = create_store_from_config()

        _memory_manager = MemoryManager(config=config, store=store, checkpointer=checkpointer)

    return _memory_manager


def reset_memory_manager():
    """Reset memory manager (mainly for testing)"""
    global _memory_manager
    _memory_manager = None
