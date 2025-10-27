"""
Memory Search Tools
Enables agents to search historical analyses and learnings
"""
import logging
import os
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
from langgraph.store.base import SearchItem

from arbee.utils.memory import get_memory_manager
from config.system_constants import (
    SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT,
    SEARCH_SIMILAR_MARKETS_LIMIT_MAX,
    SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT,
    GET_BASE_RATES_LIMIT_DEFAULT,
    NAMESPACE_KNOWLEDGE_BASE,
    NAMESPACE_EPISODE_MEMORY,
    NAMESPACE_STRATEGIES
)

logger = logging.getLogger(__name__)


@tool
async def search_similar_markets_tool(
    market_question: str,
    limit: int = SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT
) -> List[Dict[str, Any]]:
    """
    Search for similar market questions analyzed in the past.

    Use this to find analogous cases that can inform your current analysis.
    Helps with setting priors, identifying relevant evidence types, and
    learning from successful strategies.

    Args:
        market_question: Current market question to find similar cases for
        limit: Maximum number of similar markets to return

    Returns:
        List of similar market analyses with question, outcome, prior, posterior, etc.

    Example:
        >>> similar = await search_similar_markets_tool("Will Trump win 2024 election?")
        >>> for market in similar:
        ...     print(f"{market['question']}: prior={market['prior']}, outcome={market['outcome']}")
    """
    try:
        logger.info(f"🔍 Searching for similar markets to: '{market_question[:60]}'")

        limit = max(1, min(limit, SEARCH_SIMILAR_MARKETS_LIMIT_MAX))

        store_results = await _search_similar_markets_in_store(
            query=market_question,
            limit=limit
        )
        if store_results:
            return store_results

        logger.info("No similar markets found in LangGraph store")
        return []

    except Exception as e:
        logger.error(f"❌ Similar markets search failed: {e}")
        return []


@tool
async def search_historical_evidence_tool(
    topic: str,
    evidence_type: Optional[str] = None,
    limit: int = SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT
) -> List[Dict[str, Any]]:
    """
    Search historical evidence database for relevant past findings.

    Use this to check if evidence about this topic has been gathered before,
    avoiding redundant research and building on previous work.

    Args:
        topic: Topic or subclaim to search for
        evidence_type: Optional filter by evidence type (poll, news, study, etc.)
        limit: Maximum results

    Returns:
        List of historical evidence items

    Example:
        >>> evidence = await search_historical_evidence_tool("Arizona swing state polls")
        >>> print(f"Found {len(evidence)} relevant historical evidence items")
    """
    try:
        logger.info(f"🔍 Searching historical evidence: '{topic[:60]}'")

        memory_manager = get_memory_manager()
        store = getattr(memory_manager, "store", None)

        if not store:
            logger.debug("No LangGraph store configured; cannot search historical evidence")
            return []

        # Build filter
        filter_dict = {"content_type": "evidence_item"}
        if evidence_type:
            filter_dict["evidence_type"] = evidence_type

        # Search knowledge base namespace for evidence
        try:
            search_results = await store.asearch(
                (NAMESPACE_KNOWLEDGE_BASE,),
                query=topic,
                filter=filter_dict,
                limit=limit
            )
        except Exception as err:
            logger.warning(f"LangGraph store search failed: {err}")
            return []

        # Parse results
        evidence_list = []
        for item in search_results:
            value = item.value or {}
            content = value.get("content")

            # Extract evidence details
            if isinstance(content, dict):
                evidence_item = {
                    "id": value.get("id") or item.key,
                    "title": content.get("title", "Unknown"),
                    "url": content.get("url", ""),
                    "llr": content.get("LLR", 0.0),
                    "verifiability": content.get("verifiability_score", 0.5),
                    "independence": content.get("independence_score", 0.8),
                    "recency": content.get("recency_score", 0.7),
                    "support": content.get("support", "neutral"),
                    "claim_summary": content.get("claim_summary", ""),
                    "stored_at": item.updated_at.isoformat() if item.updated_at else "",
                    "relevance_score": item.score
                }
                evidence_list.append(evidence_item)

        logger.info(f"✅ Found {len(evidence_list)} historical evidence items for '{topic[:40]}'")
        return evidence_list

    except Exception as e:
        logger.error(f"❌ Historical evidence search failed: {e}")
        return []


@tool
async def get_base_rates_tool(
    event_category: str,
    time_range: Optional[str] = None,
    limit: int = GET_BASE_RATES_LIMIT_DEFAULT
) -> Dict[str, Any]:
    """
    Retrieve historical base rates for event category.

    Use this to inform prior probability selection with reference class data.

    Args:
        event_category: Category of event (e.g., "US presidential elections", "tech IPOs")
        time_range: Optional time range (e.g., "2000-2020")

    Returns:
        Dict with base_rate, sample_size, confidence, and examples

    Example:
        >>> rates = await get_base_rates_tool("incumbent party wins presidential election")
        >>> print(f"Base rate: {rates['base_rate']:.1%}")
    """
    try:
        logger.info(f"📊 Looking up base rates for: '{event_category}'")

        memory_manager = get_memory_manager()
        store = getattr(memory_manager, "store", None)

        if store:
            # Search knowledge base for stored base rates
            try:
                filter_dict = {"content_type": "base_rate"}
                if time_range:
                    filter_dict["time_range"] = time_range

                search_results = await store.asearch(
                    (NAMESPACE_KNOWLEDGE_BASE,),
                    query=event_category,
                    filter=filter_dict,
                    limit=limit
                )

                # If we found stored base rates, aggregate them
                if search_results and len(search_results) > 0:
                    rates = []
                    sources = []
                    for item in search_results:
                        value = item.value or {}
                        content = value.get("content")
                        if isinstance(content, dict):
                            rate = content.get("base_rate")
                            if rate and 0.0 <= rate <= 1.0:
                                rates.append(rate)
                                sources.append(content.get("source", "Unknown"))

                    if rates:
                        # Aggregate by averaging (could use weighted average based on sample sizes)
                        avg_rate = sum(rates) / len(rates)
                        logger.info(f"✅ Found {len(rates)} stored base rates, average: {avg_rate:.2%}")

                        return {
                            'event_category': event_category,
                            'base_rate': avg_rate,
                            'sample_size': len(rates),
                            'confidence': 'moderate' if len(rates) >= 3 else 'low',
                            'sources': sources[:3],
                            'note': f'Aggregated from {len(rates)} historical analyses'
                        }

            except Exception as err:
                logger.warning(f"Store search failed: {err}")

        # No stored base rates found - return neutral prior
        logger.info(f"📡 No stored base rates found for '{event_category}', using neutral prior")
        return {
            'event_category': event_category,
            'base_rate': 0.5,
            'sample_size': 0,
            'confidence': 'low',
            'note': 'No stored base rate data found, using neutral 50% prior'
        }

    except Exception as e:
        logger.error(f"❌ Base rates lookup failed: {e}")
        return {'error': str(e), 'base_rate': 0.5, 'confidence': 'low'}


@tool
async def store_successful_strategy_tool(
    strategy_type: str,
    description: str,
    effectiveness: float,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Store a successful research or analysis strategy for future reference.

    Use this to record what worked well so future agents can learn from it.

    Args:
        strategy_type: Type of strategy (e.g., "search_strategy", "evidence_filtering")
        description: Description of the strategy
        effectiveness: How effective it was (0.0 to 1.0)
        metadata: Optional additional context

    Returns:
        True if stored successfully

    Example:
        >>> await store_successful_strategy_tool(
        ...     "search_strategy",
        ...     "Combining '538 poll' with state name gives high-quality polling data",
        ...     effectiveness=0.9
        ... )
    """
    try:
        logger.info(f"💾 Storing successful strategy: {strategy_type}")

        memory_manager = get_memory_manager()
        store = getattr(memory_manager, "store", None)

        if not store:
            logger.warning("No LangGraph store configured; cannot store strategy")
            return False

        # Validate effectiveness
        if not (0.0 <= effectiveness <= 1.0):
            logger.warning(f"Invalid effectiveness {effectiveness}, clamping to [0, 1]")
            effectiveness = max(0.0, min(1.0, effectiveness))

        # Prepare strategy data
        strategy_data = {
            "content_type": "strategy",
            "strategy_type": strategy_type,
            "description": description,
            "effectiveness": effectiveness,
            "stored_at": "current_session",
            "metadata": metadata or {}
        }

        # Generate unique key
        import hashlib
        import time
        key_str = f"{strategy_type}_{description[:30]}_{time.time()}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
        key = f"strategy_{key_hash}"

        # Store in strategies namespace
        try:
            await store.aput(
                NAMESPACE_STRATEGIES,
                key,
                strategy_data
            )

            logger.info(f"✅ Stored strategy '{strategy_type}' with effectiveness {effectiveness:.1%}")
            return True

        except Exception as err:
            logger.error(f"Failed to store in LangGraph store: {err}")
            return False

    except Exception as e:
        logger.error(f"❌ Strategy storage failed: {e}")
        return False


async def _search_similar_markets_in_store(
    query: str,
    limit: int
) -> List[Dict[str, Any]]:
    """
    Search the LangGraph store for similar market analyses using semantic search.

    Args:
        query: Market question to search for
        limit: Maximum number of results to return

    Returns:
        List of similar market analyses dictionaries
    """
    memory_manager = get_memory_manager()
    store = getattr(memory_manager, "store", None)

    if not store:
        logger.debug("No LangGraph store configured; skipping memory search lookup")
        return []

    try:
        # Primary namespace for long-term knowledge
        search_results = await store.asearch(
            (NAMESPACE_KNOWLEDGE_BASE,),
            query=query,
            filter={"content_type": "market_analysis"},
            limit=limit
        )
    except Exception as err:
        logger.warning(f"LangGraph store search failed: {err}")
        return []

    parsed = [
        result
        for item in search_results
        if (result := _parse_market_search_item(item)) is not None
    ]

    return parsed[:limit]


def _parse_market_search_item(item: SearchItem) -> Optional[Dict[str, Any]]:
    """
    Normalize LangGraph search items into the tool output structure.

    Args:
        item: SearchItem returned by LangGraph store

    Returns:
        Parsed dictionary or None if the data is incomplete
    """
    value = item.value or {}
    content = value.get("content")

    if isinstance(content, dict):
        content_dict = content.copy()
    elif isinstance(content, str):
        content_dict = {"analysis": content}
    else:
        content_dict = {}

    question = (
        content_dict.get("market_question")
        or content_dict.get("question")
        or value.get("market_question")
        or value.get("question")
    )

    if not question:
        return None

    metadata = {}
    value_metadata = value.get("metadata")
    if isinstance(value_metadata, dict):
        metadata.update(value_metadata)
    content_metadata = content_dict.get("metadata")
    if isinstance(content_metadata, dict):
        metadata.update(content_metadata)

    result: Dict[str, Any] = {
        "id": value.get("id") or item.key,
        "question": question,
        "analysis": content_dict or value,
        "score": item.score,
        "stored_at": item.updated_at.isoformat(),
    }

    for field in ("prior", "posterior", "outcome", "market_url", "market_id", "workflow_id"):
        if field in content_dict:
            result[field] = content_dict[field]
        elif field in value:
            result[field] = value[field]

    if metadata:
        result["metadata"] = metadata

    return result
