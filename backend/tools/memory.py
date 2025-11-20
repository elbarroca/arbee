"""
Memory Tools
Enables agents to search historical analyses and learnings, plus diagnostic tools.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langgraph.store.base import SearchItem
from utils.rich_logging import (
    log_tool_start,
    log_tool_success,
    log_tool_error,
)
from utils.memory import get_memory_manager
from config.system_constants import (
    SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT,
    SEARCH_SIMILAR_MARKETS_LIMIT_MAX,
    SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT,
    GET_BASE_RATES_LIMIT_DEFAULT,
    NAMESPACE_KNOWLEDGE_BASE,
    NAMESPACE_STRATEGIES,
    NAMESPACE_EPISODE_MEMORY,
    DEFAULT_VERIFIABILITY_SCORE,
    DEFAULT_INDEPENDENCE_SCORE,
    DEFAULT_RECENCY_SCORE,
)
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Internal helpers
# ============================================================================

def _clamp_limit(limit: int, default: int, max_val: int) -> int:
    """Clamp user-provided limits to safe bounds."""
    try:
        return max(1, min(int(limit), int(max_val)))
    except Exception:
        return int(default)


def _get_store():
    """Return the LangGraph store if configured, else None."""
    mm = get_memory_manager()
    return getattr(mm, "store", None)


# ============================================================================
# Memory Search Tools (from memory_search.py)
# ============================================================================

@tool
async def search_similar_markets_tool(
    market_question: str,
    limit: int = SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Find past market analyses similar to the current market question.

    Args:
        market_question: Market question to match.
        limit: Maximum number of results.

    Returns:
        List of dicts with keys like: id, question, analysis, score, stored_at, and optional prior/posterior/outcome/market_url/market_id/workflow_id.
    """
    assert isinstance(market_question, str) and market_question.strip(), "market_question is required"
    limit = _clamp_limit(limit, SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT, SEARCH_SIMILAR_MARKETS_LIMIT_MAX)
    return await _search_similar_markets_in_store(query=market_question, limit=limit)


@tool
async def search_historical_evidence_tool(
    topic: str,
    evidence_type: Optional[str] = None,
    limit: int = SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Search previously stored evidence items in the knowledge base.

    Args:
        topic: Topic or subclaim to search for.
        evidence_type: Optional evidence subtype (e.g., poll, news, study).
        limit: Maximum results.

    Returns:
        List of simplified evidence dicts.
    """
    try:
        log_tool_start("search_historical_evidence_tool", {"topic": topic[:80], "evidence_type": evidence_type, "limit": limit})
        assert isinstance(topic, str) and topic.strip(), "topic is required"
        limit = max(1, int(limit))

        store = _get_store()
        if not store:
            log_tool_success("search_historical_evidence_tool", {"results_count": 0, "note": "No store configured"})
            return []

        filt: Dict[str, Any] = {"content_type": "evidence_item"}
        if evidence_type:
            filt["evidence_type"] = evidence_type

        results = await store.asearch((NAMESPACE_KNOWLEDGE_BASE,), query=topic, filter=filt, limit=limit)
        
        out: List[Dict[str, Any]] = []
        for item in results:
            val = item.value or {}
            content = val.get("content")
            if isinstance(content, dict):
                out.append(
                    {
                        "id": val.get("id") or item.key,
                        "title": content.get("title", "Unknown"),
                        "url": content.get("url", ""),
                        "llr": content.get("LLR", 0.0),
                        "verifiability": content.get("verifiability_score", DEFAULT_VERIFIABILITY_SCORE),
                        "independence": content.get("independence_score", DEFAULT_INDEPENDENCE_SCORE),
                        "recency": content.get("recency_score", DEFAULT_RECENCY_SCORE),
                        "support": content.get("support", "neutral"),
                        "claim_summary": content.get("claim_summary", ""),
                        "stored_at": item.updated_at.isoformat() if getattr(item, "updated_at", None) else "",
                        "relevance_score": getattr(item, "score", None),
                    }
                )
        
        log_tool_success("search_historical_evidence_tool", {"results_count": len(out)})
        return out
    except Exception as e:
        log_tool_error("search_historical_evidence_tool", e, f"Topic: {topic[:50]}")
        return []


@tool
async def get_base_rates_tool(
    event_category: str,
    time_range: Optional[str] = None,
    limit: int = GET_BASE_RATES_LIMIT_DEFAULT,
) -> Dict[str, Any]:
    """
    Retrieve historical base rates for a reference class.

    Args:
        event_category: Reference class label (e.g., "incumbent wins presidential election").
        time_range: Optional time range tag to match stored base rates (e.g., "2000-2020").
        limit: Max stored items to aggregate.

    Returns:
        Dict with base_rate (0..1), sample_size, confidence, and optional sources list.
    """
    assert isinstance(event_category, str) and event_category.strip(), "event_category is required"
    limit = max(1, int(limit))

    try:
        log_tool_start("get_base_rates_tool", {"event_category": event_category[:80], "time_range": time_range, "limit": limit})
        store = _get_store()
        if store:
            filt: Dict[str, Any] = {"content_type": "base_rate"}
            if time_range:
                filt["time_range"] = time_range

            try:
                results = await store.asearch((NAMESPACE_KNOWLEDGE_BASE,), query=event_category, filter=filt, limit=limit)
            except Exception:
                results = []

            if results:
                rates: List[float] = []
                sources: List[str] = []
                for it in results:
                    val = it.value or {}
                    content = val.get("content")
                    if isinstance(content, dict):
                        rate = content.get("base_rate")
                        if isinstance(rate, (int, float)) and 0.0 <= rate <= 1.0:
                            rates.append(float(rate))
                            src = content.get("source", "Unknown")
                            if isinstance(src, str):
                                sources.append(src)
                if rates:
                    avg = sum(rates) / len(rates)
                    result = {
                        "event_category": event_category,
                        "base_rate": avg,
                        "sample_size": len(rates),
                        "confidence": "moderate" if len(rates) >= 3 else "low",
                        "sources": sources[:3],
                        "note": f"Aggregated from {len(rates)} stored base-rate items",
                    }
                    log_tool_success("get_base_rates_tool", {"base_rate": avg, "sample_size": len(rates), "confidence": result["confidence"]})
                    return result

        # Fallback neutral prior when no stored base rates available
        from config.system_constants import PRIOR_ESTIMATION_MIN, PRIOR_ESTIMATION_MAX
        neutral_prior = (PRIOR_ESTIMATION_MIN + PRIOR_ESTIMATION_MAX) / 2.0  # Use midpoint of estimation range
        result = {
            "event_category": event_category,
            "base_rate": neutral_prior,
            "sample_size": 0,
            "confidence": "low",
            "note": f"No stored base-rate data; returning neutral {neutral_prior:.0%} prior",
        }
        log_tool_success("get_base_rates_tool", {"base_rate": neutral_prior, "sample_size": 0, "note": "No data found"})
        return result
    except Exception as e:
        log_tool_error("get_base_rates_tool", e, f"Category: {event_category[:50]}")
        from config.system_constants import PRIOR_ESTIMATION_MIN, PRIOR_ESTIMATION_MAX
        neutral_prior = (PRIOR_ESTIMATION_MIN + PRIOR_ESTIMATION_MAX) / 2.0
        return {
            "event_category": event_category,
            "base_rate": neutral_prior,
            "sample_size": 0,
            "confidence": "low",
            "note": f"No stored base-rate data; returning neutral {neutral_prior:.0%} prior",
            "error": str(e),
        }


@tool
async def store_successful_strategy_tool(
    strategy_type: str,
    description: str,
    effectiveness: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Persist a successful research/analysis strategy for future reuse.

    Args:
        strategy_type: Strategy category (e.g., "search_strategy").
        description: Human-readable description.
        effectiveness: Effectiveness in [0.0, 1.0].
        metadata: Optional extra context.

    Returns:
        True if stored successfully, else False.
    """
    assert strategy_type and description, "strategy_type and description are required"
    try:
        eff = float(effectiveness)
    except Exception:
        eff = 0.0
    eff = max(0.0, min(1.0, eff))

    store = _get_store()
    if not store:
        return False

    import hashlib
    import time

    key_src = f"{strategy_type}_{description[:40]}_{time.time()}"
    key = f"strategy_{hashlib.md5(key_src.encode()).hexdigest()[:12]}"

    data = {
        "content_type": "strategy",
        "strategy_type": strategy_type,
        "description": description,
        "effectiveness": eff,
        "stored_at": "current_session",
        "metadata": metadata or {},
    }

    try:
        await store.aput(NAMESPACE_STRATEGIES, key, data)
        return True
    except Exception:
        return False


# ============================================================================
# Memory Diagnostic Tools (from memory_diagnostics.py)
# ============================================================================

@tool
async def verify_memory_system_tool() -> Dict[str, Any]:
    """
    Verify that the memory system is properly configured and accessible.
    
    Returns:
        Dictionary with verification results including:
        - store_configured: Whether store is configured
        - storage_accessible: Whether storage backend is accessible
        - integrity_report: Integrity report from memory manager
        - status: Overall status ("healthy", "degraded", "unavailable")
    """
    try:
        log_tool_start("verify_memory_system_tool", {})
        
        mm = get_memory_manager()
        store_configured = mm.store is not None
        
        result = {
            "store_configured": store_configured,
            "storage_accessible": False,
            "integrity_report": mm.integrity_report(),
            "status": "unavailable",
        }
        
        if store_configured:
            # Verify storage accessibility
            storage_accessible = await mm.verify_storage()
            result["storage_accessible"] = storage_accessible
            
            if storage_accessible:
                result["status"] = "healthy"
            else:
                result["status"] = "degraded"
                result["error"] = "Storage backend not accessible"
        else:
            result["status"] = "unavailable"
            result["error"] = "Memory store not configured"
            result["note"] = "Set ENABLE_MEMORY_PERSISTENCE=True and configure database credentials"
        
        log_tool_success("verify_memory_system_tool", {
            "status": result["status"],
            "store_configured": store_configured,
            "storage_accessible": result.get("storage_accessible", False),
        })
        
        return result
    except Exception as e:
        log_tool_error("verify_memory_system_tool", e)
        return {
            "store_configured": False,
            "storage_accessible": False,
            "status": "error",
            "error": str(e),
        }


@tool
async def get_memory_stats_tool() -> Dict[str, Any]:
    """
    Get memory system usage statistics.
    
    Returns:
        Dictionary with statistics including:
        - store_type: Type of store backend
        - total_items: Total number of items stored
        - knowledge_base_items: Number of knowledge base items
        - episode_memory_items: Number of episode memory items
        - strategy_items: Number of strategy items
        - namespaces: Dictionary with item counts per namespace
    """
    try:
        log_tool_start("get_memory_stats_tool", {})
        
        mm = get_memory_manager()
        stats = await mm.get_memory_stats()
        
        log_tool_success("get_memory_stats_tool", {
            "total_items": stats.get("total_items", 0),
            "store_type": stats.get("store_type", "unknown"),
        })
        
        return stats
    except Exception as e:
        log_tool_error("get_memory_stats_tool", e)
        return {
            "error": str(e),
            "store_configured": False,
            "total_items": 0,
        }


@tool
async def test_memory_query_tool(query: str, namespace: str = "knowledge_base", limit: int = 5) -> Dict[str, Any]:
    """
    Test memory query functionality with a sample query.
    
    Args:
        query: Test query string
        namespace: Namespace to search ("knowledge_base", "episode_memory", or "strategies")
        limit: Maximum number of results to return
    
    Returns:
        Dictionary with query results including:
        - query: The query used
        - namespace: Namespace searched
        - results_count: Number of results found
        - results: List of result summaries
        - success: Whether query succeeded
    """
    try:
        log_tool_start("test_memory_query_tool", {
            "query": query[:80],
            "namespace": namespace,
            "limit": limit,
        })
        
        mm = get_memory_manager()
        
        if mm.store is None:
            return {
                "query": query,
                "namespace": namespace,
                "results_count": 0,
                "results": [],
                "success": False,
                "error": "Memory store not configured",
            }
        
        # Map namespace string to tuple
        namespace_map = {
            "knowledge_base": NAMESPACE_KNOWLEDGE_BASE,
            "episode_memory": NAMESPACE_EPISODE_MEMORY,
            "strategies": NAMESPACE_STRATEGIES,
        }
        
        ns_tuple = namespace_map.get(namespace, NAMESPACE_KNOWLEDGE_BASE)
        
        # Perform search
        results = await mm.store.asearch(ns_tuple, query=query, limit=limit)
        
        # Format results
        formatted_results = []
        for item in results:
            result_summary = {
                "key": getattr(item, "key", "unknown"),
                "score": getattr(item, "score", None),
            }
            
            # Extract content preview
            value = getattr(item, "value", None)
            if isinstance(value, dict):
                content = value.get("content", {})
                if isinstance(content, dict):
                    result_summary["preview"] = {
                        "title": content.get("title", content.get("market_question", "N/A"))[:100],
                        "type": content.get("content_type", "unknown"),
                    }
            
            formatted_results.append(result_summary)
        
        result = {
            "query": query,
            "namespace": namespace,
            "results_count": len(results),
            "results": formatted_results,
            "success": True,
        }
        
        log_tool_success("test_memory_query_tool", {
            "results_count": len(results),
            "namespace": namespace,
        })
        
        return result
    except Exception as e:
        log_tool_error("test_memory_query_tool", e, f"Query: {query[:50]}")
        return {
            "query": query,
            "namespace": namespace,
            "results_count": 0,
            "results": [],
            "success": False,
            "error": str(e),
        }


@tool
async def list_memory_namespaces_tool() -> Dict[str, Any]:
    """
    List all memory namespaces and their keys.
    
    Returns:
        Dictionary mapping namespace names to lists of keys (limited to 100 per namespace).
    """
    try:
        log_tool_start("list_memory_namespaces_tool", {})
        
        mm = get_memory_manager()
        namespaces = await mm.list_namespaces()
        
        # Count total keys
        total_keys = sum(len(keys) for keys in namespaces.values())
        
        # Get limit from config
        max_keys_note = min(100, SEARCH_SIMILAR_MARKETS_LIMIT_MAX)
        
        log_tool_success("list_memory_namespaces_tool", {
            "namespaces": list(namespaces.keys()),
            "total_keys": total_keys,
        })
        
        return {
            "namespaces": namespaces,
            "total_keys": total_keys,
            "note": f"Limited to first {max_keys_note} keys per namespace",
        }
    except Exception as e:
        log_tool_error("list_memory_namespaces_tool", e)
        return {
            "namespaces": {},
            "total_keys": 0,
            "error": str(e),
        }


# ============================================================================
# Store-backed search helpers
# ============================================================================

async def _search_similar_markets_in_store(query: str, limit: int) -> List[Dict[str, Any]]:
    """Semantic search for similar market analyses in the knowledge base."""
    store = _get_store()
    if not store:
        return []
    try:
        results = await store.asearch(
            (NAMESPACE_KNOWLEDGE_BASE,),
            query=query,
            filter={"content_type": "market_analysis"},
            limit=limit,
        )
    except Exception:
        return []
    parsed = [res for item in results if (res := _parse_market_search_item(item)) is not None]
    return parsed[:limit]


def _parse_market_search_item(item: SearchItem) -> Optional[Dict[str, Any]]:
    """Normalize a SearchItem into a lightweight dict."""
    val = item.value or {}
    content = val.get("content")
    if isinstance(content, dict):
        cdict = content.copy()
    elif isinstance(content, str):
        cdict = {"analysis": content}
    else:
        cdict = {}

    question = cdict.get("market_question") or cdict.get("question") or val.get("market_question") or val.get("question")
    if not question:
        return None

    out: Dict[str, Any] = {
        "id": val.get("id") or item.key,
        "question": question,
        "analysis": cdict or val,
        "score": getattr(item, "score", None),
        "stored_at": item.updated_at.isoformat() if getattr(item, "updated_at", None) else "",
    }

    for f in ("prior", "posterior", "outcome", "market_url", "market_id", "workflow_id"):
        if f in cdict:
            out[f] = cdict[f]
        elif f in val:
            out[f] = val[f]

    meta: Dict[str, Any] = {}
    for m in (val.get("metadata"), cdict.get("metadata")):
        if isinstance(m, dict):
            meta.update(m)
    if meta:
        out["metadata"] = meta

    return out


# ============================================================================
# Integrity probes
# ============================================================================

def integrity_report() -> Dict[str, Any]:
    """Compact integrity and coverage snapshot for this module."""
    store = _get_store()
    return {
        "store_available": bool(store),
        "limits": {
            "similar_markets_default": SEARCH_SIMILAR_MARKETS_LIMIT_DEFAULT,
            "similar_markets_max": SEARCH_SIMILAR_MARKETS_LIMIT_MAX,
            "historical_evidence_default": SEARCH_HISTORICAL_EVIDENCE_LIMIT_DEFAULT,
            "base_rates_default": GET_BASE_RATES_LIMIT_DEFAULT,
        },
        "violations": [] if store else ["store_unavailable"],
        "tools_available": [
            "search_similar_markets_tool",
            "search_historical_evidence_tool",
            "get_base_rates_tool",
            "store_successful_strategy_tool",
            "verify_memory_system_tool",
            "get_memory_stats_tool",
            "test_memory_query_tool",
            "list_memory_namespaces_tool",
        ],
    }

