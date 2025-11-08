"""
Web Search Tools for Research Agents
Provides web search capabilities using Valyu and optionally Tavily
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from clients.valyu import ValyuResearchClient
from utils.rich_logging import (
    log_tool_start,
    log_tool_success,
    log_tool_error,
    log_search_results,
)
import logging
import time

logger = logging.getLogger(__name__)


@tool
async def web_search_tool(
    query: str,
    max_results: int = 10,
    date_range_days: Optional[int] = 90
) -> List[Dict[str, Any]]:
    """
    Search the web for information using Valyu Research API.

    Use this tool when you need to find current information, news articles,
    research papers, or any web content related to the market question.

    Args:
        query: Search query string (be specific for better results)
        max_results: Maximum number of results to return (default 10)
        date_range_days: Only return results from last N days (default 90, None for all time)

    Returns:
        List of search results with title, URL, snippet, and published date

    Example:
        >>> results = await web_search_tool("Trump 2024 election polls Arizona", max_results=5)
        >>> print(results[0]['title'])
    """
    start_time = time.time()
    try:
        log_tool_start("web_search_tool", {"query": query, "max_results": max_results, "date_range_days": date_range_days})

        client = ValyuResearchClient()

        # Execute search
        results = await client.multi_query_search(
            queries=[query],
            max_results_per_query=max_results
        )

        # Extract results for this query
        search_results = results.get(query, [])
        execution_time = time.time() - start_time

        log_search_results("web_search_tool", query, search_results, max_preview=3)
        log_tool_success("web_search_tool", {"results_count": len(search_results), "execution_time": execution_time})

        # Format results consistently
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                'title': result.get('title', 'N/A'),
                'url': result.get('url', ''),
                'snippet': result.get('snippet', ''),
                'published_date': result.get('published_date', ''),
                'source': result.get('source', ''),
                'content': result.get('content', result.get('snippet', ''))
            })

        return formatted_results

    except Exception as e:
        log_tool_error("web_search_tool", e, f"Query: {query}")
        logger.error(f"❌ Web search failed for '{query}': {e}")
        return [{"error": str(e), "query": query}]

@tool
async def multi_query_search_tool(
    queries: List[str],
    max_results_per_query: int = 5,
    parallel: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Execute multiple search queries in parallel for comprehensive research.

    Use this when you need to research multiple angles or subclaims simultaneously.
    More efficient than calling web_search_tool multiple times.

    Args:
        queries: List of search queries
        max_results_per_query: Max results per query
        parallel: Execute queries in parallel (faster)

    Returns:
        Dictionary mapping each query to its results

    Example:
        >>> results = await multi_query_search_tool([
        ...     "Trump Arizona polls 2024",
        ...     "Harris Arizona polls 2024",
        ...     "Arizona swing state trends"
        ... ])
        >>> print(len(results))  # 3 queries
    """
    start_time = time.time()
    try:
        log_tool_start("multi_query_search_tool", {"queries": queries, "max_results_per_query": max_results_per_query, "parallel": parallel})

        client = ValyuResearchClient()

        # Execute all queries
        results = await client.multi_query_search(
            queries=queries,
            max_results_per_query=max_results_per_query
        )

        total_results = sum(len(r) for r in results.values())
        execution_time = time.time() - start_time
        
        # Log results for each query
        for query, query_results in results.items():
            log_search_results("multi_query_search_tool", query, query_results, max_preview=2)
        
        log_tool_success("multi_query_search_tool", {"queries_count": len(queries), "total_results": total_results, "execution_time": execution_time})

        return results

    except Exception as e:
        log_tool_error("multi_query_search_tool", e, f"Queries: {len(queries)} queries")
        logger.error(f"❌ Multi-query search failed: {e}")
        return {query: [] for query in queries}
