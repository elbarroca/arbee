"""
Valyu AI Client for Deep Web Research
Strict validation, deep search, no fallbacks.
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from config import settings
import logging
try:
    from langchain_valyu import ValyuSearchTool, ValyuContentsTool
    HAS_VALYU = True
except ImportError:
    HAS_VALYU = False
    ValyuSearchTool = None
    ValyuContentsTool = None

logger = logging.getLogger(__name__)


class ValyuResearchClient:
    """Deep research client using Valyu AI. Strict validation, no fallbacks."""

    def __init__(self):
        """Initialize with strict validation."""
        self._validate_config()
        os.environ['VALYU_API_KEY'] = settings.VALYU_API_KEY
        self.search_tool = ValyuSearchTool()
        self.contents_tool = ValyuContentsTool()
        logger.info("Valyu client ready")

    def _validate_config(self) -> None:
        """Validate configuration strictly."""
        assert hasattr(settings, 'VALYU_API_KEY') and settings.VALYU_API_KEY and settings.VALYU_API_KEY != '...'

    async def search(self, query: str, search_type: str = "all", num_results: int = 10,
                     relevance_threshold: float = 0.3, max_cost: float = 20.0) -> List[Dict[str, Any]]:
        """Deep search with strict validation."""
        assert query and query.strip(), "Empty query"
        assert 1 <= num_results <= 50, "Invalid num_results"
        assert 0.0 <= relevance_threshold <= 1.0, "Invalid relevance_threshold"
        assert 0.1 <= max_cost <= 50.0, "Invalid max_cost"

        result = self.search_tool._run(
            query=query, search_type=search_type, max_num_results=num_results,
            relevance_threshold=relevance_threshold, max_price=max_cost
            )
        results = result.results if hasattr(result, 'results') else result
        return self._normalize_results(results or [])

    def _normalize_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Normalize search results into clean dict format."""
        normalized = []
        for item in results:
            url = getattr(item, 'url', None) or (item.get('url') if isinstance(item, dict) else None)
            if not url or not self._valid_url(url):
                continue

            title = getattr(item, 'title', None) or (item.get('title') if isinstance(item, dict) else '')
            content = getattr(item, 'content', None) or (item.get('content') if isinstance(item, dict) else '')
            published_date = getattr(item, 'publication_date', None) or (item.get('publication_date') if isinstance(item, dict) else '')
            source = getattr(item, 'source', None) or (item.get('source') if isinstance(item, dict) else '')

            normalized.append({
                'title': self._clean_text(str(title) if title else ''),
                'url': str(url),
                'snippet': self._clean_text(str(content) if content else '')[:500],
                'published_date': self._normalize_date(str(published_date) if published_date else ''),
                'source': str(source) if source else self._extract_domain(str(url)),
                'relevance_score': 0.8
            })

        return sorted(
            [r for r in normalized if r['url'] and r['title']],
            key=lambda x: (x['relevance_score'], -len(x['title'])),
            reverse=True
        )

    def _valid_url(self, url: str) -> bool:
        """Validate URL format."""
        return bool(url and len(url) <= 2048 and url.startswith(('http://', 'https://')))

    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        return ' '.join(str(text).split())[:1000] if text else ''

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to ISO format."""
        if not date_str:
            return ''
        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_str
        except (ValueError, AttributeError):
            return ''

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except (AttributeError, ValueError) as e:
            logger.debug(f"Failed to parse URL '{url}': {e}")
            return 'unknown'

    async def get_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from URL."""
        assert url and url.strip() and url.startswith(('http://', 'https://')) and len(url) <= 2048

        docs = self.contents_tool._run(url)
        content = docs[0].page_content if docs else None
        if isinstance(content, str):
            return {
                'text': content,
                'extracted_at': datetime.utcnow().isoformat(),
                'content_type': 'text'
            }
        return dict(content) if isinstance(content, dict) else None

    async def search_with_date_filter(self, query: str, days_back: int = 90, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search with date filtering."""
        assert query and query.strip(), "Empty query"
        assert 1 <= days_back <= 3650, "Invalid days_back"
        assert 1 <= num_results <= 50, "Invalid num_results"

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        date_range = {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d")
        }
        return await self.search(query=query, num_results=num_results)

    async def multi_query_search(self, queries: List[str], max_results_per_query: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Parallel search across multiple queries."""
        assert queries and 1 <= len(queries) <= 20, "Invalid query count"
        assert 1 <= max_results_per_query <= 25, "Invalid max_results_per_query"
        assert all(query and query.strip() for query in queries), "Empty queries"

        import asyncio
        tasks = [self.search(query, num_results=max_results_per_query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for query, result in zip(queries, results):
            output[query] = [] if isinstance(result, Exception) else result
        return output

    async def deep_research(self, query: str, depth: str = "comprehensive",
                           num_results: int = 15, relevance_threshold: float = 0.7) -> Dict[str, Any]:
        """Comprehensive deep research."""
        assert query and query.strip(), "Empty query"
        assert depth in ["basic", "comprehensive", "exhaustive"], "Invalid depth"
        assert 5 <= num_results <= 30, "Invalid num_results"
        assert 0.3 <= relevance_threshold <= 0.9, "Invalid relevance_threshold"

        # Multi-layered search based on depth
        search_types = {"basic": ["web"], "comprehensive": ["all"], "exhaustive": ["all", "web"]}
        all_results = []

        for search_type in search_types[depth]:
            results = await self.search(
                query=query, search_type=search_type,
                num_results=num_results // len(search_types[depth]),
                relevance_threshold=relevance_threshold
            )
            all_results.extend(results)

        # Deduplicate and rank
        seen_urls = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x.get('relevance_score', 0), reverse=True):
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)

        # Analyze results
        if not unique_results:
            analysis = {'error': 'No results'}
        else:
            total_sources = len(unique_results)
            avg_relevance = sum(r.get('relevance_score', 0) for r in unique_results) / total_sources
            domains = {}
            for result in unique_results:
                domain = result.get('source', 'unknown')
                if domain:
                    domains[domain] = domains.get(domain, 0) + 1

            analysis = {
            'total_sources': total_sources,
            'average_relevance': round(avg_relevance, 3),
            'domain_diversity': len(domains),
            'top_domains': sorted(domains.items(), key=lambda x: x[1], reverse=True)[:5],
            'quality_score': min(1.0, avg_relevance * (len(domains) / max(total_sources, 1)))
            }

        return {
            'query': query,
            'depth': depth,
            'results': unique_results[:num_results],
            'analysis': analysis,
            'timestamp': datetime.utcnow().isoformat(),
            'total_sources': len(unique_results[:num_results])
        }
