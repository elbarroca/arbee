"""
Bayesian Analysis Tools
Wraps existing BayesianCalculator as tools for agent use
"""
from typing import List, Dict, Any
from langchain_core.tools import tool
from arbee.utils.bayesian import BayesianCalculator
import logging

logger = logging.getLogger(__name__)

# Singleton calculator instance
_calculator = None

def get_calculator() -> BayesianCalculator:
    """Get or create Bayesian calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = BayesianCalculator()
    return _calculator


@tool
async def bayesian_calculate_tool(
    prior_p: float,
    evidence_items: List[Dict[str, Any]],
    correlation_clusters: List[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform Bayesian aggregation with VALIDATED evidence schema.

    Use this tool to calculate the posterior probability from a prior and evidence items.
    It handles LLR adjustments, correlation shrinkage, and sensitivity analysis.

    Args:
        prior_p: Prior probability (0.0 to 1.0)
        evidence_items: List of evidence dicts with 'LLR', 'verifiability_score', 'independence_score', 'recency_score'
        correlation_clusters: Optional list of correlated evidence ID clusters

    Returns:
        Dict with p_bayesian, log_odds_prior, log_odds_posterior, evidence_summary, etc.
        - evidence_summary: List[Dict] with keys ['id', 'LLR', 'weight', 'adjusted_LLR']

    Example:
        >>> result = await bayesian_calculate_tool(
        ...     prior_p=0.5,
        ...     evidence_items=[
        ...         {'id': 'ev1', 'LLR': 0.8, 'verifiability_score': 0.9,
        ...          'independence_score': 0.8, 'recency_score': 1.0}
        ...     ]
        ... )
        >>> print(result['p_bayesian'])
    """
    try:
        logger.info(f"🧮 Bayesian calculation: prior={prior_p:.2%}, {len(evidence_items)} items")

        calculator = get_calculator()

        # SCHEMA VALIDATION (NEW)
        normalized_items: List[Dict[str, Any]] = []
        for idx, item in enumerate(evidence_items):
            if item is None:
                logger.warning(f"Skipping None evidence item at index {idx}")
                continue

            # Convert to dict if needed
            if hasattr(item, "model_dump"):
                normalized = item.model_dump()
            elif hasattr(item, "dict"):
                normalized = item.dict()
            else:
                normalized = dict(item)

            # VALIDATE REQUIRED KEYS
            required_keys = ['verifiability_score', 'independence_score', 'recency_score']
            missing_keys = [k for k in required_keys if k not in normalized]
            if missing_keys:
                logger.error(
                    f"Evidence item {idx} missing required keys: {missing_keys}. "
                    f"Available keys: {list(normalized.keys())}"
                )
                # Set defaults for missing scores
                normalized.setdefault('verifiability_score', 0.5)
                normalized.setdefault('independence_score', 0.8)
                normalized.setdefault('recency_score', 0.5)

            # Ensure LLR key exists
            if 'LLR' not in normalized and 'estimated_LLR' in normalized:
                normalized['LLR'] = normalized['estimated_LLR']
            elif 'LLR' not in normalized:
                logger.error(
                    f"Evidence item {idx} has no LLR or estimated_LLR. Keys: {list(normalized.keys())}"
                )
                normalized['LLR'] = 0.0

            # Ensure ID exists (for evidence_summary)
            if 'id' not in normalized:
                if 'subclaim_id' in normalized:
                    normalized['id'] = normalized['subclaim_id']
                elif 'title' in normalized:
                    normalized['id'] = normalized['title'][:50]
                else:
                    normalized['id'] = f"evidence_{idx}"

            # Sign correction based on support direction
            support = str(normalized.get('support', '')).lower()
            if support in {'pro', 'con', 'neutral'}:
                try:
                    llr_value = float(normalized.get('LLR', 0.0))
                except (TypeError, ValueError):
                    llr_value = 0.0
                if support == 'neutral':
                    normalized['LLR'] = 0.0
                elif llr_value < 0 and support == 'pro':
                    normalized['LLR'] = abs(llr_value)
                elif llr_value > 0 and support == 'con':
                    normalized['LLR'] = -abs(llr_value)
                else:
                    normalized['LLR'] = llr_value

            normalized_items.append(normalized)

        if not normalized_items:
            raise ValueError("No valid evidence items after normalization")

        # Perform aggregation
        result = calculator.aggregate_evidence(
            prior_p=prior_p,
            evidence_items=normalized_items,
            correlation_clusters=correlation_clusters or []
        )

        # VALIDATE OUTPUT SCHEMA (NEW)
        required_output_keys = [
            'p_bayesian', 'log_odds_prior', 'log_odds_posterior', 'evidence_summary'
        ]
        missing_output_keys = [k for k in required_output_keys if k not in result]
        if missing_output_keys:
            logger.error(
                f"BayesianCalculator output missing keys: {missing_output_keys}. "
                f"Returned keys: {list(result.keys())}"
            )
            # Add defaults
            result.setdefault('p_bayesian', prior_p)
            result.setdefault('log_odds_prior', 0.0)
            result.setdefault('log_odds_posterior', 0.0)
            result.setdefault('evidence_summary', [])

        # VALIDATE evidence_summary structure (NEW)
        if 'evidence_summary' in result:
            for summary_idx, summary_item in enumerate(result['evidence_summary']):
                if not isinstance(summary_item, dict):
                    logger.error(
                        f"evidence_summary[{summary_idx}] is not a dict: {type(summary_item)}. "
                        f"This will cause 'list indices must be integers' error!"
                    )
                    # Try to convert
                    if hasattr(summary_item, 'model_dump'):
                        result['evidence_summary'][summary_idx] = summary_item.model_dump()
                    elif hasattr(summary_item, 'dict'):
                        result['evidence_summary'][summary_idx] = summary_item.dict()
                    else:
                        # Create minimal dict
                        result['evidence_summary'][summary_idx] = {
                            'id': f"evidence_{summary_idx}",
                            'LLR': 0.0,
                            'weight': 0.0,
                            'adjusted_LLR': 0.0
                        }

                # Validate dict has required keys
                summary_dict = result['evidence_summary'][summary_idx]
                required_summary_keys = ['id', 'LLR', 'weight', 'adjusted_LLR']
                missing_summary_keys = [k for k in required_summary_keys if k not in summary_dict]
                if missing_summary_keys:
                    logger.warning(
                        f"evidence_summary[{summary_idx}] missing keys: {missing_summary_keys}"
                    )
                    # Add defaults
                    summary_dict.setdefault('id', f"evidence_{summary_idx}")
                    summary_dict.setdefault('LLR', 0.0)
                    summary_dict.setdefault('weight', 1.0)
                    summary_dict.setdefault('adjusted_LLR', 0.0)

        logger.info(f"✅ Bayesian result: p={result['p_bayesian']:.2%}")

        return result

    except Exception as e:
        logger.error(f"❌ Bayesian calculation failed: {e}", exc_info=True)
        # Return safe fallback
        return {
            'error': str(e),
            'p_bayesian': prior_p,
            'log_odds_prior': 0.0,
            'log_odds_posterior': 0.0,
            'evidence_summary': [],
            'correlation_adjustments': {'method': 'error', 'details': str(e)}
        }


@tool
async def sensitivity_analysis_tool(
    prior_p: float,
    evidence_items: List[Dict[str, Any]],
    scenarios: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Run sensitivity analysis to test robustness of conclusions.

    Use this to understand how sensitive your Bayesian result is to assumptions
    about evidence quality or prior probability.

    Args:
        prior_p: Prior probability
        evidence_items: Evidence items list
        scenarios: Optional list of scenario names (default: standard scenarios)

    Returns:
        List of sensitivity results with scenario name and resulting probability

    Example:
        >>> results = await sensitivity_analysis_tool(0.5, evidence_items)
        >>> for r in results:
        ...     print(f"{r['scenario']}: {r['p']:.1%}")
    """
    try:
        logger.info(f"📊 Sensitivity analysis: {len(evidence_items)} evidence items")

        calculator = get_calculator()

        # Run sensitivity analysis
        results = calculator.sensitivity_analysis(
            prior_p=prior_p,
            evidence_items=evidence_items
        )

        logger.info(f"✅ Sensitivity analysis complete: {len(results)} scenarios")

        return results

    except Exception as e:
        logger.error(f"❌ Sensitivity analysis failed: {e}")
        return [{'scenario': 'error', 'p': prior_p, 'error': str(e)}]


@tool
async def correlation_detector_tool(
    evidence_items: List[Dict[str, Any]],
    similarity_threshold: float = 0.7
) -> List[List[str]]:
    """
    Detect correlated evidence clusters based on content similarity.

    Use this to identify evidence items that likely share underlying sources
    or information, which should be downweighted to avoid double-counting.

    Args:
        evidence_items: Evidence items with 'id', 'content', 'url'
        similarity_threshold: Cosine similarity threshold for correlation (0-1)

    Returns:
        List of evidence ID clusters that are highly correlated

    Example:
        >>> clusters = await correlation_detector_tool(evidence_items)
        >>> print(f"Found {len(clusters)} correlated clusters")
    """
    try:
        logger.info(f"🔗 Detecting correlations in {len(evidence_items)} items")

        # Simplified correlation detection (in production, use embeddings)
        clusters = []

        # Group by domain (simple heuristic)
        domain_groups = {}
        for item in evidence_items:
            url = item.get('url', '')
            domain = url.split('/')[2] if '/' in url and len(url.split('/')) > 2 else 'unknown'

            if domain not in domain_groups:
                domain_groups[domain] = []
            # Use id if available, otherwise use title or a default
            evidence_id = item.get('id', item.get('title', f"evidence_{len(domain_groups[domain])}"))
            domain_groups[domain].append(evidence_id)

        # Any domain with 2+ items is a potential cluster
        for domain, ids in domain_groups.items():
            if len(ids) >= 2:
                clusters.append(ids)

        # Also group by content similarity (simple keyword matching)
        content_groups = {}
        for i, item1 in enumerate(evidence_items):
            for j, item2 in enumerate(evidence_items[i+1:], i+1):
                content1 = item1.get('content', item1.get('title', '')).lower()
                content2 = item2.get('content', item2.get('title', '')).lower()

                # Simple similarity: check for common words
                words1 = set(content1.split())
                words2 = set(content2.split())

                if words1 and words2:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > similarity_threshold:
                        id1 = item1.get('id', item1.get('title', f"evidence_{i}"))
                        id2 = item2.get('id', item2.get('title', f"evidence_{j}"))
                        cluster_key = tuple(sorted([id1, id2]))

                        if cluster_key not in content_groups:
                            content_groups[cluster_key] = [id1, id2]

        # Add content-based clusters
        for cluster_ids in content_groups.values():
            if len(cluster_ids) >= 2:
                clusters.append(cluster_ids)

        logger.info(f"✅ Found {len(clusters)} correlation clusters")

        return clusters

    except Exception as e:
        logger.error(f"❌ Correlation detection failed: {e}")
        return []


@tool
async def store_critique_results_tool(
    missing_topics: List[str],
    over_represented_sources: List[str],
    follow_up_search_seeds: List[str],
    duplicate_clusters: List[List[str]],
    correlation_warnings: List[List[str]],
    analysis_process: str
) -> Dict[str, Any]:
    """
    Store critique analysis results for use by downstream agents.

    This tool MUST be called by the Critic agent after completing analysis.
    It captures all findings (gaps, duplicates, correlations, recommendations)
    and makes them available for the Analyst and other agents.

    Args:
        missing_topics: List of topics/subclaims with insufficient coverage
        over_represented_sources: List of sources that appear too frequently
        follow_up_search_seeds: List of recommended search queries to fill gaps
        duplicate_clusters: List of duplicate evidence clusters
        correlation_warnings: List of correlated evidence clusters from correlation_detector_tool
        analysis_process: Brief summary of the critique analysis process

    Returns:
        Dict confirming results were stored

    Example:
        >>> result = await store_critique_results_tool(
        ...     missing_topics=["Subclaim 3 has no evidence"],
        ...     over_represented_sources=["CNN.com"],
        ...     follow_up_search_seeds=["query about X", "query about Y"],
        ...     duplicate_clusters=[["ev1", "ev2"]],
        ...     correlation_warnings=[["ev3", "ev4"]],
        ...     analysis_process="Analyzed 15 evidence items, found 2 gaps"
        ... )
    """
    try:
        logger.info(
            f"📥 Storing critique results: {len(missing_topics)} gaps, "
            f"{len(correlation_warnings)} correlations, {len(follow_up_search_seeds)} follow-ups"
        )

        # Package results for return
        results = {
            'missing_topics': missing_topics or [],
            'over_represented_sources': over_represented_sources or [],
            'follow_up_search_seeds': follow_up_search_seeds or [],
            'duplicate_clusters': duplicate_clusters or [],
            'correlation_warnings': correlation_warnings or [],
            'analysis_process': analysis_process or 'Critique completed'
        }

        logger.info(f"✅ Critique results packaged for storage")

        # Return results wrapped in a structure that handle_tool_message can extract
        return {
            'status': 'success',
            'results_stored': results,
            'message': f"Critique analysis stored: {len(missing_topics)} gaps, {len(correlation_warnings)} correlation warnings"
        }

    except Exception as e:
        logger.error(f"❌ Store critique results failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'results_stored': {}
        }


@tool
async def validate_llr_calibration_tool(
    llr: float,
    source_type: str
) -> Dict[str, Any]:
    """
    Validate that an LLR is properly calibrated for its source type.

    Use this to check if evidence LLRs follow calibration guidelines from CLAUDE.md.

    Args:
        llr: Log-likelihood ratio
        source_type: "primary", "high_quality_secondary", "secondary", or "weak"

    Returns:
        Dict with is_valid, expected_range, and feedback

    Example:
        >>> validation = await validate_llr_calibration_tool(0.8, "high_quality_secondary")
        >>> if not validation['is_valid']:
        ...     print(validation['feedback'])
    """
    try:
        # Calibration ranges from CLAUDE.md
        ranges = {
            'primary': (1.0, 3.0),
            'high_quality_secondary': (0.3, 1.0),
            'secondary': (0.1, 0.5),
            'weak': (0.01, 0.2)
        }

        expected_range = ranges.get(source_type, (0.1, 0.5))
        min_llr, max_llr = expected_range

        abs_llr = abs(llr)
        is_valid = min_llr <= abs_llr <= max_llr

        if is_valid:
            feedback = f"LLR {llr:+.2f} is properly calibrated for {source_type} source"
        else:
            feedback = (
                f"LLR {llr:+.2f} is outside expected range [{min_llr}, {max_llr}] "
                f"for {source_type} source. Consider adjusting."
            )

        return {
            'is_valid': is_valid,
            'llr': llr,
            'source_type': source_type,
            'expected_range': expected_range,
            'feedback': feedback
        }

    except Exception as e:
        logger.error(f"❌ LLR validation failed: {e}")
        return {'is_valid': False, 'error': str(e)}
