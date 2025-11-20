"""
Bayesian Analysis Tools
Wraps existing BayesianCalculator as tools for agent use
"""
import logging
from typing import Any, Dict, List

from langchain_core.tools import tool

from utils.bayesian import BayesianCalculator
from utils.rich_logging import (
    log_bayesian_result,
    log_tool_error,
    log_tool_start,
    log_tool_success,
)

logger = logging.getLogger(__name__)

# Singleton calculator instance
_calculator = None

def get_calculator() -> BayesianCalculator:
    """Get or create Bayesian calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = BayesianCalculator()
    assert _calculator is not None, "Calculator initialization failed"
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
    assert 0.0 <= prior_p <= 1.0, f"Prior probability must be between 0 and 1, got {prior_p}"
    assert evidence_items, "Evidence items list required"
    assert isinstance(evidence_items, list), "Evidence items must be a list"
    
    log_tool_start("bayesian_calculate_tool", {"prior_p": prior_p, "evidence_count": len(evidence_items), "correlation_clusters": len(correlation_clusters) if correlation_clusters else 0})
    logger.info(f"🧮 Bayesian calculation: prior={prior_p:.2%}, {len(evidence_items)} items")

    calculator = get_calculator()

    # SCHEMA VALIDATION
    normalized_items: List[Dict[str, Any]] = []
    for idx, item in enumerate(evidence_items):
        assert item is not None, f"Evidence item at index {idx} cannot be None"

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
        assert not missing_keys, f"Evidence item {idx} missing required keys: {missing_keys}. Available keys: {list(normalized.keys())}"

        # Ensure LLR key exists
        if 'LLR' not in normalized and 'estimated_LLR' in normalized:
            normalized['LLR'] = normalized['estimated_LLR']
        assert 'LLR' in normalized or 'estimated_LLR' in normalized, f"Evidence item {idx} has no LLR or estimated_LLR. Keys: {list(normalized.keys())}"
        if 'LLR' not in normalized:
            normalized['LLR'] = normalized['estimated_LLR']

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
            llr_value = float(normalized.get('LLR', 0.0))
            if support == 'neutral':
                normalized['LLR'] = 0.0
            elif llr_value < 0 and support == 'pro':
                normalized['LLR'] = abs(llr_value)
            elif llr_value > 0 and support == 'con':
                normalized['LLR'] = -abs(llr_value)
            else:
                normalized['LLR'] = llr_value

        normalized_items.append(normalized)

    assert normalized_items, "No valid evidence items after normalization"

    # Perform aggregation
    result = calculator.aggregate_evidence(
        prior_p=prior_p,
        evidence_items=normalized_items,
        correlation_clusters=correlation_clusters or []
    )

    # VALIDATE OUTPUT SCHEMA
    required_output_keys = [
        'p_bayesian', 'log_odds_prior', 'log_odds_posterior', 'evidence_summary', 'correlation_adjustments'
    ]
    missing_output_keys = [k for k in required_output_keys if k not in result]
    assert not missing_output_keys, f"BayesianCalculator output missing keys: {missing_output_keys}. Returned keys: {list(result.keys())}"

    # Ensure correlation_adjustments is always a dict
    assert isinstance(result.get('correlation_adjustments'), dict), f"correlation_adjustments must be dict, got {type(result.get('correlation_adjustments'))}"

    # VALIDATE evidence_summary structure
    assert 'evidence_summary' in result, "evidence_summary required in result"
    assert isinstance(result['evidence_summary'], list), "evidence_summary must be a list"
    
    for summary_idx, summary_item in enumerate(result['evidence_summary']):
        assert isinstance(summary_item, dict), f"evidence_summary[{summary_idx}] must be dict, got {type(summary_item)}"
        
        # Validate dict has required keys
        required_summary_keys = ['id', 'LLR', 'weight', 'adjusted_LLR']
        missing_summary_keys = [k for k in required_summary_keys if k not in summary_item]
        assert not missing_summary_keys, f"evidence_summary[{summary_idx}] missing keys: {missing_summary_keys}"

    logger.info(f"✅ Bayesian result: p={result['p_bayesian']:.2%}")
    
    log_bayesian_result(
        "bayesian_calculate_tool",
        prior_p,
        result['p_bayesian'],
        len(normalized_items)
    )
    log_tool_success("bayesian_calculate_tool", {
        "p_bayesian": result['p_bayesian'],
        "evidence_count": len(normalized_items),
        "log_odds_posterior": result.get('log_odds_posterior', 0.0)
    })
    
    return result


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
    assert 0.0 <= prior_p <= 1.0, f"Prior probability must be between 0 and 1, got {prior_p}"
    assert evidence_items, "Evidence items list required"
    assert isinstance(evidence_items, list), "Evidence items must be a list"
    
    log_tool_start("sensitivity_analysis_tool", {"prior_p": prior_p, "evidence_count": len(evidence_items), "scenarios": len(scenarios) if scenarios else "default"})
    logger.info(f"📊 Sensitivity analysis: {len(evidence_items)} evidence items")

    calculator = get_calculator()

    # Run sensitivity analysis
    results = calculator.sensitivity_analysis(
        prior_p=prior_p,
        evidence_items=evidence_items
    )

    assert results, "Sensitivity analysis must return results"
    assert isinstance(results, list), "Sensitivity results must be a list"
    
    logger.info(f"✅ Sensitivity analysis complete: {len(results)} scenarios")
    log_tool_success("sensitivity_analysis_tool", {"scenarios_count": len(results), "prior_p": prior_p})
    
    return results


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
    assert isinstance(missing_topics, list), "missing_topics must be a list"
    assert isinstance(over_represented_sources, list), "over_represented_sources must be a list"
    assert isinstance(follow_up_search_seeds, list), "follow_up_search_seeds must be a list"
    assert isinstance(duplicate_clusters, list), "duplicate_clusters must be a list"
    assert isinstance(correlation_warnings, list), "correlation_warnings must be a list"
    assert analysis_process and isinstance(analysis_process, str), "analysis_process must be non-empty string"
    
    logger.info(
        f"📥 Storing critique results: {len(missing_topics)} gaps, "
        f"{len(correlation_warnings)} correlations, {len(follow_up_search_seeds)} follow-ups"
    )

    # Package results for return
    results = {
        'missing_topics': missing_topics,
        'over_represented_sources': over_represented_sources,
        'follow_up_search_seeds': follow_up_search_seeds,
        'duplicate_clusters': duplicate_clusters,
        'correlation_warnings': correlation_warnings,
        'analysis_process': analysis_process
    }

    logger.info(f"✅ Critique results packaged for storage")

    # Return results wrapped in a structure that handle_tool_message can extract
    return {
        'status': 'success',
        'results_stored': results,
        'message': f"Critique analysis stored: {len(missing_topics)} gaps, {len(correlation_warnings)} correlation warnings"
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
    assert isinstance(llr, (int, float)), f"LLR must be numeric, got {type(llr)}"
    assert source_type in {'primary', 'high_quality_secondary', 'secondary', 'weak'}, f"Invalid source_type: {source_type}"
    
    # Calibration ranges from CLAUDE.md
    ranges = {
        'primary': (1.0, 3.0),
        'high_quality_secondary': (0.3, 1.0),
        'secondary': (0.1, 0.5),
        'weak': (0.01, 0.2)
    }

    expected_range = ranges[source_type]
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
