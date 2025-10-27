"""
Tool Framework for POLYSEER Autonomous Agents
Defines interfaces and base classes for agent tools
"""
from typing import List
from langchain_core.tools import BaseTool

# Import all tools here as they are created
from arbee.tools.search import web_search_tool, multi_query_search_tool
from arbee.tools.evidence import extract_evidence_tool, verify_source_tool
from arbee.tools.memory_search import (
    search_similar_markets_tool,
    search_historical_evidence_tool,
    get_base_rates_tool,
    store_successful_strategy_tool
)
from arbee.tools.bayesian import (
    bayesian_calculate_tool,
    sensitivity_analysis_tool,
    correlation_detector_tool,
    store_critique_results_tool,
    validate_llr_calibration_tool
)

__all__ = [
    'get_all_tools',
    'get_research_tools',
    'get_analysis_tools',
    'get_memory_tools',
    # Individual tools
    'web_search_tool',
    'multi_query_search_tool',
    'extract_evidence_tool',
    'verify_source_tool',
    'search_similar_markets_tool',
    'search_historical_evidence_tool',
    'get_base_rates_tool',
    'store_successful_strategy_tool',
    'bayesian_calculate_tool',
    'sensitivity_analysis_tool',
    'correlation_detector_tool',
    'store_critique_results_tool',
    'validate_llr_calibration_tool',
]


def get_research_tools() -> List[BaseTool]:
    """Get all research-related tools"""
    return [
        web_search_tool,
        multi_query_search_tool,
        extract_evidence_tool,
        verify_source_tool,
    ]


def get_analysis_tools() -> List[BaseTool]:
    """Get all analysis-related tools"""
    return [
        bayesian_calculate_tool,
        sensitivity_analysis_tool,
        correlation_detector_tool,
        store_critique_results_tool,
        validate_llr_calibration_tool,
    ]


def get_memory_tools() -> List[BaseTool]:
    """Get all memory-related tools"""
    return [
        search_similar_markets_tool,
        search_historical_evidence_tool,
        get_base_rates_tool,
        store_successful_strategy_tool,
    ]


def get_all_tools() -> List[BaseTool]:
    """Get all available tools"""
    return (
        get_research_tools() +
        get_analysis_tools() +
        get_memory_tools()
    )
