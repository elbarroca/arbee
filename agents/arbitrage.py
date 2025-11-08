"""
Autonomous ArbitrageDetector Agent
Finds mispricing opportunities with autonomous validation and edge detection
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain_core.tools import BaseTool
import logging

from agents.base import AutonomousReActAgent, AgentState
from agents.schemas import ArbitrageOpportunity, EdgeSignal, CompositeEdgeScore
from tools.edge_detection import (
    information_asymmetry_tool,
    market_inefficiency_tool,
    sentiment_edge_tool,
    base_rate_violation_tool,
    composite_edge_score_tool,
    analyze_mentions_market_tool,
)
from config.settings import settings

logger = logging.getLogger(__name__)


class AutonomousArbitrageAgent(AutonomousReActAgent):
    """
    Autonomous Arbitrage Detector - Finds mispricing opportunities

    Note: This agent has simpler tool needs since ArbitrageDetector
    already has good logic. Main improvements:
    - Autonomous retry if API calls fail
    - Validation of opportunities before returning
    - Can query multiple exchanges iteratively
    """

    def __init__(self, min_edge_threshold: float = 0.02, **kwargs):
        """
        Initialize Autonomous Arbitrage Agent

        Args:
            min_edge_threshold: Minimum edge to report
            **kwargs: Additional args
        """
        super().__init__(**kwargs)
        self.min_edge_threshold = min_edge_threshold

    def get_system_prompt(self) -> str:
        """System prompt for arbitrage detection with edge detection"""
        return f"""You are an Autonomous Arbitrage Detector in POLYSEER.

Your mission: Find mispricing opportunities between Bayesian estimate and market prices,
AND detect edge opportunities beyond simple arbitrage.

## Process

1. **Compare p_bayesian vs market prices**
   - Calculate edge = p_bayesian - market_implied_probability
   - Calculate EV considering transaction costs
   - Apply Kelly criterion for position sizing

2. **Detect Edge Signals** (use edge detection tools)
   - information_asymmetry_tool: Check for insider wallet activity
   - market_inefficiency_tool: Detect volume spikes, orderbook anomalies
   - sentiment_edge_tool: Compare Bayesian vs market price divergence
   - base_rate_violation_tool: Compare market price vs historical base rates
   - analyze_mentions_market_tool: Analyze mentions markets for mispricing
   - composite_edge_score_tool: Combine all edge signals (REQUIRES edge_signals parameter)

3. **Validate opportunities**
   - Check edge > {self.min_edge_threshold} (minimum threshold)
   - Verify liquidity is sufficient
   - Consider transaction costs and slippage
   - Factor in edge signals (insider activity, inefficiencies, etc.)

4. **Generate recommendations**
   - For each opportunity: market, provider, edge, suggested_stake
   - Include edge signals in trade_rationale
   - Include rationale and risk warnings
   - Add disclaimer: NOT FINANCIAL ADVICE

## CRITICAL: Store Results

After calling each tool, you MUST store the results in intermediate_results:

1. After calling edge detection tools (sentiment_edge_tool, base_rate_violation_tool, etc.):
   - The tool results are AUTOMATICALLY stored in intermediate_results['edge_signals']
   - Each edge signal dict is appended to the list
   - Example: intermediate_results['edge_signals'] = [{{"edge_type": "sentiment_edge", "strength": 0.9, ...}}]

2. **BEFORE calling composite_edge_score_tool**:
   - You MUST first collect all edge_signals from intermediate_results
   - Extract: edge_signals = intermediate_results.get('edge_signals', [])
   - Then call: composite_edge_score_tool(edge_signals=edge_signals, weights=None)
   - The composite result is AUTOMATICALLY stored in intermediate_results['composite_edge_score']

3. After calculating arbitrage opportunities:
   - Store list in intermediate_results['opportunities']
   - Example: intermediate_results['opportunities'] = [{{"market_id": "...", "edge": 0.15, ...}}]

4. **IMPORTANT**: Do NOT call the same tool multiple times with identical parameters.
   - Check intermediate_results before calling tools
   - If edge_signals already contains signals of a given type, skip calling that tool again
   - Track which tools you've already called

## Tool Call Sequence

1. Call individual edge detection tools (sentiment_edge_tool, base_rate_violation_tool, etc.)
   - These automatically append to intermediate_results['edge_signals']
   
2. After collecting edge signals, call composite_edge_score_tool:
   - First, extract edge_signals from intermediate_results
   - Then call: composite_edge_score_tool(edge_signals=<extracted_list>, weights=None)
   
3. Calculate arbitrage opportunities and store in intermediate_results['opportunities']

Complete when:
- All edge detection tools have been called (or skipped if already done)
- Edge signals stored in intermediate_results['edge_signals']
- Composite edge score computed and stored (if multiple signals)
- Opportunities calculated and stored in intermediate_results['opportunities']
"""

    def get_tools(self) -> List[BaseTool]:
        """Return edge detection tools"""
        if getattr(settings, "ENABLE_EDGE_DETECTION", True):
            return [
                information_asymmetry_tool,
                market_inefficiency_tool,
                sentiment_edge_tool,
                base_rate_violation_tool,
                analyze_mentions_market_tool,
                composite_edge_score_tool,
            ]
        return []

    async def is_task_complete(self, state: AgentState) -> bool:
        """Check if arbitrage detection complete (with edge signals)"""
        # Ensure intermediate_results exists
        if 'intermediate_results' not in state:
            state['intermediate_results'] = {}
        
        results = state.get('intermediate_results', {})
        
        # Check for opportunities (can be empty list if none found)
        has_opportunities_key = 'opportunities' in results
        opportunities = results.get('opportunities', [])
        has_opportunities = has_opportunities_key and isinstance(opportunities, list)
        
        # If edge detection enabled, also check for edge signals
        if getattr(settings, "ENABLE_EDGE_DETECTION", True):
            has_edge_signals_key = 'edge_signals' in results
            edge_signals = results.get('edge_signals', [])
            has_edge_signals = has_edge_signals_key and isinstance(edge_signals, list)
            
            # Also check for composite score (optional but preferred)
            has_composite = 'composite_edge_score' in results
            
            # Complete if we have opportunities stored AND edge signals stored
            # (even if lists are empty, as long as they exist)
            complete = has_opportunities and (has_edge_signals or has_composite)
            
            if not complete:
                self.logger.debug(
                    f"Arbitrage incomplete: opportunities={has_opportunities}, "
                    f"edge_signals={has_edge_signals}, composite={has_composite}"
                )
            
            return complete
        
        # If edge detection disabled, just check for opportunities
        return has_opportunities

    async def extract_final_output(self, state: AgentState) -> List[ArbitrageOpportunity]:
        """Extract arbitrage opportunities from state"""
        results = state.get('intermediate_results', {})
        opportunities_data = results.get('opportunities', [])

        opportunities = []
        for opp_data in opportunities_data:
            if isinstance(opp_data, dict):
                opportunities.append(ArbitrageOpportunity(**opp_data))

        # Log edge signals if present
        edge_signals = results.get('edge_signals', [])
        if edge_signals:
            for signal in edge_signals[:5]:  # Log first 5 signals
                if isinstance(signal, dict):
                    self.rich_logger.log_edge_detection(
                        edge_type=signal.get('edge_type', 'unknown'),
                        strength=signal.get('strength', 0.0),
                        confidence=signal.get('confidence', 0.0),
                        evidence=signal.get('evidence', []),
                        metadata={k: v for k, v in signal.items() if k not in ['edge_type', 'strength', 'confidence', 'evidence']},
                    )
        
        # Log composite edge score if present
        composite_score = results.get('composite_edge_score')
        if composite_score and isinstance(composite_score, dict):
            self.rich_logger.log_edge_detection(
                edge_type="composite",
                strength=composite_score.get('composite_score', 0.0),
                confidence=composite_score.get('confidence', 0.0),
                evidence=composite_score.get('evidence_summary', []),
                metadata=composite_score.get('weighted_components', {}),
            )

        self.logger.info(f"📤 Found {len(opportunities)} arbitrage opportunities")
        return opportunities

    async def detect_arbitrage(
        self,
        p_bayesian: float,
        market_slug: str,
        market_question: str,
        providers: List[str],
        bankroll: float,
        max_kelly: float,
        min_edge_threshold: float,
        market_price: Optional[float] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities autonomously

        Args:
            p_bayesian: Bayesian posterior probability
            market_slug: Market identifier
            market_question: Market question
            providers: List of providers to check
            bankroll: Available capital
            max_kelly: Max Kelly fraction
            min_edge_threshold: Min edge to report
            market_price: Optional market price (0-1)
            market_data: Optional market data dict

        Returns:
            List of arbitrage opportunities
        """
        return await self.run(
            task_description="Find mispricing opportunities across prediction markets",
            task_input={
                'p_bayesian': p_bayesian,
                'market_slug': market_slug,
                'market_question': market_question,
                'providers': providers,
                'bankroll': bankroll,
                'max_kelly': max_kelly,
                'min_edge_threshold': min_edge_threshold,
                'market_price': market_price or 0.0,
                'market_data': market_data,
            }
        )
