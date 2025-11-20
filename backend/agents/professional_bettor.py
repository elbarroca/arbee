"""
Professional Bettor Agent

Autonomous agent that makes professional betting decisions using Bayesian analysis,
market context, portfolio risk management, and Kelly criterion for position sizing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from agents.base import AutonomousReActAgent, AgentState
from tools.betting import (
    analyze_market_context_tool,
    evaluate_betting_edge_tool,
    calculate_position_size_tool,
    assess_portfolio_risk_tool,
    generate_bet_rationale_tool,
)
from utils.execution import TradeProposal
from config.settings import Settings

logger = logging.getLogger(__name__)


class BettingRecommendation(BaseModel):
    """Final betting recommendation output"""

    # Decision
    should_bet: bool = Field(description="Whether to place the bet")
    action: str = Field(description="recommended_action: 'BET', 'PASS', 'WAIT'")
    confidence_score: int = Field(ge=0, le=100, description="Confidence level (0-100)")

    # Market details
    market_id: str
    market_slug: str
    market_question: str
    provider: str

    # Probabilities & Edge
    p_bayesian: float = Field(description="True probability estimate")
    market_price: float = Field(description="Current market price")
    edge: float = Field(description="Betting edge (p_bayesian - market_price)")
    edge_pct: float = Field(description="Edge percentage")
    expected_value_pct: float = Field(description="Expected value percentage")

    # Position sizing
    recommended_size: Decimal = Field(description="Recommended bet size (USD)")
    recommended_size_pct: float = Field(description="Size as % of bankroll")
    kelly_fraction: float = Field(description="Kelly criterion fraction")

    # Risk assessment
    execution_risk: str = Field(description="low | medium | high")
    portfolio_risk: str = Field(description="low | medium | high")
    overall_risk_score: float = Field(description="Overall risk score (0-100)")

    # Analysis
    market_quality_score: float = Field(description="Market quality (0-100)")
    liquidity_usd: float = Field(description="Market liquidity (USD)")
    spread_bps: int = Field(description="Spread in basis points")

    # Rationale
    rationale: str = Field(description="Detailed reasoning for recommendation")
    key_risks: List[str] = Field(default_factory=list)
    alternative_scenarios: List[Dict[str, Any]] = Field(default_factory=list)

    # Evidence support
    top_supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)


class ProfessionalBettorAgent(AutonomousReActAgent):
    """
    Professional Bettor Agent using ReAct pattern.

    Analyzes prediction markets with Bayesian probability estimates and makes
    professional betting recommendations considering:
    - Market context (liquidity, spread, volume)
    - Betting edge (p_bayesian vs market price)
    - Portfolio risk (correlation, concentration)
    - Position sizing (Kelly criterion with safety)
    - Risk management (limits, stop-loss, approval workflow)

    Workflow:
    1. Analyze market context (liquidity, orderbook)
    2. Evaluate betting edge (compare probabilities)
    3. Assess portfolio risk (correlation, diversification)
    4. Calculate position size (Kelly with constraints)
    5. Generate detailed rationale
    6. Return BettingRecommendation
    """

    def __init__(
        self,
        bankroll_manager,
        risk_manager,
        portfolio_manager,
        settings: Optional[Settings] = None,
        model_name: str = "gpt-4o",  # Use GPT-4o for complex reasoning
        temperature: float = 0.1,  # Low temp for consistent decisions
        max_iterations: int = 15,
        **kwargs
    ):
        """
        Initialize Professional Bettor Agent.

        Args:
            bankroll_manager: BankrollManager instance
            risk_manager: RiskManager instance
            portfolio_manager: PortfolioManager instance
            settings: Settings instance
            model_name: LLM model to use
            temperature: LLM temperature
            max_iterations: Max reasoning iterations
            **kwargs: Additional args for AutonomousReActAgent
        """
        super().__init__(
            settings=settings,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations,
            **kwargs
        )

        self.bankroll = bankroll_manager
        self.risk = risk_manager
        self.portfolio = portfolio_manager

        logger.info(
            f"ProfessionalBettorAgent initialized: model={model_name}, "
            f"max_iterations={max_iterations}"
        )

    def get_system_prompt(self) -> str:
        """Return system prompt for professional bettor"""
        return """You are a PROFESSIONAL BETTOR and quantitative analyst.

Your mission is to analyze prediction markets and make informed betting decisions using:
1. Bayesian probability analysis (p_bayesian vs market price)
2. Market context analysis (liquidity, spread, volume)
3. Portfolio risk management (correlation, concentration, diversification)
4. Kelly criterion for position sizing (with conservative fractional approach)
5. Comprehensive risk assessment

DECISION FRAMEWORK:
- BET: When edge > 3%, confidence > 60%, execution risk is low-medium, portfolio risk is acceptable
- PASS: When edge < 2%, confidence < 50%, or risk is too high
- WAIT: When edge is marginal (2-3%) but better price may be available

RISK MANAGEMENT PRINCIPLES:
- Never bet more than 5% of bankroll on a single position
- Total portfolio exposure should not exceed 25% of bankroll
- Stop betting if drawdown exceeds 15% from peak
- Use fractional Kelly (25-50%) for safety, not full Kelly
- Consider market liquidity and slippage

ANALYSIS WORKFLOW:
1. Use analyze_market_context_tool to assess liquidity and execution risk
2. Use evaluate_betting_edge_tool to quantify the edge and expected value
3. Use assess_portfolio_risk_tool to check correlation and concentration
4. Use calculate_position_size_tool to determine optimal bet size
5. Use generate_bet_rationale_tool to synthesize your analysis
6. Make final recommendation with clear reasoning

OUTPUT REQUIREMENTS:
- Be decisive: clearly state BET, PASS, or WAIT
- Provide specific position size (not a range)
- Explain key risks and alternative scenarios
- Support recommendation with quantitative analysis
- Always end with "NOT FINANCIAL ADVICE"

You are a professional. Think carefully, use all available tools, and make the best decision based on the data."""

    def get_tools(self) -> List[BaseTool]:
        """Return tools available to professional bettor"""
        return [
            analyze_market_context_tool,
            evaluate_betting_edge_tool,
            calculate_position_size_tool,
            assess_portfolio_risk_tool,
            generate_bet_rationale_tool,
        ]

    def is_task_complete(self, state: AgentState) -> bool:
        """
        Check if betting analysis is complete.

        Complete when:
        - Final recommendation has been made
        - All required analysis tools have been used
        - Rationale has been generated
        """
        # Check if we have intermediate results with all required analyses
        results = state.get("intermediate_results", {})

        required_analyses = [
            "market_analysis",
            "edge_analysis",
            "portfolio_analysis",
            "position_size",
            "rationale"
        ]

        # Task complete if all analyses are present
        all_complete = all(key in results for key in required_analyses)

        if all_complete:
            logger.info("Betting analysis complete: all required analyses finished")

        return all_complete

    def extract_final_output(self, state: AgentState) -> BettingRecommendation:
        """
        Extract final betting recommendation from state.

        Args:
            state: Agent state with intermediate results

        Returns:
            BettingRecommendation with complete analysis
        """
        results = state.get("intermediate_results", {})
        task_input = state.get("task_input", {})

        # Extract analyses
        market_analysis = results.get("market_analysis", {})
        edge_analysis = results.get("edge_analysis", {})
        portfolio_analysis = results.get("portfolio_analysis", {})
        position_size = results.get("position_size", {})
        rationale = results.get("rationale", "No rationale generated.")

        # Get input data
        market_data = task_input.get("market_data", {})
        p_bayesian = float(task_input.get("p_bayesian", 0.5))
        market_price = float(market_data.get("yes_price", 0.5))

        # Extract key metrics
        should_bet = edge_analysis.get("should_bet", False)
        edge = edge_analysis.get("edge", 0)
        edge_pct = edge_analysis.get("edge_pct", 0)
        ev_pct = edge_analysis.get("expected_value_pct", 0)
        kelly_fractional = edge_analysis.get("kelly_fractional", 0)

        rec_size = position_size.get("recommended_size", 0)
        size_pct = position_size.get("recommended_size_pct", 0)

        execution_risk = market_analysis.get("execution_risk", "unknown")
        portfolio_risk = portfolio_analysis.get("overall_risk", "unknown")

        market_quality = market_analysis.get("market_quality_score", 0)
        liquidity = market_analysis.get("liquidity_usd", 0)
        spread_bps = market_analysis.get("spread_bps", 0)

        # Determine action
        if should_bet and portfolio_risk != "high" and execution_risk != "high":
            action = "BET"
            confidence = 85 if edge_pct >= 10 else 75 if edge_pct >= 5 else 65
        elif should_bet:
            action = "BET_CAUTIOUS"
            confidence = 60
        elif edge_pct > 2:
            action = "WAIT"
            confidence = 50
        else:
            action = "PASS"
            confidence = 30

        # Calculate overall risk score
        risk_scores = {
            "low": 90,
            "medium": 60,
            "high": 30,
            "unknown": 50
        }
        overall_risk_score = (
            risk_scores.get(execution_risk, 50) * 0.4 +
            risk_scores.get(portfolio_risk, 50) * 0.6
        )

        # Extract key risks
        key_risks = []
        if execution_risk in ("medium", "high"):
            key_risks.append(f"Execution risk: {execution_risk}")
        if portfolio_risk in ("medium", "high"):
            key_risks.append(f"Portfolio risk: {portfolio_risk}")
        if edge_pct < 5:
            key_risks.append("Edge is relatively small")

        portfolio_warnings = portfolio_analysis.get("warnings", [])
        key_risks.extend(portfolio_warnings[:2])  # Add top 2 warnings

        # Alternative scenarios
        alternative_scenarios = [
            {
                "scenario": "Don't bet",
                "reason": "Preserve capital if edge insufficient or risk too high",
                "outcome": "No gain/loss"
            },
            {
                "scenario": f"Half size (${rec_size * 0.5:.2f})",
                "reason": "Reduce risk while maintaining exposure",
                "outcome": f"Lower potential profit but safer"
            }
        ]

        # Get top evidence from task input
        evidence = task_input.get("evidence", [])
        top_evidence = sorted(
            evidence,
            key=lambda e: abs(e.get("estimated_LLR", 0)),
            reverse=True
        )[:5] if evidence else []

        recommendation = BettingRecommendation(
            should_bet=should_bet,
            action=action,
            confidence_score=confidence,
            market_id=market_data.get("market_id", ""),
            market_slug=market_data.get("slug", ""),
            market_question=market_data.get("question", ""),
            provider=market_data.get("provider", "unknown"),
            p_bayesian=p_bayesian,
            market_price=market_price,
            edge=edge,
            edge_pct=edge_pct,
            expected_value_pct=ev_pct,
            recommended_size=Decimal(str(rec_size)),
            recommended_size_pct=size_pct,
            kelly_fraction=kelly_fractional,
            execution_risk=execution_risk,
            portfolio_risk=portfolio_risk,
            overall_risk_score=overall_risk_score,
            market_quality_score=market_quality,
            liquidity_usd=liquidity,
            spread_bps=spread_bps,
            rationale=rationale,
            key_risks=key_risks,
            alternative_scenarios=alternative_scenarios,
            top_supporting_evidence=top_evidence
        )

        logger.info(
            f"Betting recommendation: {action}, "
            f"size=${rec_size:.2f}, confidence={confidence}%"
        )

        return recommendation

    async def analyze_and_recommend(
        self,
        market_data: Dict[str, Any],
        p_bayesian: float,
        confidence: int,
        evidence: List[Dict[str, Any]] = None,
        orderbook: Optional[Dict[str, Any]] = None
    ) -> BettingRecommendation:
        """
        Analyze market and generate betting recommendation.

        Args:
            market_data: Market information (slug, question, price, volume, liquidity, etc.)
            p_bayesian: Bayesian probability estimate (0-1)
            confidence: Confidence in probability estimate (0-100)
            evidence: Supporting evidence items
            orderbook: Optional orderbook data

        Returns:
            BettingRecommendation with complete analysis
        """
        # Get current portfolio state
        bankroll_state = await self.bankroll.get_current_state()
        portfolio_summary = await self.bankroll.get_position_summary()

        # Get open positions
        open_positions_query = """
            SELECT market_slug, provider, entry_size, metadata
            FROM positions
            WHERE status = 'open'
        """
        open_positions = await self.bankroll.db.fetch(open_positions_query)

        positions_list = [
            {
                "market_slug": pos["market_slug"],
                "provider": pos["provider"],
                "size": float(pos["entry_size"]),
                "category": (pos["metadata"] or {}).get("category", "unknown")
            }
            for pos in open_positions
        ]

        # Prepare task input
        task_input = {
            "market_data": market_data,
            "p_bayesian": p_bayesian,
            "confidence": confidence,
            "evidence": evidence or [],
            "orderbook": orderbook,
            "bankroll_state": {
                "total_equity": float(bankroll_state.total_equity),
                "cash_balance": float(bankroll_state.cash_balance),
                "total_exposure": float(bankroll_state.total_exposure),
                "exposure_pct": float(bankroll_state.exposure_pct),
                "num_positions": bankroll_state.num_open_positions
            },
            "open_positions": positions_list,
            "portfolio_summary": {
                "open_positions": portfolio_summary["open_positions"],
                "total_unrealized_pnl": float(portfolio_summary["total_unrealized_pnl"]),
                "win_rate": float(portfolio_summary["win_rate"])
            },
            "risk_limits": {
                "max_position_size_pct": float(self.risk.limits.max_position_size_pct),
                "max_total_exposure_pct": float(self.risk.limits.max_total_exposure_pct),
                "max_drawdown_pct": float(self.risk.limits.max_drawdown_pct)
            }
        }

        # Create human message with task description
        task_description = f"""
Analyze this prediction market and provide a professional betting recommendation:

**Market**: {market_data.get('question', 'Unknown')}
**Provider**: {market_data.get('provider', 'unknown')}
**Current Price**: {market_data.get('yes_price', 0):.1%}
**Your Probability (Bayesian)**: {p_bayesian:.1%}
**Confidence**: {confidence}%

**Market Stats**:
- Volume: ${market_data.get('volume', 0):,.0f}
- Liquidity: ${market_data.get('liquidity', 0):,.0f}
- Spread: {market_data.get('spread', 0):.1%}

**Current Portfolio**:
- Total Equity: ${bankroll_state.total_equity:,.2f}
- Available Cash: ${bankroll_state.cash_balance:,.2f}
- Open Positions: {bankroll_state.num_open_positions}
- Current Exposure: {bankroll_state.exposure_pct:.1f}%

**Your Task**:
1. Analyze market context (liquidity, execution risk)
2. Evaluate betting edge (p_bayesian vs market price)
3. Assess portfolio risk (correlation, concentration)
4. Calculate optimal position size (Kelly with safety)
5. Generate comprehensive rationale
6. Make final recommendation: BET, PASS, or WAIT

Use all available tools systematically. Be thorough but decisive.
"""

        # Run agent
        result = await self.run_task(
            task_description=task_description,
            task_input=task_input
        )

        return result
