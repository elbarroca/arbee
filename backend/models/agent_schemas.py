"""
Pydantic schemas for POLYSEER agent outputs
All schemas follow the JSON structures defined in CLAUDE.MD
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Dict, Any, Annotated
from datetime import datetime, date


# ============================================================================
# SIMPLIFIED SCHEMAS (NEW ARCHITECTURE)
# ============================================================================

class SubjectProfile(BaseModel):
    """
    Profile of the subject being researched (person/team/event).
    Built during research PHASE 1: PROFILING
    """
    entity_name: str = Field(..., description="Name of the subject (e.g., 'Diplo')")
    entity_type: Literal["person", "team", "event", "organization"] = Field(
        ...,
        description="Type of entity"
    )
    key_facts: Dict[str, str] = Field(
        default_factory=dict,
        description="Extracted facts (e.g., {'age': '46', 'profession': 'DJ/producer'})"
    )
    baseline_capabilities: Optional[str] = Field(
        None,
        description="Assessment of baseline ability (e.g., 'Recreational runner, no competitive history')"
    )


class EvidenceItem(BaseModel):
    """
    Simplified evidence model - replaces complex LLR calibration with 1-10 relevance scoring.
    Focuses on: How meaningful is this information for predicting the outcome?
    """
    source_url: str = Field(..., description="URL to source")
    published_date: Optional[str] = Field(None, description="Publication date YYYY-MM-DD or None")
    key_fact: str = Field(..., description="One sentence summary of the factual claim")
    relevance_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="How useful is this for predicting outcome? (9-10: direct answer, 7-8: strong proxy, 5-6: context, 3-4: weak signal, 1-2: barely relevant)"
    )
    support_direction: Literal["YES", "NO", "NEUTRAL"] = Field(
        ...,
        description="Does this make the market outcome more likely (YES), less likely (NO), or neither (NEUTRAL)?"
    )
    is_primary: bool = Field(
        ...,
        description="True = direct measurement/race result, False = news article/secondary source"
    )
    extraction_reasoning: str = Field(
        ...,
        description="WHY this score and direction were chosen"
    )


class ResearchPhase(BaseModel):
    """
    Tracks research progress through 3 phases with confidence-based termination.
    Replaces iteration counting with question answering.
    """
    phase: Literal["profiling", "benchmarking", "evidence_gathering", "complete"] = Field(
        default="profiling",
        description="Current research phase"
    )
    questions_answered: List[str] = Field(
        default_factory=list,
        description="Research questions that have been answered"
    )
    questions_remaining: List[str] = Field(
        default_factory=list,
        description="Research questions still to be answered"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in having enough information (stop at 0.7+)"
    )


class SimplifiedPlannerOutput(BaseModel):
    """
    Simplified planner output - generates QUESTIONS not search seeds.
    Focus on: What do we need to know to estimate this probability?
    """
    market_slug: str
    market_question: str
    market_type: Literal["sports", "politics", "finance", "entertainment", "other"]
    subject_to_profile: SubjectProfile
    core_research_questions: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="Key questions to answer (not search queries)"
    )
    baseline_prior: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Starting probability (default 0.5 unless obvious)"
    )
    prior_reasoning: str = Field(..., description="Why this prior makes sense")


class SimplifiedResearcherOutput(BaseModel):
    """
    Simplified researcher output - tracks adaptive search through phases.
    """
    subject_profile: Optional[SubjectProfile] = None
    benchmark_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Baseline/benchmark data (e.g., 'average 5k time for 46yo')"
    )
    specific_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Subject-specific evidence (e.g., 'Diplo's actual race times')"
    )
    research_phase: ResearchPhase
    search_queries_used: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Log of queries with reasoning: [{'query': '...', 'reasoning': '...'}]"
    )
    total_searches: int = Field(default=0)


class SimplifiedAnalystOutput(BaseModel):
    """
    Simplified analyst output - weighted scoring instead of complex LLR aggregation.
    """
    probability: float = Field(..., ge=0.0, le=1.0, description="Final probability estimate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in estimate (0-1)")
    yes_evidence_weight: float = Field(default=0.0, description="Total weight of YES evidence")
    no_evidence_weight: float = Field(default=0.0, description="Total weight of NO evidence")
    neutral_evidence_weight: float = Field(default=0.0, description="Total weight of NEUTRAL evidence")
    total_evidence_count: int = Field(default=0, description="Total number of evidence items")
    primary_evidence_count: int = Field(default=0, description="Number of primary sources")
    baseline_prior: float = Field(default=0.5, ge=0.0, le=1.0, description="Starting probability")
    reasoning: str = Field(..., description="Human-readable explanation of probability")
    top_yes_evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top YES evidence items with weights"
    )
    top_no_evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top NO evidence items with weights"
    )
    sensitivity_range: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability range (low/high) for sensitivity analysis"
    )


# ============================================================================
# ORIGINAL PLANNER AGENT SCHEMAS (LEGACY - TO BE DEPRECATED)
# ============================================================================

class Subclaim(BaseModel):
    """A subclaim decomposed from the main market question"""
    id: str = Field(..., description="Unique identifier for subclaim")
    text: str = Field(..., description="The subclaim text")
    direction: Literal["pro", "con"] = Field(..., description="Whether this supports YES or NO")


class SearchSeeds(BaseModel):
    """Search query seeds for evidence gathering"""
    pro: List[str] = Field(default_factory=list, description="Queries supporting YES")
    con: List[str] = Field(default_factory=list, description="Queries supporting NO")
    general: List[str] = Field(default_factory=list, description="Neutral/contextual queries")


class PlannerOutput(BaseModel):
    """Output from the Planner Agent"""
    market_slug: str = Field(..., description="Unique market identifier")
    market_question: str = Field(..., description="The prediction market question")
    p0_prior: float = Field(..., ge=0.0, le=1.0, description="Initial prior probability")
    prior_justification: str = Field(..., description="Reasoning for the prior")
    subclaims: List[Subclaim] = Field(..., min_length=4, max_length=10)
    key_variables: List[str] = Field(..., description="Critical factors affecting outcome")
    search_seeds: SearchSeeds
    decision_criteria: List[str] = Field(..., description="What would resolve the question")
    reasoning_trace: str = Field(..., description="Step-by-step chain-of-thought reasoning process")


# ============================================================================
# RESEARCHER AGENT SCHEMAS
# ============================================================================

class Evidence(BaseModel):
    """A single piece of evidence from research"""
    subclaim_id: str = Field(..., description="References Subclaim.id")
    title: str = Field(..., description="Article/source title")
    url: str = Field(..., description="Full URL to source")
    published_date: date = Field(..., description="Publication date YYYY-MM-DD")
    source_type: Literal["primary", "high_quality_secondary", "secondary", "weak"]
    claim_summary: str = Field(..., max_length=500, description="Key claim extracted")
    support: Literal["pro", "con", "neutral"] = Field(..., description="Direction of evidence")
    verifiability_score: float = Field(..., ge=0.0, le=1.0)
    independence_score: float = Field(..., ge=0.0, le=1.0)
    recency_score: float = Field(..., ge=0.0, le=1.0)
    estimated_LLR: float = Field(..., description="Log-likelihood ratio estimate")
    extraction_notes: str = Field(..., description="Context and caveats")

    @field_validator('estimated_LLR')
    @classmethod
    def validate_llr_range(cls, v: float, info) -> float:
        """Validate LLR is within calibrated ranges"""
        source_type = info.data.get('source_type')
        if source_type == 'primary' and abs(v) > 3.0:
            raise ValueError(f"Primary source LLR should be ±1-3, got {v}")
        elif source_type in ['high_quality_secondary', 'secondary'] and abs(v) > 1.0:
            raise ValueError(f"Secondary source LLR should be ±0.1-1.0, got {v}")
        return v


class ResearcherOutput(BaseModel):
    """Output from Researcher Agents (aggregated)"""
    evidence_items: List[Evidence] = Field(..., max_length=30)
    total_pro_count: int = Field(default=0)
    total_con_count: int = Field(default=0)
    total_pro_llr: float = Field(
        default=0.0,
        description="Sum of positive (pro) LLR contributions"
    )
    total_con_llr: float = Field(
        default=0.0,
        description="Sum of absolute negative (con) LLR contributions"
    )
    net_llr: float = Field(
        default=0.0,
        description="Signed sum of all LLR contributions"
    )
    context_alignment_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of directional evidence aligned with its subclaim goal"
    )
    research_timestamp: datetime = Field(default_factory=datetime.utcnow)
    search_strategy: str = Field(default="", description="Explanation of research approach taken")


# ============================================================================
# CRITIC AGENT SCHEMAS
# ============================================================================

class CorrelationWarning(BaseModel):
    """Warning about correlated evidence"""
    cluster: List[str] = Field(..., description="List of evidence IDs in cluster")
    note: str = Field(..., description="Explanation of correlation")


class CriticOutput(BaseModel):
    """Output from the Critic Agent"""
    duplicate_clusters: List[List[str]] = Field(
        default_factory=list,
        description="Groups of duplicate/near-duplicate evidence IDs"
    )
    missing_topics: List[str] = Field(
        default_factory=list,
        description="Important angles not covered"
    )
    over_represented_sources: List[str] = Field(
        default_factory=list,
        description="Sources appearing too frequently"
    )
    correlation_warnings: List[CorrelationWarning] = Field(default_factory=list)
    follow_up_search_seeds: List[str] = Field(
        default_factory=list,
        description="Additional queries to fill gaps"
    )
    analysis_process: str = Field(..., description="Step-by-step analysis of how evidence was critiqued")


# ============================================================================
# ANALYST AGENT SCHEMAS
# ============================================================================

class EvidenceSummaryItem(BaseModel):
    """Summary of a single evidence item's contribution"""
    id: str = Field(..., description="Evidence ID or subclaim_id")
    LLR: float = Field(..., description="Original log-likelihood ratio")
    weight: float = Field(..., ge=0.0, le=1.0, description="Quality weight applied")
    adjusted_LLR: float = Field(..., description="Final LLR after adjustments")


class CorrelationAdjustment(BaseModel):
    """Details on how correlation was handled"""
    method: str = Field(..., description="E.g., 'shrinkage', 'cluster_averaging'")
    details: str = Field(..., description="Explanation of adjustments made")


class SensitivityScenario(BaseModel):
    """Alternative probability under different assumptions"""
    scenario: str = Field(..., description="E.g., '+25% LLR', 'remove_weakest_sources'")
    p: float = Field(..., ge=0.0, le=1.0, description="Resulting probability")


class AnalystOutput(BaseModel):
    """Output from the Analyst Agent (Bayesian aggregator)"""
    p0: float = Field(..., ge=0.0, le=1.0, description="Prior probability")
    log_odds_prior: float = Field(..., description="ln(p0 / (1-p0))")
    evidence_summary: List[EvidenceSummaryItem]
    correlation_adjustments: CorrelationAdjustment
    log_odds_posterior: float = Field(..., description="After evidence aggregation")
    p_bayesian: float = Field(..., ge=0.0, le=1.0, description="Final probability estimate")
    p_neutral: float = Field(..., ge=0.0, le=1.0, description="Uncertainty-adjusted estimate")
    sensitivity_analysis: List[SensitivityScenario] = Field(default_factory=list)
    calculation_steps: List[str] = Field(..., description="Step-by-step mathematical reasoning trace")

    @field_validator('p_bayesian', 'p_neutral')
    @classmethod
    def clamp_extremes(cls, v: float) -> float:
        """Prevent probabilities too close to 0 or 1"""
        if v < 0.01:
            return 0.01
        if v > 0.99:
            return 0.99
        return v


# ============================================================================
# ARBITRAGE DETECTOR SCHEMAS
# ============================================================================

class PlatformSide(BaseModel):
    """Specific side of a trade on a platform"""
    platform: Literal["polymarket", "kalshi", "calci"]
    market_id: str = Field(..., description="Market identifier")
    outcome: Literal["YES", "NO"] = Field(..., description="Which outcome to bet on")
    price: float = Field(..., ge=0.0, le=1.0, description="Price for this outcome")
    stake: float = Field(..., ge=0.0, description="Amount to stake on this side")


class ArbitrageOpportunity(BaseModel):
    """A potential arbitrage opportunity"""
    arbitrage_type: Literal["mispricing", "cross_platform"] = Field(
        ...,
        description="Type: mispricing (single-sided based on Bayesian) or cross_platform (opposite sides, guaranteed)"
    )

    # Single platform fields (for mispricing arbitrage)
    market_id: Optional[str] = Field(None, description="Market identifier from provider")
    provider: Optional[Literal["polymarket", "kalshi", "calci"]] = None
    price: Optional[float] = Field(None, ge=0.0, le=1.0, description="Current market price")
    implied_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    edge: Optional[float] = Field(None, description="p_bayesian - implied_probability")

    # Cross-platform fields (for true arbitrage)
    platform_pair: Optional[List[str]] = Field(
        None,
        description="List of platforms involved in cross-platform arb"
    )
    side_a: Optional[PlatformSide] = Field(
        None,
        description="First side of the arbitrage (e.g., YES on Polymarket)"
    )
    side_b: Optional[PlatformSide] = Field(
        None,
        description="Second side of the arbitrage (e.g., NO on Kalshi)"
    )
    total_cost: Optional[float] = Field(
        None,
        description="Total cost to enter both sides (with fees)"
    )
    guaranteed_profit: Optional[float] = Field(
        None,
        description="Guaranteed profit per dollar (for cross-platform arb)"
    )

    # Common fields
    transaction_costs: float = Field(default=0.0, ge=0.0)
    slippage_estimate: float = Field(default=0.0, ge=0.0)
    expected_value_per_dollar: float = Field(..., description="EV after costs")
    kelly_fraction: float = Field(..., ge=0.0, le=1.0, description="Optimal bet fraction")
    suggested_stake: float = Field(..., ge=0.0, description="Dollar amount to stake")
    trade_rationale: str = Field(..., description="Why this is/isn't a good trade")


class ArbitrageDetectorOutput(BaseModel):
    """Output from Arbitrage Detector"""
    opportunities: List[ArbitrageOpportunity] = Field(default_factory=list)
    best_opportunity: Optional[ArbitrageOpportunity] = None
    total_expected_value: float = Field(default=0.0)
    disclaimer: str = Field(default="NOT FINANCIAL ADVICE")


# ============================================================================
# REPORTER AGENT SCHEMAS
# ============================================================================

class TopDriver(BaseModel):
    """A key factor influencing the forecast"""
    direction: Literal["pro", "con"]
    summary: str = Field(..., max_length=200)
    strength: Literal["strong", "moderate", "weak"]


class ReporterOutput(BaseModel):
    """Final output from Reporter Agent"""
    market_question: str
    p_bayesian: float = Field(..., ge=0.0, le=1.0)
    confidence_interval: List[float] = Field(
        ...,
        description="(lower, upper) bounds"
    )
    top_pro_drivers: List[TopDriver] = Field(..., max_length=3)
    top_con_drivers: List[TopDriver] = Field(..., max_length=3)
    arbitrage_summary: str
    next_steps: List[str] = Field(default_factory=list)
    tldr: str = Field(..., max_length=300, description="1-2 sentence summary")
    executive_summary: str = Field(
        ...,
        min_length=200,
        max_length=600,
        description="Markdown summary"
    )
    full_json: Dict[str, Any] = Field(..., description="Complete data package")
    disclaimer: str = Field(default="NOT FINANCIAL ADVICE")


# ============================================================================
# COMBINED WORKFLOW STATE
# ============================================================================

from typing import TypedDict
from langgraph.graph.message import AnyMessage, add_messages


def append_traces(
    state_traces: Optional[List[Dict[str, Any]]],
    new_traces: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Reducer for accumulating agent trace logs in WorkflowState.
    """
    existing = state_traces or []
    return [*existing, *new_traces]

class WorkflowState(TypedDict, total=False):
    """State passed through LangGraph workflow"""
    # Input (required)
    market_question: str
    market_url: str
    market_slug: str
    providers: List[str]
    bankroll: float
    max_kelly: float
    min_edge_threshold: float
    save_to_db: bool

    # Workflow metadata
    workflow_id: str
    timestamp: str
    context: Dict[str, Any]
    messages: Annotated[List[AnyMessage], add_messages]
    agent_traces: Annotated[List[Dict[str, Any]], append_traces]

    # Intermediate values
    p0_prior: float
    search_seeds: Any
    subclaims: List[Any]
    all_evidence: List[Any]
    p_bayesian: float

    # Agent outputs (populated during workflow)
    planner_output: PlannerOutput
    researcher_output: Dict[str, ResearcherOutput]
    critic_output: CriticOutput
    analyst_output: AnalystOutput
    arbitrage_output: List[Any]
    reporter_output: ReporterOutput


# ============================================================================
# BETTING WORKFLOW SCHEMAS (NEW)
# ============================================================================

class BettingEvent(BaseModel):
    """Betting event with markets from multiple providers"""
    title: str = Field(..., description="Event title")
    slug: str = Field(..., description="Event slug/ID")
    markets: List[Dict[str, Any]] = Field(default_factory=list, description="Markets for this event")
    providers: List[str] = Field(default_factory=list, description="Providers offering this event")
    total_volume: float = Field(default=0.0, description="Total volume across all providers")
    total_liquidity: float = Field(default=0.0, description="Total liquidity across all providers")


class MarketDataByProvider(BaseModel):
    """Market data grouped by provider"""
    provider: str = Field(..., description="Provider name (polymarket, kalshi, etc)")
    market_id: str = Field(..., description="Provider-specific market ID")
    slug: str = Field(..., description="Market slug")
    question: str = Field(..., description="Market question")
    yes_price: Optional[float] = Field(None, ge=0.0, le=1.0, description="YES outcome price")
    no_price: Optional[float] = Field(None, ge=0.0, le=1.0, description="NO outcome price")
    mid_price: Optional[float] = Field(None, ge=0.0, le=1.0, description="Mid price")
    volume: float = Field(default=0.0, description="24h volume")
    liquidity: float = Field(default=0.0, description="Available liquidity")
    spread: Optional[float] = Field(None, description="Bid-ask spread")
    spread_bps: Optional[int] = Field(None, description="Spread in basis points")
    orderbook: Optional[Dict[str, Any]] = Field(None, description="Orderbook data if available")


class ResearchSummary(BaseModel):
    """Condensed research findings for betting workflow"""
    key_insights: List[str] = Field(default_factory=list, description="Main insights from research")
    sentiment_signals: Dict[str, Any] = Field(default_factory=dict, description="Sentiment indicators")
    insider_activity: Optional[Dict[str, Any]] = Field(None, description="Insider/whale activity detected")
    information_edges: List[str] = Field(default_factory=list, description="Information advantages identified")
    total_evidence_count: int = Field(default=0, description="Number of evidence items analyzed")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Research confidence level")


class AdjustedProbability(BaseModel):
    """Market price adjusted based on research"""
    market_price: float = Field(..., ge=0.0, le=1.0, description="Current market price (prior)")
    research_adjustment: float = Field(default=0.0, description="Adjustment from research (+/-)")
    adjusted_probability: float = Field(..., ge=0.0, le=1.0, description="Final adjusted probability")
    confidence_low: float = Field(..., ge=0.0, le=1.0, description="Lower bound (95% CI)")
    confidence_high: float = Field(..., ge=0.0, le=1.0, description="Upper bound (95% CI)")
    adjustment_reasoning: str = Field(..., description="Why this adjustment was made")
    edge_vs_market: float = Field(..., description="adjusted_probability - market_price")


class BettingRecommendation(BaseModel):
    """Final betting recommendation from Professional Bettor agent"""
    decision: Literal["BET", "PASS", "WAIT"] = Field(..., description="Recommendation")
    action: Optional[str] = Field(None, description="Specific action (e.g., 'BET YES on Polymarket')")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")

    # Probabilities
    p_bayesian: float = Field(..., ge=0.0, le=1.0, description="Adjusted probability estimate")
    market_price: float = Field(..., ge=0.0, le=1.0, description="Current market price")
    edge: float = Field(..., description="p_bayesian - market_price")
    expected_value_pct: float = Field(..., description="Expected value as percentage")

    # Position sizing
    recommended_size: float = Field(..., ge=0.0, description="Recommended stake in dollars")
    kelly_fraction: float = Field(..., ge=0.0, le=1.0, description="Kelly fraction used")
    size_as_pct_bankroll: float = Field(..., ge=0.0, le=1.0, description="Size as % of bankroll")

    # Market selection
    selected_provider: Optional[str] = Field(None, description="Best provider to use")
    selected_market_id: Optional[str] = Field(None, description="Specific market ID")
    selected_threshold: Optional[int] = Field(None, description="Threshold level if applicable")

    # Risk assessment
    execution_risk: Literal["low", "medium", "high"] = Field(..., description="Execution risk")
    portfolio_risk: Literal["low", "medium", "high"] = Field(..., description="Portfolio impact")
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk (0=safe, 1=risky)")

    # Analysis
    market_quality_score: float = Field(..., ge=0.0, le=1.0, description="Liquidity/spread quality")
    liquidity_usd: float = Field(default=0.0, description="Available liquidity")
    spread_bps: int = Field(default=0, description="Spread in basis points")

    # Rationale
    rationale: str = Field(..., description="Detailed reasoning for recommendation")
    key_risks: List[str] = Field(default_factory=list, description="Main risks to consider")
    alternative_scenarios: List[str] = Field(default_factory=list, description="Other outcomes considered")
    research_sources: List[str] = Field(default_factory=list, description="Sources supporting decision")


class BettingReporterOutput(BaseModel):
    """Reporter output for betting workflow"""
    market_question: str
    decision: Literal["BET", "PASS", "WAIT"]
    recommended_position: Optional[Dict[str, Any]] = Field(None, description="Position details if BET")

    # Summary
    tldr: str = Field(..., max_length=300, description="1-2 sentence summary")
    executive_summary: str = Field(..., min_length=200, max_length=800, description="Full summary")

    # Supporting data
    research_highlights: List[str] = Field(default_factory=list, description="Key research findings")
    arbitrage_opportunities: List[Dict[str, Any]] = Field(default_factory=list, description="Arb opportunities")
    threshold_analysis: Optional[Dict[str, Any]] = Field(None, description="Threshold market analysis")

    # Provenance
    all_sources: List[str] = Field(default_factory=list, description="All research sources")
    agent_traces: List[Dict[str, Any]] = Field(default_factory=list, description="Agent execution log")

    disclaimer: str = Field(default="NOT FINANCIAL ADVICE. DO YOUR OWN RESEARCH.")


class BettingWorkflowState(TypedDict, total=False):
    """State for betting-focused workflow (market price as prior)"""
    # Input (required)
    market_question: str
    market_url: str
    market_slug: str
    providers: List[str]
    bankroll: float
    max_kelly: float
    min_edge_threshold: float

    # Workflow metadata
    workflow_id: str
    timestamp: str
    context: Dict[str, Any]
    messages: Annotated[List[AnyMessage], add_messages]
    agent_traces: Annotated[List[Dict[str, Any]], append_traces]

    # Event Fetcher outputs
    events: List[BettingEvent]
    target_markets: List[Dict[str, Any]]

    # Market Analyzer outputs
    market_data_by_provider: Dict[str, MarketDataByProvider]
    related_threshold_markets: List[Dict[str, Any]]
    best_prices: Dict[str, Any]

    # Researcher outputs
    research_summary: ResearchSummary

    # Market Pricer outputs
    adjusted_probability: AdjustedProbability

    # Professional Bettor outputs
    betting_recommendation: BettingRecommendation

    # Reporter outputs
    reporter_output: BettingReporterOutput
