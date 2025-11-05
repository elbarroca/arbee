"""
Supabase Database Client for POLYSEER
Handles all database operations with proper error handling and type safety
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from config.settings import settings
from supabase import create_client, Client
import os


class SupabaseClient:
    """Client for Supabase database operations"""

    def __init__(self):
        """Initialize Supabase client"""
        try:
            # Use uppercase attribute names to match Settings class
            self.client: Client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY
            )
        except ImportError:
            print("Warning: Supabase library not installed. Install with: pip install supabase")
            self.client = None

    # ========================================================================
    # MARKETS
    # ========================================================================

    async def create_market(
        self,
        provider: str,
        market_slug: str,
        question: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new market entry

        Args:
            provider: Platform name (polymarket, kalshi, calci)
            market_slug: Unique market identifier
            question: Market question text
            **kwargs: Additional fields (description, category, end_date, etc.)

        Returns:
            Created market dict or None
        """
        if not self.client:
            return None

        data = {
            "provider": provider,
            "market_slug": market_slug,
            "question": question,
            **kwargs
        }

        try:
            result = self.client.table("markets").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error creating market: {e}")
            return None

    async def get_market_by_slug(
        self,
        provider: str,
        market_slug: str
    ) -> Optional[Dict[str, Any]]:
        """Get market by provider and slug"""
        if not self.client:
            return None

        try:
            result = self.client.table("markets").select("*").eq(
                "provider", provider
            ).eq("market_slug", market_slug).execute()

            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error fetching market: {e}")
            return None

    async def upsert_market(
        self,
        provider: str,
        market_slug: str,
        question: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Insert or update market"""
        if not self.client:
            return None

        data = {
            "provider": provider,
            "market_slug": market_slug,
            "question": question,
            **kwargs
        }

        try:
            result = self.client.table("markets").upsert(
                data,
                on_conflict="provider,market_slug"
            ).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error upserting market: {e}")
            return None

    # ========================================================================
    # MARKET PRICES (Time-Series)
    # ========================================================================

    async def insert_price_snapshot(
        self,
        market_id: UUID,
        price: float,
        implied_prob: float,
        volume: Optional[float] = None,
        liquidity: Optional[float] = None,
        source: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Insert a price snapshot"""
        if not self.client:
            return None

        data = {
            "market_id": str(market_id),
            "price": price,
            "implied_prob": implied_prob,
            "volume": volume,
            "liquidity": liquidity,
            "source": source,
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            result = self.client.table("market_prices").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error inserting price: {e}")
            return None

    # ========================================================================
    # RESEARCH PLANS
    # ========================================================================

    async def create_research_plan(
        self,
        market_id: UUID,
        workflow_id: str,
        p0_prior: float,
        prior_justification: str,
        subclaims: List[Dict[str, Any]],
        search_seeds: Dict[str, List[str]],
        key_variables: List[str],
        decision_criteria: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Create research plan from Planner Agent output"""
        if not self.client:
            return None

        data = {
            "market_id": str(market_id),
            "workflow_id": workflow_id,
            "p0_prior": p0_prior,
            "prior_justification": prior_justification,
            "subclaims_json": subclaims,
            "search_seeds_json": search_seeds,
            "key_variables": key_variables,
            "decision_criteria": decision_criteria
        }

        try:
            result = self.client.table("research_plans").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error creating research plan: {e}")
            return None

    # ========================================================================
    # EVIDENCE
    # ========================================================================

    async def insert_evidence_batch(
        self,
        research_plan_id: UUID,
        evidence_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Insert multiple evidence items at once"""
        if not self.client:
            return []

        # Prepare data
        data = []
        for item in evidence_items:
            data.append({
                "research_plan_id": str(research_plan_id),
                **item
            })

        try:
            result = self.client.table("evidence").insert(data).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error inserting evidence batch: {e}")
            return []

    # ========================================================================
    # BAYESIAN ANALYSIS
    # ========================================================================

    async def create_bayesian_analysis(
        self,
        research_plan_id: UUID,
        p0: float,
        log_odds_prior: float,
        log_odds_posterior: float,
        p_bayesian: float,
        p_neutral: float,
        evidence_summary: List[Dict[str, Any]],
        correlation_adjustments: Dict[str, Any],
        sensitivity_analysis: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Create Bayesian analysis from Analyst Agent output"""
        if not self.client:
            return None

        data = {
            "research_plan_id": str(research_plan_id),
            "p0": p0,
            "log_odds_prior": log_odds_prior,
            "log_odds_posterior": log_odds_posterior,
            "p_bayesian": p_bayesian,
            "p_neutral": p_neutral,
            "evidence_summary_json": evidence_summary,
            "correlation_adjustments_json": correlation_adjustments,
            "sensitivity_analysis_json": sensitivity_analysis
        }

        try:
            result = self.client.table("bayesian_analysis").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error creating Bayesian analysis: {e}")
            return None

    # ========================================================================
    # ARBITRAGE OPPORTUNITIES
    # ========================================================================

    async def insert_arbitrage_opportunities(
        self,
        market_id: UUID,
        analysis_id: UUID,
        opportunities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Insert arbitrage opportunities"""
        if not self.client:
            return []

        data = []
        for opp in opportunities:
            data.append({
                "market_id": str(market_id),
                "analysis_id": str(analysis_id),
                **opp
            })

        try:
            result = self.client.table("arbitrage_opportunities").insert(data).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error inserting arbitrage opportunities: {e}")
            return []

    # ========================================================================
    # WORKFLOW EXECUTIONS
    # ========================================================================

    async def create_workflow_execution(
        self,
        workflow_id: str,
        market_id: Optional[UUID] = None
    ) -> Optional[Dict[str, Any]]:
        """Create a new workflow execution record"""
        if not self.client:
            return None

        data = {
            "workflow_id": workflow_id,
            "market_id": str(market_id) if market_id else None,
            "status": "running"
        }

        try:
            result = self.client.table("workflow_executions").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error creating workflow execution: {e}")
            return None

    async def update_workflow_execution(
        self,
        workflow_id: str,
        status: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Update workflow execution"""
        if not self.client:
            return None

        data = {"status": status, **kwargs}

        if status in ["completed", "failed", "cancelled"]:
            data["completed_at"] = datetime.utcnow().isoformat()

        try:
            result = self.client.table("workflow_executions").update(data).eq(
                "workflow_id", workflow_id
            ).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error updating workflow execution: {e}")
            return None

    # ========================================================================
    # VIEWS
    # ========================================================================

    async def get_latest_market_analysis(
        self,
        market_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query latest_market_analysis view"""
        if not self.client:
            return []

        try:
            query = self.client.table("latest_market_analysis").select("*")

            if market_id:
                query = query.eq("market_id", str(market_id))

            result = query.limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error querying latest market analysis: {e}")
            return []

    async def get_best_arbitrage_opportunities(
        self,
        min_edge: float = 0.02,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Query best_arbitrage_opportunities view"""
        if not self.client:
            return []

        try:
            result = self.client.table("best_arbitrage_opportunities").select("*").gte(
                "edge", min_edge
            ).limit(limit).execute()

            return result.data if result.data else []
        except Exception as e:
            print(f"Error querying arbitrage opportunities: {e}")
            return []
