"""
Supabase Database Client for POLYSEER
Handles all database operations with proper error handling and type safety
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from config import settings
from supabase import create_client, Client
import os


class SupabaseClient:
    """Client for Supabase database operations"""

    def __init__(self):
        """Initialize Supabase client"""
        try:

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

    # TRACKED TRADERS

    async def upsert_tracked_trader(
        self,
        wallet_address: str,
        composite_score: float,
        early_betting_pct: float = 0.0,
        volume_consistency: float = 0.0,
        win_rate: float = 0.0,
        edge_score: float = 0.0,
        activity_level: float = 0.0,
        pnl_30d: float = 0.0,
        pnl_90d: float = 0.0,
        trade_count: int = 0,
        sharpe_equivalent: float = 0.0,
        status: str = "active"
    ) -> Optional[Dict[str, Any]]:
        """Insert or update tracked trader"""
        if not self.client:
            return None

        data = {
            "wallet_address": wallet_address.lower(),
            "composite_score": composite_score,
            "early_betting_pct": early_betting_pct,
            "volume_consistency": volume_consistency,
            "win_rate": win_rate,
            "edge_score": edge_score,
            "activity_level": activity_level,
            "pnl_30d": pnl_30d,
            "pnl_90d": pnl_90d,
            "trade_count": trade_count,
            "sharpe_equivalent": sharpe_equivalent,
            "status": status
        }

        try:
            result = self.client.table("tracked_traders").upsert(
                data,
                on_conflict="wallet_address"
            ).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error upserting tracked trader: {e}")
            return None

    async def get_tracked_traders(
        self,
        status: Optional[str] = None,
        min_score: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get tracked traders with optional filters"""
        if not self.client:
            return []

        try:
            query = self.client.table("tracked_traders").select("*")

            if status:
                query = query.eq("status", status)

            if min_score is not None:
                query = query.gte("composite_score", min_score)

            result = query.order("composite_score", desc=True).limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error fetching tracked traders: {e}")
            return []

    async def insert_trader_activity(
        self,
        wallet_address: str,
        score_snapshot: float,
        trade_count_snapshot: int = 0,
        pnl_snapshot: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Insert trader activity snapshot"""
        if not self.client:
            return None

        data = {
            "wallet_address": wallet_address.lower(),
            "score_snapshot": score_snapshot,
            "trade_count_snapshot": trade_count_snapshot,
            "pnl_snapshot": pnl_snapshot,
            "metadata": metadata or {}
        }

        try:
            result = self.client.table("trader_activity").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error inserting trader activity: {e}")
            return None

    async def insert_onchain_trade(
        self,
        transaction_hash: str,
        wallet_address: str,
        market_slug: Optional[str] = None,
        token_address: Optional[str] = None,
        token_id: Optional[str] = None,
        side: str = "BUY",
        size_usd: float = 0.0,
        price: float = 0.0,
        timestamp: Optional[datetime] = None,
        block_number: Optional[int] = None,
        market_created_at: Optional[datetime] = None,
        raw_event_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Insert on-chain trade record"""
        if not self.client:
            return None

        data = {
            "transaction_hash": transaction_hash,
            "wallet_address": wallet_address.lower(),
            "market_slug": market_slug,
            "token_address": token_address,
            "token_id": token_id,
            "side": side,
            "size_usd": size_usd,
            "price": price,
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "block_number": block_number,
            "market_created_at": market_created_at.isoformat() if market_created_at else None,
            "raw_event_data": raw_event_data or {}
        }

        try:
            result = self.client.table("onchain_trades").upsert(
                data,
                on_conflict="transaction_hash"
            ).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error inserting onchain trade: {e}")
            return None

    # ========================================================================
    # PAPER TRADING
    # ========================================================================

    async def insert_paper_trading_log(
        self,
        signal_id: str,
        wallet_address: str,
        market_slug: str,
        side: str,
        size_usd: float,
        expected_price: float,
        fill_price: float,
        slippage_bps: int = 0,
        status: str = "filled",
        pnl_realized: Optional[float] = None,
        pnl_unrealized: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Insert paper trading log entry"""
        if not self.client:
            return None

        data = {
            "signal_id": signal_id,
            "wallet_address": wallet_address.lower(),
            "market_slug": market_slug,
            "side": side,
            "size_usd": size_usd,
            "expected_price": expected_price,
            "fill_price": fill_price,
            "slippage_bps": slippage_bps,
            "status": status,
            "pnl_realized": pnl_realized,
            "pnl_unrealized": pnl_unrealized,
            "metadata": metadata or {}
        }

        try:
            result = self.client.table("paper_trading_logs").insert(data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error inserting paper trading log: {e}")
            return None

    async def upsert_paper_trading_summary(
        self,
        date: datetime,
        total_trades: int = 0,
        filled_trades: int = 0,
        rejected_trades: int = 0,
        total_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        sharpe_ratio: Optional[float] = None,
        sortino_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        win_rate: float = 0.0,
        total_volume_usd: float = 0.0,
        avg_trade_size_usd: float = 0.0,
        avg_slippage_bps: float = 0.0,
        max_slippage_bps: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Insert or update paper trading daily summary"""
        if not self.client:
            return None

        data = {
            "date": date.date().isoformat() if isinstance(date, datetime) else date,
            "total_trades": total_trades,
            "filled_trades": filled_trades,
            "rejected_trades": rejected_trades,
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_volume_usd": total_volume_usd,
            "avg_trade_size_usd": avg_trade_size_usd,
            "avg_slippage_bps": avg_slippage_bps,
            "max_slippage_bps": max_slippage_bps
        }

        try:
            result = self.client.table("paper_trading_summary").upsert(
                data,
                on_conflict="date"
            ).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error upserting paper trading summary: {e}")
            return None

    async def get_paper_trading_logs(
        self,
        wallet_address: Optional[str] = None,
        market_slug: Optional[str] = None,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get paper trading logs with optional filters"""
        if not self.client:
            return []

        try:
            query = self.client.table("paper_trading_logs").select("*")

            if wallet_address:
                query = query.eq("wallet_address", wallet_address.lower())

            if market_slug:
                query = query.eq("market_slug", market_slug)

            if start_date:
                query = query.gte("timestamp", start_date.isoformat())

            if end_date:
                query = query.lte("timestamp", end_date.isoformat())

            result = query.order("timestamp", desc=True).limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error fetching paper trading logs: {e}")
            return []
