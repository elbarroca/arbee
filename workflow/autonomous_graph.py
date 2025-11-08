"""
Autonomous LangGraph Workflow for POLYSEER
Uses autonomous reasoning agents with tool use, memory, and iterative loops.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

from models.schemas import WorkflowState
from agents.planner import AutonomousPlannerAgent
from agents.researcher import run_parallel_autonomous_research
from agents.critic import AutonomousCriticAgent
from agents.analyst import AutonomousAnalystAgent
from agents.arbitrage import AutonomousArbitrageAgent
from agents.reporter import AutonomousReporterAgent
from clients.polymarket import PolymarketClient
from clients.web3.wallet_tracker import WalletTrackerClient
from clients.trade.trade_executor import TradeExecutor
from agents.copy_trading_agent import CopyTradingAgent
from utils.copy_trading import TradeSignalProcessor
from utils.memory import create_store_from_config

from utils.memory import get_memory_manager
from config.settings import settings
from utils.rich_logging import (
    setup_rich_logging,
    log_workflow_transition,
    log_workflow_progress,
    log_workflow_summary,
    log_agent_separator,
    log_agent_output_full,
)

# Keep logging minimal per module: creation + finalization of workflow.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
workflow_logger = setup_rich_logging("POLYSEER Workflow")


# -----------------------------
# Small helpers
# -----------------------------
def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _trace(agent: str, action: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"agent_name": agent, "timestamp": _now_iso(), "action": action, "details": details}


def _msg(text: str) -> List[tuple[str, str]]:
    return [("assistant", text)]


# -----------------------------
# AUTONOMOUS AGENT NODE FUNCTIONS
# -----------------------------
async def autonomous_planner_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Plan subclaims and initial priors with bounded iteration and memory.
    """
    assert "market_question" in state, "state.market_question is required"

    cfg = (state.get("context") or {}).get("planner_agent", {})
    agent = AutonomousPlannerAgent(
        model_name=cfg.get("model_name", "gpt-4o-mini"),
        temperature=cfg.get("temperature", 0.1),
        enable_auto_memory_query=store is not None,
        store=store,
        max_iterations=cfg.get("max_iterations", 10),
        min_subclaims=cfg.get("min_subclaims", 4),
        max_subclaims=cfg.get("max_subclaims", 10),
        auto_extend_iterations=cfg.get("auto_extend_iterations", True),
        iteration_extension=cfg.get("iteration_extension", 5),
        max_iteration_cap=cfg.get("max_iteration_cap", 40),
        recursion_limit=cfg.get("recursion_limit", 100),
    )

    result = await agent.plan(
        market_question=state["market_question"],
        market_url=state.get("market_url", ""),
        market_slug=state.get("market_slug", ""),
        context=state.get("context", {}),
        max_iterations=cfg.get("run_max_iterations"),
    )

    # Store brief planning memory when available
    if store:
        mm = get_memory_manager(store=store)
        await mm.store_episode_memory(
            episode_id=state.get("workflow_id", ""),
            market_question=state["market_question"],
            memory_type="planning_strategy",
            content={
                "prior": result.p0_prior,
                "subclaim_count": len(result.subclaims),
                "balance": {
                    "pro": sum(1 for sc in result.subclaims if sc.direction == "pro"),
                    "con": sum(1 for sc in result.subclaims if sc.direction == "con"),
                },
            },
            effectiveness=0.9,
        )

    stats = agent.get_stats()
    trace = {
        "subclaim_count": len(result.subclaims),
        "search_seed_counts": {
            "pro": len(result.search_seeds.pro),
            "con": len(result.search_seeds.con),
            "general": len(result.search_seeds.general),
        },
        "prior": result.p0_prior,
        "iterations": stats.get("average_iterations", 0),
        "tool_calls": stats.get("total_tool_calls", 0),
    }

    # Discover related threshold markets
    related_markets = []
    market_slug = state.get("market_slug", "")
    market_question = state.get("market_question", "")
    
    if market_slug and market_question:
        client = PolymarketClient()
        related_markets = await client.get_related_markets(market_slug, market_question)
        logger.info(f"📊 Discovered {len(related_markets)} related threshold markets")

    # Log full planner output
    log_agent_output_full("Planner", result)
    
    return {
        "planner_output": result,
        "p0_prior": result.p0_prior,
        "search_seeds": result.search_seeds,
        "subclaims": result.subclaims,
        "related_markets": related_markets,  # Store related markets in state
        "messages": _msg(f"Autonomous Planner: {trace['subclaim_count']} subclaims, {trace['iterations']:.1f} iterations"),
        "agent_traces": [_trace("AutonomousPlannerAgent", "autonomous_planning_completed", trace)],
    }


async def autonomous_researcher_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Run pro/con/general researcher agents in parallel and collect evidence.
    """
    assert "planner_output" in state, "state.planner_output is required"
    planner_out = state["planner_output"]
    seeds = planner_out.search_seeds
    subclaims = [sc.model_dump() for sc in planner_out.subclaims]

    results = await run_parallel_autonomous_research(
        search_seeds_pro=seeds.pro,
        search_seeds_con=seeds.con,
        search_seeds_general=seeds.general,
        subclaims=subclaims,
        market_question=state["market_question"],
        store=store,
        min_evidence_items=5,
        max_search_attempts=10,
        max_iterations=15,
    )

    all_evidence: List[Any] = []
    counts: Dict[str, int] = {}
    for d in ("pro", "con", "general"):
        items = results[d].evidence_items
        all_evidence.extend(items)
        counts[d] = len(items)

    if store and len(all_evidence) >= 10:
        mm = get_memory_manager(store=store)
        await mm.store_episode_memory(
            episode_id=state.get("workflow_id", ""),
            market_question=state["market_question"],
            memory_type="research_strategy",
            content={
                "search_seeds_used": {"pro": seeds.pro, "con": seeds.con, "general": seeds.general},
                "evidence_counts": counts,
                "total_evidence": len(all_evidence),
            },
            effectiveness=min(1.0, len(all_evidence) / 15),
        )

    details = {"total_items": len(all_evidence), "direction_counts": counts}
    
    # Log full researcher outputs
    for direction, output in results.items():
        log_agent_output_full(f"Researcher ({direction.upper()})", output)
    
    return {
        "researcher_output": results,
        "all_evidence": all_evidence,
        "messages": _msg(f"Autonomous Researchers: {len(all_evidence)} evidence items ({counts['pro']}P, {counts['con']}C, {counts['general']}G)"),
        "agent_traces": [_trace("AutonomousResearcherAgents", "autonomous_research_completed", details)],
    }


async def autonomous_critic_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Critique evidence for quality, duplication, and gaps.
    """
    assert "all_evidence" in state and "planner_output" in state, "state missing evidence/planner_output"

    agent = AutonomousCriticAgent(store=store, max_iterations=15, min_correlation_check_items=3, enable_auto_memory_query=store is not None)
    result = await agent.critique(
        evidence_items=state["all_evidence"],
        planner_output=state["planner_output"].model_dump(),
        market_question=state["market_question"],
    )

    stats = agent.get_stats()
    trace = {
        "duplicate_clusters": len(result.duplicate_clusters),
        "missing_topics": len(result.missing_topics),
        "over_represented_sources": len(result.over_represented_sources),
        "iterations": stats.get("average_iterations", 0),
    }
    
    # Log full critic output
    log_agent_output_full("Critic", result)

    return {
        "critic_output": result,
        "messages": _msg(f"Autonomous Critic: {trace['duplicate_clusters']} correlations, {trace['missing_topics']} gaps"),
        "agent_traces": [_trace("AutonomousCriticAgent", "autonomous_critique_completed", trace)],
    }


async def autonomous_analyst_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Aggregate evidence Bayesianly with sensitivity checks.
    """
    assert "p0_prior" in state and "all_evidence" in state and "critic_output" in state, "state missing prior/evidence/critic_output"

    agent = AutonomousAnalystAgent(store=store, max_iterations=15, max_sensitivity_range=0.3, enable_auto_memory_query=store is not None)
    result = await agent.analyze(
        prior_p=state["p0_prior"],
        evidence_items=state["all_evidence"],
        critic_output=state["critic_output"],
        market_question=state["market_question"],
    )

    # Derive threshold-specific probabilities if related markets exist
    related_markets = state.get("related_markets", [])
    threshold_probabilities = {}
    
    if related_markets:
        main_p_bayesian = result.p_bayesian
        logger.info(f"📊 Deriving threshold-specific probabilities from main p_bayesian={main_p_bayesian:.2%}")
        
        for market_info in related_markets:
            threshold = market_info.get("threshold")
            market_price = market_info.get("market_price")
            
            assert threshold is not None, "Threshold required for related market"
            assert market_price is not None, f"Market price required for threshold {threshold}%"
            
            # Derive threshold-specific probability
            # Use market price as strong prior, adjust based on main p_bayesian
            adjustment_factor = 0.3
            threshold_weight = 0.7
            main_weight = adjustment_factor
            
            threshold_adjustment = (100 - threshold) / 100.0
            adjusted_main_p = main_p_bayesian * (1 + (threshold_adjustment - 0.5) * 0.2)
            p_threshold = (threshold_weight * market_price) + (main_weight * adjusted_main_p)
            p_threshold = max(0.0, min(1.0, p_threshold))
            
            threshold_probabilities[threshold] = {
                "threshold": threshold,
                "p_bayesian": p_threshold,
                "market_price": market_price,
                "edge": p_threshold - market_price,
                "market_slug": market_info.get("slug", ""),
            }
            
            logger.info(
                f"📊 Threshold {threshold}%+: "
                f"p_bayesian={p_threshold:.2%}, "
                f"market_price={market_price:.2%}, "
                f"edge={p_threshold - market_price:.2%}"
            )
        
        # Store threshold probabilities in result
        assert hasattr(result, 'model_dump'), "Result must have model_dump method"
        result_dict = result.model_dump()
        result_dict['threshold_probabilities'] = threshold_probabilities

    stats = agent.get_stats()
    trace = {
        "p_bayesian": result.p_bayesian,
        "evidence_items": len(result.evidence_summary),
        "sensitivity_scenarios": len(result.sensitivity_analysis),
        "iterations": stats.get("average_iterations", 0),
        "threshold_markets": len(threshold_probabilities),
    }
    
    # Log full analyst output
    log_agent_output_full("Analyst", result)

    return {
        "analyst_output": result,
        "p_bayesian": result.p_bayesian,
        "threshold_probabilities": threshold_probabilities,  # Store threshold probabilities in state
        "messages": _msg(f"Autonomous Analyst: p={result.p_bayesian:.2%} (from {result.p0:.2%})"),
        "agent_traces": [_trace("AutonomousAnalystAgent", "autonomous_analysis_completed", trace)],
    }


async def autonomous_arbitrage_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Detect actionable mispricing opportunities and edge signals.
    """
    assert "p_bayesian" in state and "market_question" in state, "state missing p_bayesian/market_question"
    
    # Log p_bayesian value received from Analyst
    p_bayesian = state["p_bayesian"]
    logger.info(f"📊 Arbitrage node received p_bayesian={p_bayesian:.2%} from Analyst")
    
    # Verify consistency
    analyst_output = state.get("analyst_output")
    assert analyst_output is not None, "analyst_output required in state"
    assert hasattr(analyst_output, "p_bayesian"), "analyst_output must have p_bayesian attribute"
    analyst_p = analyst_output.p_bayesian
    assert abs(analyst_p - p_bayesian) < 0.001, f"p_bayesian mismatch: state={p_bayesian:.2%} vs analyst_output={analyst_p:.2%}"
    logger.info(f"📊 Arbitrage node verified analyst_output.p_bayesian={analyst_p:.2%}")

    # Fetch market prices if not already in state
    market_price = state.get("market_price")
    market_data = state.get("market_data")
    
    if market_price is None or market_data is None:
        market_slug = state.get("market_slug", "")
        providers = state.get("providers", ["polymarket"])
        
        assert "polymarket" in providers and market_slug, "Polymarket provider and market_slug required for price fetch"
        
        client = PolymarketClient()
        market = await client.gamma.get_market(market_slug)
        assert market is not None, f"Market not found: {market_slug}"
        
        market_price = None
        price_source = None
        
        # Strategy 1: Use orderbook first (most accurate)
        token_ids = market.get("clobTokenIds", [])
        if token_ids and len(token_ids) >= 2:
            yes_token_id = token_ids[1]
            orderbook = client.clob.get_orderbook(yes_token_id, depth=5)
            market_price = orderbook.get("mid_price", None)
            if market_price:
                price_source = "orderbook_mid_price"
                logger.info(f"📊 Using orderbook mid_price: {market_price:.2%} for {market_slug}")
        
        # Strategy 2: Use outcomePrices if orderbook unavailable
        if market_price is None:
            prices = market.get("outcomePrices", [])
            outcomes = market.get("outcomes", [])
            assert prices and len(prices) >= 2, f"Insufficient price data for {market_slug}"
            
            price_floats = [float(p) for p in prices if p is not None and p != '']
            assert len(price_floats) >= 2, f"Insufficient valid prices for {market_slug}"
            
            if len(outcomes) > 2:
                market_price = max(price_floats)
                price_source = "outcomePrices_max"
                logger.info(f"📊 Using max outcome price from multi-outcome market: {market_price:.2%} (outcomes: {outcomes})")
            else:
                market_price = price_floats[1] if len(price_floats) > 1 else price_floats[0]
                price_source = "outcomePrices_yes"
                logger.info(f"📊 Using outcomePrices[1] (YES): {market_price:.2%}")
        
        assert market_price is not None, f"Failed to extract market price for {market_slug}"
        logger.info(f"📊 Market price extracted: {market_price:.2%} (source: {price_source})")
        
        market_data = market
        state["market_price"] = market_price
        state["market_data"] = market_data

    agent = AutonomousArbitrageAgent(
        store=store,
        max_iterations=10,
        min_edge_threshold=state.get("min_edge_threshold", settings.MIN_EDGE_THRESHOLD),
        enable_auto_memory_query=store is not None,
    )
    
    # Collect all markets to analyze (main + threshold markets)
    all_markets_to_analyze = []
    
    # Add main market
    all_markets_to_analyze.append({
        "slug": state.get("market_slug", ""),
        "p_bayesian": state["p_bayesian"],
        "market_price": market_price,
        "market_data": market_data,
        "threshold": None,  # Main market has no threshold
        "is_main": True,
    })
    
    # Add threshold markets if they exist
    threshold_probabilities = state.get("threshold_probabilities", {})
    related_markets = state.get("related_markets", [])
    
    for market_info in related_markets:
        threshold = market_info.get("threshold")
        if threshold and threshold in threshold_probabilities:
            threshold_data = threshold_probabilities[threshold]
            all_markets_to_analyze.append({
                "slug": market_info.get("slug", ""),
                "p_bayesian": threshold_data.get("p_bayesian", state["p_bayesian"]),
                "market_price": threshold_data.get("market_price", 0.5),
                "market_data": market_info.get("market_data"),
                "threshold": threshold,
                "is_main": False,
            })
    
    # Detect arbitrage for each market
    all_opportunities = []
    edge_signals = []
    composite_edge_score = None
    
    for market_info in all_markets_to_analyze:
        market_slug = market_info["slug"]
        p_bayesian_market = market_info["p_bayesian"]
        market_price_market = market_info["market_price"]
        market_data_market = market_info.get("market_data")
        threshold = market_info.get("threshold")
        
        assert market_slug, "Market slug required for arbitrage analysis"
        
        logger.info(
            f"📊 Analyzing arbitrage for market: {market_slug} "
            f"(threshold={threshold}%, p_bayesian={p_bayesian_market:.2%}, market_price={market_price_market:.2%})"
        )
        
        opportunities = await agent.detect_arbitrage(
            p_bayesian=p_bayesian_market,
            market_slug=market_slug,
            market_question=state["market_question"],
            providers=state.get("providers", ["polymarket", "kalshi"]),
            bankroll=state.get("bankroll", settings.DEFAULT_BANKROLL),
            max_kelly=state.get("max_kelly", settings.MAX_KELLY_FRACTION),
            min_edge_threshold=state.get("min_edge_threshold", settings.MIN_EDGE_THRESHOLD),
            market_price=market_price_market,
            market_data=market_data_market,
        )
        
        # Add threshold info to opportunities
        for opp in opportunities:
            assert hasattr(opp, 'model_dump') or hasattr(opp, '__dict__'), "Opportunity must be serializable"
            opp_dict = opp.model_dump() if hasattr(opp, 'model_dump') else opp.__dict__
            opp_dict['threshold'] = threshold
            opp_dict['market_slug'] = market_slug
            all_opportunities.append(opp)
        
        logger.info(f"📊 Found {len(opportunities)} opportunities for {market_slug}")
    
    details = {
        "opportunity_count": len(all_opportunities),
        "markets_analyzed": len(all_markets_to_analyze),
        "threshold_markets": len([m for m in all_markets_to_analyze if m.get("threshold")]),
        "providers_checked": state.get("providers", []),
        "edge_detection_enabled": getattr(settings, "ENABLE_EDGE_DETECTION", True),
        "p_bayesian_used": p_bayesian,
        "market_price_used": market_price,
    }
    
    # Log full arbitrage output (opportunities)
    max_opportunities_to_log = getattr(settings, "ARBITRAGE_MAX_OPPORTUNITIES_TO_LOG", 5)
    if all_opportunities:
        for i, opp in enumerate(all_opportunities[:max_opportunities_to_log], 1):
            log_agent_output_full(f"Arbitrage Opportunity {i}", opp)
    else:
        logger.info("No arbitrage opportunities found")
    
    return {
        "arbitrage_output": all_opportunities,
        "edge_signals": edge_signals,
        "composite_edge_score": composite_edge_score,
        "market_price": market_price,
        "market_data": market_data,
        "messages": _msg(f"Autonomous Arbitrage: {len(all_opportunities)} opportunities across {len(all_markets_to_analyze)} markets"),
        "agent_traces": [_trace("AutonomousArbitrageAgent", "autonomous_arbitrage_completed", details)],
    }


async def autonomous_copy_trading_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Process copy trading signals and execute trades if conditions are met.
    Runs after arbitrage detection to check for copy trading opportunities.
    """
    if not getattr(settings, "ENABLE_COPY_TRADING", False):
        logger.info("Copy trading disabled - skipping copy trading node")
        return {
            "copy_trading_signals": [],
            "executed_copy_trades": [],
            "messages": _msg("Copy Trading: Disabled"),
        }
    
    assert "arbitrage_output" in state and "market_question" in state, "state missing arbitrage_output/market_question"
    
    # Initialize copy trading components
    copy_agent = CopyTradingAgent(
        min_pnl_30d=getattr(settings, "COPY_TRADER_MIN_PNL_30D", 0.0),
        min_sharpe=getattr(settings, "COPY_TRADER_MIN_SHARPE", 0.7),
        min_trades=getattr(settings, "COPY_TRADER_MIN_TRADES", 200),
        max_wallet_age_days=getattr(settings, "COPY_TRADER_MAX_WALLET_AGE_DAYS", None),
        min_win_rate=getattr(settings, "COPY_TRADER_MIN_WIN_RATE", 0.5)
    )
    
    wallet_tracker = WalletTrackerClient()
    trade_processor = TradeSignalProcessor(
        copy_agent=copy_agent,
        wallet_tracker=wallet_tracker,
        min_trade_size_usd=getattr(settings, "COPY_TRADING_MIN_TRADE_SIZE_USD", 10.0),
        min_liquidity_usd=getattr(settings, "COPY_TRADING_MIN_LIQUIDITY_USD", 1000.0),
        max_slippage_bps=getattr(settings, "COPY_TRADING_MAX_SLIPPAGE_BPS", 50)
    )
    
    trade_executor = TradeExecutor(
        max_slippage_bps=getattr(settings, "COPY_TRADING_MAX_SLIPPAGE_BPS", 50),
        max_size_per_wallet=getattr(settings, "COPY_TRADING_MAX_SIZE_PER_WALLET", 1000.0),
        cooldown_seconds=getattr(settings, "COPY_TRADING_COOLDOWN_SECONDS", 60),
        min_ev_threshold=getattr(settings, "COPY_TRADING_MIN_EV_THRESHOLD", 0.02),
        adverse_fill_limit=getattr(settings, "COPY_TRADING_ADVERSE_FILL_LIMIT", 3),
        adverse_fill_window_minutes=getattr(settings, "COPY_TRADING_ADVERSE_FILL_WINDOW_MINUTES", 10)
    )
    
    # Get p_bayesian from analyst output for EV calculation
    p_bayesian = state.get("p_bayesian")
    market_slug = state.get("market_slug", "")
    
    # Check if any copy-list wallets have trades on current market
    # In production, this would check webhook events
    # For now, we'll check if there are any active copy traders
    active_traders = copy_agent.get_active_traders()
    
    copy_trading_signals = []
    executed_trades = []
    
    if active_traders and market_slug:
        # Check for recent trades from copy-list wallets on this market
        # In production, this would check webhook events
        # Limit traders checked based on settings
        max_traders_to_check = getattr(settings, "COPY_TRADING_MAX_TRADERS_TO_CHECK", 5)
        for trader in active_traders[:max_traders_to_check]:
            try:
                # Check if trader has recent trades on this market
                trades = await wallet_tracker.get_wallet_trades(
                    trader.wallet_address,
                    lookback_hours=24,
                    market_slug=market_slug
                )
                
                if trades:
                    # Process most recent trade
                    latest_trade = trades[0]
                    
                    # Create webhook-like event structure
                    webhook_event = {
                        "type": "trade_signal",
                        "wallet_address": trader.wallet_address,
                        "transaction_hash": latest_trade.get("transaction_hash", ""),
                        "market_slug": market_slug,
                        "raw_event": latest_trade
                    }
                    
                    # Process signal
                    signal = await trade_processor.process_webhook_event(webhook_event)
                    
                    if signal:
                        copy_trading_signals.append(signal)
                        
                        # Calculate EV and Kelly
                        ev_metrics = trade_processor.calculate_ev_and_kelly(signal, p_bayesian)
                        
                        # Execute if EV meets threshold
                        if ev_metrics.get("ev_per_dollar", 0) >= trade_processor.min_ev_threshold:
                            executed = await trade_executor.execute_trade(signal, ev_metrics)
                            executed_trades.append(executed)
            except Exception as e:
                logger.warning(f"Error processing copy trade for {trader.wallet_address[:8]}...: {e}")
                continue
    
    details = {
        "signals_processed": len(copy_trading_signals),
        "trades_executed": len([t for t in executed_trades if t.status == "filled"]),
        "trades_rejected": len([t for t in executed_trades if t.status == "rejected"]),
        "active_traders": len(active_traders)
    }
    
    logger.info(
        f"Copy Trading: {details['signals_processed']} signals, "
        f"{details['trades_executed']} executed, "
        f"{details['trades_rejected']} rejected"
    )
    
    return {
        "copy_trading_signals": [s.model_dump() if hasattr(s, 'model_dump') else s for s in copy_trading_signals],
        "executed_copy_trades": [t.model_dump() if hasattr(t, 'model_dump') else t for t in executed_trades],
        "messages": _msg(f"Copy Trading: {details['signals_processed']} signals, {details['trades_executed']} executed"),
        "agent_traces": [_trace("CopyTradingAgent", "copy_trading_completed", details)],
    }


async def autonomous_reporter_node(state: Dict[str, Any], store: Optional[BaseStore] = None) -> Dict[str, Any]:
    """
    Generate a final report from all upstream outputs.
    """
    assert {"market_question", "planner_output", "researcher_output", "critic_output", "analyst_output", "arbitrage_output", "copy_trading_signals"} <= state.keys(), "state missing required keys for report"
    
    # Verify analyst_output consistency
    analyst_output = state.get("analyst_output")
    assert analyst_output is not None, "analyst_output required in state"
    assert hasattr(analyst_output, "p_bayesian"), "analyst_output must have p_bayesian attribute"
    p_bayesian = analyst_output.p_bayesian
    state_p_bayesian = state.get("p_bayesian")
    assert state_p_bayesian is not None, "p_bayesian required in state"
    assert abs(state_p_bayesian - p_bayesian) < 0.001, f"p_bayesian mismatch: state={state_p_bayesian:.2%} vs analyst_output={p_bayesian:.2%}"
    logger.info(f"📊 Reporter node verified p_bayesian={p_bayesian:.2%}")

    agent = AutonomousReporterAgent(store=store, max_iterations=10, enable_auto_memory_query=store is not None)
    
    # Prepare analyst_output with threshold_probabilities if available
    analyst_output_dict = state["analyst_output"].model_dump()
    threshold_probabilities = state.get("threshold_probabilities", {})
    if threshold_probabilities:
        analyst_output_dict['threshold_probabilities'] = threshold_probabilities
    
    result = await agent.generate_report(
        market_question=state["market_question"],
        planner_output=state["planner_output"].model_dump(),
        researcher_output={k: v.model_dump() for k, v in state["researcher_output"].items()},
        critic_output=state["critic_output"].model_dump(),
        analyst_output=analyst_output_dict,  # Include threshold_probabilities
        arbitrage_opportunities=[opp.model_dump() if hasattr(opp, 'model_dump') else opp for opp in state["arbitrage_output"]],
        timestamp=state.get("timestamp", datetime.now().isoformat()),
        workflow_id=state.get("workflow_id", ""),
    )
    
    # Log full reporter output
    log_agent_output_full("Reporter", result)

    return {
        "reporter_output": result,
        "messages": _msg("Autonomous Reporter: Final report complete"),
        "agent_traces": [_trace("AutonomousReporterAgent", "autonomous_report_completed", {"summary_length": len(result.executive_summary or '')})],
    }


# -----------------------------
# AUTONOMOUS WORKFLOW CONSTRUCTION
# -----------------------------
def create_autonomous_polyseer_workflow(store: Optional[BaseStore] = None) -> StateGraph:
    """
    Create and compile the POLYSEER workflow with autonomous agents.
    START → planner → researcher → critic → analyst → arbitrage → copy_trading → reporter → END
    """
    logger.info("🚀 Creating AUTONOMOUS POLYSEER workflow")
    
    # Auto-create store if not provided and memory persistence is enabled
    if store is None:
        from config.settings import settings as global_settings
        from config.system_constants import ENABLE_MEMORY_PERSISTENCE
        if getattr(global_settings, "ENABLE_MEMORY_PERSISTENCE", ENABLE_MEMORY_PERSISTENCE):
            try:
                store = create_store_from_config()
                if store is not None:
                    logger.info(f"Auto-created memory store for workflow: {type(store).__name__}")
                else:
                    logger.warning("Memory persistence enabled but store creation failed - workflow will run without memory")
            except Exception as e:
                logger.warning(f"Failed to auto-create memory store for workflow: {e} - workflow will run without memory")
                store = None
        else:
            logger.debug("Memory persistence disabled - workflow will run without memory")
    
    workflow_logger.log_agent_start(
        task_description="POLYSEER Autonomous Workflow",
        input_info={"store_configured": store is not None},
    )

    workflow = StateGraph(WorkflowState)
    
    # Total steps: planner, researcher, critic, analyst, arbitrage, copy_trading, reporter
    total_steps = 7

    async def planner_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        log_workflow_transition("START", "Planner", f"1/{total_steps}", {"market_question": s.get('market_question', 'Unknown')[:80]})
        log_workflow_progress(1, total_steps, "Planner", "running")
        log_agent_separator("Planner Agent")
        result = await autonomous_planner_node(s, store)
        log_workflow_progress(1, total_steps, "Planner", "completed")
        return result

    async def researcher_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        log_workflow_transition("Planner", "Researcher", f"2/{total_steps}", {"subclaims": len(s.get('planner_output', {}).get('subclaims', [])) if isinstance(s.get('planner_output'), dict) else 0})
        log_workflow_progress(2, total_steps, "Researcher", "running")
        log_agent_separator("Researcher Agents (PRO/CON/GENERAL)")
        result = await autonomous_researcher_node(s, store)
        log_workflow_progress(2, total_steps, "Researcher", "completed")
        return result

    async def critic_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        log_workflow_transition("Researcher", "Critic", f"3/{total_steps}")
        log_workflow_progress(3, total_steps, "Critic", "running")
        log_agent_separator("Critic Agent")
        result = await autonomous_critic_node(s, store)
        log_workflow_progress(3, total_steps, "Critic", "completed")
        return result

    async def analyst_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        log_workflow_transition("Critic", "Analyst", f"4/{total_steps}")
        log_workflow_progress(4, total_steps, "Analyst", "running")
        log_agent_separator("Analyst Agent")
        result = await autonomous_analyst_node(s, store)
        log_workflow_progress(4, total_steps, "Analyst", "completed")
        return result

    async def arbitrage_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        p_bayesian = s.get('analyst_output', {}).get('p_bayesian', 0.5) if isinstance(s.get('analyst_output'), dict) else 0.5
        log_workflow_transition("Analyst", "Arbitrage", f"5/{total_steps}", {"p_bayesian": f"{p_bayesian:.2%}"})
        log_workflow_progress(5, total_steps, "Arbitrage", "running")
        log_agent_separator("Arbitrage Agent")
        result = await autonomous_arbitrage_node(s, store)
        log_workflow_progress(5, total_steps, "Arbitrage", "completed")
        return result

    async def copy_trading_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        log_workflow_transition("Arbitrage", "Copy Trading", f"6/{total_steps}")
        log_workflow_progress(6, total_steps, "Copy Trading", "running")
        log_agent_separator("Copy Trading Agent")
        result = await autonomous_copy_trading_node(s, store)
        log_workflow_progress(6, total_steps, "Copy Trading", "completed")
        return result

    async def reporter_wrapper(s: Dict[str, Any]) -> Dict[str, Any]:
        log_workflow_transition("Copy Trading", "Reporter", f"7/{total_steps}")
        log_workflow_progress(7, total_steps, "Reporter", "running")
        log_agent_separator("Reporter Agent")
        result = await autonomous_reporter_node(s, store)
        log_workflow_progress(7, total_steps, "Reporter", "completed")
        return result

    workflow.add_node("planner", planner_wrapper)
    workflow.add_node("researcher", researcher_wrapper)
    workflow.add_node("critic", critic_wrapper)
    workflow.add_node("analyst", analyst_wrapper)
    workflow.add_node("arbitrage", arbitrage_wrapper)
    workflow.add_node("copy_trading", copy_trading_wrapper)
    workflow.add_node("reporter", reporter_wrapper)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    workflow.add_edge("critic", "analyst")
    workflow.add_edge("analyst", "arbitrage")
    workflow.add_edge("arbitrage", "copy_trading")
    workflow.add_edge("copy_trading", "reporter")
    workflow.add_edge("reporter", END)

    app = workflow.compile(checkpointer=MemorySaver(), debug=False)
    logger.info("✅ Autonomous workflow compiled successfully")
    return app


# -----------------------------
# CONVENIENCE RUNNER
# -----------------------------
async def run_autonomous_workflow(
    market_question: str,
    market_url: str = "",
    market_slug: str = "",
    providers: Optional[List[str]] = None,
    bankroll: Optional[float] = None,
    max_kelly: Optional[float] = None,
    min_edge_threshold: Optional[float] = None,
    store: Optional[BaseStore] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Execute the complete autonomous workflow and return the final state.
    """
    assert isinstance(market_question, str) and market_question.strip(), "market_question is required"

    # Auto-create store if not provided and memory persistence is enabled
    if store is None:
        from config.settings import settings as global_settings
        from config.system_constants import ENABLE_MEMORY_PERSISTENCE
        if getattr(global_settings, "ENABLE_MEMORY_PERSISTENCE", ENABLE_MEMORY_PERSISTENCE):
            try:
                store = create_store_from_config()
                if store is not None:
                    logger.info(f"Auto-created memory store for workflow execution: {type(store).__name__}")
            except Exception as e:
                logger.warning(f"Failed to auto-create memory store: {e} - workflow will run without memory")
                store = None

    workflow_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    logger.info(f"🚀 Starting AUTONOMOUS POLYSEER workflow {workflow_id}")
    if store:
        logger.info("✅ Memory store configured - agents will query memory")
    else:
        logger.info("⚠️  No memory store configured - agents will run without memory")

    app = create_autonomous_polyseer_workflow(store=store)
    initial_state = {
        "workflow_id": workflow_id,
        "timestamp": timestamp,
        "market_question": market_question,
        "market_url": market_url,
        "market_slug": market_slug or market_question.lower().replace(" ", "-")[:50],
        "providers": providers or ["polymarket", "kalshi"],
        "bankroll": bankroll or settings.DEFAULT_BANKROLL,
        "max_kelly": max_kelly or settings.MAX_KELLY_FRACTION,
        "min_edge_threshold": min_edge_threshold or settings.MIN_EDGE_THRESHOLD,
        "context": kwargs,
    }

    config = {"configurable": {"thread_id": workflow_id}, "recursion_limit": 150}
    final_state = await app.ainvoke(initial_state, config)
    logger.info(f"✅ Autonomous workflow {workflow_id} completed successfully")
    
    # Calculate execution stats for summary
    execution_stats = {
        "workflow_id": workflow_id,
        "timestamp": timestamp,
    }
    
    # Extract stats from final state
    analyst_output = final_state.get('analyst_output')
    assert analyst_output is not None, "analyst_output required in final state"
    assert hasattr(analyst_output, 'p_bayesian'), "analyst_output must have p_bayesian"
    execution_stats['p_bayesian'] = analyst_output.p_bayesian
    
    # Count evidence items
    researcher_output = final_state.get('researcher_output')
    assert researcher_output is not None, "researcher_output required in final state"
    assert isinstance(researcher_output, dict), "researcher_output must be dict"
    total_evidence = sum(
        len(r.get('evidence_items', [])) if isinstance(r, dict) else 0
        for r in researcher_output.values()
    )
    execution_stats['evidence_items'] = total_evidence
    
    # Count opportunities
    arbitrage_output = final_state.get('arbitrage_output')
    assert arbitrage_output is not None, "arbitrage_output required in final state"
    assert isinstance(arbitrage_output, list), "arbitrage_output must be list"
    execution_stats['opportunities'] = len(arbitrage_output)
    
    # Count tool calls and iterations from agent traces
    total_tool_calls = 0
    total_iterations = 0
    agent_times = {}
    tool_call_counts = {}
    
    for key in ['planner_output', 'researcher_output', 'critic_output', 'analyst_output', 'arbitrage_output', 'reporter_output']:
        output = final_state.get(key)
        assert output is not None, f"{key} required in final state"
        assert isinstance(output, dict), f"{key} must be dict"
        if 'tool_calls' in output:
            assert isinstance(output['tool_calls'], list), f"{key}.tool_calls must be list"
            count = len(output['tool_calls'])
            agent_name = key.replace('_output', '').title()
            tool_call_counts[agent_name] = count
            total_tool_calls += count
    
    execution_stats['total_tool_calls'] = total_tool_calls
    execution_stats['total_iterations'] = total_iterations
    
    # Log workflow summary
    log_workflow_summary(execution_stats, agent_times, tool_call_counts)
    
    return final_state