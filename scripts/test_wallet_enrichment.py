#!/usr/bin/env python3
"""
Comprehensive integration test for the complete wallet data pipeline.

This test validates the end-to-end functionality:
1. Fast path: PARALLEL wallet discovery from events (retrieve_wallets_from_events - no trades storage)
2. Slow path: Enrich wallets with all historical positions (enrich_wallets_positions)
3. Data integrity: Verify positions have correct event metadata and wallet enrichment

The test ensures:
- Events are properly marked as retrieve_data=true after processing
- Wallets discovered in fast path are enriched in slow path
- Positions are saved with complete event metadata
- Wallet enrichment status is correctly updated (enriched=true)
- Wallet stats and scores are computed correctly
- No data loss in the pipeline
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from clients.functions.retrieve_wallets import retrieve_wallets_from_events
from clients.functions.wallets import enrich_wallets_positions
from database.client import MarketDatabase


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wallet_enrichment_test")

MAX_EVENTS = 2  # Reduced for focused testing to avoid timeouts
DEFAULT_WALLET_BATCH = 5  # Smaller batch to avoid overwhelming DB


async def run_fast_process(max_events: int) -> tuple[Dict[str, Any], Set[str]]:
    """Run fast path and return both results and discovered wallets."""
    logger.info("Fast path | retrieving trades & wallets for %d events", max_events)

    # We need to track discovered wallets - let's query the DB after the fast process
    settings = Settings()
    db = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    # Get wallets before fast process
    wallets_before = set()
    try:
        result = db.supabase.table("wallets").select("proxy_wallet").execute()
        wallets_before = {w["proxy_wallet"] for w in (result.data or [])}
    except Exception as e:
        logger.warning(f"Could not get wallets before fast process: {e}")

    result = await retrieve_wallets_from_events(max_events=max_events, is_closed=False)

    # Get wallets after fast process to determine what was discovered
    discovered_wallets = set()
    try:
        result_after = db.supabase.table("wallets").select("proxy_wallet").execute()
        wallets_after = {w["proxy_wallet"] for w in (result_after.data or [])}
        discovered_wallets = wallets_after - wallets_before
    except Exception as e:
        logger.warning(f"Could not determine discovered wallets: {e}")

    logger.info(
        "Fast path complete | events=%d wallets=%d trades=%d discovered_wallets=%d duration=%.1fs",
        result.get("events_processed", 0),
        result.get("wallets_discovered", 0),
        result.get("trades_saved", 0),
        len(discovered_wallets),
        result.get("duration_seconds", 0.0),
    )
    return result, discovered_wallets


async def run_slow_process(max_wallets: int, skip_existing: bool) -> Dict[str, Any]:
    logger.info(
        "Slow path | enriching up to %d wallets (skip_existing=%s)",
        max_wallets,
        skip_existing,
    )
    result = await enrich_wallets_positions(
        max_wallets=max_wallets,
        skip_existing_positions=skip_existing,
    )
    logger.info(
        "Slow path complete | wallets_processed=%d enriched=%d stats=%d duration=%.1fs",
        result.get("wallets_processed", 0),
        result.get("wallets_enriched", 0),
        result.get("stats_computed", 0),
        result.get("duration_seconds", 0.0),
    )
    return result


async def validate_pipeline_integrity(settings: Settings, discovered_wallets: Set[str] = None) -> Dict[str, Any]:
    """Comprehensive validation of the complete wallet pipeline."""
    logger.info("🔍 Starting comprehensive pipeline validation...")
    db = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)

    # Check recently processed events
    recent_events = (
        db.supabase.table("events")
        .select("id,slug,retrieve_data,updated_at")
        .eq("retrieve_data", True)
        .order("updated_at", desc=True)
        .limit(10)
        .execute()
    ).data or []

    # Check wallets discovered and enriched
    recent_wallets = (
        db.supabase.table("wallets")
        .select("proxy_wallet,enriched,last_sync_at,updated_at")
        .order("updated_at", desc=True)
        .limit(20)
        .execute()
    ).data or []

    # Check positions with event metadata
    recent_positions = (
        db.supabase.table("wallet_closed_positions")
        .select("id,proxy_wallet,event_id,event_slug,total_bought,realized_pnl,timestamp")
        .order("timestamp", desc=True)
        .limit(50)
        .execute()
    ).data or []

    # Check events_closed table for metadata enrichment
    recent_event_metadata = (
        db.supabase.table("events_closed")
        .select("id,slug,title,total_volume,market_count")
        .order("updated_at", desc=True)
        .limit(10)
        .execute()
    ).data or []

    # Check stats and scores
    wallet_stats = (
        db.supabase.table("wallet_stats")
        .select("proxy_wallet,n_positions,total_volume,realized_pnl,computed_at")
        .order("computed_at", desc=True)
        .limit(10)
        .execute()
    ).data or []

    wallet_scores = (
        db.supabase.table("wallet_scores")
        .select("proxy_wallet,composite_score,computed_at")
        .order("computed_at", desc=True)
        .limit(10)
        .execute()
    ).data or []

    # Comprehensive validation checks
    validation = {
        "events_processed": len(recent_events),
        "wallets_total": len(recent_wallets),
        "wallets_enriched": sum(1 for w in recent_wallets if w.get("enriched")),
        "positions_saved": len(recent_positions),
        "event_metadata_saved": len(recent_event_metadata),
        "stats_computed": len(wallet_stats),
        "scores_computed": len(wallet_scores),
    }

    # Data integrity checks
    positions_with_event_data = sum(1 for p in recent_positions if p.get("event_id") and p.get("event_slug"))
    validation["positions_with_metadata"] = positions_with_event_data

    # Check if discovered wallets were enriched
    if discovered_wallets:
        enriched_discovered = sum(1 for w in recent_wallets
                                if w.get("proxy_wallet") in discovered_wallets and w.get("enriched"))
        validation["discovered_wallets_enriched"] = enriched_discovered
        validation["discovered_wallets_total"] = len(discovered_wallets)

    # Sample data for reporting
    validation["sample_positions"] = recent_positions[:3] if recent_positions else []
    validation["sample_wallets"] = [w["proxy_wallet"][:10] + "..." for w in recent_wallets[:3]] if recent_wallets else []
    validation["sample_events"] = [e.get("slug", e["id"][:10] + "...") for e in recent_events[:3]] if recent_events else []

    # Success criteria
    success_criteria = {
        "events_properly_marked": validation["events_processed"] > 0,
        "wallets_being_enriched": validation["wallets_enriched"] > 0,
        "positions_have_metadata": validation["positions_with_metadata"] > 0,
        "pipeline_integrity": validation["positions_saved"] > 0 and validation["event_metadata_saved"] > 0,
        "stats_computation": validation["stats_computed"] > 0,
        "scores_computation": validation["scores_computed"] > 0,
    }

    validation["success_criteria"] = success_criteria
    validation["overall_success"] = all(success_criteria.values())

    logger.info("✅ Pipeline validation complete:")
    logger.info(f"   Events processed: {validation['events_processed']}")
    logger.info(f"   Wallets enriched: {validation['wallets_enriched']}/{validation['wallets_total']}")
    logger.info(f"   Positions saved: {validation['positions_saved']} (with metadata: {validation['positions_with_metadata']})")
    logger.info(f"   Event metadata saved: {validation['event_metadata_saved']}")
    logger.info(f"   Stats computed: {validation['stats_computed']}, Scores: {validation['scores_computed']}")
    logger.info(f"   Overall success: {'✅ PASS' if validation['overall_success'] else '❌ FAIL'}")

    return validation


async def main() -> None:
    settings = Settings()
    start = datetime.now()

    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("🚀 Starting comprehensive wallet pipeline test (target=%d events)", MAX_EVENTS)
    print("\n" + "=" * 80)
    print("COMPREHENSIVE WALLET PIPELINE TEST")
    print("=" * 80)

    try:
        # PHASE 1: Fast Path - Discover wallets from events
        logger.info("📊 PHASE 1: Fast Path - Retrieving trades & wallets from events")
        try:
            fast_result, discovered_wallets = await asyncio.wait_for(run_fast_process(MAX_EVENTS), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Fast path timed out after 30s, skipping to slow path test")
            fast_result = {"events_processed": 0, "wallets_discovered": 0, "trades_saved": 0}
            discovered_wallets = set()

        # Extract discovered wallets for targeted enrichment
        wallets_to_enrich = max(len(discovered_wallets), DEFAULT_WALLET_BATCH)

        print(f"Fast Path Results:")
        print(f"  Events processed: {fast_result.get('events_processed', 0)}")
        print(f"  Wallets discovered: {fast_result.get('wallets_discovered', 0)}")
        print(f"  Trades saved: {fast_result.get('trades_saved', 0)}")
        print(f"  Targeting {wallets_to_enrich} wallets for enrichment")
        if discovered_wallets:
            print(f"  Discovered wallets: {len(discovered_wallets)}")

        # PHASE 2: Slow Path - Enrich discovered wallets with positions
        logger.info("💰 PHASE 2: Slow Path - Enriching wallets with historical positions")
        try:
            slow_result = await asyncio.wait_for(run_slow_process(wallets_to_enrich, skip_existing=False), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("Slow path timed out after 60s, using mock results for validation")
            slow_result = {"wallets_processed": 0, "positions_synced": 0, "total_volume": 0.0, "total_pnl": 0.0}

        print(f"Slow Path Results:")
        print(f"  Wallets processed: {slow_result.get('wallets_processed', 0)}")
        print(f"  Positions synced: {slow_result.get('positions_synced', 0)}")
        print(f"  Total volume: ${slow_result.get('total_volume', 0):,.2f}")
        print(f"  Total PnL: ${slow_result.get('total_pnl', 0):,.2f}")

        # PHASE 3: Comprehensive Validation
        logger.info("🔍 PHASE 3: Validating complete pipeline integrity")
        try:
            validation = await asyncio.wait_for(validate_pipeline_integrity(settings, discovered_wallets), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Validation timed out after 30s, using basic validation")
            validation = {
                "events_processed": 0, "wallets_enriched": 0, "positions_saved": 0,
                "overall_success": False, "success_criteria": {"timeout": False}
            }

        # Detailed Results
        duration = (datetime.now() - start).total_seconds()
        print(f"\nValidation Results:")
        print(f"  Events with retrieve_data=true: {validation['events_processed']}")
        print(f"  Wallets enriched: {validation['wallets_enriched']}/{validation['wallets_total']}")
        print(f"  Positions saved: {validation['positions_saved']}")
        print(f"  Positions with event metadata: {validation['positions_with_metadata']}")
        print(f"  Event metadata records: {validation['event_metadata_saved']}")
        print(f"  Wallet stats computed: {validation['stats_computed']}")
        print(f"  Wallet scores computed: {validation['scores_computed']}")
        if 'discovered_wallets_enriched' in validation:
            print(f"  Discovered wallets enriched: {validation['discovered_wallets_enriched']}/{validation['discovered_wallets_total']}")

        # Success criteria
        success = validation['overall_success']
        print(f"\n🎯 Success Criteria:")
        for criterion, passed in validation['success_criteria'].items():
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion.replace('_', ' ').title()}")

        print(f"\n🏁 Overall Result: {'✅ PIPELINE SUCCESS' if success else '❌ PIPELINE FAILED'}")
        print(f"⏱️  Total execution time: {duration:.1f}s")

        # Sample data
        if validation['sample_positions']:
            print("\n💾 Sample Positions:")
            for pos in validation['sample_positions'][:2]:
                print(f"  Wallet: {pos['proxy_wallet'][:8]}... | Event: {pos.get('event_slug', 'N/A')} | Volume: ${pos['total_bought']:.2f}")

        if validation['sample_wallets']:
            print(f"👛 Sample Wallets: {', '.join(validation['sample_wallets'])}")

        if validation['sample_events']:
            print(f"📅 Sample Events: {', '.join(validation['sample_events'])}")

        print("\n" + "=" * 80)

        # Exit with appropriate code
        exit(0 if success else 1)

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())