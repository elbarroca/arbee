"""
Elite End-to-End Pipeline Test

Complete flow validation:
1. Data Sync (events + markets)
2. Wallet Discovery (from closed events)
3. Performance Analytics (stats + scoring)
4. Trader Analysis (best traders + copy suggestions)

This test validates the entire POLYSEER copy-trading intelligence pipeline.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from config.settings import Settings
from database.client import MarketDatabase

# Import all pipeline modules
from clients.functions.data import PolymarketDataCollector, sync_polymarket_data
from clients.functions.wallets import PolymarketWalletCollector, execute_full_wallet_sync
from clients.functions.traders import PolymarketTraderAnalyzer, execute_full_trader_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PipelineTestOrchestrator:
    """
    Elite pipeline test orchestrator.

    Coordinates end-to-end validation of:
    1. Market data intelligence
    2. Wallet discovery operations
    3. Performance analytics computation
    4. Copy-trade signal generation
    """

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.test_results = {}
        self.start_time = None

    async def execute_full_pipeline_test(
        self,
        event_limit: int = 10,
        wallet_limit: int = 50,
        trader_limit: int = 20,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete pipeline test with database validation.

        Args:
            event_limit: Max events to sync
            wallet_limit: Max wallets to process
            trader_limit: Max traders to analyze
            verbose: Print detailed progress

        Returns:
            Complete test results report
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info("🚀 ELITE PIPELINE TEST INITIATED")
        logger.info("=" * 60)

        # Pre-test database snapshot
        pre_snapshot = await self._get_database_snapshot()
        self._print_snapshot("PRE-TEST DATABASE STATE", pre_snapshot)

        # =====================================================================
        # PHASE 1: DATA SYNC (Events + Markets)
        # =====================================================================
        logger.info("\n📍 PHASE 1: DATA SYNC (Events + Markets)")
        logger.info("-" * 60)

        phase1_result = await self._test_data_sync(event_limit, verbose)
        self.test_results["phase1_data_sync"] = phase1_result

        if verbose:
            self._print_phase_result("DATA SYNC", phase1_result)

        # Post-Phase 1 snapshot
        post_phase1 = await self._get_database_snapshot()
        self._print_delta("PHASE 1 CHANGES", pre_snapshot, post_phase1)

        # =====================================================================
        # PHASE 2: WALLET DISCOVERY (From Closed Events)
        # =====================================================================
        logger.info("\n📍 PHASE 2: WALLET DISCOVERY")
        logger.info("-" * 60)

        phase2_result = await self._test_wallet_discovery(event_limit, verbose)
        self.test_results["phase2_wallet_discovery"] = phase2_result

        if verbose:
            self._print_phase_result("WALLET DISCOVERY", phase2_result)

        # Post-Phase 2 snapshot
        post_phase2 = await self._get_database_snapshot()
        self._print_delta("PHASE 2 CHANGES", post_phase1, post_phase2)

        # =====================================================================
        # PHASE 3: PERFORMANCE ANALYTICS
        # =====================================================================
        logger.info("\n📍 PHASE 3: PERFORMANCE ANALYTICS")
        logger.info("-" * 60)

        phase3_result = await self._test_performance_analytics(wallet_limit, verbose)
        self.test_results["phase3_performance_analytics"] = phase3_result

        if verbose:
            self._print_phase_result("PERFORMANCE ANALYTICS", phase3_result)

        # Post-Phase 3 snapshot
        post_phase3 = await self._get_database_snapshot()
        self._print_delta("PHASE 3 CHANGES", post_phase2, post_phase3)

        # =====================================================================
        # PHASE 4: WALLET SCORING
        # =====================================================================
        logger.info("\n📍 PHASE 4: WALLET SCORING & TIER ASSIGNMENT")
        logger.info("-" * 60)

        phase4_result = await self._test_wallet_scoring(verbose)
        self.test_results["phase4_wallet_scoring"] = phase4_result

        if verbose:
            self._print_phase_result("WALLET SCORING", phase4_result)

        # Post-Phase 4 snapshot
        post_phase4 = await self._get_database_snapshot()
        self._print_delta("PHASE 4 CHANGES", post_phase3, post_phase4)

        # =====================================================================
        # PHASE 5: TRADER ANALYSIS & COPY SUGGESTIONS
        # =====================================================================
        logger.info("\n📍 PHASE 5: TRADER ANALYSIS & COPY SUGGESTIONS")
        logger.info("-" * 60)

        phase5_result = await self._test_trader_analysis(trader_limit, verbose)
        self.test_results["phase5_trader_analysis"] = phase5_result

        if verbose:
            self._print_phase_result("TRADER ANALYSIS", phase5_result)

        # Post-Phase 5 snapshot
        post_phase5 = await self._get_database_snapshot()
        self._print_delta("PHASE 5 CHANGES", post_phase4, post_phase5)

        # =====================================================================
        # FINAL REPORT
        # =====================================================================
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PIPELINE TEST COMPLETE")
        logger.info("=" * 60)

        final_report = self._generate_final_report(pre_snapshot, post_phase5)
        self._print_final_report(final_report)

        return final_report

    async def _test_data_sync(self, limit: int, verbose: bool) -> Dict[str, Any]:
        """Test Phase 1: Data synchronization."""
        logger.info(f"🔄 Syncing Polymarket data (limit={limit})")

        try:
            collector = PolymarketDataCollector()
            result = await collector.collect_new_events_and_markets(limit=limit)

            return {
                "success": True,
                "events_processed": result.get("events_processed", 0),
                "events_saved": result.get("events_saved", 0),
                "markets_saved": result.get("markets_saved", 0),
                "markets_moved_to_closed": result.get("markets_moved_to_closed", 0),
                "events_moved_to_closed": result.get("events_moved_to_closed", 0),
                "outcome_stats": result.get("outcome_stats", {}),
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ Data sync failed: {e}")
            return {"success": False, "error": str(e)}

    async def _test_wallet_discovery(self, limit: int, verbose: bool) -> Dict[str, Any]:
        """Test Phase 2: Wallet discovery from closed events."""
        logger.info(f"🔍 Discovering wallets from closed events (limit={limit})")

        try:
            collector = PolymarketWalletCollector()
            result = await collector.sync_wallets_from_closed_events(limit=limit, save_trades=True)

            return {
                "success": True,
                "events_processed": result.get("events_processed", 0),
                "trades_fetched": result.get("trades_fetched", 0),
                "wallets_discovered": result.get("wallets_discovered", 0),
                "events_synced": result.get("events_synced", 0),
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ Wallet discovery failed: {e}")
            return {"success": False, "error": str(e)}

    async def _test_performance_analytics(self, wallet_limit: int, verbose: bool) -> Dict[str, Any]:
        """Test Phase 3: Performance analytics computation."""
        logger.info(f"📊 Computing performance analytics (max_wallets={wallet_limit})")

        try:
            collector = PolymarketWalletCollector()
            result = await collector.sync_wallet_performance_data(max_wallets=wallet_limit)

            return {
                "success": True,
                "wallets_processed": result.get("wallets_processed", 0),
                "positions_synced": result.get("positions_synced", 0),
                "total_volume": result.get("total_volume", 0),
                "total_pnl": result.get("total_pnl", 0),
                "stats_computed": result.get("stats_computed", 0),
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ Performance analytics failed: {e}")
            return {"success": False, "error": str(e)}

    async def _test_wallet_scoring(self, verbose: bool) -> Dict[str, Any]:
        """Test Phase 4: Wallet scoring and tier assignment."""
        logger.info("🏆 Computing wallet scores and tier assignments")

        try:
            collector = PolymarketWalletCollector()
            result = await collector.compute_and_save_wallet_scores()

            return {
                "success": True,
                "scores_computed": result.get("scores_computed", 0),
                "tier_a_count": result.get("tier_a_count", 0),
                "tier_b_count": result.get("tier_b_count", 0),
                "tier_c_count": result.get("tier_c_count", 0),
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ Wallet scoring failed: {e}")
            return {"success": False, "error": str(e)}

    async def _test_trader_analysis(self, trader_limit: int, verbose: bool) -> Dict[str, Any]:
        """Test Phase 5: Trader analysis and copy suggestions."""
        logger.info(f"🎯 Analyzing best traders (limit={trader_limit})")

        try:
            analyzer = PolymarketTraderAnalyzer()

            # Filter best traders
            best_traders = await analyzer.filter_best_traders(
                min_roi=0.05,
                min_win_rate=0.60,
                limit=trader_limit
            )

            # Get open positions
            positions_report = await analyzer.get_best_traders_open_positions(
                min_roi=0.05,
                min_win_rate=0.60,
                max_traders=min(trader_limit, 20)
            )

            # Get copy trade suggestions
            suggestions = await analyzer.get_copy_trade_suggestions(
                min_roi=0.05,
                min_win_rate=0.60,
                min_elite_traders=1,  # Lower threshold for testing
                top_n=10
            )

            return {
                "success": True,
                "best_traders_found": len(best_traders),
                "open_positions_count": positions_report.get("total_positions", 0),
                "traders_with_positions": positions_report.get("traders_with_positions", 0),
                "total_position_value": positions_report.get("total_position_value", 0),
                "copy_suggestions_count": len(suggestions),
                "top_suggestion": suggestions[0] if suggestions else None,
                "error": None
            }
        except Exception as e:
            logger.error(f"❌ Trader analysis failed: {e}")
            return {"success": False, "error": str(e)}

    async def _get_database_snapshot(self) -> Dict[str, int]:
        """Get current database table counts."""
        tables = [
            "events", "events_closed", "markets", "markets_closed",
            "wallet", "trades", "wallet_positions", "wallet_stats",
            "wallet_tag_stats", "wallet_market_stats", "wallet_scores",
            "wallet_open_positions", "trader_tag_credibility"
        ]

        snapshot = {}
        for table in tables:
            try:
                result = self.db.supabase.table(table).select("*", count="exact").limit(0).execute()
                snapshot[table] = result.count or 0
            except Exception:
                snapshot[table] = -1  # Table might not exist

        return snapshot

    def _print_snapshot(self, title: str, snapshot: Dict[str, int]) -> None:
        """Print database snapshot."""
        logger.info(f"\n{title}:")
        for table, count in snapshot.items():
            if count >= 0:
                logger.info(f"   {table}: {count:,} rows")

    def _print_delta(self, title: str, before: Dict[str, int], after: Dict[str, int]) -> None:
        """Print changes between snapshots."""
        logger.info(f"\n{title}:")
        changes = []
        for table in before.keys():
            delta = after.get(table, 0) - before.get(table, 0)
            if delta != 0:
                changes.append(f"   {table}: +{delta:,}" if delta > 0 else f"   {table}: {delta:,}")

        if changes:
            for change in changes:
                logger.info(change)
        else:
            logger.info("   No changes detected")

    def _print_phase_result(self, phase_name: str, result: Dict[str, Any]) -> None:
        """Print phase result details."""
        if result.get("success"):
            logger.info(f"✅ {phase_name} SUCCESS")
            for key, value in result.items():
                if key not in ["success", "error", "top_suggestion"]:
                    if isinstance(value, float):
                        logger.info(f"   {key}: ${value:,.2f}")
                    elif isinstance(value, dict):
                        logger.info(f"   {key}: {value}")
                    else:
                        logger.info(f"   {key}: {value}")
        else:
            logger.error(f"❌ {phase_name} FAILED: {result.get('error')}")

    def _generate_final_report(self, pre_snapshot: Dict[str, int], post_snapshot: Dict[str, int]) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()

        # Calculate total changes
        total_rows_added = sum(
            max(0, post_snapshot.get(table, 0) - pre_snapshot.get(table, 0))
            for table in pre_snapshot.keys()
        )

        # Check phase successes
        all_success = all(
            phase.get("success", False) for phase in self.test_results.values()
        )

        return {
            "test_status": "PASSED" if all_success else "FAILED",
            "total_duration_seconds": round(duration, 2),
            "pre_test_snapshot": pre_snapshot,
            "post_test_snapshot": post_snapshot,
            "total_rows_added": total_rows_added,
            "phase_results": self.test_results,
            "summary": {
                "events_processed": self.test_results.get("phase1_data_sync", {}).get("events_processed", 0),
                "wallets_discovered": self.test_results.get("phase2_wallet_discovery", {}).get("wallets_discovered", 0),
                "positions_synced": self.test_results.get("phase3_performance_analytics", {}).get("positions_synced", 0),
                "tier_a_wallets": self.test_results.get("phase4_wallet_scoring", {}).get("tier_a_count", 0),
                "copy_suggestions": self.test_results.get("phase5_trader_analysis", {}).get("copy_suggestions_count", 0),
            },
            "completed_at": end_time.isoformat()
        }

    def _print_final_report(self, report: Dict[str, Any]) -> None:
        """Print formatted final report."""
        status = report["test_status"]
        duration = report["total_duration_seconds"]
        summary = report["summary"]

        if status == "PASSED":
            logger.info("🎉 TEST STATUS: PASSED")
        else:
            logger.error("❌ TEST STATUS: FAILED")

        logger.info(f"⏱️  Total Duration: {duration:.2f} seconds")
        logger.info(f"📊 Total Rows Added: {report['total_rows_added']:,}")

        logger.info("\n📈 PIPELINE SUMMARY:")
        logger.info(f"   Events processed: {summary['events_processed']}")
        logger.info(f"   Wallets discovered: {summary['wallets_discovered']}")
        logger.info(f"   Positions synced: {summary['positions_synced']}")
        logger.info(f"   Elite (Tier A) wallets: {summary['tier_a_wallets']}")
        logger.info(f"   Copy-trade suggestions: {summary['copy_suggestions']}")

        # Print top suggestion if available
        top_suggestion = self.test_results.get("phase5_trader_analysis", {}).get("top_suggestion")
        if top_suggestion:
            logger.info("\n🏆 TOP COPY-TRADE SUGGESTION:")
            logger.info(f"   Market: {top_suggestion.get('market_title', 'N/A')}")
            logger.info(f"   Elite Traders: {top_suggestion.get('n_elite_traders', 0)}")
            logger.info(f"   Avg ROI: {top_suggestion.get('avg_trader_roi', 0) * 100:.1f}%")
            logger.info(f"   Conviction Score: {top_suggestion.get('conviction_score', 0):.3f}")


# ========================================================================
# STANDALONE TEST FUNCTIONS
# ========================================================================

async def test_complete_pipeline(
    event_limit: int = 10,
    wallet_limit: int = 50,
    trader_limit: int = 20
) -> Dict[str, Any]:
    """
    Execute complete end-to-end pipeline test.

    Args:
        event_limit: Max events to sync
        wallet_limit: Max wallets to process
        trader_limit: Max traders to analyze

    Returns:
        Complete test results
    """
    orchestrator = PipelineTestOrchestrator()
    return await orchestrator.execute_full_pipeline_test(
        event_limit=event_limit,
        wallet_limit=wallet_limit,
        trader_limit=trader_limit,
        verbose=True
    )


async def test_data_sync_only(limit: int = 10) -> Dict[str, Any]:
    """Test only the data sync phase."""
    logger.info("🚀 Testing Data Sync Phase Only")
    orchestrator = PipelineTestOrchestrator()
    return await orchestrator._test_data_sync(limit, verbose=True)


async def test_wallet_flow_only(event_limit: int = 10, wallet_limit: int = 50) -> Dict[str, Any]:
    """Test wallet discovery and performance analytics."""
    logger.info("🚀 Testing Wallet Flow (Discovery + Analytics)")
    orchestrator = PipelineTestOrchestrator()

    # Phase 2: Discovery
    discovery = await orchestrator._test_wallet_discovery(event_limit, verbose=True)

    # Phase 3: Analytics
    analytics = await orchestrator._test_performance_analytics(wallet_limit, verbose=True)

    # Phase 4: Scoring
    scoring = await orchestrator._test_wallet_scoring(verbose=True)

    return {
        "discovery": discovery,
        "analytics": analytics,
        "scoring": scoring
    }


async def test_trader_analysis_only(trader_limit: int = 20) -> Dict[str, Any]:
    """Test only the trader analysis phase."""
    logger.info("🚀 Testing Trader Analysis Phase Only")
    orchestrator = PipelineTestOrchestrator()
    return await orchestrator._test_trader_analysis(trader_limit, verbose=True)


# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Elite Pipeline Test Orchestrator")
    parser.add_argument("--events", type=int, default=10, help="Max events to process")
    parser.add_argument("--wallets", type=int, default=50, help="Max wallets to process")
    parser.add_argument("--traders", type=int, default=20, help="Max traders to analyze")
    parser.add_argument("--phase", type=str, default="all",
                       choices=["all", "data", "wallets", "traders"],
                       help="Which phase to test")

    args = parser.parse_args()

    async def main():
        if args.phase == "all":
            result = await test_complete_pipeline(
                event_limit=args.events,
                wallet_limit=args.wallets,
                trader_limit=args.traders
            )
        elif args.phase == "data":
            result = await test_data_sync_only(limit=args.events)
        elif args.phase == "wallets":
            result = await test_wallet_flow_only(
                event_limit=args.events,
                wallet_limit=args.wallets
            )
        elif args.phase == "traders":
            result = await test_trader_analysis_only(trader_limit=args.traders)

        return result

    # Run the test
    final_result = asyncio.run(main())

    # Print JSON result for programmatic access
    import json
    print("\n" + "=" * 60)
    print("JSON RESULT:")
    print("=" * 60)
    print(json.dumps(final_result, indent=2, default=str))
