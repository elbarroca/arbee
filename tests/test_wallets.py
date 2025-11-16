"""
Individual Wallet Functionality Tests

Focused testing of PolymarketWalletCollector components:
1. Wallet discovery from closed events
2. Performance analytics computation
3. Wallet scoring and tier assignment

Run with: python tests/test_wallets.py --help
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from config.settings import Settings
from database.client import MarketDatabase

# Import wallet collector
from clients.functions.wallets import PolymarketWalletCollector, execute_full_wallet_sync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class WalletTestOrchestrator:
    """
    Focused wallet testing orchestrator.

    Tests individual wallet components without full pipeline dependencies.
    """

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.test_results = {}
        self.start_time = None

    async def execute_wallet_discovery_test(
        self,
        event_limit: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test wallet discovery from closed events.

        Args:
            event_limit: Max closed events to process
            verbose: Print detailed progress

        Returns:
            Discovery test results
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info("🚀 WALLET DISCOVERY TEST INITIATED")
        logger.info("=" * 60)

        # Get pre-test snapshot
        pre_snapshot = await self._get_wallet_snapshot()

        try:
            collector = PolymarketWalletCollector()
            result = await collector.sync_wallets_from_closed_events(
                limit=event_limit,
                save_trades=True
            )

            success = result.get("success", True)
            self.test_results["discovery"] = result

            if verbose:
                self._print_discovery_result(result)

            # Post-test snapshot
            post_snapshot = await self._get_wallet_snapshot()
            changes = self._calculate_changes(pre_snapshot, post_snapshot)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - self.start_time).total_seconds()

            return {
                "test_status": "PASSED" if success else "FAILED",
                "duration_seconds": round(duration, 2),
                "pre_snapshot": pre_snapshot,
                "post_snapshot": post_snapshot,
                "changes": changes,
                "result": result,
                "completed_at": end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Wallet discovery test failed: {e}")
            return {
                "test_status": "FAILED",
                "error": str(e),
                "duration_seconds": round((datetime.now(timezone.utc) - self.start_time).total_seconds(), 2)
            }

    async def execute_performance_analytics_test(
        self,
        wallet_limit: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test performance analytics computation.

        Args:
            wallet_limit: Max wallets to process
            verbose: Print detailed progress

        Returns:
            Analytics test results
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info("🚀 PERFORMANCE ANALYTICS TEST INITIATED")
        logger.info("=" * 60)

        # Get pre-test snapshot
        pre_snapshot = await self._get_performance_snapshot()

        try:
            collector = PolymarketWalletCollector()
            result = await collector.sync_wallet_performance_data(max_wallets=wallet_limit)

            success = result.get("success", True)
            self.test_results["analytics"] = result

            if verbose:
                self._print_analytics_result(result)

            # Post-test snapshot
            post_snapshot = await self._get_performance_snapshot()
            changes = self._calculate_changes(pre_snapshot, post_snapshot)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - self.start_time).total_seconds()

            return {
                "test_status": "PASSED" if success else "FAILED",
                "duration_seconds": round(duration, 2),
                "pre_snapshot": pre_snapshot,
                "post_snapshot": post_snapshot,
                "changes": changes,
                "result": result,
                "completed_at": end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Performance analytics test failed: {e}")
            return {
                "test_status": "FAILED",
                "error": str(e),
                "duration_seconds": round((datetime.now(timezone.utc) - self.start_time).total_seconds(), 2)
            }

    async def execute_wallet_scoring_test(
        self,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test wallet scoring and tier assignment.

        Args:
            verbose: Print detailed progress

        Returns:
            Scoring test results
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info("🚀 WALLET SCORING TEST INITIATED")
        logger.info("=" * 60)

        # Get pre-test snapshot
        pre_snapshot = await self._get_scoring_snapshot()

        try:
            collector = PolymarketWalletCollector()
            result = await collector.compute_and_save_wallet_scores()

            success = result.get("success", True)
            self.test_results["scoring"] = result

            if verbose:
                self._print_scoring_result(result)

            # Post-test snapshot
            post_snapshot = await self._get_scoring_snapshot()
            changes = self._calculate_changes(pre_snapshot, post_snapshot)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - self.start_time).total_seconds()

            return {
                "test_status": "PASSED" if success else "FAILED",
                "duration_seconds": round(duration, 2),
                "pre_snapshot": pre_snapshot,
                "post_snapshot": post_snapshot,
                "changes": changes,
                "result": result,
                "completed_at": end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Wallet scoring test failed: {e}")
            return {
                "test_status": "FAILED",
                "error": str(e),
                "duration_seconds": round((datetime.now(timezone.utc) - self.start_time).total_seconds(), 2)
            }

    async def execute_full_wallet_flow_test(
        self,
        event_limit: int = 10,
        wallet_limit: int = 50,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test complete wallet flow: discovery → analytics → scoring.

        Args:
            event_limit: Max events for discovery
            wallet_limit: Max wallets for analytics
            verbose: Print detailed progress

        Returns:
            Complete wallet flow test results
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info("🚀 FULL WALLET FLOW TEST INITIATED")
        logger.info("=" * 60)

        # Get pre-test snapshot
        pre_snapshot = await self._get_full_wallet_snapshot()

        # Phase 1: Discovery
        logger.info("\n📍 PHASE 1: WALLET DISCOVERY")
        discovery_result = await self.execute_wallet_discovery_test(event_limit, verbose)
        if discovery_result["test_status"] != "PASSED":
            return {
                "test_status": "FAILED",
                "failed_phase": "discovery",
                "error": discovery_result.get("error", "Discovery failed")
            }

        # Phase 2: Analytics
        logger.info("\n📍 PHASE 2: PERFORMANCE ANALYTICS")
        analytics_result = await self.execute_performance_analytics_test(wallet_limit, verbose)
        if analytics_result["test_status"] != "PASSED":
            return {
                "test_status": "FAILED",
                "failed_phase": "analytics",
                "error": analytics_result.get("error", "Analytics failed")
            }

        # Phase 3: Scoring
        logger.info("\n📍 PHASE 3: WALLET SCORING")
        scoring_result = await self.execute_wallet_scoring_test(verbose)
        if scoring_result["test_status"] != "PASSED":
            return {
                "test_status": "FAILED",
                "failed_phase": "scoring",
                "error": scoring_result.get("error", "Scoring failed")
            }

        # Final snapshot and summary
        post_snapshot = await self._get_full_wallet_snapshot()
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()

        return {
            "test_status": "PASSED",
            "total_duration_seconds": round(duration, 2),
            "phases": {
                "discovery": discovery_result,
                "analytics": analytics_result,
                "scoring": scoring_result
            },
            "pre_snapshot": pre_snapshot,
            "post_snapshot": post_snapshot,
            "summary": {
                "wallets_discovered": discovery_result["result"].get("wallets_discovered", 0),
                "positions_synced": analytics_result["result"].get("positions_synced", 0),
                "scores_computed": scoring_result["result"].get("scores_computed", 0),
                "tier_a_count": scoring_result["result"].get("tier_a_count", 0)
            },
            "completed_at": end_time.isoformat()
        }

    async def _get_wallet_snapshot(self) -> Dict[str, int]:
        """Get wallet-related table counts."""
        tables = ["wallets", "trades"]
        return await self._get_table_counts(tables)

    async def _get_performance_snapshot(self) -> Dict[str, int]:
        """Get performance-related table counts."""
        tables = ["wallet_closed_positions", "wallet_stats", "wallet_tag_stats", "wallet_market_stats"]
        return await self._get_table_counts(tables)

    async def _get_scoring_snapshot(self) -> Dict[str, int]:
        """Get scoring-related table counts."""
        tables = ["wallet_scores"]
        return await self._get_table_counts(tables)

    async def _get_full_wallet_snapshot(self) -> Dict[str, int]:
        """Get all wallet-related table counts."""
        tables = [
            "wallets", "trades", "wallet_closed_positions", "wallet_stats",
            "wallet_tag_stats", "wallet_market_stats", "wallet_scores"
        ]
        return await self._get_table_counts(tables)

    async def _get_table_counts(self, tables: list) -> Dict[str, int]:
        """Get row counts for specified tables."""
        snapshot = {}
        for table in tables:
            try:
                result = self.db.supabase.table(table).select("*", count="exact").limit(0).execute()
                snapshot[table] = result.count or 0
            except Exception:
                snapshot[table] = -1  # Table might not exist
        return snapshot

    def _calculate_changes(self, before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
        """Calculate changes between snapshots."""
        changes = {}
        for table in before.keys():
            delta = after.get(table, 0) - before.get(table, 0)
            changes[table] = delta
        return changes

    def _print_discovery_result(self, result: Dict[str, Any]) -> None:
        """Print wallet discovery results."""
        if result.get("success", True):
            logger.info("✅ WALLET DISCOVERY SUCCESS")
            for key, value in result.items():
                if key not in ["success", "error"]:
                    logger.info(f"   {key}: {value}")
        else:
            logger.error(f"❌ WALLET DISCOVERY FAILED: {result.get('error')}")

    def _print_analytics_result(self, result: Dict[str, Any]) -> None:
        """Print performance analytics results."""
        if result.get("success", True):
            logger.info("✅ PERFORMANCE ANALYTICS SUCCESS")
            for key, value in result.items():
                if key not in ["success", "error"]:
                    if isinstance(value, float):
                        logger.info(f"   {key}: ${value:,.2f}")
                    else:
                        logger.info(f"   {key}: {value}")
        else:
            logger.error(f"❌ PERFORMANCE ANALYTICS FAILED: {result.get('error')}")

    def _print_scoring_result(self, result: Dict[str, Any]) -> None:
        """Print wallet scoring results."""
        if result.get("success", True):
            logger.info("✅ WALLET SCORING SUCCESS")
            for key, value in result.items():
                if key not in ["success", "error"]:
                    logger.info(f"   {key}: {value}")
        else:
            logger.error(f"❌ WALLET SCORING FAILED: {result.get('error')}")


# ========================================================================
# STANDALONE TEST FUNCTIONS
# ========================================================================

async def test_wallet_discovery_only(event_limit: int = 10) -> Dict[str, Any]:
    """Test only wallet discovery functionality."""
    logger.info("🚀 Testing Wallet Discovery Only")
    orchestrator = WalletTestOrchestrator()
    return await orchestrator.execute_wallet_discovery_test(event_limit=event_limit, verbose=True)


async def test_performance_analytics_only(wallet_limit: int = 50) -> Dict[str, Any]:
    """Test only performance analytics functionality."""
    logger.info("🚀 Testing Performance Analytics Only")
    orchestrator = WalletTestOrchestrator()
    return await orchestrator.execute_performance_analytics_test(wallet_limit=wallet_limit, verbose=True)


async def test_wallet_scoring_only() -> Dict[str, Any]:
    """Test only wallet scoring functionality."""
    logger.info("🚀 Testing Wallet Scoring Only")
    orchestrator = WalletTestOrchestrator()
    return await orchestrator.execute_wallet_scoring_test(verbose=True)


async def test_full_wallet_flow(event_limit: int = 10, wallet_limit: int = 50) -> Dict[str, Any]:
    """Test complete wallet flow."""
    logger.info("🚀 Testing Full Wallet Flow")
    orchestrator = WalletTestOrchestrator()
    return await orchestrator.execute_full_wallet_flow_test(
        event_limit=event_limit,
        wallet_limit=wallet_limit,
        verbose=True
    )


# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wallet Functionality Tests")
    parser.add_argument("--events", type=int, default=10, help="Max events for discovery")
    parser.add_argument("--wallets", type=int, default=50, help="Max wallets for analytics")
    parser.add_argument("--test", type=str, default="discovery",
                       choices=["discovery", "analytics", "scoring", "full"],
                       help="Which wallet test to run")

    args = parser.parse_args()

    async def main():
        if args.test == "discovery":
            result = await test_wallet_discovery_only(event_limit=args.events)
        elif args.test == "analytics":
            result = await test_performance_analytics_only(wallet_limit=args.wallets)
        elif args.test == "scoring":
            result = await test_wallet_scoring_only()
        elif args.test == "full":
            result = await test_full_wallet_flow(event_limit=args.events, wallet_limit=args.wallets)

        return result

    # Run the test
    final_result = asyncio.run(main())

    # Print JSON result for programmatic access
    import json
    print("\n" + "=" * 60)
    print("JSON RESULT:")
    print("=" * 60)
    print(json.dumps(final_result, indent=2, default=str))
