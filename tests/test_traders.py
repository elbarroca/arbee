"""
Individual Trader Functionality Tests

Focused testing of PolymarketTraderAnalyzer components:
1. Best trader filtering by performance metrics
2. Open positions surveillance
3. Copy-trade suggestions generation

Run with: python tests/test_traders.py --help
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from config.settings import Settings
from database.client import MarketDatabase

# Import trader analyzer
from clients.functions.traders import PolymarketTraderAnalyzer, execute_full_trader_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class TraderTestOrchestrator:
    """
    Focused trader testing orchestrator.

    Tests individual trader analysis components without full pipeline dependencies.
    """

    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        self.test_results = {}
        self.start_time = None

    async def execute_trader_filtering_test(
        self,
        trader_limit: int = 20,
        min_roi: float = 0.05,
        min_win_rate: float = 0.60,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test trader filtering by performance metrics.

        Args:
            trader_limit: Max traders to return
            min_roi: Minimum ROI threshold
            min_win_rate: Minimum win rate threshold
            verbose: Print detailed progress

        Returns:
            Filtering test results
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info("🚀 TRADER FILTERING TEST INITIATED")
        logger.info("=" * 60)

        try:
            analyzer = PolymarketTraderAnalyzer()
            traders = await analyzer.filter_best_traders(
                min_roi=min_roi,
                min_win_rate=min_win_rate,
                limit=trader_limit
            )

            success = True
            result = {
                "success": True,
                "traders_found": len(traders),
                "traders": traders[:5] if traders else [],  # Show first 5 for brevity
                "min_roi_threshold": min_roi,
                "min_win_rate_threshold": min_win_rate,
                "limit": trader_limit
            }

            self.test_results["filtering"] = result

            if verbose:
                self._print_filtering_result(result)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - self.start_time).total_seconds()

            return {
                "test_status": "PASSED" if success else "FAILED",
                "duration_seconds": round(duration, 2),
                "result": result,
                "completed_at": end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Trader filtering test failed: {e}")
            return {
                "test_status": "FAILED",
                "error": str(e),
                "duration_seconds": round((datetime.now(timezone.utc) - self.start_time).total_seconds(), 2)
            }

    async def execute_open_positions_test(
        self,
        trader_limit: int = 20,
        min_roi: float = 0.05,
        min_win_rate: float = 0.60,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test open positions surveillance for elite traders.

        Args:
            trader_limit: Max traders to analyze
            min_roi: Minimum ROI threshold
            min_win_rate: Minimum win rate threshold
            verbose: Print detailed progress

        Returns:
            Open positions test results
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info("🚀 OPEN POSITIONS TEST INITIATED")
        logger.info("=" * 60)

        try:
            analyzer = PolymarketTraderAnalyzer()
            positions_report = await analyzer.get_best_traders_open_positions(
                min_roi=min_roi,
                min_win_rate=min_win_rate,
                max_traders=trader_limit
            )

            success = positions_report.get("success", True)
            result = positions_report

            self.test_results["positions"] = result

            if verbose:
                self._print_positions_result(result)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - self.start_time).total_seconds()

            return {
                "test_status": "PASSED" if success else "FAILED",
                "duration_seconds": round(duration, 2),
                "result": result,
                "completed_at": end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Open positions test failed: {e}")
            return {
                "test_status": "FAILED",
                "error": str(e),
                "duration_seconds": round((datetime.now(timezone.utc) - self.start_time).total_seconds(), 2)
            }

    async def execute_copy_suggestions_test(
        self,
        trader_limit: int = 20,
        min_roi: float = 0.05,
        min_win_rate: float = 0.60,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test copy-trade suggestions generation.

        Args:
            trader_limit: Max traders to analyze
            min_roi: Minimum ROI threshold
            min_win_rate: Minimum win rate threshold
            verbose: Print detailed progress

        Returns:
            Copy suggestions test results
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info("🚀 COPY SUGGESTIONS TEST INITIATED")
        logger.info("=" * 60)

        try:
            analyzer = PolymarketTraderAnalyzer()
            suggestions = await analyzer.get_copy_trade_suggestions(
                min_roi=min_roi,
                min_win_rate=min_win_rate,
                min_elite_traders=1,
                top_n=10
            )

            success = True
            result = {
                "success": True,
                "suggestions_count": len(suggestions),
                "suggestions": suggestions,
                "top_suggestion": suggestions[0] if suggestions else None,
                "min_roi_threshold": min_roi,
                "min_win_rate_threshold": min_win_rate
            }

            self.test_results["suggestions"] = result

            if verbose:
                self._print_suggestions_result(result)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - self.start_time).total_seconds()

            return {
                "test_status": "PASSED" if success else "FAILED",
                "duration_seconds": round(duration, 2),
                "result": result,
                "completed_at": end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Copy suggestions test failed: {e}")
            return {
                "test_status": "FAILED",
                "error": str(e),
                "duration_seconds": round((datetime.now(timezone.utc) - self.start_time).total_seconds(), 2)
            }

    async def execute_full_trader_analysis_test(
        self,
        trader_limit: int = 20,
        min_roi: float = 0.05,
        min_win_rate: float = 0.60,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test complete trader analysis flow: filtering → positions → suggestions.

        Args:
            trader_limit: Max traders to analyze
            min_roi: Minimum ROI threshold
            min_win_rate: Minimum win rate threshold
            verbose: Print detailed progress

        Returns:
            Complete trader analysis test results
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info("🚀 FULL TRADER ANALYSIS TEST INITIATED")
        logger.info("=" * 60)

        # Phase 1: Filtering
        logger.info("\n📍 PHASE 1: TRADER FILTERING")
        filtering_result = await self.execute_trader_filtering_test(trader_limit, min_roi, min_win_rate, verbose)
        if filtering_result["test_status"] != "PASSED":
            return {
                "test_status": "FAILED",
                "failed_phase": "filtering",
                "error": filtering_result.get("error", "Filtering failed")
            }

        # Phase 2: Open Positions
        logger.info("\n📍 PHASE 2: OPEN POSITIONS SURVEILLANCE")
        positions_result = await self.execute_open_positions_test(trader_limit, min_roi, min_win_rate, verbose)
        if positions_result["test_status"] != "PASSED":
            return {
                "test_status": "FAILED",
                "failed_phase": "positions",
                "error": positions_result.get("error", "Positions failed")
            }

        # Phase 3: Copy Suggestions
        logger.info("\n📍 PHASE 3: COPY-TRADE SUGGESTIONS")
        suggestions_result = await self.execute_copy_suggestions_test(trader_limit, min_roi, min_win_rate, verbose)
        if suggestions_result["test_status"] != "PASSED":
            return {
                "test_status": "FAILED",
                "failed_phase": "suggestions",
                "error": suggestions_result.get("error", "Suggestions failed")
            }

        # Final summary
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()

        return {
            "test_status": "PASSED",
            "total_duration_seconds": round(duration, 2),
            "phases": {
                "filtering": filtering_result,
                "positions": positions_result,
                "suggestions": suggestions_result
            },
            "summary": {
                "traders_found": filtering_result["result"]["traders_found"],
                "open_positions_count": positions_result["result"].get("total_positions", 0),
                "traders_with_positions": positions_result["result"].get("traders_with_positions", 0),
                "copy_suggestions_count": suggestions_result["result"]["suggestions_count"]
            },
            "completed_at": end_time.isoformat()
        }

    def _print_filtering_result(self, result: Dict[str, Any]) -> None:
        """Print trader filtering results."""
        if result.get("success", True):
            logger.info("✅ TRADER FILTERING SUCCESS")
            logger.info(f"   Traders found: {result['traders_found']}")
            logger.info(f"   Min ROI: {result['min_roi_threshold']*100:.0f}%")
            logger.info(f"   Min Win Rate: {result['min_win_rate_threshold']*100:.0f}%")
            if result['traders_found'] > 0:
                logger.info("   Sample traders:")
                for trader in result['traders'][:3]:  # Show first 3
                    roi = trader.get('total_roi', 0) * 100
                    win_rate = trader.get('win_rate', 0) * 100
                    volume = trader.get('total_volume', 0)
                    logger.info(f"     - ROI: {roi:.1f}%, Win Rate: {win_rate:.1f}%, Volume: ${volume:,.0f}")
        else:
            logger.error(f"❌ TRADER FILTERING FAILED: {result.get('error')}")

    def _print_positions_result(self, result: Dict[str, Any]) -> None:
        """Print open positions results."""
        if result.get("success", True):
            logger.info("✅ OPEN POSITIONS SUCCESS")
            logger.info(f"   Total positions: {result.get('total_positions', 0)}")
            logger.info(f"   Traders with positions: {result.get('traders_with_positions', 0)}")
            logger.info(f"   Total position value: ${result.get('total_position_value', 0):,.2f}")
        else:
            logger.error(f"❌ OPEN POSITIONS FAILED: {result.get('error')}")

    def _print_suggestions_result(self, result: Dict[str, Any]) -> None:
        """Print copy suggestions results."""
        if result.get("success", True):
            logger.info("✅ COPY SUGGESTIONS SUCCESS")
            logger.info(f"   Suggestions generated: {result['suggestions_count']}")
            if result['top_suggestion']:
                suggestion = result['top_suggestion']
                logger.info("   Top suggestion:")
                logger.info(f"     Market: {suggestion.get('market_title', 'N/A')}")
                logger.info(f"     Elite Traders: {suggestion.get('n_elite_traders', 0)}")
                logger.info(f"     Avg ROI: {suggestion.get('avg_trader_roi', 0) * 100:.1f}%")
                logger.info(f"     Conviction Score: {suggestion.get('conviction_score', 0):.3f}")
        else:
            logger.error(f"❌ COPY SUGGESTIONS FAILED: {result.get('error')}")


# ========================================================================
# STANDALONE TEST FUNCTIONS
# ========================================================================

async def test_trader_filtering_only(trader_limit: int = 20) -> Dict[str, Any]:
    """Test only trader filtering functionality."""
    logger.info("🚀 Testing Trader Filtering Only")
    orchestrator = TraderTestOrchestrator()
    return await orchestrator.execute_trader_filtering_test(trader_limit=trader_limit, verbose=True)


async def test_open_positions_only(trader_limit: int = 20) -> Dict[str, Any]:
    """Test only open positions surveillance."""
    logger.info("🚀 Testing Open Positions Only")
    orchestrator = TraderTestOrchestrator()
    return await orchestrator.execute_open_positions_test(trader_limit=trader_limit, verbose=True)


async def test_copy_suggestions_only(trader_limit: int = 20) -> Dict[str, Any]:
    """Test only copy-trade suggestions generation."""
    logger.info("🚀 Testing Copy Suggestions Only")
    orchestrator = TraderTestOrchestrator()
    return await orchestrator.execute_copy_suggestions_test(trader_limit=trader_limit, verbose=True)


async def test_full_trader_analysis(trader_limit: int = 20) -> Dict[str, Any]:
    """Test complete trader analysis flow."""
    logger.info("🚀 Testing Full Trader Analysis")
    orchestrator = TraderTestOrchestrator()
    return await orchestrator.execute_full_trader_analysis_test(trader_limit=trader_limit, verbose=True)


# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trader Functionality Tests")
    parser.add_argument("--traders", type=int, default=20, help="Max traders to analyze")
    parser.add_argument("--min-roi", type=float, default=0.05, help="Minimum ROI threshold (0.05 = 5%)")
    parser.add_argument("--min-win-rate", type=float, default=0.60, help="Minimum win rate (0.60 = 60%)")
    parser.add_argument("--test", type=str, default="filtering",
                       choices=["filtering", "positions", "suggestions", "full"],
                       help="Which trader test to run")

    args = parser.parse_args()

    async def main():
        if args.test == "filtering":
            result = await test_trader_filtering_only(trader_limit=args.traders)
        elif args.test == "positions":
            result = await test_open_positions_only(trader_limit=args.traders)
        elif args.test == "suggestions":
            result = await test_copy_suggestions_only(trader_limit=args.traders)
        elif args.test == "full":
            result = await test_full_trader_analysis(trader_limit=args.traders)

        return result

    # Run the test
    final_result = asyncio.run(main())

    # Print JSON result for programmatic access
    import json
    print("\n" + "=" * 60)
    print("JSON RESULT:")
    print("=" * 60)
    print(json.dumps(final_result, indent=2, default=str))
