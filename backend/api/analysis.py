from fastapi import APIRouter, BackgroundTasks
from clients.functions.elite_traders import EliteTraderManager
from clients.functions.get_elite_open_positions import WalletOpenPositionsCollector
import logging

router = APIRouter()
logger = logging.getLogger("api.analysis")

# Initialize Engines
elite_manager = EliteTraderManager()
positions_collector = WalletOpenPositionsCollector()

async def execute_ranking_logic():
    """
    Core Logic:
    1. Calculate Metrics (Win Rate, ROI) -> DB Function.
    2. Rank Traders (S/A/B Tier) -> DB Function.
    3. Fetch Live Positions for S/A/B Tier traders.
    """
    logger.info("🔥 [RANK] Starting Elite Trader Pipeline...")
    
    # 1. Run the Ranking Pipeline (Heavy DB Calculations)
    await elite_manager.run_pipeline()
    logger.info("🏆 [RANK] Ranking Complete.")

    # 2. Fetch Positions for Elites
    logger.info("🔥 [POSITIONS] Sniping Elite Positions...")
    pos_stats = await positions_collector.run()
    logger.info(f"🎯 [POSITIONS] Result: {pos_stats}")

@router.post("/rank")
async def trigger_trader_ranking(background_tasks: BackgroundTasks):
    background_tasks.add_task(execute_ranking_logic)
    return {"status": "accepted", "task": "trader_ranking_initiated"}