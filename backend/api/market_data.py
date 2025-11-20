import logging
from fastapi import APIRouter, BackgroundTasks
from clients.functions.data import PolymarketDataCollector

router = APIRouter()
logger = logging.getLogger("api.market_data")
collector = PolymarketDataCollector()

async def execute_sync_protocol():
    """
    Executes the complete Data Protocol:
    1. Hygiene (Archive Expired/Closed)
    2. Intelligence (Fetch New)
    3. Deployment (Save)
    4. Resolution (Populate Outcomes)
    """
    logger.info("🔥 [SYNC] Protocol Initiated...")
    
    # No try-catch. If DB/API fails, we want a loud crash log.
    # The collector handles the full lifecycle internally now.
    stats = await collector.collect_new_events_and_markets(limit=None)
    
    logger.info(f"✅ [SYNC] Mission Accomplished. Stats: {stats}")

@router.post("/sync")
async def trigger_sync(background_tasks: BackgroundTasks):
    """
    Trigger Async Data Pipeline.
    """
    background_tasks.add_task(execute_sync_protocol)
    return {"status": "accepted", "protocol": "market_sync_v2"}