from fastapi import APIRouter, BackgroundTasks
from clients.functions.wallets import PolymarketWalletCollector
import logging

router = APIRouter()
logger = logging.getLogger("api.wallets")

# Initialize Logic Engine
wallet_collector = PolymarketWalletCollector()

async def execute_wallet_enrichment_logic():
    """
    Core Logic:
    1. Discover wallets from new Event trades.
    2. Deep scan specific high-value wallets.
    """
    logger.info("🔥 [WALLETS] Starting Discovery Phase...")
    
    # 1. Discovery (Fast scan of new events)
    discovery_stats = await wallet_collector.enrich_unenriched_events(max_events=50)
    logger.info(f"🕵️ [WALLETS] Discovery: {discovery_stats}")

    # 2. Enrichment (Deep history scan of flagged wallets)
    logger.info("🔥 [WALLETS] Starting Deep Enrichment...")
    enrich_stats = await wallet_collector.enrich_wallets_positions(max_wallets=50)
    logger.info(f"💎 [WALLETS] Enrichment: {enrich_stats}")

@router.post("/enrich")
async def trigger_wallet_enrichment(background_tasks: BackgroundTasks):
    background_tasks.add_task(execute_wallet_enrichment_logic)
    return {"status": "accepted", "task": "wallet_enrichment_initiated"}