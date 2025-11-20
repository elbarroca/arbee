"""
TEST SCRIPT: Wallet Pipeline
----------------------------
1. DISCOVERY: Scans 2 active events for new wallets.
2. ENRICHMENT: Picks 2 unenriched wallets and deep scans them.
"""
import asyncio
import logging
import json
from clients.functions.wallets import enrich_unenriched_events, enrich_wallets_positions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("TEST")

async def run_test():
    logger.info("🧪 STARTING WALLET PIPELINE TEST")

    # 1. Test Discovery (Active Only)
    logger.info("--- [1/2] Discovery Phase (Active Events) ---")
    try:
        disc_res = await enrich_unenriched_events(max_events=2)
        logger.info(f"✅ Discovery Result: {json.dumps(disc_res, indent=2)}")
    except Exception as e:
        logger.error(f"❌ Discovery Failed: {e}")

    # 2. Test Enrichment (Deep Scan)
    logger.info("\n--- [2/2] Enrichment Phase (Deep Scan) ---")
    try:
        enrich_res = await enrich_wallets_positions(max_wallets=2)
        logger.info(f"✅ Enrichment Result: {json.dumps(enrich_res, indent=2)}")
        
        if enrich_res['processed'] > 0:
            logger.info(f"🎉 Success! Enriched {enrich_res['processed']} wallets with {enrich_res['positions']} positions.")
        else:
            logger.info("ℹ️ No unenriched wallets pending in DB.")
            
    except Exception as e:
        logger.error(f"❌ Enrichment Failed: {e}")

    logger.info("\n🏁 TEST COMPLETE")

if __name__ == "__main__":
    asyncio.run(run_test())