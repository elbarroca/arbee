#!/usr/bin/env python3
"""
Elite Trader Analytics Pipeline (High Performance).
Uses Async Concurrency to process 100k+ wallets efficiently.
"""

import asyncio
import logging
from datetime import datetime, timezone

from config.settings import Settings
from database.client import MarketDatabase

# Configure Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EliteTraderManager:
    """Manages the calculation and ranking of elite trader data."""

    def __init__(self):
        settings = Settings()
        self.db = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        # Performance Tuning
        self.BATCH_SIZE = 250  # Larger batches = fewer DB calls
        self.CONCURRENCY = 10   # Run 10 batches at the same time

    async def _process_batch_worker(self, semaphore, batch, batch_index, total_wallets):
        """Helper to process a single batch with a concurrency limit."""
        async with semaphore:
            try:
                self.db.supabase.rpc('fn_update_wallet_analytics_batch', {'batch_wallets': batch}).execute()
                # Log progress every few batches to reduce noise
                if batch_index % 5 == 0:
                    progress = min((batch_index * self.BATCH_SIZE) + self.BATCH_SIZE, total_wallets)
                    logger.info(f"   - Processed {progress}/{total_wallets} wallets...")
            except Exception as e:
                logger.error(f"❌ Batch {batch_index} failed: {e}")
                raise e

    async def run_pipeline(self):
        """
        Executes the pipeline with high concurrency.
        """
        start_time = datetime.now(timezone.utc)
        logger.info("🚀 Starting High-Performance Elite Trader Pipeline...")

        # --- STEP 1: Concurrent Wallet Analytics ---
        logger.info("Step 1/4: Calculating Wallet Analytics (Parallel Mode)...")
        
        # A. Fetch all source wallets
        res = self.db.supabase.rpc('fn_get_all_trader_wallets').execute()
        all_wallets = [row['wallet'] for row in res.data] if res.data else []
        total_wallets = len(all_wallets)
        
        logger.info(f"   Found {total_wallets} source wallets. Processing in batches of {self.BATCH_SIZE} with {self.CONCURRENCY}x concurrency.")

        # B. Create Tasks
        semaphore = asyncio.Semaphore(self.CONCURRENCY)
        tasks = []
        
        for i in range(0, total_wallets, self.BATCH_SIZE):
            batch = all_wallets[i : i + self.BATCH_SIZE]
            batch_index = i // self.BATCH_SIZE
            task = self._process_batch_worker(semaphore, batch, batch_index, total_wallets)
            tasks.append(task)

        # C. Execute All Batches concurrently
        await asyncio.gather(*tasks)

        # --- STEP 2: Sync to Main Profile ---
        logger.info("Step 2/4: Syncing Stats to Main Wallets Table...")
        self.db.supabase.rpc('fn_sync_wallets_summary').execute()

        # --- STEP 3: Ranking ---
        logger.info("Step 3/4: Ranking Elite Traders (Stricter Rules)...")
        self.db.supabase.rpc('fn_rank_elite_traders').execute()

        # --- STEP 4: Tag Analytics ---
        logger.info("Step 4/4: Calculating Tag & Market Stats...")
        self.db.supabase.rpc('fn_calculate_tag_comparisons').execute()

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"✅ Pipeline Completed Successfully in {duration:.2f}s")

    async def get_stats(self) -> dict:
        """Returns count of records."""
        tables = ["wallet_analytics", "elite_traders", "elite_tag_comparisons"]
        results = {}
        for table in tables:
            res = self.db.supabase.table(table).select("*", count="exact").limit(1).execute()
            results[table] = res.count
        return results

# --- Execution Entry Point ---

async def main():
    manager = EliteTraderManager()
    await manager.run_pipeline()
    stats = await manager.get_stats()
    print(f"📊 Pipeline Results: {stats}")

if __name__ == "__main__":
    asyncio.run(main())