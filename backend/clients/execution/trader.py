#!/usr/bin/env python3
import asyncio
import logging
import uuid
from datetime import datetime, timezone

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs

from config.settings import Settings as AppSettings
from database.client import MarketDatabase
from .config import settings as exec_settings
from .security import vault

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExecutionEngine:
    def __init__(self):
        s = AppSettings()
        self.db = MarketDatabase(s.SUPABASE_URL, s.SUPABASE_KEY)

    async def run(self):
        mode = "📝 PAPER" if exec_settings.PAPER_TRADING else "🔥 LIVE"
        logger.info(f"🚀 Execution Engine Started [{mode} MODE]")

        # 1. Fetch Signals
        signals = await self._get_fresh_signals()
        if not signals:
            logger.info("   No fresh signals.")
            return

        # 2. Fetch Wallets
        wallets = await self._get_funded_wallets()
        if not wallets:
            logger.info("   No funded wallets found.")
            return

        logger.info(f"   Processing {len(signals)} signals for {len(wallets)} wallets...")

        # 3. Execute
        tasks = []
        for sig in signals:
            for wall in wallets:
                tasks.append(self._process_trade(wall, sig))
        
        results = await asyncio.gather(*tasks)
        success = sum(1 for r in results if r)
        logger.info(f"✅ Batch Complete. {success}/{len(tasks)} executed.")

    async def _process_trade(self, wallet, signal):
        try:
            user_short = wallet['user_id'][:5]
            slug_short = signal['market_slug'][:15]
            
            # 1. Idempotency Check
            if await self._already_traded(wallet['wallet_id'], signal['signal_id']):
                logger.info(f"   ⏭️  Skipping {slug_short} for {user_short}: Already Traded")
                return False

            # 2. Sizing Logic
            balance = float(wallet.get('current_balance_usdc') or 0)
            pct = float(signal.get('recommended_bet_size_pct', 1)) / 100.0
            risk = float(wallet.get('risk_multiplier', 1))
            
            amount = balance * pct * risk
            amount = min(amount, exec_settings.MAX_TRADE_USDC)

            # DEBUG LOG
            # logger.info(f"   🔍 Calc: Bal ${balance} * {pct*100:.2f}% * Risk {risk} = ${amount:.2f}")

            if amount < 1.0: 
                logger.warning(f"   ⚠️ Skipping {slug_short} for {user_short}: Size ${amount:.2f} < Min $1.00")
                return False

            # 3. Execution Switch
            if exec_settings.PAPER_TRADING:
                return await self._paper_trade(wallet, signal, amount)
            else:
                return await self._live_trade(wallet, signal, amount)

        except Exception as e:
            logger.error(f"Error processing: {e}")
            return False

    async def _paper_trade(self, wallet, signal, amount):
        """Simulates a filled order."""
        logger.info(f"   📝 [PAPER] User {wallet['user_id'][:5]} 'bought' ${amount:.2f} of {signal['market_slug'][:20]}...")
        
        # Simulate Balance Update
        new_bal = float(wallet['current_balance_usdc']) - amount
        self.db.supabase.table("managed_wallets")\
            .update({"current_balance_usdc": new_bal})\
            .eq("wallet_id", wallet['wallet_id']).execute()

        # Log Execution
        self.db.supabase.table("trade_executions").insert({
            "signal_id": signal['signal_id'],
            "wallet_id": wallet['wallet_id'],
            "market_slug": signal['market_slug'],
            "condition_id": signal.get('position_id'),
            "asset_id": "PAPER_ASSET_ID",
            "side": "BUY",
            "size_usdc": amount,
            "token_amount": amount / (float(signal['current_price']) or 0.5),
            "avg_fill_price": float(signal['current_price']),
            "status": "FILLED",
            "transaction_hash": f"PAPER_TX_{uuid.uuid4()}"
        }).execute()
        
        return True

    async def _live_trade(self, wallet, signal, amount):
        # ... (Keep existing Live Logic)
        pass

    # --- Helpers ---
    async def _get_fresh_signals(self):
        res = self.db.supabase.table("copy_trading_signals").select("*").order("created_at", desc=True).limit(5).execute()
        return res.data

    async def _get_funded_wallets(self):
        res = self.db.supabase.table("managed_wallets").select("*").eq("is_active", True).gt("current_balance_usdc", 1).execute()
        return res.data

    async def _already_traded(self, wid, sid):
        res = self.db.supabase.table("trade_executions").select("execution_id").eq("wallet_id", wid).eq("signal_id", sid).execute()
        return len(res.data) > 0

if __name__ == "__main__":
    asyncio.run(ExecutionEngine().run())