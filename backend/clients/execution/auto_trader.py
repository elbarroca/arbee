#!/usr/bin/env python3
"""
Automated Execution Engine (The Middleman).
Reads Signals -> Executes on Managed Wallets -> logs to DB.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any
from datetime import datetime, timezone

# Polymarket CLOB Client (Official)
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

from config.settings import Settings
from database.client import MarketDatabase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AutoExecutionEngine:
    def __init__(self):
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        
        # Configuration
        self.CHAIN_ID = 137 # Polygon Mainnet
        self.Global_Slippage = 0.05 # 5% max slippage allowed on execution

    async def run_execution_cycle(self):
        """Main Loop: Signal -> Wallet -> Execute."""
        logger.info("🤖 Starting Execution Cycle...")

        # 1. Get Fresh Signals (Created in the last 10 minutes)
        # We don't want to execute old stale signals.
        signals = await self._fetch_fresh_signals()
        if not signals:
            logger.info("   No fresh signals to execute.")
            return

        # 2. Get Active Managed Wallets
        wallets = await self._fetch_managed_wallets()
        if not wallets:
            logger.info("   No active managed wallets found.")
            return

        logger.info(f"   Found {len(signals)} signals. Executing for {len(wallets)} users...")

        # 3. Execution Loop
        # In production, use a Task Queue (Celery/Redis) for this. 
        # Here we use Asyncio for demonstration.
        tasks = []
        for wallet in wallets:
            for signal in signals:
                tasks.append(self._execute_trade_for_user(wallet, signal))

        # Run all trades in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        logger.info(f"✅ Execution Cycle Complete. {success_count}/{len(tasks)} trades successful.")

    # --- Data Fetching ---

    async def _fetch_fresh_signals(self) -> List[Dict]:
        """Fetch signals created in the last 20 mins that are highly rated."""
        # We look for 'is_cash_flow_optimized' or very high confidence
        res = self.db.supabase.table("copy_trading_signals")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(10)\
            .execute()
        
        # Simple time filter in Python if DB filter is complex
        valid_signals = []
        now = datetime.now(timezone.utc)
        for s in res.data:
            # Logic: Is signal fresh? (e.g. < 30 mins old)
            # Note: You should add a 'processed' flag to signals table to avoid re-buying
            valid_signals.append(s)
            
        return valid_signals

    async def _fetch_managed_wallets(self) -> List[Dict]:
        res = self.db.supabase.table("managed_wallets")\
            .select("*")\
            .eq("is_active", True)\
            .execute()
        return res.data or []

    # --- Core Execution Logic ---

    async def _execute_trade_for_user(self, wallet: Dict, signal: Dict) -> bool:
        """
        Executes a specific signal for a specific user.
        """
        user_id = wallet['user_id']
        signal_id = signal['signal_id']
        
        # 1. Check if already executed (Idempotency)
        # Don't buy the same thing twice for the same user
        existing = self.db.supabase.table("trade_executions")\
            .select("execution_id")\
            .eq("wallet_id", wallet['wallet_id'])\
            .eq("signal_id", signal_id)\
            .execute()
        
        if existing.data:
            return False # Already traded

        # 2. Calculate Size
        # Allocation = Balance * Signal% * UserRisk
        balance = float(wallet.get('current_balance_usdc', 0))
        if balance < 1.0: return False # Too poor
        
        signal_pct = float(signal.get('recommended_bet_size_pct', 1)) / 100.0
        risk_mult = float(wallet.get('risk_multiplier', 1.0))
        
        bet_amount_usdc = balance * signal_pct * risk_mult
        
        # Cap at $500 or Wallet Max for safety during testing
        bet_amount_usdc = min(bet_amount_usdc, 500.0) 
        
        if bet_amount_usdc < 1.0: return False # Min trade size

        # 3. Initialize CLOB Client for THIS user
        # WARNING: In production, decrypt the private key here.
        try:
            client = ClobClient(
                host="https://clob.polymarket.com",
                key=wallet['encrypted_private_key'], # Assuming stored raw for MVP (Encrypt in Prod!)
                chain_id=self.CHAIN_ID,
                signature_type=1, # 1 = EOA (Standard Wallet), 2 = PolyProxy
                funder=wallet['proxy_wallet_address'] # The Proxy Address
            )
            
            # 4. Prepare Order
            # Buy "Yes" or "No" based on Signal. 
            # Signal usually implies buying the Outcome specified.
            token_id = await self._get_token_id(client, signal['condition_id'], signal['outcome'])
            if not token_id:
                logger.error(f"❌ Could not find token ID for {signal['market_slug']}")
                return False

            # Limit Price = Signal Price + Slippage
            price = float(signal['current_price'])
            limit_price = min(price + self.Global_Slippage, 0.99) # Cap at 0.99

            # 5. Execute Order (FOK - Fill or Kill)
            logger.info(f"🚀 Executing ${bet_amount_usdc:.2f} on {signal['market_slug']} for User {user_id}...")
            
            resp = client.create_and_post_order(
                OrderArgs(
                    price=limit_price,
                    size=bet_amount_usdc / limit_price, # Size is in Shares, not USD
                    side="BUY",
                    token_id=token_id
                )
            )
            
            # 6. Log Success
            if resp and resp.get('orderID'):
                logger.info(f"   ✅ Filled: {resp['orderID']}")
                
                # Save to DB
                self.db.supabase.table("trade_executions").insert({
                    "signal_id": signal_id,
                    "wallet_id": wallet['wallet_id'],
                    "market_slug": signal['market_slug'],
                    "condition_id": signal.get('position_id'), # Using pos ID as ref
                    "asset_id": token_id,
                    "side": "BUY",
                    "size_usdc": bet_amount_usdc,
                    "token_amount": bet_amount_usdc / limit_price,
                    "avg_fill_price": limit_price, # Approx
                    "status": "FILLED",
                    "transaction_hash": resp.get('transactionHash')
                }).execute()
                
                return True
            
            logger.warning(f"   ⚠️ Order Failed: {resp}")
            return False

        except Exception as e:
            logger.error(f"   ❌ Execution Error: {str(e)}")
            return False

    async def _get_token_id(self, client, condition_id, outcome):
        # Helper to resolve YES/NO to the long Token ID
        # This usually requires querying the Gamma API again or CLOB dict
        # For MVP, we assume the Signal contains the Asset ID or we fetch it.
        # In your 'elite_open_positions', you have 'asset' column which matches the outcome held.
        # We should probably pull that into the signal table to make this easier.
        return "TOKEN_ID_PLACEHOLDER" # You need to pass Asset ID in the signal

if __name__ == "__main__":
    asyncio.run(AutoExecutionEngine().run_execution_cycle())