#!/usr/bin/env python3
"""
Balance Monitor.
Scans the blockchain for real USDC balances and updates the database.
Run this every 1-5 minutes.
"""

import asyncio
import logging
import json
from web3 import Web3
from config.settings import Settings
from database.client import MarketDatabase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BalanceMonitor:
    def __init__(self):
        settings = Settings()
        self.db = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        
        # Polygon RPC (Use a paid one in production, public for testing)
        self.w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
        
        # USDC.e Contract on Polygon (Standard for Polymarket)
        self.USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        self.USDC_ABI = json.loads('[{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":false,"type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":false,"type":"function"}]')
        
        self.usdc_contract = self.w3.eth.contract(address=self.USDC_ADDRESS, abi=self.USDC_ABI)

    async def scan_and_update(self):
        """Fetch all active wallets and update their balances."""
        logger.info("🔍 Starting Balance Scan...")

        # 1. Get Wallets
        res = self.db.supabase.table("managed_wallets").select("*").eq("is_active", True).execute()
        wallets = res.data or []
        
        if not wallets:
            logger.info("   No active wallets to scan.")
            return

        updates = []
        for w in wallets:
            addr = w['proxy_wallet_address']
            
            try:
                # 2. Call Blockchain
                # Note: CheckSum address is required for Web3.py
                checksum_addr = Web3.to_checksum_address(addr)
                raw_balance = self.usdc_contract.functions.balanceOf(checksum_addr).call()
                
                # USDC has 6 decimals
                human_balance = raw_balance / 1_000_000
                
                # 3. Only update if changed
                db_balance = float(w.get('current_balance_usdc') or 0)
                
                if abs(human_balance - db_balance) > 0.01:
                    logger.info(f"   💰 Wallet {addr[:6]}... updated: ${db_balance} -> ${human_balance}")
                    updates.append({
                        "wallet_id": w['wallet_id'],
                        "current_balance_usdc": human_balance,
                        # If balance went up, assume deposit (simplified logic)
                        "total_deposited": float(w.get('total_deposited', 0)) + max(0, human_balance - db_balance)
                    })
                    
            except Exception as e:
                logger.error(f"   ❌ Error scanning {addr}: {e}")

        # 4. Batch Update DB
        if updates:
            for update in updates:
                self.db.supabase.table("managed_wallets").update(update).eq("wallet_id", update['wallet_id']).execute()
            logger.info(f"✅ Updated {len(updates)} wallet balances.")
        else:
            logger.info("   No balance changes detected.")

if __name__ == "__main__":
    asyncio.run(BalanceMonitor().scan_and_update())