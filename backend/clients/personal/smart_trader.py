#!/usr/bin/env python3
"""
Personal Smart Trader (The Hedge Fund Bot).
1. Authenticates via Private Key.
2. Auto-Approves USDC (One-time, requires MATIC).
3. Reads Signals & Rebalances Portfolio (Gasless Trading via CLOB).
"""

import asyncio
import logging
from datetime import datetime, timezone
import os
import json
from typing import Dict, List

from web3 import Web3
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from dotenv import load_dotenv

from config.settings import Settings
from database.client import MarketDatabase

# Load Env (Standard .env file)
load_dotenv()

logger = logging.getLogger("PersonalTrader")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PersonalTrader:
    def __init__(self):
        # 1. Setup Database
        self.settings = Settings()
        self.db = MarketDatabase(self.settings.SUPABASE_URL, self.settings.SUPABASE_KEY)
        
        # 2. Setup Wallet
        self.pk = os.getenv("PERSONAL_PRIVATE_KEY")
        if not self.pk:
            raise ValueError("❌ Missing PERSONAL_PRIVATE_KEY in .env")
        
        # 3. Constants
        self.host = "https://clob.polymarket.com"
        self.chain_id = 137 # Polygon
        self.usdc_address = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        self.poly_exchange = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
        
        # 4. Web3 Init (For Balance Checks & Approvals only)
        self.w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
        self.account = self.w3.eth.account.from_key(self.pk)
        self.address = self.account.address
        
        logger.info(f"🤖 Bot initialized for: {self.address}")

    async def run(self):
        """Main Rebalancing Loop."""
        logger.info("🔄 Starting Rebalancing Cycle...")

        # A. Check & Auto-Approve USDC (One-time setup)
        # This requires MATIC. If failed, we log error but try to trade anyway (might fail).
        await self._ensure_allowance()

        # B. Initialize CLOB Client (Gasless Trading)
        try:
            self.client = self._get_clob_client()
            # Update Creds (Important for first run to derive API keys)
            try:
                # Try to derive existing API credentials first
                creds = self.client.derive_api_key()
                logger.info("✅ Derived existing API credentials")
            except Exception:
                # If derivation fails, create new API credentials
                creds = self.client.create_api_key()
                logger.info("✅ Created new API credentials")
            logger.info("✅ Authenticated with Polymarket CLOB")
        except Exception as e:
            logger.error(f"❌ CLOB Auth Failed: {e}")
            return

        # C. Check USDC Balance (On Chain)
        balance = self._get_usdc_balance()
        logger.info(f"💰 Buying Power: ${balance:.2f} USDC")
        
        if balance < 5.0:
            logger.warning("⚠️ Balance too low (< $5.00). Fund your wallet with USDC.")
            return

        # D. Fetch Targets
        targets = await self._fetch_signals()
        if not targets:
            logger.info("😴 No active signals to trade.")
            return

        # E. Execute (Rebalance)
        await self._execute_rebalancing(targets, balance)

    # --- Setup Helpers ---

    async def _ensure_allowance(self) -> bool:
        """Checks if Polymarket is allowed to spend your USDC."""
        abi = json.loads('[{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"}]')
        contract = self.w3.eth.contract(address=self.usdc_address, abi=abi)
        
        try:
            allowance = contract.functions.allowance(self.address, self.poly_exchange).call()
            
            if allowance < 1_000_000 * 1_000_000:
                logger.info("⚙️ Approving USDC for Polymarket (One-time setup)...")
                
                # Check MATIC for gas
                matic = self.w3.eth.get_balance(self.address)
                if matic < 10**16: # 0.01 MATIC
                     logger.warning("   ❌ Not enough MATIC to approve USDC. Please send ~0.1 MATIC.")
                     return False

                tx = contract.functions.approve(
                    self.poly_exchange, 
                    115792089237316195423570985008687907853269984665640564039457584007913129639935
                ).build_transaction({
                    'chainId': 137,
                    'gas': 100000,
                    'gasPrice': self.w3.eth.gas_price,
                    'nonce': self.w3.eth.get_transaction_count(self.address)
                })
                
                signed_tx = self.w3.eth.account.sign_transaction(tx, self.pk)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                logger.info(f"   🚀 Approve Tx Sent: {self.w3.to_hex(tx_hash)}")
                logger.info("   ⏳ Waiting for confirmation...")
                self.w3.eth.wait_for_transaction_receipt(tx_hash)
                logger.info("   ✅ Approved!")
                return True
        except Exception as e:
            logger.error(f"⚠️ Approval Check Failed (Non-Critical if already done): {e}")
        return True

    # --- Trading Logic ---

    async def _execute_rebalancing(self, signals: List[Dict], total_balance: float):
        for sig in signals:
            try:
                market_title = sig['market_title']
                token_id = sig.get('asset_id')
                
                if not token_id:
                    logger.warning(f"   ⚠️ Skipping {market_title[:15]}: Missing Asset ID")
                    continue

                rec_pct = float(sig['recommended_bet_size_pct']) / 100.0
                target_size_usdc = total_balance * rec_pct
                
                # Cap Size
                target_size_usdc = min(target_size_usdc, 500.0) 

                if target_size_usdc < 1.0:
                    logger.info(f"   ⏭️  Skip {market_title[:15]}... (Size ${target_size_usdc:.2f} too small)")
                    continue

                # Check if we already bought this signal
                if await self._already_filled_today(sig['signal_id']):
                    logger.info(f"   ✅ Already filled {market_title[:15]} today.")
                    continue

                # Price Limits
                signal_price = float(sig['current_price'])
                limit_price = min(signal_price * 1.05, 0.99) # 5% Slippage
                
                # EXECUTE ORDER
                logger.info(f"   🚀 BUYING ${target_size_usdc:.2f} of '{market_title}'...")
                
                # Calculate Shares: Size ($) / Price = Shares
                size_shares = target_size_usdc / limit_price
                
                order_args = OrderArgs(
                    price=limit_price,
                    size=size_shares,
                    side="BUY",
                    token_id=token_id
                )
                
                resp = self.client.create_and_post_order(order_args)
                
                if resp and resp.get("orderID"):
                    logger.info(f"      🎉 Success! Order ID: {resp['orderID']}")
                    await self._log_fill(sig, target_size_usdc, resp['orderID'])
                else:
                    logger.warning(f"      ⚠️ Order Failed: {resp}")

            except Exception as e:
                logger.error(f"   ❌ Trade Error: {e}")

    # --- Utils ---

    def _get_clob_client(self) -> ClobClient:
        return ClobClient(self.host, key=self.pk, chain_id=self.chain_id, signature_type=1, funder=self.address)

    def _get_usdc_balance(self) -> float:
        usdc = self.w3.eth.contract(address=self.usdc_address, abi=[{"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"type":"function"}])
        return usdc.functions.balanceOf(self.address).call() / 1_000_000

    async def _fetch_signals(self) -> List[Dict]:
        # 1. Get Signals
        signals_res = self.db.supabase.table("copy_trading_signals").select("*").order("created_at", desc=True).limit(5).execute()
        signals = signals_res.data
        if not signals: return []

        # 2. Join Asset IDs (Token IDs) from positions
        pos_ids = [s['position_id'] for s in signals]
        pos_res = self.db.supabase.table("elite_open_positions").select("id, asset").in_("id", pos_ids).execute()
        asset_map = {p['id']: p['asset'] for p in pos_res.data}
        
        valid = []
        for s in signals:
            if s['position_id'] in asset_map:
                s['asset_id'] = asset_map[s['position_id']]
                valid.append(s)
        return valid

    async def _already_filled_today(self, signal_id):
        try:
            # Check if we logged this trade in our personal log table
            res = self.db.supabase.table("personal_trade_logs").select("id").eq("signal_id", signal_id).execute()
            return len(res.data) > 0
        except: return False

    async def _log_fill(self, signal, size, order_id):
        try:
            self.db.supabase.table("personal_trade_logs").insert({
                "signal_id": signal['signal_id'],
                "market_slug": signal['market_slug'],
                "size_usdc": size,
                "order_id": order_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }).execute()
        except: pass

if __name__ == "__main__":
    asyncio.run(PersonalTrader().run())