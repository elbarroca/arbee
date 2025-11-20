#!/usr/bin/env python3
"""
User Onboarding Script.
1. Checks if Wallet has MATIC for gas.
2. Approves USDC spending for Polymarket Exchange (CTF).
3. Enables Proxy (if required).
"""

import asyncio
import logging
import json
import time
from web3 import Web3
from eth_account import Account
from config.settings import Settings
from database.client import MarketDatabase
from .security import vault

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OnboardingAgent:
    def __init__(self):
        settings = Settings()
        self.db = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        self.w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
        
        # Contracts
        self.USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
        # Polymarket CTF Exchange Address (The contract that needs permission to spend user's money)
        self.POLY_EXCHANGE = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
        
        self.ERC20_ABI = json.loads('[{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"}]')

    async def onboard_user(self, wallet_id: str):
        """Prepares a wallet for trading."""
        # 1. Fetch Wallet Credentials
        res = self.db.supabase.table("managed_wallets").select("*").eq("wallet_id", wallet_id).execute()
        if not res.data:
            logger.error("Wallet not found.")
            return

        w_data = res.data[0]
        address = Web3.to_checksum_address(w_data['proxy_wallet_address'])
        private_key = vault.decrypt_private_key(w_data['encrypted_private_key'])
        
        logger.info(f"🔧 Onboarding {address}...")

        # 2. Check Gas (MATIC)
        matic_balance = self.w3.eth.get_balance(address)
        matic_readable = self.w3.from_wei(matic_balance, 'ether')
        
        if matic_readable < 0.01:
            logger.warning(f"   ❌ Insufficient Gas! Has {matic_readable:.4f} MATIC. Send MATIC to this address first.")
            return False

        logger.info(f"   ✅ Gas OK: {matic_readable:.4f} MATIC")

        # 3. Check Allowance
        contract = self.w3.eth.contract(address=self.USDC_ADDRESS, abi=self.ERC20_ABI)
        allowance = contract.functions.allowance(address, self.POLY_EXCHANGE).call()
        
        if allowance > 1_000_000 * 1_000_000: # > 1M USDC approved
            logger.info("   ✅ USDC Already Approved.")
            return True

        # 4. Execute Approve Transaction
        logger.info("   ⚙️ Approving USDC for Polymarket Exchange...")
        
        try:
            # Build Tx
            nonce = self.w3.eth.get_transaction_count(address)
            # Max Uint256
            max_amount = 115792089237316195423570985008687907853269984665640564039457584007913129639935
            
            tx = contract.functions.approve(self.POLY_EXCHANGE, max_amount).build_transaction({
                'chainId': 137,
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
            })
            
            # Sign & Send
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            logger.info(f"   🚀 Approve Tx Sent: {self.w3.to_hex(tx_hash)}")
            logger.info("   Waiting for confirmation...")
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            logger.info("   ✅ Wallet Successfully Onboarded!")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Approval Failed: {e}")
            return False

if __name__ == "__main__":
    # Example usage: onboard the specific wallet you created
    # You need to get the UUID from your database
    import sys
    if len(sys.argv) > 1:
        wid = sys.argv[1]
        asyncio.run(OnboardingAgent().onboard_user(wid))
    else:
        print("Usage: python -m clients.execution.onboarding <WALLET_UUID>")