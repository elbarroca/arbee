import secrets
import logging
from eth_account import Account
from database.client import MarketDatabase
from config.settings import Settings as AppSettings
from .security import vault

logger = logging.getLogger(__name__)

class WalletManager:
    def __init__(self):
        s = AppSettings()
        self.db = MarketDatabase(s.SUPABASE_URL, s.SUPABASE_KEY)

    async def create_wallet(self, user_id: str):
        """Creates and stores an encrypted wallet."""
        # 1. Check existing
        res = self.db.supabase.table("managed_wallets").select("wallet_id").eq("user_id", user_id).execute()
        if res.data:
            logger.info(f"User {user_id} already has a wallet.")
            return res.data[0]

        # 2. Generate
        priv = "0x" + secrets.token_hex(32)
        acct = Account.from_key(priv)
        
        # 3. Encrypt
        enc_key = vault.encrypt(priv)
        
        # 4. Mock API Keys (In production, you perform a signature exchange with Poly to get these)
        api_key = secrets.token_urlsafe(16)
        api_secret = secrets.token_urlsafe(32)
        passphrase = secrets.token_urlsafe(12)

        # 5. Store
        data = {
            "user_id": user_id,
            "proxy_wallet_address": acct.address,
            "encrypted_private_key": enc_key,
            "api_key": api_key,
            "api_secret": api_secret,
            "passphrase": passphrase,
            "is_active": True,
            "current_balance_usdc": 0, # Start at 0
            "risk_multiplier": 1.0
        }
        
        final = self.db.supabase.table("managed_wallets").insert(data).execute()
        logger.info(f"✅ Created wallet {acct.address} for user {user_id}")
        return final.data[0]