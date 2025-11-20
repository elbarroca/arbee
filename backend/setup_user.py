# setup_user.py

import asyncio
import uuid

from execution.manager import WalletManager



async def main():

    mgr = WalletManager()

    # Create a wallet for a fake user ID (UUID format)

    user_id = str(uuid.uuid4())

    wallet = await mgr.create_managed_wallet_for_user(user_id)

    print(f"Created Wallet: {wallet['proxy_wallet_address']}")

    print(f"User ID: {user_id}")

    print("Please send some MATIC/USDC to this address on Polygon to test.")



if __name__ == "__main__":

    asyncio.run(main())
