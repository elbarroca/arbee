import os
from dotenv import load_dotenv

load_dotenv()

class ExecutionConfig:
    # Feature Flags
    PAPER_TRADING = True  # <--- SET TO TRUE FOR TESTING WITHOUT MONEY
    
    # Security
    MASTER_KEY = os.getenv("EXECUTION_MASTER_KEY")
    
    # Chain Settings
    CHAIN_ID = 137 # Polygon
    RPC_URL = "https://polygon-rpc.com"
    
    # Polymarket Contracts
    USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    CLOB_HOST = "https://clob.polymarket.com"

    # Risk
    GLOBAL_SLIPPAGE = 0.05 # 5%
    MAX_TRADE_USDC = 500.0

settings = ExecutionConfig()