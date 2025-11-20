import uvicorn
from fastapi import FastAPI
from config.settings import settings
import logging

# Import Routes
from api.market_data import market_data
from api.wallets import wallets
from api.analysis import analysis

# Configure Logging (Global)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="POLYSEER API Node",
    version="2.0.0",
    description="Async Data Pipeline for Prediction Markets"
)

# Register Routers
app.include_router(market_data.router, prefix="/api/v1/markets", tags=["Markets"])
app.include_router(wallets.router, prefix="/api/v1/wallets", tags=["Wallets"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])

@app.get("/")
def root():
    return {"system": "POLYSEER", "status": "operational"}

if __name__ == "__main__":
    # Run with reload enabled for development
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )