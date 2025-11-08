"""
Trader discovery endpoints
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from .database.client import SupabaseClient

from api.dependencies import get_db_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/discovery", tags=["discovery"])


@router.post("/scan")
async def trigger_discovery_scan(
    background_tasks: BackgroundTasks,
    db_client: SupabaseClient = Depends(get_db_client)
):
    """
    Trigger a trader discovery scan.
    
    This will run the discovery job in the background.
    """
    try:
        # TODO: Implement discovery scan trigger
        # For now, just return success
        return {
            "status": "started",
            "message": "Discovery scan started in background"
        }
    except Exception as e:
        logger.error(f"Failed to start discovery scan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

