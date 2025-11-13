"""
Pydantic schemas for webhook endpoints
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class WebhookEvent(BaseModel):
    """Generic webhook event payload"""
    type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None


class AlchemyWebhook(BaseModel):
    """Alchemy webhook payload structure"""
    webhookId: Optional[str] = None
    id: Optional[str] = None
    createdAt: Optional[str] = None
    type: Optional[str] = None
    event: Dict[str, Any] = Field(default_factory=dict)


class QuickNodeWebhook(BaseModel):
    """QuickNode webhook payload structure"""
    events: List[Dict[str, Any]] = Field(default_factory=list)
    subscription: Optional[str] = None
    timestamp: Optional[str] = None


class MoralisWebhook(BaseModel):
    """Moralis webhook payload structure"""
    tag: Optional[str] = None
    streamId: Optional[str] = None
    confirmed: Optional[bool] = None
    erc20Transfers: List[Dict[str, Any]] = Field(default_factory=list)
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    txs: List[Dict[str, Any]] = Field(default_factory=list)
    abi: Optional[Dict[str, Any]] = None


class WebhookProcessingResponse(BaseModel):
    """Webhook processing response"""
    status: str = Field(..., description="success, skipped, filtered, error")
    action: str = Field(..., description="processed, ignored, logged_dry_run, etc.")
    reason: Optional[str] = None
    trader: Optional[str] = None
    market: Optional[str] = None
    side: Optional[str] = None
    size_usd: Optional[float] = None










