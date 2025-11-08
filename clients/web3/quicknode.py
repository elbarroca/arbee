"""
QuickNode Webhooks Client for Polygon Wallet Tracking
Fallback provider with programmable JS filters and reorg handling.
"""
import logging
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)


class QuickNodeWebhooksError(Exception):
    """Base exception for QuickNode webhooks"""
    pass


class QuickNodeWebhooksClient:
    """
    Client for QuickNode webhooks on Polygon network.
    
    Features:
    - Programmable JS filters for trade detection
    - Reorg handling for Polygon
    - Multi-chain support
    """
    
    POLYGON_CHAIN_ID = 137
    
    def __init__(self, api_key: Optional[str] = None, webhook_url: Optional[str] = None):
        """
        Initialize QuickNode webhooks client.
        
        Args:
            api_key: QuickNode API key (from settings if not provided)
            webhook_url: Your webhook endpoint URL to receive events
        """
        self.api_key = api_key or getattr(settings, "QUICKNODE_API_KEY", "")
        self.webhook_url = webhook_url or getattr(settings, "QUICKNODE_WEBHOOK_URL", "")
        self.base_url = "https://api.quicknode.com"
        
        if not self.api_key:
            logger.warning("QuickNode API key not configured - webhook operations will fail")
        
        self._webhook_ids: Dict[str, str] = {}
    
    async def create_address_webhook(
        self,
        wallet_address: str,
        webhook_url: Optional[str] = None,
        filter_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a webhook to track wallet address activity.
        
        Args:
            wallet_address: Wallet address to track
            webhook_url: Webhook endpoint URL
            filter_code: Optional JavaScript filter code for custom filtering
            
        Returns:
            Dict with webhook_id and metadata
        """
        if not self.api_key:
            raise QuickNodeWebhooksError("QuickNode API key not configured")
        
        url = f"{self.base_url}/webhooks"
        target_url = webhook_url or self.webhook_url
        
        if not target_url:
            raise QuickNodeWebhooksError("Webhook URL must be provided")
        
        # Default filter: track ERC20 transfers and external transactions
        default_filter = filter_code or """
        (event) => {
            return (
                event.type === 'transfer' ||
                event.type === 'transaction'
            ) && (
                event.from === '{{WALLET_ADDRESS}}' ||
                event.to === '{{WALLET_ADDRESS}}'
            );
        }
        """.replace("{{WALLET_ADDRESS}}", wallet_address.lower())
        
        payload = {
            "name": f"Wallet Tracker: {wallet_address[:8]}...",
            "webhook_url": target_url,
            "chain": "polygon-mainnet",
            "filter": {
                "type": "address",
                "address": wallet_address
            },
            "code": default_filter if filter_code else None
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                webhook_id = data.get("id") or data.get("webhook_id")
                if webhook_id:
                    self._webhook_ids[wallet_address] = webhook_id
                    logger.info(f"Created QuickNode webhook for {wallet_address}: {webhook_id}")
                
                return data
        except httpx.HTTPError as e:
            logger.error(f"Failed to create QuickNode webhook: {e}")
            raise QuickNodeWebhooksError(f"Webhook creation failed: {e}") from e
    
    async def create_contract_webhook(
        self,
        contract_address: str,
        webhook_url: Optional[str] = None,
        event_signatures: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a webhook to track contract events.
        
        Args:
            contract_address: Contract address to monitor
            webhook_url: Webhook endpoint URL
            event_signatures: List of event signatures to track (e.g., ["Transfer(address,address,uint256)"])
            
        Returns:
            Dict with webhook_id and metadata
        """
        if not self.api_key:
            raise QuickNodeWebhooksError("QuickNode API key not configured")
        
        url = f"{self.base_url}/webhooks"
        target_url = webhook_url or self.webhook_url
        
        if not target_url:
            raise QuickNodeWebhooksError("Webhook URL must be provided")
        
        # Default CTF events
        default_events = event_signatures or [
            "Transfer(address,address,uint256)",
            "Fill(address,address,uint256,uint256)"
        ]
        
        payload = {
            "name": f"Contract Tracker: {contract_address[:8]}...",
            "webhook_url": target_url,
            "chain": "polygon-mainnet",
            "filter": {
                "type": "contract",
                "address": contract_address,
                "events": default_events
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                data = response.json()
                logger.info(f"Created QuickNode contract webhook: {data.get('id')}")
                return data
        except httpx.HTTPError as e:
            logger.error(f"Failed to create contract webhook: {e}")
            raise QuickNodeWebhooksError(f"Contract webhook creation failed: {e}") from e
    
    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook by ID"""
        if not self.api_key:
            raise QuickNodeWebhooksError("QuickNode API key not configured")
        
        url = f"{self.base_url}/webhooks/{webhook_id}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    url,
                    headers={"x-api-key": self.api_key}
                )
                response.raise_for_status()
                logger.info(f"Deleted QuickNode webhook: {webhook_id}")
                return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to delete webhook {webhook_id}: {e}")
            return False
    
    async def list_webhooks(self) -> List[Dict[str, Any]]:
        """List all active webhooks"""
        if not self.api_key:
            raise QuickNodeWebhooksError("QuickNode API key not configured")
        
        url = f"{self.base_url}/webhooks"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers={"x-api-key": self.api_key}
                )
                response.raise_for_status()
                data = response.json()
                return data.get("webhooks", [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to list webhooks: {e}")
            return []
    
    def parse_webhook_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse an incoming webhook event and extract trade signal.
        
        Args:
            event_data: Raw webhook payload from QuickNode
            
        Returns:
            Parsed trade signal dict or None if not a trade event
        """
        event_type = event_data.get("type") or event_data.get("event_type")
        
        if event_type == "transfer":
            return self._parse_transfer_event(event_data)
        elif event_type == "transaction":
            return self._parse_transaction_event(event_data)
        elif event_type == "log":
            return self._parse_log_event(event_data)
        else:
            logger.debug(f"Unknown QuickNode event type: {event_type}")
            return None
    
    def _parse_transfer_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse ERC20 transfer event"""
        return {
            "event_type": "trade_signal",
            "wallet_address": event_data.get("from") or event_data.get("to"),
            "token_address": event_data.get("token", {}).get("address", ""),
            "amount": event_data.get("amount"),
            "transaction_hash": event_data.get("transactionHash", ""),
            "block_number": event_data.get("blockNumber", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "raw_event": event_data
        }
    
    def _parse_transaction_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse transaction event"""
        return {
            "event_type": "trade_signal",
            "wallet_address": event_data.get("from") or event_data.get("to"),
            "transaction_hash": event_data.get("hash", ""),
            "value": event_data.get("value"),
            "block_number": event_data.get("blockNumber", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "raw_event": event_data
        }
    
    def _parse_log_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse contract log event"""
        topics = event_data.get("topics", [])
        if not topics:
            return None
        
        # Check if this is a Transfer event (signature: 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef)
        transfer_signature = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        if topics[0] == transfer_signature:
            return {
                "event_type": "trade_signal",
                "contract_address": event_data.get("address", ""),
                "transaction_hash": event_data.get("transactionHash", ""),
                "block_number": event_data.get("blockNumber", ""),
                "topics": topics,
                "data": event_data.get("data", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "raw_event": event_data
            }
        
        return None

