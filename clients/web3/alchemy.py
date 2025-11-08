"""
Alchemy Webhooks Client for Polygon Wallet Tracking
Subscribes to wallet addresses and CTF contract events for copy trading.
"""
import logging
import httpx
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)


class AlchemyWebhooksError(Exception):
    """Base exception for Alchemy webhooks"""
    pass


class AlchemyWebhooksClient:
    """
    Client for Alchemy webhooks on Polygon network.
    
    Supports:
    - Address activity webhooks (wallet tracking)
    - CTF (Conditional Token Framework) contract events
    - Trade event parsing and validation
    """
    
    # Polygon CTF contract addresses (Polymarket)
    POLYGON_CTF_EXCHANGE = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"  # Main CTF Exchange
    POLYGON_NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # Neg Risk CTF (multi-outcome)
    POLYGON_USDC = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"  # USDC.e (bridged)
    POLYGON_CHAIN_ID = 137

    # Event signatures for trade detection
    TRANSFER_SIGNATURE = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
    
    def __init__(self, api_key: Optional[str] = None, webhook_url: Optional[str] = None):
        """
        Initialize Alchemy webhooks client.
        
        Args:
            api_key: Alchemy API key (from settings if not provided)
            webhook_url: Your webhook endpoint URL to receive events
        """
        self.api_key = api_key or getattr(settings, "ALCHEMY_API_KEY", "")
        self.webhook_url = webhook_url or getattr(settings, "ALCHEMY_WEBHOOK_URL", "")
        self.base_url = f"https://polygon-mainnet.g.alchemy.com/v2/{self.api_key}"
        
        if not self.api_key:
            logger.warning("Alchemy API key not configured - webhook operations will fail")
        
        self._webhook_ids: Dict[str, str] = {}  # wallet_address -> webhook_id
    
    async def create_address_activity_webhook(
        self,
        wallet_address: str,
        webhook_url: Optional[str] = None,
        app_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a webhook to track address activity for a wallet.
        
        Args:
            wallet_address: Wallet address to track (checksummed)
            webhook_url: Webhook endpoint URL (uses instance default if not provided)
            app_id: Optional Alchemy app ID
            
        Returns:
            Dict with webhook_id and other metadata
        """
        if not self.api_key:
            raise AlchemyWebhooksError("Alchemy API key not configured")
        
        url = f"{self.base_url}/webhooks"
        target_url = webhook_url or self.webhook_url
        
        if not target_url:
            raise AlchemyWebhooksError("Webhook URL must be provided")
        
        # Alchemy webhook payload structure
        payload = {
            "webhook_type": "ADDRESS_ACTIVITY",
            "app_id": app_id or "",
            "webhook_url": target_url,
            "addresses": [wallet_address],
            "network": "POLYGON_MAINNET"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
                webhook_id = data.get("id") or data.get("webhook_id")
                if webhook_id:
                    self._webhook_ids[wallet_address] = webhook_id
                    logger.info(f"Created Alchemy webhook for {wallet_address}: {webhook_id}")
                
                return data
        except httpx.HTTPError as e:
            logger.error(f"Failed to create Alchemy webhook: {e}")
            raise AlchemyWebhooksError(f"Webhook creation failed: {e}") from e
    
    async def create_ctf_contract_webhook(
        self,
        contract_address: Optional[str] = None,
        webhook_url: Optional[str] = None,
        app_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a webhook to track CTF contract events (token transfers, fills).
        
        Args:
            contract_address: CTF contract address (uses default if not provided)
            webhook_url: Webhook endpoint URL
            app_id: Optional Alchemy app ID
            
        Returns:
            Dict with webhook_id and metadata
        """
        if not self.api_key:
            raise AlchemyWebhooksError("Alchemy API key not configured")
        
        url = f"{self.base_url}/webhooks"
        target_url = webhook_url or self.webhook_url
        contract = contract_address or self.POLYGON_CTF_EXCHANGE
        
        if not target_url:
            raise AlchemyWebhooksError("Webhook URL must be provided")
        
        payload = {
            "webhook_type": "CONTRACT_ACTIVITY",
            "app_id": app_id or "",
            "webhook_url": target_url,
            "contract_address": contract,
            "network": "POLYGON_MAINNET"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
                logger.info(f"Created Alchemy CTF contract webhook: {data.get('id')}")
                return data
        except httpx.HTTPError as e:
            logger.error(f"Failed to create CTF webhook: {e}")
            raise AlchemyWebhooksError(f"CTF webhook creation failed: {e}") from e
    
    async def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete a webhook by ID.
        
        Args:
            webhook_id: Webhook ID to delete
            
        Returns:
            True if successful
        """
        if not self.api_key:
            raise AlchemyWebhooksError("Alchemy API key not configured")
        
        url = f"{self.base_url}/webhooks/{webhook_id}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(url)
                response.raise_for_status()
                logger.info(f"Deleted Alchemy webhook: {webhook_id}")
                return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to delete webhook {webhook_id}: {e}")
            return False
    
    async def list_webhooks(self) -> List[Dict[str, Any]]:
        """
        List all active webhooks.
        
        Returns:
            List of webhook dicts
        """
        if not self.api_key:
            raise AlchemyWebhooksError("Alchemy API key not configured")
        
        url = f"{self.base_url}/webhooks"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to list webhooks: {e}")
            return []
    
    def parse_webhook_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse an incoming webhook event and extract trade signal.
        
        Args:
            event_data: Raw webhook payload from Alchemy
            
        Returns:
            Parsed trade signal dict or None if not a trade event
        """
        event_type = event_data.get("type") or event_data.get("event_type")
        
        if event_type == "ADDRESS_ACTIVITY":
            return self._parse_address_activity(event_data)
        elif event_type == "CONTRACT_ACTIVITY":
            return self._parse_contract_activity(event_data)
        else:
            logger.debug(f"Unknown webhook event type: {event_type}")
            return None
    
    def _parse_address_activity(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse address activity event into trade signal"""
        activity = event_data.get("activity", [])
        if not activity:
            return None
        
        # Look for token transfers (CTF trades)
        for item in activity:
            if item.get("category") == "token" and item.get("from") and item.get("to"):
                # Check if this is a CTF token transfer
                token_address = item.get("rawContract", {}).get("address", "")
                if "0x" in token_address.lower():  # Basic check - enhance with actual CTF detection
                    return {
                        "event_type": "trade_signal",
                        "wallet_address": item.get("from") or item.get("to"),
                        "token_address": token_address,
                        "transaction_hash": item.get("hash", ""),
                        "block_number": item.get("blockNum", ""),
                        "timestamp": datetime.utcnow().isoformat(),
                        "raw_event": item
                    }
        
        return None
    
    def _parse_contract_activity(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse contract activity event into trade signal"""
        events = event_data.get("events", [])
        if not events:
            return None
        
        # Look for Transfer events or Fill events
        for event in events:
            event_name = event.get("eventName", "")
            if event_name in ["Transfer", "Fill", "OrderFilled"]:
                return {
                    "event_type": "trade_signal",
                    "contract_address": event.get("contractAddress", ""),
                    "transaction_hash": event.get("transactionHash", ""),
                    "block_number": event.get("blockNumber", ""),
                    "event_name": event_name,
                    "event_data": event.get("eventData", {}),
                    "timestamp": datetime.utcnow().isoformat(),
                    "raw_event": event
                }
        
        return None
    
    async def get_address_transactions(
        self,
        wallet_address: str,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent transactions for an address (fallback if webhooks unavailable).
        
        Args:
            wallet_address: Wallet address to query
            from_block: Starting block number
            to_block: Ending block number
            limit: Maximum number of transactions
            
        Returns:
            List of transaction dicts
        """
        if not self.api_key:
            raise AlchemyWebhooksError("Alchemy API key not configured")
        
        url = f"{self.base_url}/getAssetTransfers"
        params = {
            "fromAddress": wallet_address,
            "category": ["erc20", "external"],
            "maxCount": str(limit),
            "excludeZeroValue": "true"
        }
        
        if from_block:
            params["fromBlock"] = str(from_block)
        if to_block:
            params["toBlock"] = str(to_block)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                return data.get("transfers", [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to get address transactions: {e}")
            return []

