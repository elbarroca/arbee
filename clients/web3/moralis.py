"""
Moralis Streams Client for Polygon Wallet Tracking
Fast Polygon streams with easy webhook setup for multiple wallets.
"""
import logging
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)


class MoralisStreamsError(Exception):
    """Base exception for Moralis streams"""
    pass


class MoralisStreamsClient:
    """
    Client for Moralis streams on Polygon network.
    
    Features:
    - Fast Polygon streams
    - Easy webhook setup for multiple wallets
    - Trade event parsing
    """
    
    POLYGON_CHAIN_ID = 137
    
    def __init__(self, api_key: Optional[str] = None, webhook_url: Optional[str] = None):
        """
        Initialize Moralis streams client.
        
        Args:
            api_key: Moralis API key (from settings if not provided)
            webhook_url: Your webhook endpoint URL to receive events
        """
        self.api_key = api_key or getattr(settings, "MORALIS_API_KEY", "")
        self.webhook_url = webhook_url or getattr(settings, "MORALIS_WEBHOOK_URL", "")
        self.base_url = "https://api.moralis.io/api/v2"
        
        if not self.api_key:
            logger.warning("Moralis API key not configured - stream operations will fail")
        
        self._stream_ids: Dict[str, str] = {}
    
    async def create_address_stream(
        self,
        wallet_addresses: List[str],
        webhook_url: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a stream to track multiple wallet addresses.
        
        Args:
            wallet_addresses: List of wallet addresses to track
            webhook_url: Webhook endpoint URL
            description: Optional stream description
            
        Returns:
            Dict with stream_id and metadata
        """
        if not self.api_key:
            raise MoralisStreamsError("Moralis API key not configured")
        
        url = f"{self.base_url}/streams"
        target_url = webhook_url or self.webhook_url
        
        if not target_url:
            raise MoralisStreamsError("Webhook URL must be provided")
        
        # Normalize addresses (checksum)
        normalized_addresses = [addr.lower() for addr in wallet_addresses]
        
        payload = {
            "webhookUrl": target_url,
            "description": description or f"Wallet tracker for {len(wallet_addresses)} addresses",
            "tag": "polymarket_copy_trading",
            "chains": [f"0x{self.POLYGON_CHAIN_ID:x}"],  # 0x89 for Polygon
            "includeNativeTxs": True,
            "includeContractLogs": True,
            "includeInternalTxs": True,
            "address": normalized_addresses  # Moralis supports multiple addresses
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "X-API-Key": self.api_key,
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                stream_id = data.get("id") or data.get("streamId")
                if stream_id:
                    for addr in wallet_addresses:
                        self._stream_ids[addr.lower()] = stream_id
                    logger.info(f"Created Moralis stream for {len(wallet_addresses)} addresses: {stream_id}")
                
                return data
        except httpx.HTTPError as e:
            logger.error(f"Failed to create Moralis stream: {e}")
            raise MoralisStreamsError(f"Stream creation failed: {e}") from e
    
    async def add_addresses_to_stream(
        self,
        stream_id: str,
        wallet_addresses: List[str]
    ) -> Dict[str, Any]:
        """
        Add addresses to an existing stream.
        
        Args:
            stream_id: Existing stream ID
            wallet_addresses: List of addresses to add
            
        Returns:
            Updated stream data
        """
        if not self.api_key:
            raise MoralisStreamsError("Moralis API key not configured")
        
        url = f"{self.base_url}/streams/{stream_id}/address"
        
        normalized_addresses = [addr.lower() for addr in wallet_addresses]
        payload = {
            "address": normalized_addresses
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "X-API-Key": self.api_key,
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                data = response.json()
                logger.info(f"Added {len(wallet_addresses)} addresses to stream {stream_id}")
                return data
        except httpx.HTTPError as e:
            logger.error(f"Failed to add addresses to stream: {e}")
            raise MoralisStreamsError(f"Add addresses failed: {e}") from e
    
    async def delete_stream(self, stream_id: str) -> bool:
        """Delete a stream by ID"""
        if not self.api_key:
            raise MoralisStreamsError("Moralis API key not configured")
        
        url = f"{self.base_url}/streams/{stream_id}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    url,
                    headers={"X-API-Key": self.api_key}
                )
                response.raise_for_status()
                logger.info(f"Deleted Moralis stream: {stream_id}")
                return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to delete stream {stream_id}: {e}")
            return False
    
    async def list_streams(self) -> List[Dict[str, Any]]:
        """List all active streams"""
        if not self.api_key:
            raise MoralisStreamsError("Moralis API key not configured")
        
        url = f"{self.base_url}/streams"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    url,
                    headers={"X-API-Key": self.api_key}
                )
                response.raise_for_status()
                data = response.json()
                return data.get("result", [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to list streams: {e}")
            return []
    
    def parse_webhook_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse an incoming webhook event and extract trade signal.
        
        Args:
            event_data: Raw webhook payload from Moralis
            
        Returns:
            Parsed trade signal dict or None if not a trade event
        """
        # Moralis sends events in different formats
        if "erc20Transfers" in event_data:
            return self._parse_erc20_transfers(event_data)
        elif "txs" in event_data:
            return self._parse_transactions(event_data)
        elif "logs" in event_data:
            return self._parse_logs(event_data)
        else:
            # Try direct event parsing
            return self._parse_direct_event(event_data)
    
    def _parse_erc20_transfers(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse ERC20 transfer events"""
        transfers = event_data.get("erc20Transfers", [])
        if not transfers:
            return None
        
        # Take the first transfer (can be enhanced to aggregate)
        transfer = transfers[0]
        return {
            "event_type": "trade_signal",
            "wallet_address": transfer.get("fromAddress") or transfer.get("toAddress"),
            "token_address": transfer.get("tokenAddress", ""),
            "amount": transfer.get("value"),
            "transaction_hash": transfer.get("transactionHash", ""),
            "block_number": transfer.get("blockNumber", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "raw_event": transfer
        }
    
    def _parse_transactions(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse transaction events"""
        txs = event_data.get("txs", [])
        if not txs:
            return None
        
        tx = txs[0]
        return {
            "event_type": "trade_signal",
            "wallet_address": tx.get("fromAddress") or tx.get("toAddress"),
            "transaction_hash": tx.get("hash", ""),
            "value": tx.get("value"),
            "block_number": tx.get("blockNumber", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "raw_event": tx
        }
    
    def _parse_logs(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse contract log events"""
        logs = event_data.get("logs", [])
        if not logs:
            return None
        
        log = logs[0]
        topics = log.get("topic", [])
        
        # Transfer event signature
        transfer_signature = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        if topics and topics[0] == transfer_signature:
            return {
                "event_type": "trade_signal",
                "contract_address": log.get("address", ""),
                "transaction_hash": log.get("transactionHash", ""),
                "block_number": log.get("blockNumber", ""),
                "topics": topics,
                "data": log.get("data", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "raw_event": log
            }
        
        return None
    
    def _parse_direct_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse direct event format"""
        if "transactionHash" in event_data:
            return {
                "event_type": "trade_signal",
                "wallet_address": event_data.get("fromAddress") or event_data.get("toAddress"),
                "transaction_hash": event_data.get("transactionHash", ""),
                "block_number": event_data.get("blockNumber", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "raw_event": event_data
            }
        return None

