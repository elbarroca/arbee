"""
Custom API Exceptions
"""
from fastapi import HTTPException
from typing import Optional


class APIException(HTTPException):
    """Base API exception"""
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


class TraderNotFoundError(APIException):
    """Trader not found exception"""
    def __init__(self, wallet_address: str):
        super().__init__(
            status_code=404,
            detail=f"Trader with wallet address {wallet_address} not found",
            error_code="TRADER_NOT_FOUND"
        )


class TraderAlreadyExistsError(APIException):
    """Trader already exists exception"""
    def __init__(self, wallet_address: str):
        super().__init__(
            status_code=400,
            detail=f"Trader with wallet address {wallet_address} already exists",
            error_code="TRADER_ALREADY_EXISTS"
        )


class TraderCriteriaNotMetError(APIException):
    """Trader does not meet criteria exception"""
    def __init__(self, reason: str):
        super().__init__(
            status_code=400,
            detail=f"Trader does not meet criteria: {reason}",
            error_code="TRADER_CRITERIA_NOT_MET"
        )


class MarketNotFoundError(APIException):
    """Market not found exception"""
    def __init__(self, market_slug: str):
        super().__init__(
            status_code=404,
            detail=f"Market {market_slug} not found",
            error_code="MARKET_NOT_FOUND"
        )


class TradeExecutionError(APIException):
    """Trade execution error"""
    def __init__(self, reason: str):
        super().__init__(
            status_code=400,
            detail=f"Trade execution failed: {reason}",
            error_code="TRADE_EXECUTION_ERROR"
        )


class WebhookProcessingError(APIException):
    """Webhook processing error"""
    def __init__(self, reason: str):
        super().__init__(
            status_code=400,
            detail=f"Webhook processing failed: {reason}",
            error_code="WEBHOOK_PROCESSING_ERROR"
        )

