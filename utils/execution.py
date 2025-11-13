"""
Execution Manager

Handles trade execution with hybrid approval workflow:
- Auto-execute small bets within risk limits
- Require manual approval for large bets or low-confidence trades
- Track trade execution status and errors
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TradeStatus(str, Enum):
    """Trade status enum"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class TradeProposal:
    """Proposed trade"""
    provider: str
    market_id: str
    market_slug: str
    side: str  # YES, NO, BUY, SELL
    size: Decimal
    expected_price: Decimal
    confidence_score: int  # 0-100
    agent_reasoning: str
    alternative_scenarios: List[Dict]
    edge: Optional[Decimal] = None
    kelly_fraction: Optional[Decimal] = None


@dataclass
class TradeExecution:
    """Executed trade result"""
    trade_id: int
    status: TradeStatus
    size: Decimal
    fill_price: Optional[Decimal]
    slippage_bps: Optional[int]
    fee: Decimal
    total_cost: Decimal
    error_message: Optional[str]
    executed_at: Optional[datetime]


class ExecutionManager:
    """
    Manages trade execution with hybrid approval workflow.

    Features:
    - Auto-execute trades within risk/size limits
    - Require approval for large bets or low-confidence trades
    - Track pending approvals with expiry
    - Execute trades via provider APIs
    - Log all execution attempts and outcomes
    """

    def __init__(
        self,
        db_client,
        bankroll_manager,
        risk_manager,
        *,
        auto_execute_threshold: Decimal = Decimal("100.0"),
        min_confidence_for_auto: int = 75,
        approval_expiry_hours: int = 24
    ):
        """
        Initialize execution manager.

        Args:
            db_client: Database client instance
            bankroll_manager: BankrollManager instance
            risk_manager: RiskManager instance
            auto_execute_threshold: Max bet size for auto-execution (default: $100)
            min_confidence_for_auto: Min confidence for auto-execution (default: 75)
            approval_expiry_hours: Hours before pending approvals expire (default: 24)
        """
        self.db = db_client
        self.bankroll = bankroll_manager
        self.risk = risk_manager
        self.auto_execute_threshold = auto_execute_threshold
        self.min_confidence_for_auto = min_confidence_for_auto
        self.approval_expiry_hours = approval_expiry_hours
        self.logger = logger

        self.logger.info(
            f"ExecutionManager initialized: "
            f"auto_threshold=${auto_execute_threshold}, "
            f"min_confidence={min_confidence_for_auto}"
        )

    async def propose_trade(
        self,
        proposal: TradeProposal,
        position_id: Optional[int] = None
    ) -> TradeExecution:
        """
        Propose a trade for execution (with approval workflow).

        Args:
            proposal: TradeProposal object
            position_id: Optional position ID if adding to existing position

        Returns:
            TradeExecution result
        """
        # Validate bet against risk limits
        validation = await self.risk.validate_bet(
            bet_size=proposal.size,
            market_id=proposal.market_id,
            provider=proposal.provider
        )

        # Check if trade requires approval
        requires_approval = self._requires_approval(proposal, validation)

        # Create trade record
        trade_id = await self._create_trade_record(
            proposal=proposal,
            position_id=position_id,
            requires_approval=requires_approval,
            validation=validation
        )

        if not validation.is_valid:
            # Trade violates risk limits - reject
            await self._update_trade_status(
                trade_id=trade_id,
                status=TradeStatus.REJECTED,
                error_message=f"Risk violations: {', '.join([v.message for v in validation.violations])}"
            )

            return TradeExecution(
                trade_id=trade_id,
                status=TradeStatus.REJECTED,
                size=proposal.size,
                fill_price=None,
                slippage_bps=None,
                fee=Decimal("0.0"),
                total_cost=Decimal("0.0"),
                error_message="Risk limit violations",
                executed_at=None
            )

        if requires_approval:
            # Requires manual approval - mark as pending
            self.logger.info(
                f"Trade #{trade_id} requires manual approval: "
                f"size=${proposal.size}, confidence={proposal.confidence_score}"
            )

            return TradeExecution(
                trade_id=trade_id,
                status=TradeStatus.PENDING,
                size=proposal.size,
                fill_price=None,
                slippage_bps=None,
                fee=Decimal("0.0"),
                total_cost=Decimal("0.0"),
                error_message=None,
                executed_at=None
            )

        else:
            # Auto-execute
            return await self.execute_trade(trade_id)

    async def execute_trade(self, trade_id: int) -> TradeExecution:
        """
        Execute a trade (approved or auto-execute).

        Args:
            trade_id: Trade ID to execute

        Returns:
            TradeExecution result
        """
        # Get trade details
        trade = await self._get_trade(trade_id)

        if not trade:
            raise ValueError(f"Trade {trade_id} not found")

        if trade['status'] not in (TradeStatus.PENDING.value, TradeStatus.APPROVED.value):
            raise ValueError(f"Trade {trade_id} cannot be executed (status: {trade['status']})")

        try:
            # TODO: Implement actual trade execution via provider APIs
            # For now, simulate execution
            fill_price = Decimal(str(trade['expected_price']))
            slippage_bps = 20  # 20 basis points slippage
            fee = Decimal(str(trade['size'])) * Decimal("0.002")  # 0.2% fee
            total_cost = Decimal(str(trade['size'])) * fill_price + fee

            # Update trade record
            await self.db.execute(
                """
                UPDATE trades
                SET status = $1,
                    execution_timestamp = NOW(),
                    price = $2,
                    slippage_bps = $3,
                    fee = $4,
                    total_cost = $5,
                    updated_at = NOW()
                WHERE id = $6
                """,
                TradeStatus.EXECUTED.value,
                fill_price,
                slippage_bps,
                fee,
                total_cost,
                trade_id
            )

            # Create or update position
            position_id = trade['position_id']
            if position_id:
                # Add to existing position
                await self._add_to_position(position_id, trade, fill_price, fee)
            else:
                # Create new position
                position_id = await self._create_position(trade, fill_price, fee)

            # Log to bankroll history
            await self.bankroll.log_bankroll_snapshot(
                note=f"Executed trade #{trade_id}: {trade['market_slug']}"
            )

            self.logger.info(
                f"Trade #{trade_id} executed: {trade['market_slug']} "
                f"size=${trade['size']}, fill=${fill_price}"
            )

            return TradeExecution(
                trade_id=trade_id,
                status=TradeStatus.EXECUTED,
                size=Decimal(str(trade['size'])),
                fill_price=fill_price,
                slippage_bps=slippage_bps,
                fee=fee,
                total_cost=total_cost,
                error_message=None,
                executed_at=datetime.utcnow()
            )

        except Exception as e:
            self.logger.error(f"Trade #{trade_id} execution failed: {e}")

            # Update trade status to failed
            await self._update_trade_status(
                trade_id=trade_id,
                status=TradeStatus.FAILED,
                error_message=str(e)
            )

            return TradeExecution(
                trade_id=trade_id,
                status=TradeStatus.FAILED,
                size=Decimal(str(trade['size'])),
                fill_price=None,
                slippage_bps=None,
                fee=Decimal("0.0"),
                total_cost=Decimal("0.0"),
                error_message=str(e),
                executed_at=None
            )

    async def approve_trade(
        self,
        trade_id: int,
        approved_by: str,
        execute_immediately: bool = True
    ) -> TradeExecution:
        """
        Approve a pending trade.

        Args:
            trade_id: Trade ID to approve
            approved_by: Username/ID of approver
            execute_immediately: Execute immediately after approval (default: True)

        Returns:
            TradeExecution result
        """
        trade = await self._get_trade(trade_id)

        if not trade or trade['status'] != TradeStatus.PENDING.value:
            raise ValueError(f"Trade {trade_id} not found or not pending")

        # Update trade to approved
        await self.db.execute(
            """
            UPDATE trades
            SET status = $1,
                approved_by = $2,
                approved_at = NOW(),
                updated_at = NOW()
            WHERE id = $3
            """,
            TradeStatus.APPROVED.value,
            approved_by,
            trade_id
        )

        self.logger.info(f"Trade #{trade_id} approved by {approved_by}")

        if execute_immediately:
            return await self.execute_trade(trade_id)
        else:
            return TradeExecution(
                trade_id=trade_id,
                status=TradeStatus.APPROVED,
                size=Decimal(str(trade['size'])),
                fill_price=None,
                slippage_bps=None,
                fee=Decimal("0.0"),
                total_cost=Decimal("0.0"),
                error_message=None,
                executed_at=None
            )

    async def reject_trade(
        self,
        trade_id: int,
        rejection_reason: str
    ) -> None:
        """
        Reject a pending trade.

        Args:
            trade_id: Trade ID to reject
            rejection_reason: Reason for rejection
        """
        await self.db.execute(
            """
            UPDATE trades
            SET status = $1,
                rejection_reason = $2,
                updated_at = NOW()
            WHERE id = $3
            """,
            TradeStatus.REJECTED.value,
            rejection_reason,
            trade_id
        )

        self.logger.info(f"Trade #{trade_id} rejected: {rejection_reason}")

    async def get_pending_approvals(self) -> List[Dict]:
        """
        Get all pending trade approvals.

        Returns:
            List of pending trades
        """
        query = """
            SELECT
                id, provider, market_id, market_slug, side, size,
                expected_price, confidence_score, agent_reasoning,
                alternative_scenarios, created_at
            FROM trades
            WHERE status = $1
                AND requires_approval = TRUE
                AND created_at > NOW() - INTERVAL '1 day' * $2
            ORDER BY created_at ASC
        """

        rows = await self.db.fetch(query, TradeStatus.PENDING.value, self.approval_expiry_hours / 24)

        return [
            {
                "trade_id": row['id'],
                "provider": row['provider'],
                "market_id": row['market_id'],
                "market_slug": row['market_slug'],
                "side": row['side'],
                "size": Decimal(str(row['size'])),
                "expected_price": Decimal(str(row['expected_price'])),
                "confidence_score": row['confidence_score'],
                "agent_reasoning": row['agent_reasoning'],
                "alternative_scenarios": row['alternative_scenarios'],
                "created_at": row['created_at'],
                "age_hours": (datetime.utcnow() - row['created_at']).total_seconds() / 3600
            }
            for row in rows
        ]

    async def expire_pending_approvals(self) -> int:
        """
        Expire pending approvals older than expiry threshold.

        Returns:
            Number of approvals expired
        """
        expiry_cutoff = datetime.utcnow() - timedelta(hours=self.approval_expiry_hours)

        result = await self.db.execute(
            """
            UPDATE trades
            SET status = $1,
                rejection_reason = 'Approval expired',
                updated_at = NOW()
            WHERE status = $2
                AND requires_approval = TRUE
                AND created_at < $3
            """,
            TradeStatus.REJECTED.value,
            TradeStatus.PENDING.value,
            expiry_cutoff
        )

        # Extract number of rows from result (format depends on DB driver)
        expired_count = int(result.split()[-1]) if isinstance(result, str) else 0

        if expired_count > 0:
            self.logger.info(f"Expired {expired_count} pending approvals")

        return expired_count

    def _requires_approval(
        self,
        proposal: TradeProposal,
        validation
    ) -> bool:
        """Check if trade requires manual approval"""
        # Require approval if:
        # 1. Bet size >= threshold
        if proposal.size >= self.auto_execute_threshold:
            return True

        # 2. Low confidence
        if proposal.confidence_score < self.min_confidence_for_auto:
            return True

        # 3. Risk validation requires it
        if validation.requires_approval:
            return True

        # 4. Edge is low or negative
        if proposal.edge is not None and proposal.edge < Decimal("0.03"):  # < 3% edge
            return True

        return False

    async def _create_trade_record(
        self,
        proposal: TradeProposal,
        position_id: Optional[int],
        requires_approval: bool,
        validation
    ) -> int:
        """Create trade record in database"""
        query = """
            INSERT INTO trades (
                position_id,
                provider,
                market_id,
                market_slug,
                trade_type,
                side,
                size,
                expected_price,
                requires_approval,
                agent_reasoning,
                confidence_score,
                alternative_scenarios,
                status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
        """

        trade_id = await self.db.fetchval(
            query,
            position_id,
            proposal.provider,
            proposal.market_id,
            proposal.market_slug,
            "entry" if not position_id else "rebalance",
            proposal.side,
            proposal.size,
            proposal.expected_price,
            requires_approval,
            proposal.agent_reasoning,
            proposal.confidence_score,
            json.dumps(proposal.alternative_scenarios) if proposal.alternative_scenarios else '[]',
            TradeStatus.PENDING.value
        )

        return trade_id

    async def _get_trade(self, trade_id: int) -> Optional[Dict]:
        """Get trade by ID"""
        query = """
            SELECT *
            FROM trades
            WHERE id = $1
        """

        row = await self.db.fetchrow(query, trade_id)
        return dict(row) if row else None

    async def _update_trade_status(
        self,
        trade_id: int,
        status: TradeStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update trade status"""
        await self.db.execute(
            """
            UPDATE trades
            SET status = $1,
                error_message = $2,
                updated_at = NOW()
            WHERE id = $3
            """,
            status.value,
            error_message,
            trade_id
        )

    async def _create_position(
        self,
        trade: Dict,
        fill_price: Decimal,
        fee: Decimal
    ) -> int:
        """Create new position from trade"""
        query = """
            INSERT INTO positions (
                provider,
                market_id,
                market_slug,
                question,
                side,
                entry_size,
                entry_price,
                entry_fee,
                current_size,
                current_price,
                status,
                agent_reasoning,
                confidence_score,
                edge,
                kelly_fraction
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            RETURNING id
        """

        position_id = await self.db.fetchval(
            query,
            trade['provider'],
            trade['market_id'],
            trade['market_slug'],
            trade.get('question', trade['market_slug']),
            trade['side'],
            trade['size'],
            fill_price,
            fee,
            trade['size'],
            fill_price,
            'open',
            trade.get('agent_reasoning'),
            trade.get('confidence_score'),
            trade.get('edge'),
            trade.get('kelly_fraction')
        )

        # Link trade to position
        await self.db.execute(
            "UPDATE trades SET position_id = $1 WHERE id = $2",
            position_id,
            trade['id']
        )

        return position_id

    async def _add_to_position(
        self,
        position_id: int,
        trade: Dict,
        fill_price: Decimal,
        fee: Decimal
    ) -> None:
        """Add trade to existing position"""
        # Get current position
        pos = await self.db.fetchrow(
            "SELECT entry_size, entry_price, current_size, entry_fee FROM positions WHERE id = $1",
            position_id
        )

        # Calculate new weighted average entry price
        total_size = Decimal(str(pos['current_size'])) + Decimal(str(trade['size']))
        new_entry_price = (
            (Decimal(str(pos['entry_price'])) * Decimal(str(pos['current_size'])) +
             fill_price * Decimal(str(trade['size']))) / total_size
        )

        # Update position
        await self.db.execute(
            """
            UPDATE positions
            SET current_size = current_size + $1,
                entry_price = $2,
                entry_fee = entry_fee + $3,
                last_updated = NOW()
            WHERE id = $4
            """,
            trade['size'],
            new_entry_price,
            fee,
            position_id
        )


# Need to import json
import json
