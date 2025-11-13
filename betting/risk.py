"""
Risk Manager

Enforces risk limits and manages portfolio risk for the professional betting system.
Prevents excessive losses through position limits, drawdown controls, and stop-loss.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk management limits configuration"""
    max_position_size_pct: Decimal = Decimal("0.05")  # 5% of bankroll max per position
    max_total_exposure_pct: Decimal = Decimal("0.25")  # 25% max total exposure
    max_drawdown_pct: Decimal = Decimal("0.15")  # 15% max drawdown
    stop_loss_pct: Decimal = Decimal("0.50")  # 50% stop-loss per position
    min_position_size: Decimal = Decimal("10.0")  # Min $10 per position
    max_positions: int = 20  # Max 20 open positions
    max_correlated_positions: int = 5  # Max 5 positions in same category
    require_approval_threshold: Decimal = Decimal("100.0")  # Approve if bet > $100


@dataclass
class RiskViolation:
    """Represents a risk limit violation"""
    violation_type: str
    severity: str  # 'warning' | 'critical'
    message: str
    current_value: Decimal
    limit_value: Decimal
    suggested_action: str


@dataclass
class BetValidation:
    """Result of bet validation"""
    is_valid: bool
    violations: List[RiskViolation]
    requires_approval: bool
    max_allowed_size: Decimal
    adjusted_size: Optional[Decimal] = None
    warnings: List[str] = None


class RiskManager:
    """
    Manages risk limits and validates betting decisions.

    Features:
    - Position size limits (% of bankroll)
    - Total exposure limits
    - Drawdown monitoring and breaker
    - Stop-loss enforcement
    - Correlation limits
    - Approval workflow for large bets
    """

    def __init__(
        self,
        db_client,
        bankroll_manager,
        limits: Optional[RiskLimits] = None
    ):
        """
        Initialize risk manager.

        Args:
            db_client: Database client instance
            bankroll_manager: BankrollManager instance
            limits: Custom risk limits (default: RiskLimits())
        """
        self.db = db_client
        self.bankroll = bankroll_manager
        self.limits = limits or RiskLimits()
        self.logger = logger

        self.logger.info(
            f"RiskManager initialized: "
            f"max_position={self.limits.max_position_size_pct * 100}%, "
            f"max_exposure={self.limits.max_total_exposure_pct * 100}%, "
            f"max_drawdown={self.limits.max_drawdown_pct * 100}%"
        )

    async def validate_bet(
        self,
        bet_size: Decimal,
        market_id: str,
        provider: str,
        category: Optional[str] = None
    ) -> BetValidation:
        """
        Validate a proposed bet against risk limits.

        Args:
            bet_size: Proposed bet size (USD)
            market_id: Market identifier
            provider: Provider name
            category: Market category (for correlation check)

        Returns:
            BetValidation with validation result and violations
        """
        violations = []
        warnings = []

        # Get current bankroll state
        state = await self.bankroll.get_current_state()

        # Check 1: Minimum position size
        if bet_size < self.limits.min_position_size:
            violations.append(RiskViolation(
                violation_type="min_position_size",
                severity="critical",
                message=f"Bet size ${bet_size:.2f} below minimum ${self.limits.min_position_size:.2f}",
                current_value=bet_size,
                limit_value=self.limits.min_position_size,
                suggested_action="Increase bet size or skip this bet"
            ))

        # Check 2: Maximum position size (% of equity)
        max_position_size = state.total_equity * self.limits.max_position_size_pct
        if bet_size > max_position_size:
            violations.append(RiskViolation(
                violation_type="max_position_size",
                severity="critical",
                message=f"Bet size ${bet_size:.2f} exceeds max position size ${max_position_size:.2f} ({self.limits.max_position_size_pct * 100}% of equity)",
                current_value=bet_size,
                limit_value=max_position_size,
                suggested_action=f"Reduce bet size to ${max_position_size:.2f}"
            ))

        # Check 3: Total exposure limit
        new_exposure = state.total_exposure + bet_size
        max_exposure = state.total_equity * self.limits.max_total_exposure_pct

        if new_exposure > max_exposure:
            violations.append(RiskViolation(
                violation_type="max_exposure",
                severity="critical",
                message=f"New total exposure ${new_exposure:.2f} exceeds limit ${max_exposure:.2f} ({self.limits.max_total_exposure_pct * 100}% of equity)",
                current_value=new_exposure,
                limit_value=max_exposure,
                suggested_action="Close some positions before opening new ones"
            ))

        # Check 4: Drawdown breaker
        if state.drawdown >= self.limits.max_drawdown_pct * 100:
            violations.append(RiskViolation(
                violation_type="max_drawdown",
                severity="critical",
                message=f"Current drawdown {state.drawdown:.2f}% exceeds limit {self.limits.max_drawdown_pct * 100}%",
                current_value=state.drawdown,
                limit_value=self.limits.max_drawdown_pct * 100,
                suggested_action="Stop trading until drawdown recovers below threshold"
            ))

        # Check 5: Available capital
        if bet_size > state.cash_balance:
            violations.append(RiskViolation(
                violation_type="insufficient_capital",
                severity="critical",
                message=f"Bet size ${bet_size:.2f} exceeds available cash ${state.cash_balance:.2f}",
                current_value=bet_size,
                limit_value=state.cash_balance,
                suggested_action=f"Reduce bet size to ${state.cash_balance:.2f} or close positions to free capital"
            ))

        # Check 6: Maximum number of positions
        if state.num_open_positions >= self.limits.max_positions:
            violations.append(RiskViolation(
                violation_type="max_positions",
                severity="warning",
                message=f"Already at max positions ({state.num_open_positions}/{self.limits.max_positions})",
                current_value=Decimal(str(state.num_open_positions)),
                limit_value=Decimal(str(self.limits.max_positions)),
                suggested_action="Close some positions before opening new ones"
            ))

        # Check 7: Correlated positions (if category provided)
        if category:
            correlated_count = await self._count_positions_in_category(category)
            if correlated_count >= self.limits.max_correlated_positions:
                violations.append(RiskViolation(
                    violation_type="max_correlated_positions",
                    severity="warning",
                    message=f"Already have {correlated_count} positions in category '{category}' (limit: {self.limits.max_correlated_positions})",
                    current_value=Decimal(str(correlated_count)),
                    limit_value=Decimal(str(self.limits.max_correlated_positions)),
                    suggested_action="Diversify into other categories"
                ))

        # Check 8: Approval requirement
        requires_approval = bet_size >= self.limits.require_approval_threshold

        if requires_approval:
            warnings.append(
                f"Bet size ${bet_size:.2f} requires manual approval (threshold: ${self.limits.require_approval_threshold:.2f})"
            )

        # Determine if bet is valid
        critical_violations = [v for v in violations if v.severity == "critical"]
        is_valid = len(critical_violations) == 0

        # Calculate max allowed size
        max_allowed_size = min(
            max_position_size,
            max_exposure - state.total_exposure,
            state.cash_balance
        )

        # Suggest adjusted size if over limit
        adjusted_size = None
        if not is_valid and bet_size > max_allowed_size:
            adjusted_size = max_allowed_size

        return BetValidation(
            is_valid=is_valid,
            violations=violations,
            requires_approval=requires_approval,
            max_allowed_size=max_allowed_size,
            adjusted_size=adjusted_size,
            warnings=warnings or []
        )

    async def check_drawdown(self) -> Tuple[bool, Optional[RiskViolation]]:
        """
        Check if current drawdown exceeds limit.

        Returns:
            (is_breached, violation) tuple
        """
        state = await self.bankroll.get_current_state()

        if state.drawdown >= self.limits.max_drawdown_pct * 100:
            violation = RiskViolation(
                violation_type="max_drawdown",
                severity="critical",
                message=f"Drawdown {state.drawdown:.2f}% exceeds limit {self.limits.max_drawdown_pct * 100}%",
                current_value=state.drawdown,
                limit_value=self.limits.max_drawdown_pct * 100,
                suggested_action="Stop trading and review strategy"
            )

            # Log risk event
            await self._log_risk_event(
                event_type="max_drawdown_breached",
                severity="critical",
                description=violation.message,
                action_taken="Trading halted"
            )

            return (True, violation)

        return (False, None)

    async def enforce_stop_loss(self) -> List[int]:
        """
        Check all open positions for stop-loss violations and mark for closing.

        Returns:
            List of position IDs that hit stop-loss
        """
        query = """
            SELECT id, market_slug, entry_size, unrealized_pnl
            FROM positions
            WHERE status = 'open'
                AND unrealized_pnl IS NOT NULL
        """

        rows = await self.db.fetch(query)
        stopped_out = []

        for row in rows:
            position_id = row['id']
            entry_size = Decimal(str(row['entry_size']))
            unrealized_pnl = Decimal(str(row['unrealized_pnl']))

            # Check if loss exceeds stop-loss threshold
            loss_pct = abs(unrealized_pnl / entry_size) if entry_size > 0 else Decimal("0.0")

            if unrealized_pnl < 0 and loss_pct >= self.limits.stop_loss_pct:
                # Stop-loss triggered
                self.logger.warning(
                    f"Stop-loss triggered for position {position_id} ({row['market_slug']}): "
                    f"loss={loss_pct * 100:.1f}%, threshold={self.limits.stop_loss_pct * 100}%"
                )

                # Mark position as stopped out
                await self.db.execute(
                    "UPDATE positions SET status = 'stopped_out' WHERE id = $1",
                    position_id
                )

                # Log risk event
                await self._log_risk_event(
                    event_type="stop_loss_triggered",
                    severity="warning",
                    description=f"Position {position_id} hit stop-loss: {loss_pct * 100:.1f}% loss",
                    action_taken="Position marked for closing",
                    position_id=position_id
                )

                stopped_out.append(position_id)

        if stopped_out:
            self.logger.info(f"Stop-loss enforced on {len(stopped_out)} positions")

        return stopped_out

    async def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics.

        Returns:
            Dict with risk metrics
        """
        state = await self.bankroll.get_current_state()

        # Check if any limits are close to breach (> 80% of limit)
        warnings = []

        if state.exposure_pct >= self.limits.max_total_exposure_pct * 80:
            warnings.append(f"Exposure at {state.exposure_pct:.1f}% (limit: {self.limits.max_total_exposure_pct * 100}%)")

        if state.drawdown >= self.limits.max_drawdown_pct * 80 * 100:
            warnings.append(f"Drawdown at {state.drawdown:.1f}% (limit: {self.limits.max_drawdown_pct * 100}%)")

        if state.num_open_positions >= self.limits.max_positions * 0.8:
            warnings.append(f"Open positions at {state.num_open_positions} (limit: {self.limits.max_positions})")

        return {
            "total_equity": state.total_equity,
            "cash_balance": state.cash_balance,
            "total_exposure": state.total_exposure,
            "exposure_pct": state.exposure_pct,
            "exposure_limit_pct": self.limits.max_total_exposure_pct * 100,
            "drawdown_pct": state.drawdown,
            "drawdown_limit_pct": self.limits.max_drawdown_pct * 100,
            "num_open_positions": state.num_open_positions,
            "max_positions": self.limits.max_positions,
            "max_position_size": state.total_equity * self.limits.max_position_size_pct,
            "warnings": warnings
        }

    async def _count_positions_in_category(self, category: str) -> int:
        """Count open positions in a specific category"""
        query = """
            SELECT COUNT(*)
            FROM positions
            WHERE status = 'open'
                AND metadata->>'category' = $1
        """
        count = await self.db.fetchval(query, category)
        return count or 0

    async def _log_risk_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        action_taken: Optional[str] = None,
        position_id: Optional[int] = None,
        trade_id: Optional[int] = None
    ) -> int:
        """Log a risk management event"""
        query = """
            INSERT INTO risk_events (
                event_type,
                severity,
                description,
                action_taken,
                position_id,
                trade_id
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        """

        event_id = await self.db.fetchval(
            query,
            event_type,
            severity,
            description,
            action_taken,
            position_id,
            trade_id
        )

        return event_id
