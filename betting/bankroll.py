"""
Bankroll Manager

Tracks bankroll, positions, P&L, and available capital for the professional betting system.
Interfaces with database to maintain real-time bankroll state.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position"""
    id: int
    provider: str
    market_id: str
    market_slug: str
    question: str
    side: str
    entry_size: Decimal
    entry_price: Decimal
    current_size: Decimal
    current_price: Optional[Decimal]
    unrealized_pnl: Decimal
    realized_pnl: Optional[Decimal]
    status: str
    entry_timestamp: datetime
    metadata: Dict


@dataclass
class BankrollState:
    """Current bankroll state"""
    total_equity: Decimal  # Cash + unrealized P&L
    cash_balance: Decimal  # Available cash
    positions_value: Decimal  # Sum of position values
    unrealized_pnl: Decimal  # Unrealized P&L
    realized_pnl: Decimal  # Cumulative realized P&L
    peak_equity: Decimal  # High water mark
    drawdown: Decimal  # Current drawdown (%)
    max_drawdown: Decimal  # Max drawdown ever (%)
    total_exposure: Decimal  # Total position exposure
    exposure_pct: Decimal  # Exposure as % of equity
    num_open_positions: int
    timestamp: datetime


class BankrollManager:
    """
    Manages bankroll, positions, and P&L tracking.

    Features:
    - Track total equity (cash + unrealized P&L)
    - Calculate available capital
    - Mark-to-market positions
    - Calculate drawdown from peak
    - Log bankroll history
    """

    def __init__(self, db_client, initial_bankroll: Decimal = Decimal("10000.0")):
        """
        Initialize bankroll manager.

        Args:
            db_client: Database client instance
            initial_bankroll: Starting bankroll (default: $10,000)
        """
        self.db = db_client
        self.initial_bankroll = initial_bankroll
        self.logger = logger

        self.logger.info(f"BankrollManager initialized with ${initial_bankroll:,.2f}")

    async def get_current_state(self) -> BankrollState:
        """
        Get current bankroll state.

        Returns:
            BankrollState with all current metrics
        """
        # Get open positions
        open_positions = await self._get_open_positions()

        # Calculate positions value and unrealized P&L
        positions_value = Decimal("0.0")
        unrealized_pnl = Decimal("0.0")

        for pos in open_positions:
            if pos.current_price:
                # Mark-to-market
                current_value = pos.current_size * pos.current_price
                entry_value = pos.current_size * pos.entry_price
                pnl = current_value - entry_value

                # Adjust for side (YES/NO or BUY/SELL)
                if pos.side in ("NO", "SELL"):
                    pnl = -pnl

                positions_value += current_value
                unrealized_pnl += pnl

        # Get realized P&L from closed positions
        realized_pnl = await self._get_realized_pnl()

        # Calculate cash balance
        # Initial bankroll + realized P&L - current position values
        deployed_capital = sum(
            pos.entry_size for pos in open_positions
        )
        cash_balance = self.initial_bankroll + realized_pnl - deployed_capital

        # Total equity
        total_equity = cash_balance + positions_value

        # Get peak equity for drawdown calculation
        peak_equity = await self._get_peak_equity()
        if peak_equity is None or total_equity > peak_equity:
            peak_equity = total_equity

        # Calculate drawdown
        if peak_equity > 0:
            drawdown = ((peak_equity - total_equity) / peak_equity) * 100
        else:
            drawdown = Decimal("0.0")

        # Get max drawdown
        max_drawdown = await self._get_max_drawdown()
        if max_drawdown is None or drawdown > max_drawdown:
            max_drawdown = drawdown

        # Calculate exposure
        total_exposure = deployed_capital
        exposure_pct = (total_exposure / total_equity * 100) if total_equity > 0 else Decimal("0.0")

        return BankrollState(
            total_equity=total_equity,
            cash_balance=cash_balance,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            peak_equity=peak_equity,
            drawdown=drawdown,
            max_drawdown=max_drawdown,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            num_open_positions=len(open_positions),
            timestamp=datetime.utcnow()
        )

    async def get_available_capital(self) -> Decimal:
        """
        Get available capital for new positions.

        Returns:
            Available cash balance
        """
        state = await self.get_current_state()
        return state.cash_balance

    async def get_position_pnl(self, position_id: int) -> Tuple[Decimal, Decimal]:
        """
        Get P&L for a specific position.

        Args:
            position_id: Position ID

        Returns:
            (unrealized_pnl, realized_pnl) tuple
        """
        query = """
            SELECT
                current_size,
                entry_price,
                current_price,
                side,
                unrealized_pnl,
                realized_pnl,
                status
            FROM positions
            WHERE id = $1
        """

        row = await self.db.fetchrow(query, position_id)

        if not row:
            raise ValueError(f"Position {position_id} not found")

        if row['status'] == 'open' and row['current_price']:
            # Recalculate unrealized P&L
            current_value = Decimal(str(row['current_size'])) * Decimal(str(row['current_price']))
            entry_value = Decimal(str(row['current_size'])) * Decimal(str(row['entry_price']))
            unrealized_pnl = current_value - entry_value

            # Adjust for side
            if row['side'] in ("NO", "SELL"):
                unrealized_pnl = -unrealized_pnl

            # Update in database
            await self.db.execute(
                "UPDATE positions SET unrealized_pnl = $1, last_updated = NOW() WHERE id = $2",
                unrealized_pnl,
                position_id
            )

            return (unrealized_pnl, Decimal("0.0"))
        else:
            # Closed position
            realized_pnl = Decimal(str(row['realized_pnl'] or 0))
            return (Decimal("0.0"), realized_pnl)

    async def update_positions(self, market_prices: Dict[str, Decimal]) -> int:
        """
        Mark-to-market all open positions with current prices.

        Args:
            market_prices: Dict of market_id -> current_price

        Returns:
            Number of positions updated
        """
        open_positions = await self._get_open_positions()
        updated = 0

        for pos in open_positions:
            market_key = f"{pos.provider}:{pos.market_id}"
            if market_key in market_prices:
                new_price = market_prices[market_key]

                # Calculate unrealized P&L
                current_value = pos.current_size * new_price
                entry_value = pos.current_size * pos.entry_price
                unrealized_pnl = current_value - entry_value

                # Adjust for side
                if pos.side in ("NO", "SELL"):
                    unrealized_pnl = -unrealized_pnl

                # Update position
                await self.db.execute(
                    """
                    UPDATE positions
                    SET current_price = $1,
                        unrealized_pnl = $2,
                        last_updated = NOW()
                    WHERE id = $3
                    """,
                    new_price,
                    unrealized_pnl,
                    pos.id
                )

                updated += 1

        if updated > 0:
            self.logger.info(f"Updated {updated} positions with current prices")

        return updated

    async def log_bankroll_snapshot(self, note: Optional[str] = None) -> int:
        """
        Log current bankroll state to history.

        Args:
            note: Optional note for this snapshot

        Returns:
            Snapshot ID
        """
        state = await self.get_current_state()

        query = """
            INSERT INTO bankroll_history (
                total_equity,
                cash_balance,
                positions_value,
                unrealized_pnl,
                realized_pnl,
                peak_equity,
                drawdown,
                max_drawdown,
                total_exposure,
                exposure_pct,
                note
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
        """

        snapshot_id = await self.db.fetchval(
            query,
            state.total_equity,
            state.cash_balance,
            state.positions_value,
            state.unrealized_pnl,
            state.realized_pnl,
            state.peak_equity,
            state.drawdown,
            state.max_drawdown,
            state.total_exposure,
            state.exposure_pct,
            note
        )

        self.logger.debug(f"Logged bankroll snapshot #{snapshot_id}")
        return snapshot_id

    async def get_portfolio_pnl(self) -> Dict[str, Decimal]:
        """
        Get total portfolio P&L.

        Returns:
            Dict with unrealized_pnl, realized_pnl, total_pnl
        """
        state = await self.get_current_state()

        return {
            "unrealized_pnl": state.unrealized_pnl,
            "realized_pnl": state.realized_pnl,
            "total_pnl": state.unrealized_pnl + state.realized_pnl,
            "total_pnl_pct": (
                ((state.unrealized_pnl + state.realized_pnl) / self.initial_bankroll * 100)
                if self.initial_bankroll > 0
                else Decimal("0.0")
            )
        }

    async def get_position_summary(self) -> Dict[str, any]:
        """
        Get summary of all positions.

        Returns:
            Dict with position counts, avg size, etc.
        """
        query = """
            SELECT
                COUNT(*) FILTER (WHERE status = 'open') as open_count,
                COUNT(*) FILTER (WHERE status = 'closed') as closed_count,
                AVG(entry_size) FILTER (WHERE status = 'open') as avg_open_size,
                SUM(realized_pnl) FILTER (WHERE status = 'closed') as total_realized_pnl,
                SUM(unrealized_pnl) FILTER (WHERE status = 'open') as total_unrealized_pnl,
                COUNT(*) FILTER (WHERE status = 'closed' AND realized_pnl > 0) as winning_positions,
                COUNT(*) FILTER (WHERE status = 'closed' AND realized_pnl <= 0) as losing_positions
            FROM positions
        """

        row = await self.db.fetchrow(query)

        total_closed = (row['closed_count'] or 0)
        winning = (row['winning_positions'] or 0)
        win_rate = (winning / total_closed * 100) if total_closed > 0 else 0.0

        return {
            "open_positions": row['open_count'] or 0,
            "closed_positions": row['closed_count'] or 0,
            "avg_open_size": Decimal(str(row['avg_open_size'] or 0)),
            "total_realized_pnl": Decimal(str(row['total_realized_pnl'] or 0)),
            "total_unrealized_pnl": Decimal(str(row['total_unrealized_pnl'] or 0)),
            "winning_positions": winning,
            "losing_positions": row['losing_positions'] or 0,
            "win_rate": Decimal(str(win_rate))
        }

    async def _get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        query = """
            SELECT
                id, provider, market_id, market_slug, question, side,
                entry_size, entry_price, current_size, current_price,
                unrealized_pnl, realized_pnl, status, entry_timestamp, metadata
            FROM positions
            WHERE status = 'open'
            ORDER BY entry_timestamp DESC
        """

        rows = await self.db.fetch(query)

        return [
            Position(
                id=row['id'],
                provider=row['provider'],
                market_id=row['market_id'],
                market_slug=row['market_slug'],
                question=row['question'],
                side=row['side'],
                entry_size=Decimal(str(row['entry_size'])),
                entry_price=Decimal(str(row['entry_price'])),
                current_size=Decimal(str(row['current_size'])),
                current_price=Decimal(str(row['current_price'])) if row['current_price'] else None,
                unrealized_pnl=Decimal(str(row['unrealized_pnl'])),
                realized_pnl=Decimal(str(row['realized_pnl'])) if row['realized_pnl'] else None,
                status=row['status'],
                entry_timestamp=row['entry_timestamp'],
                metadata=row['metadata'] or {}
            )
            for row in rows
        ]

    async def _get_realized_pnl(self) -> Decimal:
        """Get total realized P&L from closed positions"""
        query = "SELECT COALESCE(SUM(realized_pnl), 0) FROM positions WHERE status IN ('closed', 'settled')"
        result = await self.db.fetchval(query)
        return Decimal(str(result))

    async def _get_peak_equity(self) -> Optional[Decimal]:
        """Get peak equity from history"""
        query = "SELECT MAX(peak_equity) FROM bankroll_history"
        result = await self.db.fetchval(query)
        return Decimal(str(result)) if result else None

    async def _get_max_drawdown(self) -> Optional[Decimal]:
        """Get max drawdown from history"""
        query = "SELECT MAX(max_drawdown) FROM bankroll_history"
        result = await self.db.fetchval(query)
        return Decimal(str(result)) if result else None
