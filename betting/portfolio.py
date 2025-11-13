"""
Portfolio Manager

Manages portfolio optimization, rebalancing, hedging, and correlation analysis.
Helps diversify risk and optimize capital allocation across positions.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class PositionCorrelation:
    """Correlation between two positions"""
    position_1_id: int
    position_2_id: int
    correlation_score: float  # -1 to 1
    category_overlap: bool
    provider_same: bool
    description: str


@dataclass
class RebalanceRecommendation:
    """Portfolio rebalance recommendation"""
    action: str  # 'close', 'reduce', 'increase', 'hedge'
    position_id: int
    current_size: Decimal
    recommended_size: Decimal
    reason: str
    urgency: str  # 'low', 'medium', 'high'


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_return_pct: Decimal
    sharpe_ratio: Optional[Decimal]
    sortino_ratio: Optional[Decimal]
    max_drawdown_pct: Decimal
    win_rate: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: Decimal  # avg_win / avg_loss
    num_positions: int
    concentration_score: Decimal  # 0-100, higher = more concentrated


class PortfolioManager:
    """
    Manages portfolio optimization and rebalancing.

    Features:
    - Correlation analysis between positions
    - Rebalancing recommendations
    - Diversification metrics
    - Hedge suggestions
    - Performance attribution
    """

    def __init__(self, db_client, bankroll_manager, risk_manager):
        """
        Initialize portfolio manager.

        Args:
            db_client: Database client instance
            bankroll_manager: BankrollManager instance
            risk_manager: RiskManager instance
        """
        self.db = db_client
        self.bankroll = bankroll_manager
        self.risk = risk_manager
        self.logger = logger

        self.logger.info("PortfolioManager initialized")

    async def get_correlation_matrix(self) -> List[PositionCorrelation]:
        """
        Calculate correlation between open positions.

        Returns:
            List of PositionCorrelation objects
        """
        query = """
            SELECT
                id, market_slug, provider, metadata
            FROM positions
            WHERE status = 'open'
        """

        rows = await self.db.fetch(query)
        correlations = []

        # Compare each pair of positions
        for i, pos1 in enumerate(rows):
            for pos2 in rows[i+1:]:
                # Calculate correlation score based on:
                # 1. Category overlap
                # 2. Provider (markets on same platform may correlate)
                # 3. Market slug similarity

                category1 = pos1['metadata'].get('category', '') if pos1['metadata'] else ''
                category2 = pos2['metadata'].get('category', '') if pos2['metadata'] else ''

                category_overlap = (category1 == category2 and category1 != '')
                provider_same = (pos1['provider'] == pos2['provider'])

                # Simple correlation scoring
                score = 0.0
                if category_overlap:
                    score += 0.6
                if provider_same:
                    score += 0.2

                # Check for keyword overlap in market slugs
                keywords1 = set(pos1['market_slug'].lower().split('-'))
                keywords2 = set(pos2['market_slug'].lower().split('-'))
                keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1), len(keywords2), 1)
                score += keyword_overlap * 0.2

                # Only include if some correlation exists
                if score > 0.2:
                    description = []
                    if category_overlap:
                        description.append(f"same category ({category1})")
                    if provider_same:
                        description.append(f"same provider ({pos1['provider']})")
                    if keyword_overlap > 0.3:
                        description.append(f"similar markets")

                    correlations.append(PositionCorrelation(
                        position_1_id=pos1['id'],
                        position_2_id=pos2['id'],
                        correlation_score=min(score, 1.0),
                        category_overlap=category_overlap,
                        provider_same=provider_same,
                        description=", ".join(description)
                    ))

        self.logger.debug(f"Found {len(correlations)} correlated position pairs")
        return correlations

    async def suggest_rebalance(self) -> List[RebalanceRecommendation]:
        """
        Generate portfolio rebalancing recommendations.

        Returns:
            List of RebalanceRecommendation objects
        """
        recommendations = []

        # Get current state
        state = await self.bankroll.get_current_state()

        # Get all open positions
        query = """
            SELECT
                id, market_slug, entry_size, current_size,
                unrealized_pnl, entry_price, current_price, metadata
            FROM positions
            WHERE status = 'open'
            ORDER BY entry_size DESC
        """

        positions = await self.db.fetch(query)

        # 1. Check for positions that have grown too large (> 10% of portfolio)
        for pos in positions:
            position_value = Decimal(str(pos['current_size'])) * Decimal(str(pos['current_price'] or pos['entry_price']))
            position_pct = (position_value / state.total_equity * 100) if state.total_equity > 0 else Decimal("0.0")

            if position_pct > 10:  # Position > 10% of portfolio
                # Recommend reducing to 5%
                target_value = state.total_equity * Decimal("0.05")
                target_size = target_value / Decimal(str(pos['current_price'] or pos['entry_price']))

                recommendations.append(RebalanceRecommendation(
                    action="reduce",
                    position_id=pos['id'],
                    current_size=Decimal(str(pos['current_size'])),
                    recommended_size=target_size,
                    reason=f"Position too large ({position_pct:.1f}% of portfolio)",
                    urgency="medium"
                ))

        # 2. Check for positions with large unrealized losses (close to stop-loss)
        for pos in positions:
            unrealized_pnl = Decimal(str(pos['unrealized_pnl'] or 0))
            entry_size = Decimal(str(pos['entry_size']))

            if unrealized_pnl < 0:
                loss_pct = abs(unrealized_pnl / entry_size) if entry_size > 0 else Decimal("0.0")

                # Close if loss > 40% (approaching 50% stop-loss)
                if loss_pct >= Decimal("0.40"):
                    recommendations.append(RebalanceRecommendation(
                        action="close",
                        position_id=pos['id'],
                        current_size=Decimal(str(pos['current_size'])),
                        recommended_size=Decimal("0.0"),
                        reason=f"Large unrealized loss ({loss_pct * 100:.1f}%)",
                        urgency="high"
                    ))

        # 3. Check for winners to compound
        for pos in positions:
            unrealized_pnl = Decimal(str(pos['unrealized_pnl'] or 0))
            entry_size = Decimal(str(pos['entry_size']))

            if unrealized_pnl > 0:
                profit_pct = (unrealized_pnl / entry_size) if entry_size > 0 else Decimal("0.0")

                # Increase size if profit > 50% and position is still small
                position_value = Decimal(str(pos['current_size'])) * Decimal(str(pos['current_price'] or pos['entry_price']))
                position_pct = (position_value / state.total_equity * 100) if state.total_equity > 0 else Decimal("0.0")

                if profit_pct >= Decimal("0.50") and position_pct < 5:
                    # Suggest increasing to 5% of portfolio
                    target_value = state.total_equity * Decimal("0.05")
                    target_size = target_value / Decimal(str(pos['current_price'] or pos['entry_price']))

                    recommendations.append(RebalanceRecommendation(
                        action="increase",
                        position_id=pos['id'],
                        current_size=Decimal(str(pos['current_size'])),
                        recommended_size=target_size,
                        reason=f"Strong performer (+{profit_pct * 100:.1f}%), compound winnings",
                        urgency="low"
                    ))

        # 4. Check for correlated positions (suggest hedge)
        correlations = await self.get_correlation_matrix()
        high_correlations = [c for c in correlations if c.correlation_score > 0.7]

        for corr in high_correlations:
            recommendations.append(RebalanceRecommendation(
                action="hedge",
                position_id=corr.position_1_id,
                current_size=Decimal("0.0"),  # N/A for hedge
                recommended_size=Decimal("0.0"),  # N/A for hedge
                reason=f"High correlation with position {corr.position_2_id}: {corr.description}",
                urgency="low"
            ))

        self.logger.info(f"Generated {len(recommendations)} rebalancing recommendations")
        return recommendations

    async def close_position(
        self,
        position_id: int,
        reason: str = "Manual close"
    ) -> Dict:
        """
        Close a position and calculate realized P&L.

        Args:
            position_id: Position ID to close
            reason: Reason for closing

        Returns:
            Dict with close details
        """
        # Get position
        query = """
            SELECT
                id, market_slug, entry_size, entry_price, current_size,
                current_price, unrealized_pnl, side
            FROM positions
            WHERE id = $1 AND status = 'open'
        """

        pos = await self.db.fetchrow(query, position_id)

        if not pos:
            raise ValueError(f"Position {position_id} not found or already closed")

        # Calculate realized P&L
        current_price = Decimal(str(pos['current_price'] or pos['entry_price']))
        entry_price = Decimal(str(pos['entry_price']))
        size = Decimal(str(pos['current_size']))

        current_value = size * current_price
        entry_value = size * entry_price
        realized_pnl = current_value - entry_value

        # Adjust for side
        if pos['side'] in ("NO", "SELL"):
            realized_pnl = -realized_pnl

        # Update position
        await self.db.execute(
            """
            UPDATE positions
            SET status = 'closed',
                exit_price = $1,
                exit_timestamp = NOW(),
                realized_pnl = $2,
                metadata = metadata || jsonb_build_object('close_reason', $3)
            WHERE id = $4
            """,
            current_price,
            realized_pnl,
            reason,
            position_id
        )

        # Log to bankroll history
        await self.bankroll.log_bankroll_snapshot(note=f"Closed position: {pos['market_slug']}")

        self.logger.info(
            f"Closed position {position_id} ({pos['market_slug']}): "
            f"realized_pnl=${realized_pnl:.2f}"
        )

        return {
            "position_id": position_id,
            "market_slug": pos['market_slug'],
            "exit_price": current_price,
            "realized_pnl": realized_pnl,
            "pnl_pct": ((realized_pnl / entry_value * 100) if entry_value > 0 else Decimal("0.0")),
            "reason": reason
        }

    async def compound_winners(self, min_profit_pct: Decimal = Decimal("0.30")) -> List[int]:
        """
        Identify profitable positions to scale up.

        Args:
            min_profit_pct: Minimum profit % to consider (default: 30%)

        Returns:
            List of position IDs to compound
        """
        query = """
            SELECT id, market_slug, entry_size, unrealized_pnl
            FROM positions
            WHERE status = 'open'
                AND unrealized_pnl > 0
        """

        rows = await self.db.fetch(query)
        winners = []

        for row in rows:
            entry_size = Decimal(str(row['entry_size']))
            unrealized_pnl = Decimal(str(row['unrealized_pnl']))
            profit_pct = (unrealized_pnl / entry_size) if entry_size > 0 else Decimal("0.0")

            if profit_pct >= min_profit_pct:
                winners.append(row['id'])
                self.logger.debug(
                    f"Winner candidate: position {row['id']} ({row['market_slug']}) "
                    f"with +{profit_pct * 100:.1f}% profit"
                )

        return winners

    async def get_portfolio_metrics(self, days: int = 30) -> PortfolioMetrics:
        """
        Calculate portfolio performance metrics.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            PortfolioMetrics object
        """
        since_date = datetime.utcnow() - timedelta(days=days)

        # Get closed positions in period
        query = """
            SELECT
                realized_pnl,
                entry_size,
                exit_timestamp - entry_timestamp as hold_duration
            FROM positions
            WHERE status IN ('closed', 'settled')
                AND exit_timestamp >= $1
        """

        closed_positions = await self.db.fetch(query, since_date)

        # Calculate metrics
        total_closed = len(closed_positions)
        winners = [p for p in closed_positions if Decimal(str(p['realized_pnl'])) > 0]
        losers = [p for p in closed_positions if Decimal(str(p['realized_pnl'])) <= 0]

        win_rate = (len(winners) / total_closed * 100) if total_closed > 0 else Decimal("0.0")

        avg_win = (
            sum(Decimal(str(p['realized_pnl'])) for p in winners) / len(winners)
            if winners else Decimal("0.0")
        )

        avg_loss = (
            sum(Decimal(str(p['realized_pnl'])) for p in losers) / len(losers)
            if losers else Decimal("0.0")
        )

        profit_factor = (avg_win / abs(avg_loss)) if avg_loss != 0 else Decimal("0.0")

        # Get max drawdown
        state = await self.bankroll.get_current_state()
        max_drawdown = state.max_drawdown

        # Calculate total return
        total_pnl = sum(Decimal(str(p['realized_pnl'])) for p in closed_positions)
        initial_bankroll = self.bankroll.initial_bankroll
        total_return_pct = (total_pnl / initial_bankroll * 100) if initial_bankroll > 0 else Decimal("0.0")

        # Calculate concentration (0-100, where 100 = all capital in 1 position)
        num_open = state.num_open_positions
        if num_open > 0:
            # Simple concentration: inversely proportional to number of positions
            concentration_score = Decimal("100.0") / num_open
        else:
            concentration_score = Decimal("0.0")

        return PortfolioMetrics(
            total_return_pct=total_return_pct,
            sharpe_ratio=None,  # TODO: Calculate from daily returns
            sortino_ratio=None,  # TODO: Calculate from downside deviation
            max_drawdown_pct=max_drawdown,
            win_rate=Decimal(str(win_rate)),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            num_positions=num_open,
            concentration_score=concentration_score
        )

    async def get_position_attribution(self) -> List[Dict]:
        """
        Get P&L attribution by position.

        Returns:
            List of positions with P&L contribution
        """
        state = await self.bankroll.get_current_state()

        query = """
            SELECT
                market_slug,
                provider,
                unrealized_pnl,
                realized_pnl,
                status,
                entry_timestamp
            FROM positions
            ORDER BY COALESCE(unrealized_pnl, realized_pnl, 0) DESC
        """

        rows = await self.db.fetch(query)

        attribution = []
        for row in rows:
            pnl = Decimal(str(row['unrealized_pnl'] if row['status'] == 'open' else row['realized_pnl'] or 0))
            contribution_pct = (pnl / state.total_equity * 100) if state.total_equity > 0 else Decimal("0.0")

            attribution.append({
                "market_slug": row['market_slug'],
                "provider": row['provider'],
                "pnl": pnl,
                "contribution_pct": contribution_pct,
                "status": row['status'],
                "age_days": (datetime.utcnow() - row['entry_timestamp']).days
            })

        return attribution
