#!/usr/bin/env python3
"""
Copy Trading Signal Engine (The Sniper).
FIXED: "Saved 0" bug caused by portfolio dilution.
NOW: Uses "Greedy Allocation" to prioritize top EV signals.
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from config.settings import Settings
from database.client import MarketDatabase
from clients.polymarket import PolymarketClient

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CopySignalEngine:
    def __init__(self):
        settings = Settings()
        self.db = MarketDatabase(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        self.pm_client = PolymarketClient()
        
        # --- TUNABLE PARAMETERS ---
        self.MAX_TOTAL_PORTFOLIO_EXPOSURE = 1.0  # 100% of bankroll max
        self.MAX_SINGLE_POSITION_SIZE = 0.20     # Max 20% on one bet
        self.FRACTIONAL_KELLY = 0.25             # Safety factor
        
        self.MIN_TRADER_WIN_RATE = 35.0 
        self.MIN_CATEGORY_TRADES = 1
        self.CASH_FLOW_DAYS = 7
        self.MAX_PRICE_SLIPPAGE = 0.20 

    async def run_signal_generation(self):
        logger.info("📡 Starting Sniper Signal Generation...")
        
        positions = await self._fetch_open_positions()
        if not positions:
            logger.info("   No active positions found in DB.")
            return

        logger.info(f"   Fetching LIVE data for {len(positions)} markets...")
        live_market_data = await self._fetch_live_market_data(positions)

        wallets = list(set(p['proxy_wallet'] for p in positions))
        logger.info(f"   Analyzing history for {len(wallets)} traders...")
        
        # Fetch stats (fallback if not in elite_open_positions)
        general_stats = await self._fetch_general_stats(wallets)
        category_stats = await self._fetch_category_stats(wallets)

        raw_signals = []
        rejection_reasons = {} 

        for pos in positions:
            w_stats = general_stats.get(pos['proxy_wallet'])
            if not w_stats: 
                self._log_rejection(rejection_reasons, "Missing Wallet Stats")
                continue
            
            cat_name = pos.get('event_category') or 'General'
            cat_key = f"{pos['proxy_wallet']}_{cat_name}"
            c_stats = category_stats.get(cat_key, {})
            
            market_data = live_market_data.get(pos['condition_id'])

            signal, reason = self._calculate_sniper_signal(pos, w_stats, c_stats, market_data)
            
            if signal:
                raw_signals.append(signal)
            else:
                self._log_rejection(rejection_reasons, reason)

        if rejection_reasons:
            logger.info(f"   🚫 Rejection Summary: {json.dumps(rejection_reasons)}")

        logger.info(f"   Generated {len(raw_signals)} potential signals. Optimizing Portfolio...")

        # CRITICAL FIX HERE
        final_signals = self._optimize_portfolio(raw_signals)
        
        await self._save_signals(final_signals)
        logger.info(f"✅ Completed. Saved {len(final_signals)} optimized signals.")

    def _log_rejection(self, stats: Dict, reason: str):
        stats[reason] = stats.get(reason, 0) + 1

    async def _fetch_open_positions(self) -> List[Dict]:
        # Get positions AND the tier/stats stored with them if possible
        res = self.db.supabase.table("elite_open_positions").select("*").execute()
        return res.data or []

    async def _fetch_live_market_data(self, positions: List[Dict]) -> Dict[str, Dict]:
        condition_ids = list(set(p['condition_id'] for p in positions if p.get('condition_id')))
        results = {}
        
        async def fetch_one(cid):
            try:
                # 1. Try standard Market Fetch
                market = await self.pm_client.gamma.get_market_by_condition_id(cid)
                
                # 2. If failed, try Closed Market Fetch (For events ending TODAY)
                if not market:
                    try:
                        closed_data = await self.pm_client.gamma._get(
                            "/markets", {"conditionId": cid, "closed": "true", "limit": 1}
                        )
                        if isinstance(closed_data, list) and closed_data:
                            market = self.pm_client.gamma._normalize_market(closed_data[0])
                    except: pass

                if market:
                    prices = market.get('outcomePrices', [])
                    parsed_prices = []
                    for p in prices:
                        try: parsed_prices.append(float(p))
                        except: parsed_prices.append(0.0)
                    
                    # Valid if we have prices and they aren't all 0 or 1
                    is_tradable = False
                    if len(parsed_prices) >= 2:
                         if any(0.01 <= p <= 0.99 for p in parsed_prices):
                             is_tradable = True
                    
                    end_date = market.get('endDate') or market.get('close_date') or market.get('resolutionDate')

                    return cid, {
                        'prices': parsed_prices, 
                        'active': market.get('active', True), 
                        'tradable': is_tradable,
                        'endDate': end_date
                    }
            except:
                pass
            return cid, None

        # Fetch in batches to avoid rate limits
        batch_size = 5
        for i in range(0, len(condition_ids), batch_size):
            batch = condition_ids[i:i+batch_size]
            tasks = [fetch_one(cid) for cid in batch]
            batch_results = await asyncio.gather(*tasks)
            for cid, data in batch_results:
                if data: results[cid] = data
            await asyncio.sleep(0.1) 

        return results

    async def _fetch_general_stats(self, wallets: List[str]) -> Dict[str, Dict]:
        if not wallets: return {}
        res = self.db.supabase.table("wallet_analytics")\
            .select("proxy_wallet, win_rate, roi, n_positions, tier")\
            .in_("proxy_wallet", wallets)\
            .execute()
        return {row['proxy_wallet']: row for row in (res.data or [])}

    async def _fetch_category_stats(self, wallets: List[str]) -> Dict[str, Dict]:
        if not wallets: return {}
        res = self.db.supabase.table("wallet_closed_positions")\
            .select("proxy_wallet, event_category, realized_pnl, total_bought")\
            .in_("proxy_wallet", wallets)\
            .execute()
            
        agg = {}
        for row in (res.data or []):
            cat = row.get('event_category')
            if not cat: continue
            key = f"{row['proxy_wallet']}_{cat}"
            if key not in agg: agg[key] = {"wins": 0, "total": 0, "pnl": 0.0, "vol": 0.0}
            agg[key]["total"] += 1
            agg[key]["vol"] += (row['total_bought'] or 0)
            agg[key]["pnl"] += (row['realized_pnl'] or 0)
            if (row['realized_pnl'] or 0) > 0: agg[key]["wins"] += 1
        
        final_cats = {}
        for k, v in agg.items():
            wr = (v["wins"] / v["total"]) * 100 if v["total"] > 0 else 0
            roi = (v["pnl"] / v["vol"]) * 100 if v["vol"] > 0 else 0
            final_cats[k] = {"cat_win_rate": wr, "cat_roi": roi, "cat_trades": v["total"]}
        return final_cats

    def _calculate_sniper_signal(
        self, 
        pos: Dict, 
        w_stats: Dict, 
        c_stats: Dict, 
        market_data: Optional[Dict]
    ) -> Tuple[Optional[Dict], str]:
        
        if not market_data: return None, "Market Data Missing"
        
        # --- 1. Date Logic ---
        # Prefer DB date (enriched), fallback to API
        end_date_str = pos.get('event_end_date')
        if not end_date_str or end_date_str == "":
             end_date_str = market_data.get('endDate')

        is_future = True 
        days_left = 30
        
        if end_date_str:
            try:
                clean_str = end_date_str.replace("Z", "+00:00")
                if "T" not in clean_str and "+" not in clean_str:
                     dt = datetime.fromisoformat(clean_str).replace(tzinfo=timezone.utc)
                else:
                     dt = datetime.fromisoformat(clean_str)
                
                now = datetime.now(timezone.utc)
                # Allow a 2-hour buffer for "Live" events that haven't settled yet
                if dt < now:
                    # Check if it ended very recently (e.g., < 2 hours ago)
                    time_diff = (now - dt).total_seconds()
                    if time_diff > 7200: # 2 hours
                        is_future = False
                    else:
                        days_left = 0 # It's happening now
                else:
                    days_left = (dt - now).days
            except: 
                pass 

        # --- 2. Price Logic ---
        db_price = float(pos.get('current_price') or 0.5)
        live_price = None
        if market_data:
            live_prices = market_data.get('prices', [])
            idx = pos.get('outcome_index', 1)
            if len(live_prices) > idx:
                live_price = live_prices[idx]

        use_price = db_price
        
        # If it's live/future, trust the live price
        if is_future:
            if live_price and 0.01 < live_price < 0.99:
                use_price = live_price
        else:
            # If strictly in the past and API says not tradable, reject
            if not market_data.get('tradable', False):
                 return None, "Market Expired"

        if use_price <= 0.01 or use_price >= 0.99:
            return None, f"Price Extreme ({use_price:.2f})"

        # --- 3. Trader Stats (Tier Aware) ---
        # Prefer Tier from the Open Position row (it's fresher), else Analytics
        tier = pos.get('trader_tier') or w_stats.get('tier') or 'C'
        
        gen_wr = float(w_stats.get('win_rate') or 0)
        cat_wr = c_stats.get('cat_win_rate', 0)
        cat_trades = c_stats.get('cat_trades', 0)
        cat_roi = c_stats.get('cat_roi', 0)
        
        effective_wr = (gen_wr * 0.8 + cat_wr * 0.2) if cat_trades < 3 else cat_wr

        if effective_wr < self.MIN_TRADER_WIN_RATE: 
            return None, f"Low Win Rate ({effective_wr:.1f}%)"

        # --- 4. ROI & Kelly (With Elite Edge Boost) ---
        potential_roi_pct = ((1.0 / use_price) - 1.0) * 100.0
        net_odds = (1.0 / use_price) - 1.0
        
        # Edge Boost: Give S and A tier traders a benefit of doubt against the spread
        edge_boost = 0.0
        if tier == 'S': edge_boost = 5.0
        elif tier == 'A': edge_boost = 3.0
        elif tier == 'B': edge_boost = 1.0
        
        p = (effective_wr + edge_boost) / 100.0
        p = min(p, 0.99) # Cap at 99%
        
        q = 1.0 - p
        kelly = (net_odds * p - q) / net_odds if net_odds > 0 else 0
        
        if kelly <= 0:
             # Soft Filter: If Kelly is slightly neg but trader is elite, allow min bet
             if kelly > -0.05 and tier in ['S', 'A']:
                 kelly = 0.01 
             else:
                 return None, "Negative Kelly"

        # --- 5. Rank Score ---
        # Prefer High EV + High Tier
        tier_mult = 1.2 if tier == 'S' else 1.1 if tier == 'A' else 1.0
        rank_score = (p ** 2) * kelly * 100 * tier_mult

        # --- 6. Slippage ---
        entry_price = float(pos.get('avg_entry_price') or use_price)
        if entry_price <= 0.001: entry_price = use_price
        
        # If price has moved AGAINST the trader significantly
        if use_price > (entry_price + self.MAX_PRICE_SLIPPAGE):
             return None, "Slippage High"

        rationale = (
            f"Strategy: Copy {tier}-Tier. WR {effective_wr:.0f}%. "
            f"Odds {use_price:.2f}. Exp ROI {potential_roi_pct:.0f}%."
        )

        return {
            "position_id": pos['id'],
            "proxy_wallet": pos['proxy_wallet'],
            "market_title": pos.get('title'),
            "market_slug": pos.get('slug'),
            "outcome": pos.get('outcome'),
            "asset_id": pos['asset'], # Ensure this is passed through
            "current_price": use_price,
            "potential_roi_pct": round(potential_roi_pct, 1),
            "days_to_expiry": days_left,
            "implied_odds": round(net_odds, 2),
            "trader_general_win_rate": gen_wr,
            "category_win_rate": round(cat_wr, 1),
            "category_roi": round(cat_roi, 1),
            "category_trade_count": cat_trades,
            "confidence_score": round(min(effective_wr, 99.9), 1),
            "raw_kelly": kelly,
            "rationale": rationale,
            "is_cash_flow_optimized": days_left < self.CASH_FLOW_DAYS,
            "ev_score": rank_score
        }, "Success"

    def _optimize_portfolio(self, signals: List[Dict]) -> List[Dict]:
        """
        Greedy Allocation Strategy:
        Sort by EV Score. Fill bucket until MAX_TOTAL_PORTFOLIO_EXPOSURE is hit.
        Do not dilute good trades to fit bad ones.
        """
        if not signals: return []
        
        # 1. Sort by Expected Value (Best first)
        signals.sort(key=lambda x: x['ev_score'], reverse=True)

        final_output = []
        current_total_exposure = 0.0

        for s in signals:
            # Calculate Ideal Kelly Size
            raw_size = s['raw_kelly'] * self.FRACTIONAL_KELLY
            
            # Cap single position size
            allocated_size = min(raw_size, self.MAX_SINGLE_POSITION_SIZE)
            
            # Ensure we don't have microscopic bets (< 0.5%)
            if allocated_size < 0.005: 
                continue

            # Check if we have room in the portfolio
            if current_total_exposure + allocated_size > self.MAX_TOTAL_PORTFOLIO_EXPOSURE:
                # Take whatever is left in the bucket?
                remaining_room = self.MAX_TOTAL_PORTFOLIO_EXPOSURE - current_total_exposure
                if remaining_room > 0.005: # Only if meaningful room left
                    allocated_size = remaining_room
                else:
                    break # Portfolio is full, stop adding signals

            # Add to portfolio
            s['recommended_bet_size_pct'] = round(allocated_size * 100, 2)
            s['kelly_fraction'] = round(s['raw_kelly'] * 100, 2)
            
            final_output.append(s)
            current_total_exposure += allocated_size

            if current_total_exposure >= self.MAX_TOTAL_PORTFOLIO_EXPOSURE:
                break

        return final_output

    async def _save_signals(self, signals: List[Dict]):
        self.db.supabase.table("copy_trading_signals").delete().neq("position_id", "_").execute()
        if not signals: return
        
        batch_size = 500
        for i in range(0, len(signals), batch_size):
            batch = signals[i : i + batch_size]
            try:
                self.db.supabase.table("copy_trading_signals").insert(batch).execute()
            except Exception as e:
                logger.error(f"❌ Error saving signal batch: {e}")

async def main():
    engine = CopySignalEngine()
    await engine.run_signal_generation()

if __name__ == "__main__":
    asyncio.run(main())