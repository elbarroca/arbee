"""
Polymarket API Client - Clean & Simplified
Gamma API (events/markets) + CLOB API (orderbooks/prices)
"""
import json
import logging
import httpx
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta
try:
    from py_clob_client.client import ClobClient
    HAS_CLOB_CLIENT = True
except ImportError:
    HAS_CLOB_CLIENT = False
    ClobClient = None
from config import settings

# Exceptions
class PolymarketError(Exception):
    pass

class MarketNotFoundError(PolymarketError):
    pass

class InvalidOrderbookError(PolymarketError):
    pass

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolymarketGamma:
    """Gamma API - Events & Markets metadata"""

    def __init__(self):
        self.base_url = settings.POLYMARKET_GAMMA_URL
        self.markets_url = f"{self.base_url}/markets"
        self.events_url = f"{self.base_url}/events"

    async def get_events(self, active=True, limit=100, **filters) -> List[Dict]:
        """Get events with optional filters"""
        params = {"active": str(active).lower(), "limit": limit, **filters}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.events_url, params=params)
            response.raise_for_status()
            data = response.json()

            return [self._normalize_event(e) for e in data] if isinstance(data, list) else []

    async def get_markets(self, active=True, limit=100, **filters) -> List[Dict]:
        """Get markets with optional filters"""
        params = {"active": str(active).lower(), "limit": limit, **filters}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.markets_url, params=params)
            response.raise_for_status()
            data = response.json()

            return [self._normalize_market(m) for m in data] if isinstance(data, list) else []

    async def get_market(self, slug: str) -> Dict:
        """Get single market by slug"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.markets_url, params={"slug": slug})
            response.raise_for_status()
            data = response.json()

            if not data or not isinstance(data, list):
                raise MarketNotFoundError(f"Market not found: {slug}")

            return self._normalize_market(data[0])

    def _normalize_event(self, event: Dict) -> Dict:
        """Normalize event data"""
        event.setdefault('markets', [])
        event.setdefault('tags', [])

        # Normalize nested markets
        if event.get('markets'):
            event['markets'] = [self._normalize_market(m) for m in event['markets']]

        return event

    def _normalize_market(self, market: Dict) -> Dict:
        """Normalize market data"""
        # Parse string arrays
        for field in ['outcomes', 'outcomePrices', 'clobTokenIds']:
            value = market.get(field, [])
            if isinstance(value, str):
                try:
                    market[field] = json.loads(value)
                except:
                    market[field] = []

        # Parse numeric fields
        for field in ['volumeNum', 'liquidityNum', 'volume', 'liquidity']:
            if field in market:
                market[field] = float(market.get(field, 0) or 0)

        # Set defaults
        market.setdefault('active', True)
        market.setdefault('slug', '')
        market.setdefault('question', '')

        return market


class PolymarketCLOB:
    """CLOB API - Orderbooks & Prices"""

    def __init__(self):
        if not HAS_CLOB_CLIENT or ClobClient is None:
            logger.warning("py_clob_client not installed - CLOB functionality disabled")
            self.client = None
            return
        
        # Only pass key if it's set and valid (not empty, not placeholder)
        key = getattr(settings, 'POLYMARKET_PRIVATE_KEY', None)
        if key and key.strip() and key != '...' and len(key) > 10:
            self.client = ClobClient(settings.POLYMARKET_CLOB_URL, key=key, chain_id=137)
        else:
            self.client = ClobClient(settings.POLYMARKET_CLOB_URL, chain_id=137)

    def get_orderbook(self, token_id: str, depth=50) -> Dict:
        """Get orderbook for token"""
        if not token_id:
            raise ValueError("Invalid token_id")

        try:
            orderbook_obj = self.client.get_order_book(token_id)
        except Exception as e:
            raise InvalidOrderbookError(f"Failed to fetch orderbook: {e}")

        # Convert object to dict
        if hasattr(orderbook_obj, '__dict__'):
            orderbook = dict(orderbook_obj.__dict__)
        else:
            raise InvalidOrderbookError("Cannot convert orderbook to dict")

        # Convert bids/asks to dicts
        bids = self._convert_orders(orderbook.get('bids', []))
        asks = self._convert_orders(orderbook.get('asks', []))

        if not bids or not asks:
            raise InvalidOrderbookError("Empty orderbook")

        # Limit depth
        orderbook['bids'] = bids[:depth]
        orderbook['asks'] = asks[:depth]

        # Calculate metrics
        best_bid = float(bids[0]['price'])
        best_ask = float(asks[0]['price'])

        orderbook['best_bid'] = best_bid
        orderbook['best_ask'] = best_ask
        orderbook['mid_price'] = (best_bid + best_ask) / 2
        orderbook['spread'] = best_ask - best_bid
        orderbook['spread_bps'] = (best_ask - best_bid) * 10000

        # Calculate liquidity
        bid_liq = sum(float(b['price']) * float(b['size']) for b in bids[:20])
        ask_liq = sum(float(a['price']) * float(a['size']) for a in asks[:20])
        orderbook['total_liquidity'] = min(bid_liq, ask_liq)

        return orderbook

    def _convert_orders(self, orders: List) -> List[Dict]:
        """Convert OrderSummary objects to dicts"""
        result = []
        for order in orders:
            if hasattr(order, '__dict__'):
                result.append(dict(order.__dict__))
            elif isinstance(order, dict):
                result.append(order)
        return result

    def get_price(self, token_id: str, side="BUY") -> float:
        """Get current price for token"""
        if not token_id:
            raise ValueError("Invalid token_id")

        price = self.client.get_price(token_id, side=side)
        if price is None:
            raise ValueError(f"No price for token {token_id}")

        return float(price)


class PolymarketClient:
    """Main Polymarket client - simplified interface"""

    def __init__(self):
        self.gamma = PolymarketGamma()
        self.clob = PolymarketCLOB()

    # ========== Core Event/Market Retrieval ==========

    async def get_event(self, slug: str, with_orderbooks=False) -> Dict:
        """Get single event with all markets in clean JSON"""
        events = await self.gamma.get_events(limit=1, slug=slug)

        if not events:
            raise MarketNotFoundError(f"Event not found: {slug}")

        return await self._format_event(events[0], with_orderbooks)

    async def get_events(self, limit=10, with_orderbooks=False, **filters) -> List[Dict]:
        """Get multiple events with clean JSON formatting"""
        events_raw = await self.gamma.get_events(active=True, limit=limit, **filters)

        formatted = []
        for event in events_raw:
            try:
                formatted.append(await self._format_event(event, with_orderbooks))
            except Exception as e:
                logger.debug(f"Skip event: {e}")
                continue

        return formatted

    async def get_markets(self, limit=200, min_liquidity=100, max_spread=0.98, **filters) -> List[Dict]:
        """Get markets enriched with orderbooks, sorted by volume"""
        # Set default date filters
        if 'end_date_min' not in filters:
            filters['end_date_min'] = datetime.now().strftime('%Y-%m-%d')
        if 'end_date_max' not in filters:
            filters['end_date_max'] = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')

        # Fetch more markets than needed to allow for filtering
        markets = await self.gamma.get_markets(active=True, limit=limit * 3, **filters)

        # Sort by volume descending to get highest volume markets first
        markets.sort(key=lambda m: float(m.get('volumeNum', m.get('volume', 0)) or 0), reverse=True)

        enriched = []
        for market in markets[:limit * 10]:
            if len(enriched) >= limit:
                break

            result = await self._enrich_market(market, min_liquidity, max_spread)
            if result:
                enriched.append(result)

        logger.info(f"Enriched {len(enriched)} markets")
        return enriched

    # ========== Formatting & Enrichment ==========

    async def _format_event(self, event: Dict, with_orderbooks=False) -> Dict:
        """Format event into clean JSON schema"""
        markets_raw = event.get('markets', [])

        # Calculate total volume
        total_vol = sum(float(m.get('volumeNum', m.get('volume', 0))) for m in markets_raw)

        # Format each market
        formatted_markets = []
        for market in markets_raw:
            prices = market.get('outcomePrices', [])
            yes_price = float(prices[1]) if len(prices) >= 2 else 0.5
            no_price = float(prices[0]) if len(prices) >= 1 else 0.5

            market_dict = {
                "outcome": market.get('question', ''),
                "slug": market.get('slug', ''),
                "probability": round(yes_price * 100, 0),
                "probability_pct": f"{yes_price * 100:.0f}%",
                "volume": self._fmt_vol(float(market.get('volumeNum', market.get('volume', 0)))),
                "volume_raw": float(market.get('volumeNum', market.get('volume', 0))),
                "yes_price": yes_price,
                "no_price": no_price,
                "buy_yes_display": f"{yes_price * 100:.1f}¢",
                "buy_no_display": f"{no_price * 100:.1f}¢",
                "active": market.get('active', True)
            }

            # Add orderbook if requested
            if with_orderbooks:
                token_ids = market.get('clobTokenIds', [])
                if token_ids:
                    try:
                        ob = self.clob.get_orderbook(token_ids[0], depth=10)
                        market_dict["orderbook"] = {
                            "best_bid": ob['best_bid'],
                            "best_ask": ob['best_ask'],
                            "spread": f"{ob['spread'] * 100:.1f}%",
                            "liquidity": self._fmt_vol(ob['total_liquidity'])
                        }
                    except (InvalidOrderbookError, ValueError, KeyError) as e:
                        logger.debug(f"Failed to fetch orderbook for market {market.get('slug', 'unknown')}: {e}")

            formatted_markets.append(market_dict)

        # Sort by volume
        formatted_markets.sort(key=lambda m: m.get('volume_raw', 0), reverse=True)

        return {
            "event": {
                "id": event.get('id', ''),
                "slug": event.get('slug', ''),
                "title": event.get('title', ''),
                "description": event.get('description', ''),
                "total_volume": self._fmt_vol(total_vol),
                "total_volume_raw": total_vol,
                "market_count": len(formatted_markets),
                "end_date": self._fmt_date(event.get('endDate', '')),
                "status": "active" if event.get('active', True) else "closed",
                "category": event.get('category', ''),
                "markets": formatted_markets
            }
        }

    async def _enrich_market(self, market: Dict, min_liq: float, max_spread: float) -> Dict:
        """Enrich market with orderbook data"""
        try:
            token_ids = market.get('clobTokenIds', [])
            if not token_ids:
                return None

            orderbook = self.clob.get_orderbook(token_ids[0], depth=20)

            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])

            if not bids or not asks:
                return None

            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])
            spread = best_ask - best_bid

            if spread < 0 or spread > 1 or spread > max_spread:
                return None

            bid_liq = sum(float(b['price']) * float(b['size']) for b in bids[:10])
            ask_liq = sum(float(a['price']) * float(a['size']) for a in asks[:10])
            total_liq = min(bid_liq, ask_liq) * 1000

            if total_liq < min_liq:
                return None

            prices = market.get('outcomePrices', [])
            yes_prob = float(prices[1]) if len(prices) >= 2 else best_ask

            return {
                'market_slug': market.get('slug', ''),
                'question': market.get('question', ''),
                'category': market.get('category', ''),
                'yes_probability': round(yes_prob * 100, 2),
                'total_liquidity': total_liq,
                'total_volume': market.get('volumeNum', market.get('volume', 0)),
                'spread': round(spread * 100, 2),
                'spread_bps': round(spread * 10000, 0),
                'best_bid': best_bid,
                'best_ask': best_ask,
                'midpoint_price': round((best_bid + best_ask) / 2, 4),
                'end_date': market.get('endDate', ''),
                'orderbook': {
                    'bids': [{'price': b['price'], 'size': b['size']} for b in bids[:15]],
                    'asks': [{'price': a['price'], 'size': a['size']} for a in asks[:15]]
                }
            }

        except (InvalidOrderbookError, ValueError, KeyError, TypeError) as e:
            logger.debug(f"Failed to enrich market {market.get('slug', 'unknown')}: {e}")
            return None

    # ========== Utilities ==========
    def _fmt_vol(self, volume: float) -> str:
        """Format volume as readable string"""
        if volume >= 1_000_000:
            return f"${volume / 1_000_000:.2f}M"
        elif volume >= 1_000:
            return f"${volume / 1_000:.0f}K"
        else:
            return f"${volume:,.0f}"

    def _fmt_date(self, iso_date: str) -> str:
        """Format ISO date to readable"""
        if not iso_date:
            return "N/A"
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime("%b %d, %Y")
        except (ValueError, AttributeError) as e:
            logger.debug(f"Failed to parse date '{iso_date}': {e}")
            return iso_date
    
    async def get_related_markets(self, main_slug: str, question_text: str) -> List[Dict]:
        """
        Find related threshold markets for the same question.
        
        Strategy:
        1. Try to get event containing the main market, then get all markets in that event
        2. If event not found, search markets by question text similarity
        3. Extract threshold from outcome text (e.g., "30%+", "40%+")
        
        Args:
            main_slug: Slug of the main market
            question_text: The market question text
            
        Returns:
            List of related market dicts with structure:
            {
                "slug": "market-slug-30pct",
                "threshold": 30,  # extracted threshold value
                "market_price": 0.95,  # YES price
                "market_data": {...}  # full market data
            }
        """
        related_markets = []
        
        try:
            # Strategy 1: Get event containing main market
            main_market = await self.gamma.get_market(main_slug)
            event_id = main_market.get('eventId') or main_market.get('event_id')
            
            if event_id:
                # Get all markets in the same event
                # Try to get event by ID - if API doesn't support id filter, search by slug or get all markets
                events = []
                try:
                    # Try filtering by event ID
                    events = await self.gamma.get_events(limit=100, eventId=event_id)
                except Exception:
                    try:
                        # Fallback: try with id parameter
                        events = await self.gamma.get_events(limit=100, id=event_id)
                    except Exception:
                        # If event lookup fails, we'll use Strategy 2 (text similarity)
                        pass
                
                if events:
                    event = events[0]
                    markets_in_event = event.get('markets', [])
                    
                    for market in markets_in_event:
                        market = self.gamma._normalize_market(market)
                        market_slug = market.get('slug', '')
                        
                        # Skip the main market itself
                        if market_slug == main_slug:
                            continue
                        
                        # Extract threshold from question/outcome
                        threshold = self._extract_threshold(market.get('question', ''))
                        if threshold is None:
                            # Try extracting from outcomes
                            outcomes = market.get('outcomes', [])
                            for outcome in outcomes:
                                threshold = self._extract_threshold(str(outcome))
                                if threshold is not None:
                                    break
                        
                        # Only include markets with extractable thresholds
                        if threshold is not None:
                            # Extract market price
                            prices = market.get('outcomePrices', [])
                            market_price = None
                            
                            if prices and len(prices) >= 2:
                                try:
                                    market_price = float(prices[1])  # YES price
                                except (ValueError, TypeError):
                                    pass
                            
                            # Try orderbook if outcomePrices not available
                            if market_price is None:
                                token_ids = market.get('clobTokenIds', [])
                                if token_ids and len(token_ids) >= 2:
                                    try:
                                        orderbook = self.clob.get_orderbook(token_ids[1], depth=5)
                                        market_price = orderbook.get('mid_price')
                                    except Exception:
                                        pass
                            
                            if market_price is not None:
                                related_markets.append({
                                    "slug": market_slug,
                                    "threshold": threshold,
                                    "market_price": market_price,
                                    "market_data": market,
                                    "question": market.get('question', ''),
                                })
            
            # Strategy 2: If no event found, search by question text similarity
            if not related_markets:
                # Extract base question (remove threshold-specific parts)
                base_question = self._extract_base_question(question_text)
                
                # Search markets with similar question text
                all_markets = await self.gamma.get_markets(active=True, limit=500)
                
                for market in all_markets:
                    market = self.gamma._normalize_market(market)
                    market_slug = market.get('slug', '')
                    
                    # Skip main market
                    if market_slug == main_slug:
                        continue
                    
                    market_question = market.get('question', '')
                    
                    # Check if question is similar (contains base question)
                    if base_question.lower() in market_question.lower() or market_question.lower() in base_question.lower():
                        threshold = self._extract_threshold(market_question)
                        if threshold is None:
                            outcomes = market.get('outcomes', [])
                            for outcome in outcomes:
                                threshold = self._extract_threshold(str(outcome))
                                if threshold is not None:
                                    break
                        
                        if threshold is not None:
                            prices = market.get('outcomePrices', [])
                            market_price = None
                            
                            if prices and len(prices) >= 2:
                                try:
                                    market_price = float(prices[1])
                                except (ValueError, TypeError):
                                    pass
                            
                            if market_price is not None:
                                related_markets.append({
                                    "slug": market_slug,
                                    "threshold": threshold,
                                    "market_price": market_price,
                                    "market_data": market,
                                    "question": market_question,
                                })
            
            # Sort by threshold (ascending)
            related_markets.sort(key=lambda m: m.get('threshold', 0))
            
            logger.info(f"Found {len(related_markets)} related threshold markets for {main_slug}")
            return related_markets
            
        except Exception as e:
            logger.warning(f"Error discovering related markets: {e}")
            return []
    
    def _extract_threshold(self, text: str) -> Optional[int]:
        """
        Extract threshold percentage from text.
        
        Examples:
            "30%+" -> 30
            "40% or higher" -> 40
            "Score 50%+" -> 50
        """
        if not text:
            return None
        
        # Pattern: number followed by % and optionally +
        patterns = [
            r'(\d+)%\s*\+',  # "30%+", "40% +"
            r'(\d+)\s*%\s*or\s*higher',  # "30% or higher"
            r'score\s*(\d+)%',  # "score 30%"
            r'(\d+)%',  # Just "30%"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    threshold = int(match.group(1))
                    # Reasonable threshold range: 0-100
                    if 0 <= threshold <= 100:
                        return threshold
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_base_question(self, question_text: str) -> str:
        """
        Extract base question by removing threshold-specific parts.
        
        Example:
            "Google Gemini 3 score 30%+ on Humanity's Last Exam" 
            -> "Google Gemini 3 score on Humanity's Last Exam"
        """
        # Remove threshold patterns
        base = re.sub(r'\d+%\s*\+', '', question_text, flags=re.IGNORECASE)
        base = re.sub(r'\d+\s*%\s*or\s*higher', '', base, flags=re.IGNORECASE)
        base = re.sub(r'score\s*\d+%', 'score', base, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        base = re.sub(r'\s+', ' ', base).strip()
        
        return base
