export interface EliteTrader {
    proxy_wallet: string;
    tier: string;
    rank_in_tier: number;
    total_volume: number | null;
    roi: number | null;
    win_rate: number | null;
    composite_score: number | null;
    n_positions: number | null;
    n_markets: number | null;
    meets_thresholds: boolean;
  }
  
  export interface EliteTagComparison {
    tag: string;
    elite_trader_count: number;
    non_elite_trader_count: number;
    total_trader_count: number;
    elite_avg_roi: number;
    non_elite_avg_roi: number;
    elite_avg_win_rate: number;
    non_elite_avg_win_rate: number;
    elite_total_volume: number;
    non_elite_total_volume: number;
    performance_edge: number; // The gap between elite and avg
    volume_concentration: number;
    number_of_events: number; // Number of events tracked in this category
  }
  
  export interface EliteOpenPosition {
    id: string;
    proxy_wallet: string;
    title: string | null;
    event_category: string | null;
    outcome: string | null;
    size: number | null; // Amount Invested/Size
    avg_entry_price: number | null;
    current_price: number | null;
    unrealized_pnl: number | null;
    position_value: number | null;
    event_slug: string | null;
    event_end_date: string | null;
    raw_data: {
      icon?: string;
    } | null;
  }