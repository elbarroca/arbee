export interface WalletAnalytics {
    proxy_wallet: string;
    total_volume: number | null;
    realized_pnl: number | null;
    roi: number | null;
    n_positions: number | null;
    n_wins: number | null;
    n_losses: number | null;
    win_rate: number | null;
    n_markets: number | null;
    pnl_all_time: number | null;
    roi_score: number | null;
    win_rate_score: number | null;
    tier: string | null;
    last_sync_at: string | null;
  }
  
  export interface ClosedPosition {
    id: string;
    proxy_wallet: string;
    title: string | null;
    event_category: string | null;
    event_slug: string | null;
    outcome: string | null; // e.g. "Yes", "Cowboys"
    outcome_index: number | null;
    total_bought: number | null; // Initial Investment
    avg_price: number | null; // Entry
    cur_price: number | null; // Exit/Current
    realized_pnl: number | null;
    timestamp: number; // Unix timestamp
    event_tags: string[] | null; // JSONB
  }