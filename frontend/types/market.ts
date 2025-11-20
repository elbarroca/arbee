export interface MarketEvent {
  id: string;
  title: string;
  category: string;
  total_liquidity: number | null;
  total_volume: number | null;
  start_date: string | null;
  end_date: string | null;
  slug: string | null;
  market_count: number;
  raw_data: {
    icon?: string;
    image?: string;
  } | null;
}

export interface Market {
  id: string;
  event_id: string;
  title: string;
  category: string;
  p_yes: number | null;
  p_no: number | null;
  liquidity: number | null;
  volume_24h: number | null;
  total_volume: number | null;
  close_date: string | null;
  spread: number | null;
  event_slug?: string | null;
  raw_data: {
    icon?: string;
    image?: string;
    outcomes?: string[];
    slug?: string;
  } | null;
}

// New: Aggregated Stats Type
export interface GlobalMarketStats {
  totalVolume: number;
  activeEvents: number;
  expiringSoon: number; // Events ending in < 24h
  topCategory: {
    name: string;
    volume: number;
  } | null;
  categoryDistribution: {
    name: string;
    volume: number;
    count: number;
  }[];
}