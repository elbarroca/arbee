import { supabase } from '@/lib/supabase';
import { EliteTrader, EliteTagComparison, EliteOpenPosition } from '@/types/elite';

// 1. Fetch ALL Tag Comparisons for the Analysis Tab
export async function fetchEliteTags() {
  const { data, error } = await supabase
    .from('elite_tag_comparisons')
    .select('*')
    .order('number_of_events', { ascending: false }); // Rank by number of events (highest first)

  if (error) throw error;
  return data as EliteTagComparison[];
}

// 2. Fetch The Leaderboard (Keep Pagination)
export async function fetchEliteTraders(page = 1, categories: string[] = []) {
  const ITEMS = 50;
  const from = (page - 1) * ITEMS;
  const to = from + ITEMS - 1;

  let walletIds: string[] | null = null;

  // 1. If Categories selected, find wallets active in those categories
  if (categories.length > 0) {
    const { data: positions } = await supabase
      .from('elite_open_positions')
      .select('proxy_wallet')
      .in('event_category', categories); // Exact match for selected categories

    if (positions && positions.length > 0) {
      // Deduplicate wallet IDs
      walletIds = Array.from(new Set(positions.map(p => p.proxy_wallet)));
    } else {
      // No positions found, return empty result
      return {
        data: [] as EliteTrader[],
        total: 0
      };
    }
  }

  // 2. Build Main Query
  let query = supabase
    .from('elite_traders')
    .select('*', { count: 'exact' })
    .order('composite_score', { ascending: false });

  // Apply Filters
  if (walletIds !== null && walletIds.length > 0) {
    // Only show traders found in step 1
    query = query.in('proxy_wallet', walletIds);
  }

  // Pagination
  query = query.range(from, to);

  const { data, count, error } = await query;
  if (error) throw error;

  return {
    data: data as EliteTrader[],
    total: count || 0
  };
}

// 3. Fetch Active Positions
export async function fetchElitePositions(walletId: string) {
  const { data, error } = await supabase
    .from('elite_open_positions')
    .select('*')
    .eq('proxy_wallet', walletId)
    .order('unrealized_pnl', { ascending: false });

  if (error) throw error;
  return data as EliteOpenPosition[];
}

// 4. Fetch All Unique Event Categories
export async function fetchEventCategories() {
  const { data, error } = await supabase
    .from('elite_open_positions')
    .select('event_category')
    .not('event_category', 'is', null);

  if (error) throw error;

  // Get unique categories and sort alphabetically
  const uniqueCategories = Array.from(new Set(
    data.map(p => p.event_category).filter(Boolean)
  )).sort();

  return uniqueCategories as string[];
}