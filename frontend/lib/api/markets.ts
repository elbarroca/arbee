import { supabase } from '@/lib/supabase';
import { Market, MarketEvent, GlobalMarketStats } from '@/types/market';

export async function fetchMarketStats(): Promise<GlobalMarketStats> {
  // 1. Fetch all ACTIVE events with relevant columns
  // We use 'events' because they hold the aggregated volume for their markets
  // Remove default 1000 row limit by fetching in batches
  let allEvents: Array<{ category: string | null; total_volume: number | null; end_date: string | null }> = [];
  let from = 0;
  const batchSize = 1000;
  let hasMore = true;

  while (hasMore) {
    const { data, error } = await supabase
      .from('events')
      .select('category, total_volume, end_date')
      .eq('status', 'active')
      .range(from, from + batchSize - 1);

    if (error) throw error;

    if (data && data.length > 0) {
      allEvents = allEvents.concat(data);
      hasMore = data.length === batchSize;
      from += batchSize;
    } else {
      hasMore = false;
    }
  }

  const data = allEvents;

  // 2. Initialize Aggregators
  let totalVolume = 0;
  let expiringSoon = 0;
  const categoryMap: Record<string, { volume: number; count: number }> = {};
  
  const now = new Date();
  const tomorrow = new Date(now.getTime() + 24 * 60 * 60 * 1000);

  // 3. Process Data in Memory (Fast enough for < 10k events)
  data?.forEach((event) => {
    const vol = Number(event.total_volume || 0);
    const cat = event.category || 'Uncategorized';

    // Global Sums
    totalVolume += vol;

    // Expiry Check (Active AND ends within 24h)
    if (event.end_date) {
      const endDate = new Date(event.end_date);
      if (endDate > now && endDate < tomorrow) {
        expiringSoon++;
      }
    }

    // Category Aggregation
    if (!categoryMap[cat]) {
      categoryMap[cat] = { volume: 0, count: 0 };
    }
    categoryMap[cat].volume += vol;
    categoryMap[cat].count += 1;
  });

  // 4. Sort Categories
  const sortedCategories = Object.entries(categoryMap)
    .map(([name, stat]) => ({ name, ...stat }))
    .sort((a, b) => b.volume - a.volume);

  return {
    totalVolume,
    activeEvents: data?.length || 0,
    expiringSoon,
    topCategory: sortedCategories.length > 0 ? sortedCategories[0] : null,
    categoryDistribution: sortedCategories
  };
}

export async function fetchMarkets(
  page: number = 1,
  search: string = '',
  sortBy: keyof Market = 'volume_24h',
  ascending: boolean = false,
  filterExpiring: boolean = false,
  categoryFilter?: string,
  minVolume?: number,
  minLiquidity?: number
) {
  const ITEMS_PER_PAGE = 50;
  const from = (page - 1) * ITEMS_PER_PAGE;
  const to = from + ITEMS_PER_PAGE - 1;

  // Markets use status 'open' (not 'active')
  // First, fetch markets without join to ensure basic query works
  let query = supabase
    .from('markets')
    .select('*', { count: 'exact' })
    .eq('status', 'open')
    .order(sortBy, { ascending })
    .range(from, to);

  if (search) {
    query = query.ilike('title', `%${search}%`);
  }

  if (categoryFilter) {
    query = query.eq('category', categoryFilter);
  }

  if (minVolume !== undefined) {
    query = query.gte('total_volume', minVolume);
  }

  if (minLiquidity !== undefined) {
    query = query.gte('liquidity', minLiquidity);
  }

  // 48-Hour Expiry Filter
  if (filterExpiring) {
    const now = new Date();
    const in48h = new Date(now.getTime() + 48 * 60 * 60 * 1000);
    query = query
      .gt('close_date', now.toISOString())
      .lt('close_date', in48h.toISOString());
  }

  const { data, count, error } = await query;
  if (error) {
    console.error('Supabase query error:', error);
    throw error;
  }

  if (!data || data.length === 0) {
    return {
      data: [],
      total: count || 0,
      totalPages: Math.ceil((count || 0) / ITEMS_PER_PAGE),
    };
  }

  // Fetch event slugs for the markets in a separate query
  const eventIds = [...new Set(data.map((m: Market) => m.event_id).filter(Boolean))];
  const eventSlugsMap: Record<string, string> = {};

  if (eventIds.length > 0) {
    const { data: eventsData, error: eventsError } = await supabase
      .from('events')
      .select('id, slug')
      .in('id', eventIds);

    if (!eventsError && eventsData) {
      eventsData.forEach((event: { id: string; slug: string | null }) => {
        if (event.slug) {
          eventSlugsMap[event.id] = event.slug;
        }
      });
    }
  }

  // Transform data to include slug from events
  const transformedData = (data || []).map((market: Market) => {
    const eventSlug = market.event_id ? eventSlugsMap[market.event_id] || null : null;
    return {
      ...market,
      event_slug: eventSlug,
    };
  });

  return {
    data: transformedData as Market[],
    total: count || 0,
    totalPages: Math.ceil((count || 0) / ITEMS_PER_PAGE),
  };
}

export async function fetchEvents(
  page: number = 1,
  search: string = '',
  sortBy: keyof MarketEvent = 'total_volume',
  ascending: boolean = false,
  filterExpiring: boolean = false,
  categoryFilter?: string,
  minVolume?: number,
  minLiquidity?: number
) {
  const ITEMS_PER_PAGE = 50;
  const from = (page - 1) * ITEMS_PER_PAGE;
  const to = from + ITEMS_PER_PAGE - 1;

  let query = supabase
    .from('events')
    .select('*', { count: 'exact' })
    .eq('status', 'active')
    .order(sortBy, { ascending })
    .range(from, to);

  if (search) {
    query = query.ilike('title', `%${search}%`);
  }

  if (categoryFilter) {
    query = query.eq('category', categoryFilter);
  }

  if (minVolume !== undefined) {
    query = query.gte('total_volume', minVolume);
  }

  if (minLiquidity !== undefined) {
    query = query.gte('total_liquidity', minLiquidity);
  }

  // 48-Hour Expiry Filter
  if (filterExpiring) {
    const now = new Date();
    const in48h = new Date(now.getTime() + 48 * 60 * 60 * 1000);
    query = query
      .gt('end_date', now.toISOString())
      .lt('end_date', in48h.toISOString());
  }

  const { data, count, error } = await query;
  if (error) {
    console.error('Supabase query error:', error);
    throw error;
  }

  return {
    data: (data || []) as MarketEvent[],
    total: count || 0,
    totalPages: Math.ceil((count || 0) / ITEMS_PER_PAGE),
  };
}