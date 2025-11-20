'use client';

import { useState, useEffect, useCallback } from 'react';
import Image from 'next/image';
import { fetchMarkets, fetchEvents, fetchMarketStats } from '@/lib/api/markets';
import { Market, MarketEvent, GlobalMarketStats } from '@/types/market';
import MarketCharts from '@/components/scanner/MarketCharts';
import { Input } from '@/components/ui/input';
import { Search, ChevronLeft, ChevronRight, ArrowUpDown, ExternalLink, Loader2, Clock, Filter, X } from 'lucide-react';
import { cn } from '@/lib/utils';

type ViewMode = 'markets' | 'events';

export default function ScannerPage() {
  // Stats State
  const [stats, setStats] = useState<GlobalMarketStats | null>(null);
  
  // View Mode
  const [viewMode, setViewMode] = useState<ViewMode>('markets');
  
  // Table State
  const [markets, setMarkets] = useState<Market[]>([]);
  const [events, setEvents] = useState<MarketEvent[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  
  // Pagination & Filtering
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [filterExpiring, setFilterExpiring] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState<string>('');
  const [minVolume, setMinVolume] = useState<string>('');
  const [minLiquidity, setMinLiquidity] = useState<string>('');
  
  const [sortConfig, setSortConfig] = useState<{ key: string; dir: 'asc' | 'desc' }>({
    key: viewMode === 'markets' ? 'volume_24h' : 'total_volume',
    dir: 'desc'
  });

  // Initial Stats Load
  useEffect(() => {
    fetchMarketStats().then(setStats).catch(console.error);
  }, []);

  // Get unique categories for filter dropdown
  const [categories, setCategories] = useState<string[]>([]);
  
  useEffect(() => {
    if (stats?.categoryDistribution) {
      setCategories(stats.categoryDistribution.map(c => c.name));
    }
  }, [stats]);

  // Market Data Load
  const loadMarkets = useCallback(async () => {
    setLoading(true);
    try {
      const { data, total } = await fetchMarkets(
        page, 
        search, 
        sortConfig.key as keyof Market, 
        sortConfig.dir === 'asc', 
        filterExpiring,
        categoryFilter || undefined,
        minVolume ? parseFloat(minVolume) : undefined,
        minLiquidity ? parseFloat(minLiquidity) : undefined
      );
      setMarkets(data);
      setTotalCount(total);
    } catch (err) {
      console.error('Error loading markets:', err);
      console.error('Error details:', JSON.stringify(err, null, 2));
      if (err instanceof Error) {
        console.error('Error message:', err.message);
        console.error('Error stack:', err.stack);
      }
    } finally {
      setLoading(false);
    }
  }, [page, search, sortConfig, filterExpiring, categoryFilter, minVolume, minLiquidity]);

  // Events Data Load
  const loadEvents = useCallback(async () => {
    setLoading(true);
    try {
      const { data, total } = await fetchEvents(
        page,
        search,
        sortConfig.key as keyof MarketEvent,
        sortConfig.dir === 'asc',
        filterExpiring,
        categoryFilter || undefined,
        minVolume ? parseFloat(minVolume) : undefined,
        minLiquidity ? parseFloat(minLiquidity) : undefined
      );
      setEvents(data);
      setTotalCount(total);
    } catch (err) {
      console.error('Error loading events:', err);
    } finally {
      setLoading(false);
    }
  }, [page, search, sortConfig, filterExpiring, categoryFilter, minVolume, minLiquidity]);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (viewMode === 'markets') {
        loadMarkets();
      } else {
        loadEvents();
      }
    }, 300); // Debounce
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode, page, search, sortConfig, filterExpiring, categoryFilter, minVolume, minLiquidity]);

  useEffect(() => {
    setPage(1);
    setSortConfig({
      key: viewMode === 'markets' ? 'volume_24h' : 'total_volume',
      dir: 'desc'
    });
  }, [viewMode]);

  const handleSort = (key: string) => {
    setSortConfig(curr => ({ key, dir: curr.key === key && curr.dir === 'desc' ? 'asc' : 'desc' }));
    setPage(1);
  };

  const clearFilters = () => {
    setCategoryFilter('');
    setMinVolume('');
    setMinLiquidity('');
    setFilterExpiring(false);
    setPage(1);
  };

  const hasActiveFilters = categoryFilter || minVolume || minLiquidity || filterExpiring;

  // Helpers
  const currency = (n: number | null) => n ? `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '-';
  const percent = (n: number | null) => n ? `${(n * 100).toFixed(1)}%` : '-';

  return (
    <div className="flex flex-col h-full bg-[#050505] relative overflow-hidden">
      <div className="flex-1 overflow-y-auto px-6 md:px-8 py-8">
        
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-white mb-2">Market Scanner</h1>
          <p className="text-zinc-400 text-sm">Real-time analysis of active liquidity and volume across the ecosystem.</p>
        </div>

        {/* Visualizations */}
        <MarketCharts stats={stats} />

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b border-white/5">
          <button
            onClick={() => setViewMode('markets')}
            className={cn(
              "px-4 py-2 text-sm font-medium transition-colors border-b-2",
              viewMode === 'markets'
                ? "text-white border-blue-500"
                : "text-zinc-400 border-transparent hover:text-white"
            )}
          >
            Markets
          </button>
          <button
            onClick={() => setViewMode('events')}
            className={cn(
              "px-4 py-2 text-sm font-medium transition-colors border-b-2",
              viewMode === 'events'
                ? "text-white border-blue-500"
                : "text-zinc-400 border-transparent hover:text-white"
            )}
          >
            Events
          </button>
        </div>

        {/* Filter Toolbar */}
        <div className="flex flex-col gap-4 mb-6">
            <div className="flex flex-col xl:flex-row items-start xl:items-center justify-between gap-4">
                <div className="flex flex-col sm:flex-row gap-3 w-full xl:w-auto">
                    {/* Search */}
                    <div className="relative w-full sm:w-96 group">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 group-hover:text-zinc-300 transition-colors" />
                        <Input 
                            placeholder={`Find ${viewMode}...`}
                            value={search}
                            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
                            className="pl-10"
                        />
                    </div>
                    {/* Expiring Filter Toggle */}
                    <button
                        onClick={() => { setFilterExpiring(!filterExpiring); setPage(1); }}
                        className={cn(
                            "flex items-center gap-2 px-4 py-2 rounded-xl border transition-all text-sm font-medium whitespace-nowrap",
                            filterExpiring 
                                ? "bg-amber-500/10 border-amber-500/30 text-amber-400 hover:bg-amber-500/20" 
                                : "bg-zinc-900/50 border-white/10 text-zinc-400 hover:text-white hover:bg-zinc-900"
                        )}
                    >
                        <Clock className="w-4 h-4" />
                        Expiring (48h)
                    </button>
                    {/* Advanced Filters Toggle */}
                    <button
                        onClick={() => setShowFilters(!showFilters)}
                        className={cn(
                            "flex items-center gap-2 px-4 py-2 rounded-xl border transition-all text-sm font-medium whitespace-nowrap",
                            showFilters || hasActiveFilters
                                ? "bg-blue-500/10 border-blue-500/30 text-blue-400 hover:bg-blue-500/20" 
                                : "bg-zinc-900/50 border-white/10 text-zinc-400 hover:text-white hover:bg-zinc-900"
                        )}
                    >
                        <Filter className="w-4 h-4" />
                        Filters
                        {hasActiveFilters && (
                            <span className="ml-1 px-1.5 py-0.5 bg-blue-500/20 rounded text-xs">
                                {[categoryFilter, minVolume, minLiquidity, filterExpiring].filter(Boolean).length}
                            </span>
                        )}
                    </button>
                </div>
                {/* Pagination */}
                <div className="flex items-center gap-3 ml-auto">
                     <span className="text-xs text-zinc-500 font-mono">
                        {totalCount === 0 ? '0-0' : `${(page - 1) * 50 + 1}-${Math.min(page * 50, totalCount)}`} of {totalCount}
                     </span>
                     <div className="flex gap-1">
                        <button 
                            onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1 || loading}
                            className="p-2 border border-white/10 rounded-lg hover:bg-zinc-900 disabled:opacity-30 transition-colors"
                        >
                            <ChevronLeft className="w-4 h-4 text-zinc-400" />
                        </button>
                        <button 
                            onClick={() => setPage(p => p + 1)} disabled={(viewMode === 'markets' ? markets.length : events.length) < 50 || loading}
                            className="p-2 border border-white/10 rounded-lg hover:bg-zinc-900 disabled:opacity-30 transition-colors"
                        >
                            <ChevronRight className="w-4 h-4 text-zinc-400" />
                        </button>
                     </div>
                </div>
            </div>

            {/* Advanced Filters Panel */}
            {showFilters && (
                <div className="bg-zinc-900/50 border border-white/10 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-medium text-white">Advanced Filters</h3>
                        {hasActiveFilters && (
                            <button
                                onClick={clearFilters}
                                className="text-xs text-zinc-400 hover:text-white flex items-center gap-1"
                            >
                                <X className="w-3 h-3" />
                                Clear all
                            </button>
                        )}
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Category Filter */}
                        <div>
                            <label className="text-xs text-zinc-400 mb-1 block">Category</label>
                            <select
                                value={categoryFilter}
                                onChange={(e) => { setCategoryFilter(e.target.value); setPage(1); }}
                                className="w-full px-3 py-2 bg-zinc-800 border border-white/10 rounded-lg text-sm text-white focus:outline-none focus:border-blue-500"
                            >
                                <option value="">All Categories</option>
                                {categories.map(cat => (
                                    <option key={cat} value={cat}>{cat}</option>
                                ))}
                            </select>
                        </div>
                        {/* Min Volume Filter */}
                        <div>
                            <label className="text-xs text-zinc-400 mb-1 block">Min Volume ($)</label>
                            <Input
                                type="number"
                                placeholder="0"
                                value={minVolume}
                                onChange={(e) => { setMinVolume(e.target.value); setPage(1); }}
                                className="bg-zinc-800 border-white/10"
                            />
                        </div>
                        {/* Min Liquidity Filter */}
                        <div>
                            <label className="text-xs text-zinc-400 mb-1 block">Min Liquidity ($)</label>
                            <Input
                                type="number"
                                placeholder="0"
                                value={minLiquidity}
                                onChange={(e) => { setMinLiquidity(e.target.value); setPage(1); }}
                                className="bg-zinc-800 border-white/10"
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>

        {/* Data Table */}
        <div className="bg-zinc-900/20 border border-white/5 rounded-xl overflow-hidden min-h-[400px]">
            <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                    <thead className="bg-white/[0.02] text-[11px] text-zinc-500 uppercase tracking-wider font-semibold sticky top-0 z-10">
                        <tr>
                            {viewMode === 'markets' ? (
                                <>
                                    <th className="p-4 pl-6 min-w-[300px]">Market</th>
                                    <th className="p-4">Category</th>
                                    <th className="p-4 text-right cursor-pointer hover:text-white" onClick={() => handleSort('p_yes')}>
                                        Avg Price <ArrowUpDown className="w-3 h-3 inline" />
                                    </th>
                                    <th className="p-4 text-right cursor-pointer hover:text-white" onClick={() => handleSort('volume_24h')}>
                                        24h Vol <ArrowUpDown className="w-3 h-3 inline" />
                                    </th>
                                    <th className="p-4 text-right cursor-pointer hover:text-white" onClick={() => handleSort('total_volume')}>
                                        Total Vol <ArrowUpDown className="w-3 h-3 inline" />
                                    </th>
                                    <th className="p-4 text-right cursor-pointer hover:text-white" onClick={() => handleSort('liquidity')}>
                                        Liquidity <ArrowUpDown className="w-3 h-3 inline" />
                                    </th>
                                    <th className="p-4 text-right">Deadline</th>
                                    <th className="p-4"></th>
                                </>
                            ) : (
                                <>
                                    <th className="p-4 pl-6 min-w-[300px]">Event</th>
                                    <th className="p-4">Category</th>
                                    <th className="p-4 text-right">Markets</th>
                                    <th className="p-4 text-right cursor-pointer hover:text-white" onClick={() => handleSort('total_volume')}>
                                        Total Vol <ArrowUpDown className="w-3 h-3 inline" />
                                    </th>
                                    <th className="p-4 text-right cursor-pointer hover:text-white" onClick={() => handleSort('total_liquidity')}>
                                        Liquidity <ArrowUpDown className="w-3 h-3 inline" />
                                    </th>
                                    <th className="p-4 text-right">Deadline</th>
                                    <th className="p-4"></th>
                                </>
                            )}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5 text-sm">
                        {loading ? (
                            <tr>
                                <td colSpan={viewMode === 'markets' ? 8 : 7} className="h-64 relative">
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <Loader2 className="w-8 h-8 animate-spin text-blue-500"/>
                                    </div>
                                </td>
                            </tr>
                        ) : (viewMode === 'markets' ? markets.length === 0 : events.length === 0) ? (
                            <tr>
                                <td colSpan={viewMode === 'markets' ? 8 : 7} className="p-12 text-center text-zinc-500">
                                    No {viewMode} found matching criteria.
                                </td>
                            </tr>
                        ) : viewMode === 'markets' ? (
                            markets.map((m) => {
                                // Construct URL: Prefer event slug, then market raw_data slug, then market ID
                                const slug = m.event_slug || m.raw_data?.slug || m.id;
                                const marketUrl = `https://polymarket.com/event/${slug}`;
                                
                                // Get image/icon: prefer icon, fallback to image
                                const imageUrl = m.raw_data?.icon || m.raw_data?.image;
                                
                                // Average price is p_yes (probability of Yes)
                                const avgPrice = m.p_yes || 0;
                                
                                return (
                                    <tr key={m.id} className="group hover:bg-white/[0.02] transition-colors">
                                        <td className="p-4 pl-6">
                                            <div className="flex items-center gap-3 min-w-0">
                                                {imageUrl ? (
                                                    <Image src={imageUrl} alt="Market icon" width={32} height={32} className="w-8 h-8 rounded-md object-cover bg-zinc-800 flex-shrink-0" />
                                                ) : (
                                                    <div className="w-8 h-8 rounded-md bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-600 flex-shrink-0">PM</div>
                                                )}
                                                <div className="flex flex-col min-w-0 flex-1">
                                                    <span className="font-medium text-zinc-200 break-words" title={m.title}>{m.title}</span>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            <span className="px-2.5 py-1 rounded-md bg-zinc-800/50 border border-white/5 text-xs text-zinc-400 whitespace-nowrap">
                                                {m.category}
                                            </span>
                                        </td>
                                        <td className="p-4 text-right font-mono">
                                            <span className={cn("font-bold", avgPrice > 0.5 ? "text-emerald-400" : "text-blue-400")}>
                                                {percent(avgPrice)}
                                            </span>
                                        </td>
                                        <td className="p-4 text-right font-mono text-zinc-300">
                                            {currency(m.volume_24h)}
                                        </td>
                                        <td className="p-4 text-right font-mono text-zinc-300">
                                            {currency(m.total_volume)}
                                        </td>
                                        <td className="p-4 text-right font-mono text-zinc-400">
                                            {currency(m.liquidity)}
                                        </td>
                                        <td className="p-4 text-right text-xs text-zinc-500 font-mono whitespace-nowrap">
                                            {m.close_date ? new Date(m.close_date).toLocaleDateString() : '-'}
                                        </td>
                                        <td className="p-4 text-right">
                                            <a 
                                                href={marketUrl} 
                                                target="_blank" 
                                                rel="noreferrer"
                                                className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-blue-400 transition-all"
                                            >
                                                <ExternalLink className="w-4 h-4" />
                                            </a>
                                        </td>
                                    </tr>
                                );
                            })
                        ) : (
                            events.map((e) => {
                                const eventUrl = `https://polymarket.com/event/${e.slug || e.id}`;
                                const imageUrl = e.raw_data?.icon || e.raw_data?.image;
                                
                                return (
                                    <tr key={e.id} className="group hover:bg-white/[0.02] transition-colors">
                                        <td className="p-4 pl-6">
                                            <div className="flex items-center gap-3 min-w-0">
                                                {imageUrl ? (
                                                    <Image src={imageUrl} alt="Event icon" width={32} height={32} className="w-8 h-8 rounded-md object-cover bg-zinc-800 flex-shrink-0" />
                                                ) : (
                                                    <div className="w-8 h-8 rounded-md bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-600 flex-shrink-0">EV</div>
                                                )}
                                                <div className="flex flex-col min-w-0 flex-1">
                                                    <span className="font-medium text-zinc-200 break-words" title={e.title}>{e.title}</span>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="p-4">
                                            <span className="px-2.5 py-1 rounded-md bg-zinc-800/50 border border-white/5 text-xs text-zinc-400 whitespace-nowrap">
                                                {e.category}
                                            </span>
                                        </td>
                                        <td className="p-4 text-right font-mono text-zinc-300">
                                            {e.market_count || 0}
                                        </td>
                                        <td className="p-4 text-right font-mono text-zinc-300">
                                            {currency(e.total_volume)}
                                        </td>
                                        <td className="p-4 text-right font-mono text-zinc-400">
                                            {currency(e.total_liquidity)}
                                        </td>
                                        <td className="p-4 text-right text-xs text-zinc-500 font-mono whitespace-nowrap">
                                            {e.end_date ? new Date(e.end_date).toLocaleDateString() : '-'}
                                        </td>
                                        <td className="p-4 text-right">
                                            <a 
                                                href={eventUrl} 
                                                target="_blank" 
                                                rel="noreferrer"
                                                className="inline-flex items-center justify-center w-8 h-8 rounded-lg hover:bg-white/10 text-zinc-500 hover:text-blue-400 transition-all"
                                            >
                                                <ExternalLink className="w-4 h-4" />
                                            </a>
                                        </td>
                                    </tr>
                                );
                            })
                        )}
                    </tbody>
                </table>
            </div>
        </div>

      </div>
    </div>
  );
}