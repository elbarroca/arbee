'use client';

import { useState, useEffect, useCallback } from 'react';
import { fetchWallets } from '@/lib/api/wallets';
import { WalletAnalytics } from '@/types/database';
import WalletDrawer from '@/components/wallets/WalletDrawer';
import { Input } from '@/components/ui/input';
import { Search, ArrowUpDown, ChevronLeft, ChevronRight, Loader2, Trophy, TrendingUp, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function WalletsPage() {
  // Data State
  const [wallets, setWallets] = useState<WalletAnalytics[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);

  // Pagination & Filter State
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [sortConfig, setSortConfig] = useState<{ key: keyof WalletAnalytics; dir: 'asc' | 'desc' }>({
    key: 'realized_pnl',
    dir: 'desc'
  });

  // Selection State
  const [selectedWallet, setSelectedWallet] = useState<WalletAnalytics | null>(null);

  // Fetch Data Function
  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const { data, total, totalPages: pages } = await fetchWallets(page, sortConfig.key, sortConfig.dir === 'asc', search);
      setWallets(data);
      setTotalCount(total);
      setTotalPages(pages); // Set this!
    } catch (err) {
      console.error("Failed to fetch wallets", err);
    } finally {
      setLoading(false);
    }
  }, [page, sortConfig, search]); // Dependencies

  // Trigger fetch on changes
  useEffect(() => {
    // Debounce search slightly to avoid too many requests
    const timeoutId = setTimeout(() => {
      loadData();
    }, 300);
    return () => clearTimeout(timeoutId);
  }, [loadData]);

  const handleSort = (key: keyof WalletAnalytics) => {
    setSortConfig(current => ({
      key,
      dir: current.key === key && current.dir === 'desc' ? 'asc' : 'desc'
    }));
    setPage(1); // Reset to page 1 on sort change
  };

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearch(e.target.value);
    setPage(1); // Reset to page 1 on search
  };

  // Helper for currency
  const formatCurrency = (val: number | null) => 
    val ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val) : '-';

  return (
    <div className="flex flex-col h-full bg-[#050505] relative overflow-hidden">
      
      {/* 1. Top Stats Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-6 md:p-8 pb-0">
        <div className="bg-zinc-900/30 border border-white/5 rounded-xl p-5 flex items-center gap-4">
           <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
              <TrendingUp className="w-6 h-6 text-emerald-400" />
           </div>
           <div>
              <div className="text-zinc-500 text-xs uppercase tracking-wider">Total Volume Tracked</div>
              <div className="text-2xl font-mono font-bold text-white">$124M+</div>
           </div>
        </div>
        <div className="bg-zinc-900/30 border border-white/5 rounded-xl p-5 flex items-center gap-4">
           <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center border border-blue-500/20">
              <Activity className="w-6 h-6 text-blue-400" />
           </div>
           <div>
              <div className="text-zinc-500 text-xs uppercase tracking-wider">Total Wallets</div>
              <div className="text-2xl font-mono font-bold text-white">{totalCount.toLocaleString()}</div>
           </div>
        </div>
        <div className="bg-zinc-900/30 border border-white/5 rounded-xl p-5 flex items-center gap-4">
           <div className="w-12 h-12 rounded-full bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
              <Trophy className="w-6 h-6 text-amber-400" />
           </div>
           <div>
              <div className="text-zinc-500 text-xs uppercase tracking-wider">Highest ROI</div>
              <div className="text-2xl font-mono font-bold text-white">4,200%</div>
           </div>
        </div>
      </div>

      {/* 2. Filter & Controls */}
      <div className="px-6 md:px-8 py-6 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="relative w-full md:w-96 group">
           <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500 group-hover:text-zinc-300 transition-colors" />
           <Input 
              placeholder="Search wallet address..." 
              value={search}
              onChange={handleSearch}
              className="pl-10"
           />
        </div>
        
        {/* Pagination Controls */}
        <div className="flex items-center gap-3">
            <span className="text-xs text-zinc-500 font-mono">
                Page {page} of {totalPages}
            </span>
            <div className="flex items-center gap-1">
                <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1 || loading}
                    className="p-2 rounded-lg border border-white/5 hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                    <ChevronLeft className="w-4 h-4 text-zinc-400" />
                </button>
                <button
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    disabled={page >= totalPages || loading}
                    className="p-2 rounded-lg border border-white/5 hover:bg-zinc-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                    <ChevronRight className="w-4 h-4 text-zinc-400" />
                </button>
            </div>
        </div>
      </div>

      {/* 3. Main Table */}
      <div className="flex-1 overflow-y-auto px-6 md:px-8 pb-8">
        <div className="bg-zinc-900/20 border border-white/5 rounded-2xl overflow-x-auto min-h-[400px] flex flex-col">
          <table className="w-full text-left border-collapse">
            <thead className="bg-zinc-900/30 sticky top-0 z-10 backdrop-blur-md">
              <tr className="text-[11px] text-zinc-500 uppercase tracking-wider font-semibold border-b border-white/5">
                <th className="p-4 pl-6 font-medium min-w-[200px]">Wallet</th>
                <th className="p-4 cursor-pointer hover:text-white transition-colors min-w-[80px]" onClick={() => handleSort('tier')}>Tier</th>
                <th className="p-4 text-right cursor-pointer hover:text-white transition-colors min-w-[120px]" onClick={() => handleSort('win_rate')}>Win Rate</th>
                <th className="p-4 text-right cursor-pointer hover:text-white transition-colors min-w-[100px]" onClick={() => handleSort('n_positions')}>Trades</th>
                <th className="p-4 text-right cursor-pointer hover:text-white transition-colors min-w-[140px]" onClick={() => handleSort('total_volume')}>Volume</th>
                <th className="p-4 text-right cursor-pointer hover:text-white transition-colors min-w-[140px]" onClick={() => handleSort('realized_pnl')}>
                   <div className="flex items-center justify-end gap-1">
                      Realized PnL <ArrowUpDown className="w-3 h-3" />
                   </div>
                </th>
                <th className="p-4 text-right cursor-pointer hover:text-white transition-colors min-w-[100px]" onClick={() => handleSort('roi')}>ROI</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-sm relative">
              {loading ? (
                 // Loading Overlay
                 <tr>
                    <td colSpan={7} className="h-64">
                        <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/50 backdrop-blur-sm z-20">
                            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                        </div>
                    </td>
                 </tr>
              ) : wallets.length === 0 ? (
                  <tr>
                      <td colSpan={7} className="p-12 text-center text-zinc-500">
                          No wallets found matching your criteria.
                      </td>
                  </tr>
              ) : (
                 wallets.map((wallet) => (
                  <tr 
                    key={wallet.proxy_wallet} 
                    onClick={() => setSelectedWallet(wallet)}
                    className="group hover:bg-white/[0.02] transition-colors cursor-pointer"
                  >
                    <td className="p-4 pl-6">
                       <div className="font-mono text-zinc-300 group-hover:text-blue-400 transition-colors truncate w-32 md:w-48">
                          {wallet.proxy_wallet}
                       </div>
                    </td>
                    <td className="p-4">
                       <span className={cn("px-2 py-0.5 rounded text-[10px] font-bold border", 
                          wallet.tier === 'S' ? "bg-amber-500/10 text-amber-400 border-amber-500/20" :
                          wallet.tier === 'A' ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" :
                          "bg-blue-500/10 text-blue-400 border-blue-500/20"
                       )}>
                          {wallet.tier || '-'}
                       </span>
                    </td>
                    <td className="p-4 text-right font-mono text-zinc-300">
                       <div className="flex items-center justify-end gap-2">
                          <div className="w-12 h-1 bg-zinc-800 rounded-full overflow-hidden">
                             <div className="h-full bg-blue-500" style={{ width: `${wallet.win_rate}%` }}></div>
                          </div>
                          <span>{wallet.win_rate?.toFixed(1)}%</span>
                       </div>
                    </td>
                    <td className="p-4 text-right font-mono text-zinc-400">{wallet.n_positions}</td>
                    <td className="p-4 text-right font-mono text-zinc-300">{formatCurrency(wallet.total_volume)}</td>
                    <td className={cn("p-4 text-right font-mono font-bold", 
                        (wallet.realized_pnl || 0) > 0 ? "text-emerald-400" : "text-red-400"
                    )}>
                        {(wallet.realized_pnl || 0) > 0 ? '+' : ''}{formatCurrency(wallet.realized_pnl)}
                    </td>
                    <td className={cn("p-4 text-right font-mono font-bold", 
                        (wallet.roi || 0) > 0 ? "text-emerald-400" : "text-red-400"
                    )}>
                        {wallet.roi?.toFixed(2)}%
                    </td>
                  </tr>
                 ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail Drawer */}
      <WalletDrawer 
        wallet={selectedWallet} 
        onClose={() => setSelectedWallet(null)} 
      />

    </div>
  );
}