'use client';

import { useEffect, useState, useMemo } from 'react';
import { X, ExternalLink, ArrowRight, Activity, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { WalletAnalytics, ClosedPosition } from '@/types/database';
import { fetchWalletTrades } from '@/lib/api/wallets';
import { cn } from '@/lib/utils';

interface WalletDrawerProps {
  wallet: WalletAnalytics | null;
  onClose: () => void;
}

export default function WalletDrawer({ wallet, onClose }: WalletDrawerProps) {
  const [trades, setTrades] = useState<ClosedPosition[]>([]);
  const [loading, setLoading] = useState(!!wallet);
  const [openingLink, setOpeningLink] = useState<string | null>(null);

  const effectiveTrades = useMemo(() => wallet ? trades : [], [wallet, trades]);

  useEffect(() => {
    if (wallet) {
      fetchWalletTrades(wallet.proxy_wallet)
        .then((tradesData) => {
          setTrades(tradesData);
          // Prefetch the first 3 market links for faster subsequent clicks
          tradesData.slice(0, 3).forEach((trade) => {
            if (trade.event_slug) {
              const link = document.createElement('link');
              link.rel = 'prefetch';
              link.href = `https://polymarket.com/event/${trade.event_slug}`;
              link.as = 'document';
              document.head.appendChild(link);
            }
          });
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [wallet]);

  // Formatters
  const currency = (n: number | null) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n || 0);
    
  const percent = (n: number | null) =>
    `${(n || 0).toFixed(2)}%`;

  const cents = (n: number | null) =>
    (n || 0).toFixed(2); // Display prices as 0.55

  return (
    <AnimatePresence>
      {wallet && (
        <>
          {/* Backdrop */}
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />

          {/* Drawer Panel */}
          <motion.div
            initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-full md:w-[700px] bg-[#09090b] border-l border-white/10 z-50 shadow-2xl flex flex-col font-sans"
          >
            {/* 1. Header Section */}
            <div className="p-4 md:p-8 pb-4 md:pb-6 border-b border-white/10 bg-[#0c0c0c]">
              <div className="flex justify-between items-start mb-4 md:mb-6 gap-2">
                <div className="flex-1 min-w-0">
                   <div className="flex items-center gap-2 mb-2 flex-wrap">
                     <span className={cn("px-2 py-0.5 rounded text-xs font-bold border",
                       wallet.tier === 'S' ? "bg-amber-500/10 text-amber-400 border-amber-500/20" :
                       wallet.tier === 'A' ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" :
                       "bg-blue-500/10 text-blue-400 border-blue-500/20"
                     )}>Tier {wallet.tier || 'N/A'}</span>
                     <span className="text-zinc-500 text-xs font-mono">{new Date().toLocaleDateString()}</span>
                   </div>
                   <h2 className="text-base md:text-xl lg:text-2xl font-bold text-white font-mono break-words">
                     {wallet.proxy_wallet}
                   </h2>
                </div>
                <button onClick={onClose} className="p-2 bg-white/5 hover:bg-white/10 rounded-full transition-colors flex-shrink-0">
                  <X className="w-5 h-5 text-zinc-400" />
                </button>
              </div>

              {/* High Level Stats Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 md:gap-4">
                <div className="p-3 md:p-4 rounded-xl bg-zinc-900/50 border border-white/5">
                  <div className="text-xs text-zinc-500 mb-1">Realized PnL</div>
                  <div className={cn("text-base md:text-xl font-mono font-bold truncate", (wallet.realized_pnl || 0) >= 0 ? "text-emerald-400" : "text-red-400")}>
                    {(wallet.realized_pnl || 0) > 0 ? '+' : ''}{currency(wallet.realized_pnl)}
                  </div>
                </div>
                <div className="p-3 md:p-4 rounded-xl bg-zinc-900/50 border border-white/5">
                  <div className="text-xs text-zinc-500 mb-1">Win Rate</div>
                  <div className="text-base md:text-xl font-mono font-bold text-blue-400">
                    {percent(wallet.win_rate)}
                  </div>
                </div>
                <div className="p-3 md:p-4 rounded-xl bg-zinc-900/50 border border-white/5 col-span-2 sm:col-span-1">
                  <div className="text-xs text-zinc-500 mb-1">Total Volume</div>
                  <div className="text-base md:text-xl font-mono font-bold text-zinc-200 truncate">
                    {currency(wallet.total_volume)}
                  </div>
                </div>
              </div>
            </div>

            {/* 2. Trade History List */}
            <div className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8 bg-[#050505]">
              <div className="flex items-center justify-between mb-4 md:mb-6 gap-2">
                <h3 className="text-xs md:text-sm font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
                   <Activity className="w-4 h-4" /> Position History
                </h3>
                <span className="text-xs text-zinc-600 whitespace-nowrap">{effectiveTrades.length} records</span>
              </div>
              
              {loading ? (
                 <div className="space-y-4">
                   {[1,2,3,4].map(i => <div key={i} className="h-32 bg-zinc-900/50 rounded-xl animate-pulse border border-white/5" />)}
                 </div>
              ) : effectiveTrades.length === 0 ? (
                <div className="text-center py-20 text-zinc-500">No trade history available.</div>
              ) : (
                <div className="space-y-4">
                  {effectiveTrades.map((trade) => {
                    const invested = trade.total_bought || 0;
                    const pnl = trade.realized_pnl || 0;
                    const isProfit = pnl >= 0;

                    // Determine Trade Status for Context
                    const entry = trade.avg_price || 0;
                    const exit = trade.cur_price || 0;

                    let statusLabel = "Closed";
                    let statusColor = "text-zinc-500";

                    if (exit >= 0.99) { statusLabel = "WON"; statusColor = "text-emerald-400"; }
                    else if (exit <= 0.01) { statusLabel = "LOST"; statusColor = "text-red-400"; }
                    else if (isProfit) { statusLabel = "SOLD PROFIT"; statusColor = "text-emerald-400"; }
                    else { statusLabel = "SOLD LOSS"; statusColor = "text-red-400"; }

                    return (
                      <div key={trade.id} className="group relative p-3 md:p-5 rounded-xl border border-white/5 bg-zinc-900/20 hover:bg-zinc-900/40 transition-all hover:border-white/10">

                        {/* Top Row: Header */}
                        <div className="flex justify-between items-start gap-2 md:gap-4 mb-3 md:mb-4">
                          <div className="flex items-start gap-2 md:gap-3 flex-1 min-w-0">
                             <div className="w-8 h-8 md:w-10 md:h-10 rounded-md bg-zinc-800 flex items-center justify-center text-xs font-bold text-zinc-500 flex-shrink-0">PM</div>
                             <div className="min-w-0 flex-1">
                                <h4 className="text-xs md:text-sm font-medium text-zinc-200 leading-snug mb-1 line-clamp-2">
                                  {trade.title}
                                </h4>
                                <div className="flex items-center gap-1.5 text-[10px] text-zinc-500 flex-wrap">
                                   <span className="truncate">{trade.event_category}</span>
                                   <span>•</span>
                                   <span className="whitespace-nowrap">{new Date(trade.timestamp * 1000).toLocaleDateString()}</span>
                                </div>
                             </div>
                          </div>

                          <div className="text-right flex-shrink-0">
                             <div className={cn("text-base md:text-lg font-mono font-bold whitespace-nowrap", isProfit ? "text-emerald-400" : "text-red-400")}>
                                {isProfit ? '+' : ''}{currency(pnl)}
                             </div>
                             <div className={cn("text-[10px] font-bold uppercase tracking-wider", statusColor)}>
                                {statusLabel}
                             </div>
                          </div>
                        </div>

                        {/* Middle Row: The "Strategy" Visualization */}
                        <div className="bg-[#0a0a0a] rounded-lg p-2.5 md:p-3 border border-white/5 mb-2 md:mb-3">
                           <div className="flex items-center justify-between gap-2 text-xs flex-wrap sm:flex-nowrap">
                              {/* Outcome Side */}
                              <div className="flex flex-col gap-1">
                                 <span className="text-zinc-500 text-[10px] uppercase">Position</span>
                                 <span className={cn(
                                   "px-1.5 md:px-2 py-0.5 rounded text-[10px] md:text-[11px] font-bold border self-start whitespace-nowrap",
                                   trade.outcome === 'Yes' || trade.outcome === 'Over'
                                     ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                                     : "bg-blue-500/10 text-blue-400 border-blue-500/20"
                                 )}>
                                   {trade.outcome}
                                 </span>
                              </div>

                              {/* Price Movement Arrow */}
                              <div className="flex items-center gap-2 md:gap-3">
                                 <div className="text-right">
                                    <div className="text-zinc-500 text-[10px] uppercase">Entry</div>
                                    <div className="font-mono text-zinc-300 text-xs">{cents(entry)}¢</div>
                                 </div>
                                 <ArrowRight className="w-3 h-3 md:w-4 md:h-4 text-zinc-600" />
                                 <div className="text-left">
                                    <div className="text-zinc-500 text-[10px] uppercase">Exit</div>
                                    <div className={cn("font-mono font-bold text-xs", statusColor)}>
                                      {cents(exit)}¢
                                    </div>
                                 </div>
                              </div>

                              {/* Size Side */}
                              <div className="text-right">
                                 <div className="text-zinc-500 text-[10px] uppercase">Size</div>
                                 <div className="font-mono text-zinc-300 text-xs whitespace-nowrap">{currency(invested)}</div>
                              </div>
                           </div>
                        </div>

                        {/* Bottom Row: Link */}
                        <div className="flex justify-end">
                           {trade.event_slug && (
                                <button
                                  onClick={() => {
                                    setOpeningLink(trade.id);
                                    // Small delay to show loading state, then open link
                                    setTimeout(() => {
                                      window.open(`https://polymarket.com/event/${trade.event_slug}`, '_blank', 'noopener');
                                      setOpeningLink(null);
                                    }, 300);
                                  }}
                                  disabled={openingLink === trade.id}
                                  className="flex items-center gap-1 text-[11px] text-blue-500 hover:text-blue-400 transition-colors group/link disabled:opacity-50"
                                >
                                    {openingLink === trade.id ? (
                                      <>
                                        Opening... <Loader2 className="w-3 h-3 animate-spin" />
                                      </>
                                    ) : (
                                      <>
                                        Analyze Market <ExternalLink className="w-3 h-3 transition-transform group-hover/link:translate-x-0.5" />
                                      </>
                                    )}
                                </button>
                            )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}