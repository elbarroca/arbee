'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';
import { X, ExternalLink, ArrowRight, Flame } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { EliteTrader, EliteOpenPosition } from '@/types/elite';
import { fetchElitePositions } from '@/lib/api/elites';
import { cn } from '@/lib/utils';

interface DrawerProps {
  trader: EliteTrader | null;
  highlightCategory?: string;
  onClose: () => void;
}

export default function ElitePositionsDrawer({ trader, highlightCategory, onClose }: DrawerProps) {
  const [positions, setPositions] = useState<EliteOpenPosition[]>([]);
  const [loading, setLoading] = useState(!!trader);

  useEffect(() => {
    if (trader) {
      fetchElitePositions(trader.proxy_wallet)
        .then(setPositions)
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [trader]);

  const currency = (n: number | null) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(n || 0);

  const cents = (n: number | null) => (n || 0).toFixed(2);

  // Count positions in the highlighted category
  const relevantCount = highlightCategory
    ? positions.filter(p => p.event_category?.toLowerCase().includes(highlightCategory.toLowerCase())).length
    : 0;

  return (
    <AnimatePresence>
      {trader && (
        <>
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          />
          <motion.div
            initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }}
            className="fixed right-0 top-0 h-full w-full md:w-[600px] bg-[#09090b] border-l border-white/10 z-50 shadow-2xl flex flex-col"
          >
            {/* Header */}
            <div className="p-6 border-b border-white/10 bg-[#0c0c0c]">
              <div className="flex justify-between items-start mb-4">
                <div>
                    <div className="flex gap-2 mb-2">
                        <span className="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 text-xs font-bold border border-blue-500/30">
                            Rank #{trader.rank_in_tier}
                        </span>
                        <span className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 text-xs font-mono border border-white/5">
                            Tier {trader.tier}
                        </span>
                    </div>
                    <h2 className="text-lg font-mono font-bold text-white break-all">{trader.proxy_wallet}</h2>
                </div>
                <button onClick={onClose}><X className="w-5 h-5 text-zinc-500 hover:text-white" /></button>
              </div>
              <div className="grid grid-cols-2 gap-4">
                  <div className="bg-zinc-900 p-3 rounded-lg border border-white/5">
                      <div className="text-xs text-zinc-500">Total Volume</div>
                      <div className="text-lg font-mono font-bold text-white">{currency(trader.total_volume)}</div>
                  </div>
                  <div className="bg-zinc-900 p-3 rounded-lg border border-white/5">
                      <div className="text-xs text-zinc-500">Win Rate</div>
                      <div className="text-lg font-mono font-bold text-emerald-400">{trader.win_rate?.toFixed(1)}%</div>
                  </div>
                  <div className="bg-zinc-900 p-3 rounded-lg border border-white/5">
                      <div className="text-xs text-zinc-500">ROI</div>
                      <div className="text-lg font-mono font-bold text-emerald-400">+{trader.roi?.toFixed(2)}%</div>
                  </div>
                  <div className="bg-zinc-900 p-3 rounded-lg border border-white/5">
                      <div className="text-xs text-zinc-500">Composite Score</div>
                      <div className="text-lg font-mono font-bold text-purple-400">{trader.composite_score?.toFixed(1)}</div>
                  </div>
              </div>

              {/* NEW: Sector Signal Banner */}
              {highlightCategory && relevantCount > 0 && (
                  <div className="mt-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center gap-3 animate-in slide-in-from-top-2">
                      <div className="p-2 bg-amber-500/20 rounded-full text-amber-400">
                          <Flame className="w-4 h-4" />
                      </div>
                      <div>
                          <div className="text-sm font-bold text-amber-100">Sector Match Detected</div>
                          <div className="text-xs text-amber-500/80">
                              This trader has <span className="font-bold text-white">{relevantCount}</span> open positions in {highlightCategory}.
                          </div>
                      </div>
                  </div>
              )}
            </div>

            {/* Active Positions Section */}
            <div className="flex-1 overflow-y-auto p-6 bg-[#050505]">
                {/* Explicit Header Title */}
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2 uppercase tracking-wider">
                        <span className="relative flex h-2.5 w-2.5">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                          <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                        </span>
                        Live Open Positions
                    </h3>
                    <span className="text-xs text-zinc-500 bg-zinc-900 px-2 py-1 rounded border border-white/5">
                        {positions.length} Active Contracts
                    </span>
                </div>

                {loading ? (
                    <div className="space-y-3">{[1,2,3].map(i => <div key={i} className="h-24 bg-zinc-900 rounded-xl animate-pulse" />)}</div>
                ) : positions.length === 0 ? (
                    <div className="text-center text-zinc-500 py-10">No open positions currently.</div>
                ) : (
                    <div className="space-y-4">
                        {/* Sort: Put highlighted positions first */}
                        {positions.sort((a, b) => {
                            const aMatch = highlightCategory && a.event_category?.toLowerCase().includes(highlightCategory.toLowerCase());
                            const bMatch = highlightCategory && b.event_category?.toLowerCase().includes(highlightCategory.toLowerCase());
                            return (aMatch === bMatch) ? 0 : aMatch ? -1 : 1;
                        }).map(pos => {
                            const isProfit = (pos.unrealized_pnl || 0) >= 0;
                            const isMatch = highlightCategory && pos.event_category?.toLowerCase().includes(highlightCategory.toLowerCase());

                            return (
                                <div
                                    key={pos.id}
                                    className={cn(
                                        "p-4 rounded-xl border transition-all relative overflow-hidden",
                                        isMatch
                                            ? "bg-amber-950/10 border-amber-500/30 hover:bg-amber-900/20"
                                            : "bg-zinc-900/20 border-white/5 hover:bg-zinc-900/40"
                                    )}
                                >
                                    {isMatch && <div className="absolute left-0 top-0 bottom-0 w-1 bg-amber-500" />}
                                    <div className="flex justify-between items-start mb-3">
                                        <div className="flex gap-3">
                                            {pos.raw_data?.icon ? (
                                                <Image src={pos.raw_data.icon} alt={pos.title || 'Market icon'} width={40} height={40} className="w-10 h-10 rounded-md object-cover bg-zinc-800" />
                                            ) : (
                                                <div className="w-10 h-10 rounded-md bg-zinc-800 flex items-center justify-center text-xs">PM</div>
                                            )}
                                            <div>
                                                <div className="text-sm font-medium text-white line-clamp-1">{pos.title}</div>
                                                <div className="flex items-center gap-2 mt-1">
                                                    <span className={cn("text-[10px] px-1.5 py-0.5 rounded border font-bold uppercase", 
                                                        pos.outcome === 'Yes' ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-blue-500/10 text-blue-400 border-blue-500/20"
                                                    )}>{pos.outcome}</span>
                                                    <span className="text-[10px] text-zinc-500">{pos.event_category}</span>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className={cn("text-sm font-mono font-bold", isProfit ? "text-emerald-400" : "text-red-400")}>
                                                {isProfit ? '+' : ''}{currency(pos.unrealized_pnl)}
                                            </div>
                                            <div className="text-[10px] text-zinc-500">PnL</div>
                                        </div>
                                    </div>

                                    {/* Price Bar */}
                                    <div className="bg-black rounded-lg p-2.5 border border-white/5 flex items-center justify-between text-xs font-mono">
                                        <div className="text-zinc-500">
                                            Entry: <span className="text-zinc-300">{cents(pos.avg_entry_price)}</span>
                                        </div>
                                        <ArrowRight className="w-3 h-3 text-zinc-600" />
                                        <div className="text-zinc-500">
                                            Curr: <span className={cn("font-bold", isProfit ? "text-emerald-400" : "text-red-400")}>
                                                {cents(pos.current_price)}
                                            </span>
                                        </div>
                                        <div className="h-3 w-px bg-white/10 mx-2"></div>
                                        <div className="text-zinc-400">
                                            Size: {currency(pos.size)}
                                        </div>
                                    </div>
                                    
                                    <div className="mt-2 text-right">
                                        <a href={`https://polymarket.com/event/${pos.event_slug}`} target="_blank" className="text-[10px] text-blue-500 hover:text-blue-400 flex items-center justify-end gap-1">
                                            View Market <ExternalLink className="w-3 h-3" />
                                        </a>
                                    </div>
                                </div>
                            )
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