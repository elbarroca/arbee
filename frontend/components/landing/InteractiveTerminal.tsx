'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, User, CheckCircle2, ChevronLeft, Copy, Wallet } from 'lucide-react';

interface Position {
  id: number;
  market: string;
  signal: string;
  entry: number;
  curr: number;
  confidence: string;
  roi: number;
  type: string;
}

interface Wallet {
  id: string;
  winRate: string;
  pnl: string;
  vol: string;
  tokens: number;
  positions: Position[];
  tags?: string[];
}

// Extended to 8 Wallets with Full Addresses and Tags
const WALLETS = [
  { id: '0x839395e209b87466', winRate: '92%', pnl: '+$242k', vol: '$1.2M', tokens: 12, tags: ['Pro', 'Macro'], positions: [
      { id: 1, market: "Fed Rate Cut Nov", signal: "Buy YES", entry: 0.42, curr: 0.55, confidence: "92%", roi: 31, type: "Macro" },
      { id: 2, market: "BTC > $100k 2024", signal: "Sell NO", entry: 0.15, curr: 0.08, confidence: "85%", roi: 46, type: "Crypto" }
  ]},
  { id: '0x4f2b399100ac91b2', winRate: '88%', pnl: '+$185k', vol: '$850k', tokens: 8, tags: ['Pro', 'Crypto'], positions: [
      { id: 3, market: "Eth ETF Approval", signal: "Buy YES", entry: 0.30, curr: 0.65, confidence: "89%", roi: 116, type: "Crypto" }
  ]},
  { id: '0x1d9902384aa28391', winRate: '78%', pnl: '+$45k', vol: '$220k', tokens: 5, tags: ['Politics'], positions: [
      { id: 4, market: "US Election Winner", signal: "Buy YES", entry: 0.52, curr: 0.61, confidence: "95%", roi: 17, type: "Politics" }
  ]},
  { id: '0x9a88321900bb1203', winRate: '76%', pnl: '+$32k', vol: '$150k', tokens: 1, tags: ['Odd Streak', 'Sports'], positions: [
      { id: 5, market: "Super Bowl Winner", signal: "Buy YES", entry: 0.45, curr: 0.67, confidence: "82%", roi: 49, type: "Sports" }
  ]},
  { id: '0x2cff001923884122', winRate: '72%', pnl: '+$28k', vol: '$110k', tokens: 1, tags: ['Sports'], positions: [
      { id: 6, market: "World Cup 2026", signal: "Buy YES", entry: 0.38, curr: 0.52, confidence: "75%", roi: 37, type: "Sports" }
  ]},
  { id: '0x7e11293847100293', winRate: '65%', pnl: '+$12k', vol: '$90k', tokens: 1, tags: ['Odd Streak'], positions: [
      { id: 7, market: "AI Breakthrough", signal: "Buy YES", entry: 0.28, curr: 0.41, confidence: "68%", roi: 46, type: "Tech" }
  ]},
  { id: '0xbb22381900128331', winRate: '62%', pnl: '+$8.5k', vol: '$50k', tokens: 1, tags: ['Sports'], positions: [
      { id: 8, market: "NBA Finals", signal: "Buy YES", entry: 0.35, curr: 0.48, confidence: "71%", roi: 37, type: "Sports" }
  ]},
  { id: '0xaa11293840019238', winRate: '58%', pnl: '+$4.2k', vol: '$20k', tokens: 1, tags: [], positions: [
      { id: 9, market: "Oil Price Surge", signal: "Buy YES", entry: 0.42, curr: 0.55, confidence: "64%", roi: 31, type: "Commodities" }
  ]},
];

export default function InteractiveTerminal() {
  const [step, setStep] = useState(0);
  const [selectedWallet, setSelectedWallet] = useState<Wallet | null>(null);
  const [selectedPos, setSelectedPos] = useState<Position | null>(null);
  const [simAmount] = useState(1000); // Simulation amount

  const handleWalletClick = (wallet: Wallet) => {
    setSelectedWallet(wallet);
    setStep(2);
  };

  const handleCopyTrade = () => {
    setStep(4);
    setTimeout(() => {
        setStep(0);
        setSelectedWallet(null);
        setSelectedPos(null);
    }, 3000);
  };

  return (
    <div className="relative w-full max-w-4xl mx-auto aspect-[3/4] sm:aspect-[4/3] md:aspect-[16/9] lg:aspect-[16/10] bg-[#050505] rounded-xl border border-white/10 shadow-2xl shadow-blue-900/20 overflow-hidden flex flex-col font-sans select-none group cursor-default">
      
      {/* Header */}
      <div className="h-10 border-b border-white/5 bg-[#0a0a0a] flex items-center px-3 md:px-4 gap-2 md:gap-3 z-20">
        <div className="flex gap-1 md:gap-1.5">
          <div className="w-2 h-2 md:w-2.5 md:h-2.5 rounded-full bg-[#2c2c2c] border border-white/10" />
          <div className="w-2 h-2 md:w-2.5 md:h-2.5 rounded-full bg-[#2c2c2c] border border-white/10" />
          <div className="w-2 h-2 md:w-2.5 md:h-2.5 rounded-full bg-[#2c2c2c] border border-white/10" />
        </div>
        <div className="flex-1 text-[10px] md:text-xs font-mono text-zinc-600 tracking-tight truncate">
            poly-analytics-terminal — v2.4.0
        </div>
        <div className="w-8 md:w-10 flex justify-end">
             {step > 1 && <button onClick={() => setStep(step - 1)} className="text-zinc-500 hover:text-white transition-colors p-1"><ChevronLeft className="w-3 h-3 md:w-4 md:h-4"/></button>}
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden">
        
        {/* STEP 0: SEARCH */}
        <motion.div
            className={`absolute inset-0 flex flex-col items-center justify-center transition-all duration-500 px-4 ${step === 0 ? 'opacity-100 z-10' : 'opacity-0 pointer-events-none -z-10'}`}
        >
             <div onClick={() => setStep(1)} className="relative group/input cursor-pointer w-full max-w-md mx-auto">
                <div className="absolute left-3 md:left-4 top-3 md:top-3.5 text-zinc-500 group-hover/input:text-emerald-400 transition-colors">
                    <Search className="w-4 h-4" />
                </div>
                <input
                    readOnly
                    className="w-full bg-[#0f0f0f] border border-white/10 rounded-xl py-3 pl-9 md:pl-10 pr-4 text-zinc-300 shadow-2xl text-sm group-hover/input:border-white/20 transition-all"
                    placeholder="Find whales, markets, or signals..."
                />
                <div className="mt-3 md:mt-4 flex flex-wrap justify-center gap-2">
                    {['High PnL', 'Active Now', 'Accumulating'].map(tag => (
                        <span key={tag} className="text-[10px] px-2 py-1 rounded border border-white/5 bg-white/5 text-zinc-600 group-hover/input:text-zinc-500 transition-colors">{tag}</span>
                    ))}
                </div>
            </div>
        </motion.div>

        {/* STEP 1: COMPACT WALLET LIST (8 items) */}
        <AnimatePresence>
            {step === 1 && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="absolute inset-0 top-0 p-3 md:p-4 bg-[#050505] overflow-y-auto scrollbar-hide"
                >
                    <div className="flex justify-between items-center mb-3 px-1">
                        <div className="text-zinc-400 text-xs font-medium">Elite Traders (Live)</div>
                    </div>

                    <div className="space-y-2">
                        {WALLETS.map((wallet, i) => (
                            <motion.div
                                key={wallet.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.03 }}
                                onClick={() => handleWalletClick(wallet)}
                                className="p-3 rounded-lg border border-white/5 bg-[#0a0a0a] hover:bg-[#121212] hover:border-white/10 cursor-pointer group transition-all flex items-center justify-between active:scale-[0.98] touch-manipulation"
                            >
                                <div className="flex items-center gap-2 md:gap-3 flex-1 min-w-0">
                                    <div className="text-[10px] md:text-xs font-mono text-zinc-300 group-hover:text-emerald-400 transition-colors truncate flex-shrink-0">
                                        {wallet.id.slice(0, 8)}...{wallet.id.slice(-4)}
                                    </div>
                                    <div className="w-7 h-7 md:w-8 md:h-8 rounded bg-zinc-900 border border-white/5 flex items-center justify-center text-zinc-500 group-hover:text-emerald-400 group-hover:bg-zinc-800 transition-colors flex-shrink-0">
                                        <User className="w-2.5 h-2.5 md:w-3 md:h-3" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-1 md:gap-2 text-[9px] md:text-[10px] text-zinc-600 flex-wrap">
                                            <span>Win: {wallet.winRate}</span>
                                            <span className="text-zinc-700 hidden sm:inline">•</span>
                                            <span className="hidden sm:inline">Vol: {wallet.vol}</span>
                                            {wallet.tags && wallet.tags.length > 0 && (
                                                <>
                                                    <span className="text-zinc-700">•</span>
                                                    <div className="flex gap-1 flex-wrap">
                                                        {wallet.tags.map((tag, idx) => (
                                                            <span key={idx} className="px-1 py-0.5 md:px-1.5 rounded text-[8px] md:text-[9px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                                                {tag}
                                                            </span>
                                                        ))}
                                                    </div>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                </div>
                                <div className="text-right flex-shrink-0">
                                    <div className="text-emerald-400 font-mono font-bold text-xs md:text-sm">{wallet.pnl}</div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>
            )}
        </AnimatePresence>

        {/* STEP 2: POSITIONS GRID (Improved UI) */}
        <AnimatePresence>
            {step === 2 && selectedWallet && (
                <motion.div
                    initial={{ opacity: 0, x: 50 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -50 }}
                    className="absolute inset-0 top-0 p-4 md:p-6 bg-[#050505] overflow-y-auto scrollbar-hide"
                >
                    {/* Header Section */}
                    <div className="mb-6 pb-4 border-b border-white/5">
                        <div className="flex items-start justify-between mb-4">
                            <div className="flex items-center gap-4">
                                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500/20 to-zinc-900 flex items-center justify-center border border-emerald-500/20 text-emerald-400">
                                    <Wallet className="w-5 h-5" />
                                </div>
                                <div>
                                    <h3 className="text-sm font-mono text-white mb-1">{selectedWallet.id}</h3>
                                    <div className="flex items-center gap-3 text-[10px] text-zinc-500">
                                        <span>{selectedWallet.tokens} Positions</span>
                                        <span className="text-zinc-700">•</span>
                                        <span className="text-emerald-400 font-semibold">{selectedWallet.pnl} Net</span>
                                        <span className="text-zinc-700">•</span>
                                        <span>Win: {selectedWallet.winRate}</span>
                                    </div>
                                </div>
                            </div>
                            {selectedWallet.tags && selectedWallet.tags.length > 0 && (
                                <div className="flex gap-1.5 flex-wrap justify-end">
                                    {selectedWallet.tags.map((tag, idx) => (
                                        <span key={idx} className="px-2 py-1 rounded-md text-[9px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Positions Grid */}
                    <div className="space-y-3">
                        {selectedWallet.positions.length > 0 ? selectedWallet.positions.map((pos: Position, i: number) => (
                            <motion.div 
                                key={pos.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.08 }}
                                onClick={() => { setSelectedPos(pos); setStep(3); }}
                                className="p-5 rounded-xl border border-white/5 bg-[#0a0a0a] hover:border-emerald-500/30 hover:bg-[#0f0f0f] cursor-pointer group transition-all"
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <div className="flex items-center gap-2">
                                        <span className="px-2 py-1 rounded-md text-[9px] font-bold bg-white/5 text-zinc-300 border border-white/10 uppercase tracking-wider">
                                            {pos.type}
                                        </span>
                                        <span className="text-emerald-400 font-mono font-bold text-xs bg-emerald-500/10 px-2 py-1 rounded-md border border-emerald-500/20">
                                            {pos.confidence} Conf.
                                        </span>
                                    </div>
                                    <div className="text-emerald-400 font-mono font-bold text-sm">
                                        +{pos.roi}%
                                    </div>
                                </div>
                                
                                <h4 className="text-base font-bold text-white mb-2 group-hover:text-emerald-400 transition-colors">
                                    {pos.market}
                                </h4>
                                
                                <div className="flex items-center gap-2 mb-4">
                                    <span className="text-xs text-zinc-500">Signal:</span>
                                    <span className={`px-2 py-1 rounded-md text-xs font-semibold ${
                                        pos.signal.includes('Buy') 
                                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                                            : 'bg-red-500/10 text-red-400 border border-red-500/20'
                                    }`}>
                                        {pos.signal}
                                    </span>
                                </div>
                                
                                <div className="grid grid-cols-2 gap-4 pt-4 border-t border-white/5">
                                    <div>
                                        <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Entry Price</div>
                                        <div className="text-sm font-mono text-zinc-300">${pos.entry.toFixed(2)}</div>
                                    </div>
                                    <div>
                                        <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Current</div>
                                        <div className="text-sm font-mono text-emerald-400 font-semibold">${pos.curr.toFixed(2)}</div>
                                    </div>
                                </div>
                            </motion.div>
                        )) : (
                           <div className="text-center py-16">
                               <div className="w-16 h-16 rounded-full bg-zinc-900 border border-white/5 flex items-center justify-center mx-auto mb-4">
                                   <Wallet className="w-6 h-6 text-zinc-600" />
                               </div>
                               <div className="text-zinc-500 text-sm mb-1">No active positions</div>
                               <div className="text-zinc-600 text-xs">This trader has no positions available for copy trading.</div>
                           </div>
                        )}
                    </div>
                </motion.div>
            )}
        </AnimatePresence>

        {/* STEP 3: ALPHA DETECT (With Profit Calc) */}
        <AnimatePresence>
            {step === 3 && selectedPos && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 z-30 flex items-center justify-center p-3 md:p-4 bg-black/80 backdrop-blur-sm"
                >
                    <div className="bg-[#09090b] border border-white/10 w-full max-w-xs mx-4 rounded-xl shadow-2xl relative overflow-hidden">
                         <div className="h-1 w-full bg-gradient-to-r from-blue-500 to-emerald-500" />

                         <div className="p-4 md:p-5">
                             <div className="flex justify-between items-center mb-4">
                                <h3 className="text-white font-bold text-sm">Alpha Detected</h3>
                                <div className="bg-emerald-500/10 text-emerald-400 p-1.5 rounded">
                                    <CheckCircle2 className="w-4 h-4" />
                                </div>
                             </div>

                             <div className="bg-[#050505] rounded-lg p-3 border border-white/5 space-y-2 mb-4">
                                <div className="flex justify-between text-xs">
                                    <span className="text-zinc-500">Signal</span>
                                    <span className={selectedPos.signal.includes('Buy') ? 'text-emerald-400' : 'text-red-400'}>{selectedPos.signal}</span>
                                </div>
                                <div className="flex justify-between text-xs">
                                    <span className="text-zinc-500">ROI Potential</span>
                                    <span className="text-emerald-400 font-mono">+{selectedPos.roi}%</span>
                                </div>
                             </div>

                             {/* Profit Calculator Simulation */}
                             <div className="bg-zinc-900/50 rounded-lg p-3 border border-white/5 mb-4">
                                <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Simulation ($1,000 Position)</div>
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-zinc-400 text-xs">Invested</span>
                                    <span className="text-white font-mono text-xs">$1,000.00</span>
                                </div>
                                <div className="flex items-center justify-between border-t border-white/5 pt-1 mt-1">
                                    <span className="text-zinc-400 text-xs">Est. Profit</span>
                                    <span className="text-emerald-400 font-mono text-sm font-bold">
                                        +${(simAmount * (selectedPos.roi / 100)).toFixed(2)}
                                    </span>
                                </div>
                             </div>
                            
                            <button 
                                onClick={handleCopyTrade}
                                className="w-full bg-white text-black hover:bg-zinc-200 py-3 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2"
                            >
                                <Copy className="w-3 h-3" /> Copy & Execute
                            </button>
                            
                            <button onClick={() => setStep(2)} className="w-full mt-2 text-zinc-600 hover:text-zinc-400 text-[10px] py-1">Cancel</button>
                         </div>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>

        {/* STEP 4: SUCCESS */}
        <AnimatePresence>
            {step === 4 && (
                <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 z-40 flex flex-col items-center justify-center bg-[#050505]"
                >
                    <motion.div 
                        initial={{ scale: 0.5, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center mb-4 border border-emerald-500/20"
                    >
                        <CheckCircle2 className="w-8 h-8 text-emerald-500" />
                    </motion.div>
                    <h2 className="text-lg font-bold text-white mb-1">Position Copied</h2>
                    <p className="text-zinc-500 text-xs">Transaction confirmed.</p>
                </motion.div>
            )}
        </AnimatePresence>

      </div>
    </div>
  );
}