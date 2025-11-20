'use client';

import { motion } from 'framer-motion';
import { Vote, Zap, DollarSign, Trophy, Globe, Activity, ArrowUpRight } from 'lucide-react';

const positions = [
  // Live positions first
  { icon: Vote, event: "US Election 2024", date: "Nov 05", market: "Winner", outcome: "Trump", entry: "$0.52", current: "$0.61", roi: "+17.3%", status: "Live" },
  { icon: Activity, event: "FOMC Meeting", date: "Nov 15", market: "Rate Cut", outcome: "Yes", entry: "$0.42", current: "$0.89", roi: "+112.0%", status: "Live" },
  { icon: DollarSign, event: "Bitcoin Price", date: "Dec 31", market: "> $100k", outcome: "No", entry: "$0.15", current: "$0.18", roi: "+20.0%", status: "Live" },
  { icon: Globe, event: "COP28 Summit", date: "Dec 12", market: "Deal Signed", outcome: "Yes", entry: "$0.60", current: "$0.72", roi: "+20.0%", status: "Live" },
  { icon: Vote, event: "Midterm Elections", date: "Nov 08", market: "Senate Control", outcome: "Republicans", entry: "$0.48", current: "$0.67", roi: "+39.6%", status: "Live" },
  { icon: Zap, event: "NVIDIA Q3", date: "Nov 21", market: "Revenue > $22B", outcome: "Yes", entry: "$0.62", current: "$0.91", roi: "+46.8%", status: "Live" },
  { icon: Trophy, event: "World Cup 2026", date: "Jun 30", market: "Host Country", outcome: "USA", entry: "$0.42", current: "$0.38", roi: "-9.5%", status: "Live" },
  { icon: Activity, event: "Oil Prices", date: "Dec 15", market: "> $80/barrel", outcome: "No", entry: "$0.28", current: "$0.31", roi: "+10.7%", status: "Live" },
  // Closed positions
  { icon: Zap, event: "SpaceX Starship", date: "Oct 13", market: "Successful", outcome: "Yes", entry: "$0.88", current: "$0.98", roi: "+11.4%", status: "Closed" },
  { icon: Trophy, event: "Super Bowl", date: "Feb 11", market: "Winner", outcome: "Chiefs", entry: "$0.45", current: "$1.00", roi: "+122.0%", status: "Closed" },
  { icon: Activity, event: "Fed Interest Rate", date: "Oct 23", market: "Rate Decision", outcome: "Hold", entry: "$0.35", current: "$0.41", roi: "+17.1%", status: "Closed" },
  { icon: DollarSign, event: "Tesla Earnings", date: "Oct 18", market: "Beat Estimates", outcome: "Yes", entry: "$0.55", current: "$0.78", roi: "+41.8%", status: "Closed" },
  { icon: Globe, event: "UK Election 2024", date: "Jul 04", market: "Labour Win", outcome: "Yes", entry: "$0.58", current: "$0.89", roi: "+53.4%", status: "Closed" },
  { icon: DollarSign, event: "Ethereum Merge", date: "Sep 15", market: "Success", outcome: "Yes", entry: "$0.71", current: "$0.94", roi: "+32.4%", status: "Closed" },
  { icon: Vote, event: "French Election", date: "Apr 24", market: "Le Pen Win", outcome: "No", entry: "$0.32", current: "$0.35", roi: "+9.4%", status: "Closed" },
];

export default function LiveAlphaSection() {
  
  // Chart path matching actual trades chronologically - bullish trend with volatility
  // Key inflection points: Start(100) → +9.4% → -9.5%(dip) → +53.4% → +32.4% → +11.4% → 
  //                        +41.8% → +17.1% → +17.3% → +39.6% → +112.0%(BIG SPIKE) → 
  //                        +46.8% → +20.0% → +10.7% → +20.0%
  // Y-axis inverted (lower Y = higher on chart), showing strong bullish trend
  const pathData = "0,145 30,143 60,146 90,132 120,125 150,123 180,115 210,110 240,105 270,90 300,88 330,35 360,25 390,22 420,20 450,18 480,16 510,14 540,12 570,10 600,8";

  return (
    <section className="py-24 px-6 border-y border-white/5 bg-[#030303]">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="mb-10 flex flex-col md:flex-row justify-between items-end gap-6">
            <div>
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-900/20 border border-emerald-500/20 text-emerald-400 text-xs font-medium mb-4">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    Live Strategy Verification
                </div>
                <h2 className="text-3xl md:text-4xl font-bold text-white">Proof of Performance</h2>
            </div>
            <div className="text-right">
                <div className="text-5xl font-bold text-emerald-400 font-mono tracking-tighter">+34.2%</div>
                <div className="text-zinc-500 text-sm">7-Day Rolling ROI</div>
            </div>
        </div>

        {/* Main Container */}
        <div className="bg-[#080808] border border-white/10 rounded-3xl overflow-hidden shadow-2xl">
            
            {/* CHART AREA */}
            <div className="h-[400px] relative w-full border-b border-white/5 bg-[#060606]">
                {/* Grid Lines (Stock Chart Style) */}
                <div className="absolute inset-0 flex flex-col justify-between pointer-events-none opacity-20 p-8">
                    <div className="w-full h-px bg-zinc-800"></div>
                    <div className="w-full h-px bg-zinc-800"></div>
                    <div className="w-full h-px bg-zinc-800"></div>
                    <div className="w-full h-px bg-zinc-800"></div>
                    <div className="w-full h-px bg-zinc-800"></div>
                </div>
                <div className="absolute inset-0 flex justify-between pointer-events-none opacity-20 px-8">
                    <div className="h-full w-px bg-zinc-800"></div>
                    <div className="h-full w-px bg-zinc-800"></div>
                    <div className="h-full w-px bg-zinc-800"></div>
                    <div className="h-full w-px bg-zinc-800"></div>
                </div>
                
                <svg className="w-full h-full absolute inset-0 pt-8 px-0" preserveAspectRatio="none" viewBox="0 0 600 160">
                    <defs>
                        <linearGradient id="stockGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10b981" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
                        </linearGradient>
                    </defs>
                    
                    {/* Fill Area */}
                    <motion.path 
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        transition={{ duration: 1 }}
                        d={`M ${pathData} V 200 H 0 Z`} 
                        fill="url(#stockGrad)" 
                    />
                    
                    {/* The Line (Polyline for sharp, realistic look) */}
                    <motion.polyline 
                        initial={{ pathLength: 0 }}
                        whileInView={{ pathLength: 1 }}
                        transition={{ duration: 2, ease: "linear" }}
                        points={pathData}
                        fill="none" 
                        stroke="#10b981" 
                        strokeWidth="2" 
                        strokeLinejoin="round"
                        strokeLinecap="round"
                    />
                </svg>

                {/* Floating Current Price Indicator */}
                <motion.div 
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: 2 }}
                    className="absolute top-[5%] right-0 bg-emerald-600 text-white text-xs px-2 py-1 rounded-l font-mono font-bold shadow-lg flex items-center gap-2"
                >
                    <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                    $142,894
                </motion.div>
            </div>

            {/* DATA TABLE */}
            <div className="w-full overflow-x-auto">
                <table className="w-full text-left border-collapse min-w-[800px]">
                    <thead>
                        <tr className="bg-white/[0.02] text-[11px] text-zinc-500 uppercase tracking-wider border-b border-white/5 font-semibold">
                            <th className="p-5 pl-8">Event</th>
                            <th className="p-5">Date</th>
                            <th className="p-5">Market</th>
                            <th className="p-5">Outcome</th>
                            <th className="p-5 text-right">Entry</th>
                            <th className="p-5 text-right">Current</th>
                            <th className="p-5 text-right">ROI</th>
                            <th className="p-5 text-center pr-8">Status</th>
                        </tr>
                    </thead>
                    <tbody className="text-sm divide-y divide-white/5">
                        {positions.map((pos, i) => (
                            <tr key={i} className="hover:bg-white/[0.02] transition-colors group">
                                <td className="p-5 pl-8 font-medium text-white flex items-center gap-3">
                                    <div className="w-8 h-8 rounded bg-zinc-800 flex items-center justify-center text-zinc-400 border border-white/5">
                                        <pos.icon className="w-4 h-4" />
                                    </div>
                                    {pos.event}
                                </td>
                                <td className="p-5 text-zinc-500 font-mono text-xs">{pos.date}</td>
                                <td className="p-5 text-zinc-300">{pos.market}</td>
                                <td className="p-5">
                                    <span className={`px-2 py-1 rounded text-[10px] font-bold border ${
                                        pos.outcome === 'Yes' || pos.outcome === 'Trump' 
                                        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                                        : 'bg-red-500/10 text-red-400 border-red-500/20'
                                    }`}>
                                        {pos.outcome}
                                    </span>
                                </td>
                                <td className="p-5 text-right font-mono text-zinc-400">{pos.entry}</td>
                                <td className="p-5 text-right font-mono text-white">{pos.current}</td>
                                <td className="p-5 text-right font-mono font-bold">
                                    <span className={pos.roi.startsWith('+') ? 'text-emerald-400' : 'text-red-400'}>
                                        {pos.roi}
                                    </span>
                                </td>
                                <td className="p-5 text-center pr-8">
                                    <div className={`inline-flex items-center gap-1.5 text-[10px] uppercase font-bold ${pos.status === 'Live' ? 'text-emerald-500' : 'text-zinc-500'}`}>
                                        <span className={`w-1.5 h-1.5 rounded-full ${pos.status === 'Live' ? 'bg-emerald-500 animate-pulse' : 'bg-zinc-600'}`} />
                                        {pos.status}
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="p-4 bg-zinc-900/30 border-t border-white/5 text-center">
                <button className="text-xs text-zinc-500 hover:text-white transition-colors flex items-center justify-center gap-2 mx-auto group">
                    View Full Ledger 
                    <ArrowUpRight className="w-3 h-3 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                </button>
            </div>
        </div>
      </div>
    </section>
  );
}