'use client';

import { motion } from 'framer-motion';

export default function PnLChart() {
  // Simplified path for a graph going up
  const pathData = "M0,100 Q30,90 50,70 T100,60 T150,40 T200,50 T250,20 T300,10";

  return (
    <div className="w-full h-full flex flex-col">
       <div className="flex items-center justify-between mb-4 px-2">
          <div>
            <div className="text-zinc-500 text-xs uppercase tracking-widest mb-1">Portfolio Growth</div>
            <div className="text-3xl font-bold text-white font-mono">$142,894.21</div>
          </div>
          <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-2 py-1 rounded text-xs font-bold">
            +324% (YTD)
          </div>
       </div>
       
       <div className="flex-1 relative w-full min-h-[200px] bg-gradient-to-b from-emerald-900/5 to-transparent rounded-lg border border-emerald-500/10 overflow-hidden">
          <svg viewBox="0 0 300 120" className="w-full h-full absolute inset-0" preserveAspectRatio="none">
             {/* Gradient Fill */}
             <defs>
                <linearGradient id="gradient" x1="0" y1="0" x2="0" y2="1">
                   <stop offset="0%" stopColor="#10b981" stopOpacity="0.2" />
                   <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
                </linearGradient>
             </defs>
             <motion.path
                d={`${pathData} L300,120 L0,120 Z`}
                fill="url(#gradient)"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 2 }}
             />
             {/* Line */}
             <motion.path
                d={pathData}
                fill="none"
                stroke="#10b981"
                strokeWidth="3"
                initial={{ pathLength: 0 }}
                whileInView={{ pathLength: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 2, ease: "easeInOut" }}
             />
          </svg>
       </div>
    </div>
  );
}