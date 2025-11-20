'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { GlobalMarketStats } from '@/types/market';
import { TrendingUp, Activity, AlertCircle, Layers } from 'lucide-react';

const COLORS = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#6366f1', '#ec4899'];

interface TooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    payload: {
      name: string;
      volume: number;
      count: number;
    };
  }>;
}

const CustomTooltip = ({ active, payload }: TooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#09090b] border border-white/10 p-3 rounded-lg shadow-xl z-50">
        <p className="text-white font-medium text-sm">{payload[0].name}</p>
        <p className="text-emerald-400 font-mono text-xs">
          ${Number(payload[0].value).toLocaleString()} Vol
        </p>
      </div>
    );
  }
  return null;
};

export default function MarketCharts({ stats }: { stats: GlobalMarketStats | null }) {
  if (!stats) return <div className="h-64 animate-pulse bg-zinc-900/30 rounded-xl mb-8" />;

  // Prepare Chart Data (Top 5 + Other)
  const chartData = stats.categoryDistribution.slice(0, 5);
  const otherVol = stats.categoryDistribution.slice(5).reduce((acc, curr) => acc + curr.volume, 0);
  
  if (otherVol > 0) {
    chartData.push({ name: 'Other', volume: otherVol, count: 0 });
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
      
      {/* 1. Real Volume Distribution */}
      <div className="bg-zinc-900/30 border border-white/5 rounded-xl p-6 flex flex-col">
        <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-wider mb-4 flex items-center gap-2">
          <Activity className="w-4 h-4" /> Volume Distribution
        </h3>
        <div className="flex-1 min-h-[250px] relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="volume"
                  stroke="none"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} cursor={false} />
                <Legend 
                  verticalAlign="bottom" 
                  height={36} 
                  iconType="circle"
                  wrapperStyle={{ fontSize: '11px', color: '#a1a1aa', paddingTop: '20px' }}
                />
              </PieChart>
            </ResponsiveContainer>
            
            {/* Center Text: Total Volume */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center mt-[-40px]">
                    <div className="text-[10px] text-zinc-500 uppercase tracking-wider">Total Vol</div>
                    <div className="text-lg font-bold text-white font-mono">
                        ${(stats.totalVolume / 1_000_000).toFixed(1)}M
                    </div>
                </div>
            </div>
        </div>
      </div>

      {/* 2. Real Key Metrics */}
      <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
        
        {/* Active Events Card */}
        <div className="bg-zinc-900/30 border border-white/5 rounded-xl p-6 flex flex-col justify-between relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
            <div>
                <div className="flex items-center gap-2 text-blue-400 mb-2">
                    <Layers className="w-5 h-5" />
                    <span className="text-xs font-bold uppercase tracking-wider">Active Events</span>
                </div>
                <div className="text-4xl font-mono font-bold text-white">{stats.activeEvents}</div>
                <div className="text-zinc-500 text-xs mt-1">Events currently live on-chain</div>
            </div>
            
            {/* Mini Bar Chart of categories */}
            <div className="mt-6 flex items-end gap-1 h-12 opacity-50">
                {chartData.map((cat, i) => (
                   <div 
                     key={i} 
                     className="flex-1 bg-blue-500/30 rounded-t-sm"
                     style={{ height: `${(cat.volume / stats.totalVolume) * 100}%` }}
                   />
                ))}
            </div>
        </div>

        {/* Top Category Card */}
        <div className="bg-zinc-900/30 border border-white/5 rounded-xl p-6 flex flex-col justify-between relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
            <div>
                <div className="flex items-center gap-2 text-emerald-400 mb-2">
                    <TrendingUp className="w-5 h-5" />
                    <span className="text-xs font-bold uppercase tracking-wider">Top Category</span>
                </div>
                <div className="text-3xl font-bold text-white truncate">
                    {stats.topCategory?.name || 'N/A'}
                </div>
                <div className="text-zinc-500 text-xs mt-1 font-mono">
                    ${stats.topCategory?.volume.toLocaleString()} Volume
                </div>
            </div>
            <div className="mt-4">
               <div className="flex gap-2 flex-wrap">
                   {stats.categoryDistribution.slice(1, 4).map((cat, i) => (
                       <span key={i} className="px-2 py-1 rounded bg-white/5 border border-white/10 text-[10px] text-zinc-400">
                           {cat.name}
                       </span>
                   ))}
               </div>
            </div>
        </div>

        {/* Expiring Soon Card */}
        <div className="sm:col-span-2 bg-zinc-900/30 border border-white/5 rounded-xl p-5 flex items-center justify-between relative overflow-hidden group">
            <div className="absolute inset-0 bg-amber-500/5 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="flex items-center gap-4 relative z-10">
                <div className="w-12 h-12 rounded-xl bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
                    <AlertCircle className="w-6 h-6 text-amber-500" />
                </div>
                <div>
                    <div className="text-white font-bold text-sm">Approaching Expiry</div>
                    <div className="text-zinc-500 text-xs">Events resolving within 24 hours</div>
                </div>
            </div>
            <div className="text-right relative z-10">
                <div className="text-3xl font-mono font-bold text-white">{stats.expiringSoon}</div>
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider">Events</div>
            </div>
        </div>

      </div>
    </div>
  );
}