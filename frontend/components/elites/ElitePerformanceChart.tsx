'use client';

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts';
import { EliteTagComparison } from '@/types/elite';

export default function ElitePerformanceChart({ data }: { data: EliteTagComparison[] }) {
  return (
    <div className="bg-zinc-900/30 border border-white/5 rounded-xl p-6 flex flex-col h-[350px]">
      <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-wider mb-6">
        Elite Edge by Category (ROI %)
      </h3>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
          <XAxis 
            dataKey="tag" 
            stroke="#71717a" 
            fontSize={12} 
            tickLine={false} 
            axisLine={false}
          />
          <YAxis 
            stroke="#71717a" 
            fontSize={12} 
            tickLine={false} 
            axisLine={false} 
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip 
            cursor={{ fill: '#ffffff05' }}
            contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', borderRadius: '8px' }}
            itemStyle={{ fontSize: '12px', fontWeight: 'bold' }}
            formatter={(value: number) => [`${value.toFixed(1)}%`]}
          />
          <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
          <Bar dataKey="elite_avg_roi" name="Elite ROI" fill="#10b981" radius={[4, 4, 0, 0]} barSize={20} />
          <Bar dataKey="non_elite_avg_roi" name="Avg Market ROI" fill="#3f3f46" radius={[4, 4, 0, 0]} barSize={20} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}