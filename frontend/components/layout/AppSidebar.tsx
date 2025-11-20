'use client';

import {
  LayoutGrid,
  MessageSquareText,
  WalletMinimal,
  Settings2,
  LogOut,
  Command,
  Crown
} from 'lucide-react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';

// Mock History Data (In a real app, this comes from Supabase)
const recentChats = [
  { id: 1, title: "Wallet Analysis: 0x83...9a", date: "2h ago" },
  { id: 2, title: "Top Traders Q3", date: "1d ago" },
  { id: 3, title: "US Election Volatility", date: "3d ago" },
];

const navItems = [
  { name: 'Elite Wallets', icon: Crown, href: '/dashboard/elites' },
  { name: 'Market Scanner', icon: LayoutGrid, href: '/dashboard/scanner' },
  { name: 'Wallet Tracker', icon: WalletMinimal, href: '/dashboard/wallets' },
];

export function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();

  const handleSignOut = () => {
    router.push('/');
  };

  return (
    <aside className="w-[280px] flex-shrink-0 bg-[#050505] flex flex-col border-r border-white/[0.08] h-full hidden md:flex">
      
      {/* 1. Logo Area */}
      <div className="h-14 flex items-center px-4 border-b border-white/[0.08]">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-lg shadow-blue-900/20">
            <Command className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-zinc-100 tracking-tight text-sm">
            POLY<span className="text-zinc-500">ANALYTICS</span>
          </span>
        </div>
      </div>


      {/* 3. Main Navigation */}
      <div className="px-3 py-2">
        <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider px-3 mb-2">Menu</div>
        <Link href="/dashboard">
          <div className={cn(
            "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-all mb-1",
            pathname === '/dashboard' 
              ? "bg-zinc-900 text-white font-medium" 
              : "text-zinc-400 hover:bg-zinc-900/50 hover:text-zinc-200"
          )}>
            <MessageSquareText className="w-4 h-4" />
            AI Chat
          </div>
        </Link>
        
        {navItems.map((item) => (
          <Link key={item.href} href={item.href}>
            <div className={cn(
              "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-all mb-1",
              pathname === item.href 
                ? "bg-zinc-900 text-white font-medium" 
                : "text-zinc-400 hover:bg-zinc-900/50 hover:text-zinc-200"
            )}>
              <item.icon className="w-4 h-4" />
              {item.name}
            </div>
          </Link>
        ))}
      </div>

      {/* 4. History Section */}
      <div className="px-3 py-2 flex-1 overflow-y-auto">
        <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider px-3 mb-2 mt-4">Recent History</div>
        <div className="space-y-0.5">
          {recentChats.map((chat) => (
            <button key={chat.id} className="flex w-full flex-col items-start px-3 py-2 rounded-md text-zinc-400 hover:bg-zinc-900/50 hover:text-zinc-200 transition-all group">
              <span className="text-sm truncate w-full text-left">{chat.title}</span>
              <span className="text-[10px] text-zinc-600 group-hover:text-zinc-500">{chat.date}</span>
            </button>
          ))}
        </div>
      </div>

      {/* 5. User Profile Footer */}
      <div className="p-3 border-t border-white/[0.08] bg-[#0a0a0a]">
        <div className="flex items-center gap-3 w-full px-2 py-2 rounded-md hover:bg-zinc-900 transition-colors">
          <div className="w-8 h-8 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-xs font-medium text-zinc-300">
            PT
          </div>
          <div className="flex-1 text-left overflow-hidden">
            <div className="text-sm font-medium text-zinc-200 truncate">Pro Trader</div>
            <div className="text-[10px] text-emerald-500 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Connected
            </div>
          </div>
          <div className="flex gap-1">
            <Link href="/dashboard/settings">
              <button
                className="p-1 rounded-md hover:bg-zinc-800 transition-colors"
                title="Settings"
              >
                <Settings2 className="w-4 h-4 text-zinc-500 hover:text-zinc-300" />
              </button>
            </Link>
            <button
              onClick={handleSignOut}
              className="p-1 rounded-md hover:bg-zinc-800 transition-colors"
              title="Sign Out"
            >
              <LogOut className="w-4 h-4 text-zinc-500 hover:text-zinc-300" />
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
}