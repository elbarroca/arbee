'use client';

import { useState } from 'react';
import { Menu, X, Crown, LayoutGrid, WalletMinimal, MessageSquareText, Settings2, LogOut, Command } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';

export default function MobileSidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const pathname = usePathname();

  const navItems = [
    { name: 'AI Chat', icon: MessageSquareText, href: '/dashboard' },
    { name: 'Elite Wallets', icon: Crown, href: '/dashboard/elites' },
    { name: 'Market Scanner', icon: LayoutGrid, href: '/dashboard/scanner' },
    { name: 'Wallet Tracker', icon: WalletMinimal, href: '/dashboard/wallets' },
  ];

  return (
    <>
      {/* Mobile Header Trigger - Only visible on mobile */}
      <div className="md:hidden fixed top-0 left-0 w-full h-14 bg-[#050505]/90 backdrop-blur border-b border-white/10 z-50 flex items-center justify-between px-4">
         <div className="flex items-center gap-2">
             <div className="w-7 h-7 bg-gradient-to-br from-blue-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-lg shadow-blue-900/20">
                <Command className="w-3.5 h-3.5 text-white" />
             </div>
             <span className="font-bold text-zinc-100 text-sm tracking-tight">PolyAnalytics</span>
         </div>
         <Button variant="ghost" size="icon" onClick={() => setIsOpen(true)} className="text-zinc-300 hover:bg-white/10">
            <Menu className="w-5 h-5" />
         </Button>
      </div>

      {/* Slide-out Drawer */}
      <AnimatePresence>
        {isOpen && (
            <>
                {/* Backdrop */}
                <motion.div 
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    onClick={() => setIsOpen(false)}
                    className="fixed inset-0 bg-black/80 z-50 md:hidden backdrop-blur-sm"
                />
                
                {/* Sidebar Panel */}
                <motion.div
                    initial={{ x: '-100%' }} animate={{ x: 0 }} exit={{ x: '-100%' }}
                    transition={{ type: "spring", damping: 25, stiffness: 300 }}
                    className="fixed top-0 left-0 h-full w-[280px] bg-[#0a0a0a] border-r border-white/10 z-50 md:hidden flex flex-col shadow-2xl"
                >
                    <div className="h-14 px-4 border-b border-white/10 flex justify-between items-center">
                        <span className="font-bold text-zinc-100">Menu</span>
                        <button onClick={() => setIsOpen(false)} className="p-2 text-zinc-400 hover:text-white">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    <div className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
                        {navItems.map((item) => (
                            <Link key={item.href} href={item.href} onClick={() => setIsOpen(false)}>
                                <div className={cn(
                                    "flex items-center gap-3 px-3 py-3 rounded-lg text-sm transition-all",
                                    pathname === item.href 
                                    ? "bg-zinc-900 text-white font-medium border border-white/5 shadow-sm" 
                                    : "text-zinc-400 hover:bg-zinc-900/50 hover:text-zinc-200"
                                )}>
                                    <item.icon className="w-5 h-5" />
                                    {item.name}
                                </div>
                            </Link>
                        ))}
                    </div>

                    <div className="p-4 border-t border-white/10 bg-[#050505]">
                        <div className="flex gap-2">
                            <Link href="/dashboard/settings" className="flex-1" onClick={() => setIsOpen(false)}>
                                <button className="w-full flex items-center justify-center gap-2 p-2.5 rounded-lg bg-zinc-900 text-zinc-300 text-xs border border-white/5 font-medium">
                                    <Settings2 className="w-4 h-4" /> Settings
                                </button>
                            </Link>
                            <button className="flex items-center justify-center gap-2 p-2.5 rounded-lg bg-red-500/10 text-red-400 text-xs border border-red-500/20 px-3">
                                <LogOut className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </motion.div>
            </>
        )}
      </AnimatePresence>
    </>
  );
}