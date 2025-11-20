'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, User, MousePointer2, CheckCircle2 } from 'lucide-react';

export default function HeroAnimation() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    let mounted = true;
    
    const runSequence = async () => {
      // Loop forever
      while (mounted) {
        setStep(0); // Start: Search Input
        await wait(1500); // Typing time
        
        setStep(1); // Search moves up, Results appear
        await wait(800); 
        
        setStep(2); // Mouse appears and moves to target
        await wait(1200);
        
        setStep(3); // Click down
        await wait(200);
        
        setStep(4); // Click up (Action triggered)
        await wait(4000); // Show Detail View
      }
    };

    runSequence();
    return () => { mounted = false; };
  }, []);

  return (
    <div className="relative w-full max-w-4xl mx-auto aspect-[16/10] bg-[#050505] rounded-xl border border-white/10 shadow-2xl shadow-blue-900/20 overflow-hidden flex flex-col font-sans select-none">
      {/* Window Header */}
      <div className="h-10 border-b border-white/5 bg-[#0a0a0a] flex items-center px-4 gap-2 z-20">
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/50" />
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500/50" />
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/20 border border-green-500/50" />
        </div>
        <div className="ml-4 flex-1 flex justify-center">
            <div className="h-5 px-3 bg-zinc-900 rounded text-[10px] flex items-center justify-center text-zinc-500 font-mono border border-white/5">
                poly-analytics-terminal
            </div>
        </div>
      </div>

      <div className="flex-1 relative p-8 overflow-hidden">
        
        {/* --- PHASE 1: SEARCH --- */}
        <AnimatePresence mode="wait">
          {step === 0 && (
            <motion.div 
                key="search"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, y: -20 }}
                className="absolute inset-0 flex items-center justify-center z-10"
            >
                <div className="w-full max-w-lg relative">
                    <div className="absolute left-4 top-3.5 text-blue-500">
                        <Search className="w-5 h-5" />
                    </div>
                    <input 
                        readOnly
                        className="w-full bg-zinc-900/50 border border-white/10 rounded-full py-3 pl-12 pr-4 text-zinc-300 shadow-lg"
                        placeholder="Analyze top performing wallets..." 
                    />
                    {/* Typing animation mask */}
                    <motion.div 
                        initial={{ left: 50 }} 
                        animate={{ left: "100%" }} 
                        transition={{ duration: 1, ease: "linear" }}
                        className="absolute top-0 bottom-0 right-0 bg-[#050505] opacity-80"
                    />
                </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* --- PHASE 2: RESULTS LIST --- */}
        <AnimatePresence>
            {step >= 1 && (
                <motion.div 
                    key="results"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="absolute inset-0 top-16 px-8 md:px-20 space-y-3 z-0"
                >
                    {/* Search Bar (Moved to top) */}
                    <div className="w-full flex items-center gap-3 mb-8 opacity-50">
                         <div className="bg-zinc-800/50 p-2 rounded-full"><Search className="w-4 h-4 text-zinc-400"/></div>
                         <div className="h-px bg-white/10 flex-1"></div>
                    </div>

                    {/* Results */}
                    {['0x83...9a', '0x4f...2b', '0x1d...ea'].map((wallet, i) => (
                        <motion.div 
                            key={wallet}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.1 }}
                            // Hover/Active state simulation based on step
                            className={`p-4 rounded-lg border flex items-center justify-between transition-colors duration-300 ${
                                (step >= 3 && i === 0) 
                                    ? 'bg-blue-500/20 border-blue-500/50 scale-[1.02]' 
                                    : i === 0 && step >= 2 
                                        ? 'bg-zinc-800/50 border-zinc-700' 
                                        : 'bg-transparent border-white/5'
                            }`}
                        >
                            <div className="flex items-center gap-4">
                                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${i === 0 ? 'bg-blue-500/20 text-blue-400' : 'bg-zinc-900 text-zinc-600'}`}>
                                    <User className="w-5 h-5" />
                                </div>
                                <div>
                                    <div className="text-sm font-medium text-zinc-200">Wallet {wallet}</div>
                                    <div className="text-xs text-zinc-500">Win Rate: {i === 0 ? '92%' : '78%'}</div>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className="text-emerald-400 font-mono font-bold">{i === 0 ? '+$242k' : '+$45k'}</div>
                                <div className="text-xs text-zinc-600">PNL (30d)</div>
                            </div>
                        </motion.div>
                    ))}
                </motion.div>
            )}
        </AnimatePresence>

        {/* --- MOUSE CURSOR --- */}
        <AnimatePresence>
            {step >= 2 && step < 4 && (
                <motion.div
                    initial={{ x: 300, y: 300, opacity: 0 }}
                    animate={{ 
                        x: step === 2 ? 150 : 150, // X coordinate of first item
                        y: step === 2 ? 50 : 50,   // Y coordinate of first item
                        opacity: 1,
                        scale: step === 3 ? 0.9 : 1 // Click press effect
                    }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: step === 2 ? 1 : 0.1, ease: "circOut" }}
                    className="absolute top-32 left-1/2 z-50 pointer-events-none"
                >
                    <MousePointer2 className="w-6 h-6 text-white fill-white drop-shadow-xl" />
                    {step === 3 && (
                        <div className="absolute -top-2 -left-2 w-10 h-10 bg-white/20 rounded-full animate-ping" />
                    )}
                </motion.div>
            )}
        </AnimatePresence>

        {/* --- PHASE 3: DETAIL VIEW OVERLAY --- */}
        <AnimatePresence>
            {step === 4 && (
                <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-[#050505]/90 backdrop-blur-sm z-40 flex items-center justify-center p-8"
                >
                    <div className="bg-[#0a0a0a] border border-emerald-500/30 w-full max-w-sm p-6 rounded-2xl shadow-2xl shadow-emerald-900/20 relative overflow-hidden">
                         {/* Glow */}
                         <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 blur-[50px] rounded-full pointer-events-none" />
                         
                         <div className="flex items-center gap-3 mb-6">
                            <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center border border-emerald-500/30 text-emerald-400">
                                <CheckCircle2 className="w-5 h-5" />
                            </div>
                            <div>
                                <h3 className="text-white font-bold text-sm">Alpha Detected</h3>
                                <p className="text-emerald-500 text-[10px] uppercase tracking-wider font-semibold">Confidence: High</p>
                            </div>
                         </div>

                         <div className="space-y-3">
                            <div className="flex justify-between text-sm border-b border-white/5 pb-2">
                                <span className="text-zinc-500">Action</span>
                                <span className="text-white font-medium">Buy YES</span>
                            </div>
                             <div className="flex justify-between text-sm border-b border-white/5 pb-2">
                                <span className="text-zinc-500">Market</span>
                                <span className="text-zinc-300">Fed Rate Cut</span>
                            </div>
                            <div className="pt-2">
                                <div className="flex justify-between items-end mb-1">
                                    <span className="text-zinc-500 text-xs">Proj. Return</span>
                                    <span className="text-xl font-bold text-emerald-400 font-mono">+18.5%</span>
                                </div>
                                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                                    <motion.div 
                                        initial={{ width: 0 }}
                                        animate={{ width: "75%" }}
                                        transition={{ duration: 1, delay: 0.2 }}
                                        className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400" 
                                    />
                                </div>
                            </div>
                         </div>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>

      </div>
    </div>
  );
}

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));