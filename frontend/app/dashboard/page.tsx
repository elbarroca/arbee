'use client';

import { useEffect, useRef } from 'react';
import { useChat } from '@/lib/hooks/use-chat';
import { cn } from '@/lib/utils';
import { User, Sparkles, Command, RotateCcw } from 'lucide-react';
import Markdown from 'react-markdown';
import ChatInput from '@/components/chat/ChatInput'; // Ensure this path is correct based on previous steps

export default function ChatPage() {
  const { messages, status, isTyping, sendMessage, resetSession } = useChat();
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [messages, isTyping]);

  return (
    <div className="flex flex-col h-full w-full relative">
      
      {/* Floating Header for Mobile/Status */}
      <header className="absolute top-0 left-0 w-full h-14 flex items-center justify-between px-6 z-10 bg-gradient-to-b from-[#050505] to-transparent">
        <div className="md:hidden flex items-center gap-2 font-bold text-zinc-200">
          <Command className="w-4 h-4" /> PolyAnalytics
        </div>
        <div className="ml-auto flex items-center gap-2">
           <button
             onClick={resetSession}
             className="p-1.5 rounded-md hover:bg-zinc-800/50 transition-colors"
             title="Reset chat session"
           >
             <RotateCcw className="w-3.5 h-3.5 text-zinc-400 hover:text-zinc-200" />
           </button>
           <span className={cn("w-2 h-2 rounded-full shadow-[0_0_10px_currentColor]",
             status === 'connected' ? "bg-emerald-500 text-emerald-500" : "bg-amber-500 text-amber-500")}
           />
        </div>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto pt-20 pb-32 scrollbar-thin scrollbar-thumb-zinc-800/50 hover:scrollbar-thumb-zinc-700">
        {messages.length === 0 ? (
          <EmptyState onSuggestionClick={sendMessage} />
        ) : (
          <div className="max-w-3xl mx-auto w-full px-4 space-y-6">
            {messages.map((msg) => (
              <div key={msg.id} className={cn("flex gap-4 w-full animate-in fade-in slide-in-from-bottom-2", msg.role === 'user' ? "justify-end" : "justify-start")}>
                
                {/* Bot Avatar */}
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-cyan-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-blue-900/20 border border-white/10">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                )}

                {/* Message Bubble */}
                <div className={cn(
                  "relative max-w-[85%] px-5 py-3.5 text-[15px] leading-relaxed shadow-md",
                  msg.role === 'user' 
                    ? "bg-white text-black rounded-2xl rounded-br-none" 
                    : "bg-zinc-900 border border-white/5 text-zinc-100 rounded-2xl rounded-bl-none"
                )}>
                  <Markdown>{msg.content}</Markdown>
                </div>

                {/* User Avatar */}
                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-lg bg-zinc-800 border border-zinc-700 flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-zinc-400" />
                  </div>
                )}
              </div>
            ))}
            
            {isTyping && (
              <div className="flex gap-4 w-full max-w-3xl mx-auto px-4">
                <div className="w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center">
                   <Sparkles className="w-4 h-4 text-blue-500 animate-pulse" />
                </div>
                <div className="flex items-center gap-1 h-8">
                   <div className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce [animation-delay:-0.3s]" />
                   <div className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce [animation-delay:-0.15s]" />
                   <div className="w-1.5 h-1.5 bg-zinc-600 rounded-full animate-bounce" />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="absolute bottom-0 left-0 w-full bg-[#050505] pt-4 pb-6 px-4 z-20 border-t border-white/5">
        <div className="max-w-3xl mx-auto">
          <ChatInput onSend={sendMessage} disabled={status !== 'connected'} />
          <div className="text-center mt-3 flex items-center justify-center gap-2">
            <span className="text-[10px] text-zinc-600 uppercase tracking-widest">Powered by Convo AI & Polymarket Data</span>
          </div>
        </div>
      </div>

    </div>
  );
}

function EmptyState({ onSuggestionClick }: { onSuggestionClick: (text: string) => void }) {
  const suggestions = [
    { icon: "💰", label: "Active Wallets", query: "How many wallets are we currently tracking?" },
    { icon: "📈", label: "Top Traders", query: "Who are the most profitable traders this week?" },
    { icon: "🔍", label: "Market Scan", query: "List active markets with > $100k volume." },
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full px-4">
      <div className="mb-8 relative group">
        <div className="absolute inset-0 bg-blue-500/20 blur-xl rounded-full group-hover:bg-blue-500/30 transition-all" />
        <div className="w-20 h-20 bg-[#0a0a0a] rounded-2xl border border-zinc-800 flex items-center justify-center relative z-10 shadow-2xl">
          <Command className="w-10 h-10 text-white" />
        </div>
      </div>
      
      <h2 className="text-2xl font-bold text-white mb-2 tracking-tight">How can I help you trade?</h2>
      <p className="text-zinc-500 mb-10 max-w-md text-center text-sm">
        I have direct access to the Polymarket database. I can analyze wallet profitability, market trends, and volume.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-3xl">
        {suggestions.map((s, i) => (
          <button
            key={i}
            onClick={() => onSuggestionClick(s.query)}
            className="flex flex-col items-center gap-3 p-4 rounded-xl border border-zinc-800 bg-zinc-900/30 hover:bg-zinc-900 hover:border-zinc-700 transition-all group"
          >
            <span className="text-2xl group-hover:scale-110 transition-transform">{s.icon}</span>
            <span className="text-xs font-medium text-zinc-300 group-hover:text-white">{s.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}