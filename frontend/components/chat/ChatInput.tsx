'use client';

import { useState, useRef, useEffect } from 'react';
import { ArrowUp, Paperclip } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize logic
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'inherit'; // Reset
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (!input.trim() || disabled) return;
    onSend(input);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };


  return (
    <div className="relative w-full">
      <div className={cn(
        "relative flex items-end w-full p-2 rounded-2xl border shadow-lg transition-all duration-200 bg-[#09090b]",
        // Updated Styling: Neutral colors, no blue ring
        "border-white/10",
        "focus-within:border-zinc-600 focus-within:ring-1 focus-within:ring-zinc-700/50"
      )}>
        
        {/* Attachment Icon */}
        <button className="p-2.5 text-zinc-500 hover:text-zinc-300 transition-colors rounded-full hover:bg-zinc-800/50 self-end mb-0.5 mr-1">
          <Paperclip className="w-5 h-5" />
        </button>

        {/* Text Area */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about markets, wallets, or trade volume..."
          disabled={disabled}
          rows={1}
          className="w-full max-h-[120px] bg-transparent border-none focus:ring-0 resize-none text-zinc-100 placeholder:text-zinc-600 py-3 px-1 scrollbar-none text-[15px] leading-relaxed"
          style={{ minHeight: '48px' }}
        />

        {/* Right Action Area */}
        <div className="flex items-center gap-1 self-end mb-0.5 ml-1">

            <button
                onClick={handleSubmit}
                disabled={!input.trim() || disabled}
                className={cn(
                    "p-2 rounded-xl transition-all flex-shrink-0 flex items-center justify-center w-10 h-10",
                    input.trim() 
                        ? "bg-white text-black hover:bg-zinc-200 shadow-sm" 
                        : "bg-zinc-800/50 text-zinc-600 cursor-not-allowed"
                )}
            >
                <ArrowUp className="w-5 h-5" />
            </button>
        </div>
      </div>
    </div>
  );
}