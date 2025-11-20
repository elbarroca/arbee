'use client';

import { useState, useRef, useEffect } from 'react';
import { ArrowUp, Paperclip } from 'lucide-react';

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'inherit';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
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
    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  return (
    <div className="relative w-full">
      <div className="relative flex items-end w-full p-3 bg-[#1e1e1e] border border-zinc-700/50 rounded-xl shadow-lg focus-within:ring-1 focus-within:ring-blue-500/50 focus-within:border-blue-500/50 transition-all">
        
        {/* Attachment Button (Visual only for now) */}
        <button className="p-2 text-zinc-400 hover:text-white transition-colors mr-2">
          <Paperclip className="w-5 h-5" />
        </button>

        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Send a message..."
          disabled={disabled}
          rows={1}
          className="w-full max-h-[200px] bg-transparent border-none focus:ring-0 resize-none text-zinc-100 placeholder:text-zinc-500 py-2 scrollbar-none"
          style={{ minHeight: '40px' }}
        />

        <button
          onClick={handleSubmit}
          disabled={!input.trim() || disabled}
          className="p-2 ml-2 bg-white text-black rounded-lg hover:bg-zinc-200 disabled:bg-zinc-700 disabled:text-zinc-500 transition-all flex-shrink-0"
        >
          <ArrowUp className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}