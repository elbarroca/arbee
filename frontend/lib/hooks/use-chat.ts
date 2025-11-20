import { useEffect, useState, useRef, useCallback } from 'react';
import { ChatClient } from 'convo-ai-sdk';
import { toolRegistry } from '../tools/registry';
import { ToolCall, ChatMessage, ConnectionStatus } from '@/types/chat';

function getSessionId(): string {
  if (typeof window === 'undefined') return 'server_session';
  let id = localStorage.getItem('poly_session_id');
  if (!id) {
    id = `user_${crypto.randomUUID().slice(0, 8)}`;
    localStorage.setItem('poly_session_id', id);
  }
  return id;
}

export function useChat() {
  const apiKey = process.env.NEXT_PUBLIC_CONVO_API_KEY;
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>(!apiKey ? 'error' : 'disconnected');
  const [isTyping, setIsTyping] = useState(false);
  
  const clientRef = useRef<ChatClient | null>(null);
  const currentMsgId = useRef<string | null>(null);
  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current || !apiKey) return;
    initialized.current = true;

    const client = new ChatClient({
      apiKey: apiKey,
      identifier: getSessionId(),
      dynamicVariables: { app_context: "PolyAnalytics Dashboard" }
    });

    clientRef.current = client;

    // Event Handling
    client.on('statusChange', (s: string) => setStatus(s as ConnectionStatus));

    client.on('messageStart', () => {
      setIsTyping(true);
      const id = crypto.randomUUID();
      currentMsgId.current = id;
      setMessages(prev => [...prev, { 
        id, role: 'assistant', content: '', isStreaming: true, timestamp: Date.now()
      }]);
    });

    client.on('messageData', (chunk: string) => {
      if (!currentMsgId.current) return;
      setMessages(prev => prev.map(msg => 
        msg.id === currentMsgId.current ? { ...msg, content: msg.content + chunk } : msg
      ));
    });

    client.on('messageDone', () => {
      setIsTyping(false);
      setMessages(prev => prev.map(msg => 
        msg.id === currentMsgId.current ? { ...msg, isStreaming: false } : msg
      ));
      currentMsgId.current = null;
    });

    client.on('toolCall', async (data: unknown) => {
      const toolCall = data as ToolCall;
      const toolFn = toolRegistry[toolCall.name];
      let result = "";
      
      try {
        if (toolFn) {
          result = await toolFn(toolCall.args);
        } else {
          result = JSON.stringify({ error: `Tool ${toolCall.name} not implemented` });
        }
      } catch {
        result = JSON.stringify({ error: "Internal tool error" });
      }

      await client.sendToolResult(toolCall.id, toolCall.name, result);
    });

    client.connect().catch(console.error);

    return () => {
      client.disconnect();
      initialized.current = false;
    };
  }, [apiKey]);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || !clientRef.current) return;
    
    setMessages(prev => [...prev, { 
      id: crypto.randomUUID(), role: 'user', content: text, timestamp: Date.now()
    }]);

    await clientRef.current.sendMessage(text);
  }, []);

  // NEW: Reset Functionality
  const resetSession = useCallback(() => {
    setMessages([]); // Clear UI
    if (clientRef.current) {
      console.log("🔄 Resetting Convo Session...");
      clientRef.current.resetChat(); // SDK Method: Disconnects, clears, reconnects
    }
  }, []);

  return { messages, status, isTyping, sendMessage, resetSession };
}