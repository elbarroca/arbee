import { useEffect, useState, useRef, useCallback } from 'react';
import { ChatClient } from 'convo-ai-sdk';
import { toolRegistry } from '../tools/registry';
import { ToolCall, ChatMessage, ConnectionStatus, ToolInvocation } from '@/types/chat';

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

    // 1. Connection Status
    client.on('statusChange', (s: string) => setStatus(s as ConnectionStatus));

    // 2. Handle "Thinking/Reasoning" Events
    // This captures the AI's internal monologue before it decides to call a tool
    client.on('thought', (thought: string) => {
      let targetId = currentMsgId.current;
      
      // If no message exists yet, create one to hold the thought
      if (!targetId) {
        targetId = crypto.randomUUID();
        currentMsgId.current = targetId;
        setMessages(prev => [...prev, { 
          id: targetId!, 
          role: 'assistant', 
          content: '', 
          timestamp: Date.now(), 
          toolInvocations: [], 
          thoughts: [thought] // Init with thought
        }]);
      } else {
        // Append thought to existing message
        setMessages(prev => prev.map(msg => {
          if (msg.id === targetId) {
            return { ...msg, thoughts: [...(msg.thoughts || []), thought] };
          }
          return msg;
        }));
      }
    });

    // 3. Standard Text Streaming
    client.on('messageStart', () => {
      setIsTyping(true);
      if (!currentMsgId.current) {
        const id = crypto.randomUUID();
        currentMsgId.current = id;
        setMessages(prev => [...prev, { 
          id, 
          role: 'assistant', 
          content: '', 
          isStreaming: true, 
          timestamp: Date.now(), 
          toolInvocations: [] 
        }]);
      }
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

    // 4. TOOL HANDLING & WIDGET TRIGGERING
    client.on('toolCall', async (data: unknown) => {
      const toolCall = data as ToolCall;
      const toolFn = toolRegistry[toolCall.name];
      let result = "";

      // A. Ensure we have a message bubble to attach the tool log to
      let targetId = currentMsgId.current;
      if (!targetId) {
         targetId = crypto.randomUUID();
         currentMsgId.current = targetId;
         setIsTyping(true);
         setMessages(prev => [...prev, { 
            id: targetId!, 
            role: 'assistant', 
            content: '', 
            timestamp: Date.now(), 
            toolInvocations: [], 
            isStreaming: true
         }]);
      }

      // B. Set Tool Status: Pending
      setMessages(prev => prev.map(msg => {
         if (msg.id === targetId) {
            const newInvocations = [...(msg.toolInvocations || []), { 
               toolName: toolCall.name, status: 'pending', args: toolCall.args 
            } as ToolInvocation];
            return { ...msg, toolInvocations: newInvocations };
         }
         return msg;
      }));
      
      try {
        if (toolFn) {
          // C. Execute the Tool
          result = await toolFn(toolCall.args);
          
          // D. Set Tool Status: Complete
          setMessages(prev => prev.map(msg => {
             if (msg.id === targetId) {
                const updatedInvocations = (msg.toolInvocations || []).map(inv => 
                   inv.toolName === toolCall.name ? { ...inv, status: 'complete' } : inv
                );
                return { ...msg, toolInvocations: updatedInvocations as ToolInvocation[] };
             }
             return msg;
          }));

          // E. WIDGET TRIGGER LOGIC
          // We immediately attempt to parse the tool result. If it's a valid widget payload,
          // we inject a NEW system message into the stream immediately.
          try {
            const parsed = JSON.parse(result);
            
            if (parsed && parsed.type && ['positions', 'funds', 'trade_result', 'pnl'].includes(parsed.type)) {
               // Use a small timeout to ensure the "Complete" animation renders first
               setTimeout(() => {
                 const widgetId = crypto.randomUUID();
                 setMessages(prev => [...prev, {
                   id: widgetId,
                   role: 'system', // System role handles pure widget rendering
                   content: '', 
                   timestamp: Date.now(),
                   widgetType: parsed.type,
                   widgetData: parsed
                 }]);
               }, 100);
            }
          } catch {
            // Result was plain text, not JSON. That's fine, no widget to trigger.
          }

        } else {
          result = JSON.stringify({ error: `Tool ${toolCall.name} not implemented` });
        }
      } catch (err: unknown) {
        result = JSON.stringify({ error: err instanceof Error ? err.message : String(err) });
        
        // F. Set Tool Status: Error
        setMessages(prev => prev.map(msg => {
           if (msg.id === targetId) {
              const updatedInvocations = (msg.toolInvocations || []).map(inv => 
                 inv.toolName === toolCall.name ? { ...inv, status: 'error' } : inv
              );
              return { ...msg, toolInvocations: updatedInvocations as ToolInvocation[] };
           }
           return msg;
        }));
      }

      // G. Send result back to AI so conversation continues
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
    
    // Add user message immediately
    setMessages(prev => [...prev, { 
      id: crypto.randomUUID(), role: 'user', content: text, timestamp: Date.now()
    }]);

    await clientRef.current.sendMessage(text);
  }, []);

  const resetSession = useCallback(() => {
    setMessages([]);
    if (clientRef.current) {
      clientRef.current.resetChat(); 
    }
  }, []);

  return { messages, status, isTyping, sendMessage, resetSession };
}