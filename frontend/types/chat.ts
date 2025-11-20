export interface ToolCall {
    id: string;
    name: string;
    args: Record<string, unknown>;
  }
  
  export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    isStreaming?: boolean;
    timestamp: number;
  }
  
  export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';