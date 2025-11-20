import { supabase } from '../supabase';

// Define a generic type for tool arguments
export type ToolFunction = (args: Record<string, unknown>) => Promise<string>;

export const toolRegistry: Record<string, ToolFunction> = {
  /**
   * Tool: get_wallet_count
   * Returns the exact number of wallets in the database.
   */
  get_wallet_count: async (_args) => { // eslint-disable-line @typescript-eslint/no-unused-vars
    try {
      const { count, error } = await supabase
        .from('wallets')
        .select('*', { count: 'exact', head: true });

      if (error) throw new Error(error.message);

      return JSON.stringify({ 
        total_wallets: count,
        source: "supabase_production",
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown database error';
      return JSON.stringify({ error: message });
    }
  },
};