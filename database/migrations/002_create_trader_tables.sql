-- Migration 002: Create Trader Discovery Tables
-- Creates tables for tracking traders, their activity, and on-chain trades

-- ============================================================================
-- tracked_traders: Main table for trader information and scores
-- ============================================================================
CREATE TABLE IF NOT EXISTS tracked_traders (
    wallet_address VARCHAR(42) PRIMARY KEY,
    composite_score DECIMAL(5, 2) NOT NULL DEFAULT 0.0,
    early_betting_pct DECIMAL(5, 2) DEFAULT 0.0,
    volume_consistency DECIMAL(5, 2) DEFAULT 0.0,
    win_rate DECIMAL(5, 4) DEFAULT 0.0,
    edge_score DECIMAL(5, 2) DEFAULT 0.0,
    activity_level DECIMAL(10, 2) DEFAULT 0.0,
    
    -- P&L metrics
    pnl_30d DECIMAL(20, 8) DEFAULT 0.0,
    pnl_90d DECIMAL(20, 8) DEFAULT 0.0,
    trade_count INTEGER DEFAULT 0,
    sharpe_equivalent DECIMAL(10, 4) DEFAULT 0.0,
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'removed')),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Indexes
    CONSTRAINT tracked_traders_status_check CHECK (status IN ('active', 'paused', 'removed'))
);

CREATE INDEX IF NOT EXISTS idx_tracked_traders_composite_score ON tracked_traders(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_tracked_traders_status ON tracked_traders(status);
CREATE INDEX IF NOT EXISTS idx_tracked_traders_updated_at ON tracked_traders(updated_at DESC);

-- ============================================================================
-- trader_activity: Time-series snapshots of trader metrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS trader_activity (
    id SERIAL PRIMARY KEY,
    wallet_address VARCHAR(42) NOT NULL REFERENCES tracked_traders(wallet_address) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Snapshot metrics
    score_snapshot DECIMAL(5, 2) NOT NULL,
    trade_count_snapshot INTEGER DEFAULT 0,
    pnl_snapshot DECIMAL(20, 8) DEFAULT 0.0,
    
    -- Additional context
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- BRIN index for efficient time-series queries
    CONSTRAINT trader_activity_wallet_fk FOREIGN KEY (wallet_address) 
        REFERENCES tracked_traders(wallet_address) ON DELETE CASCADE
);

-- BRIN index for time-series optimization (efficient for append-only time-series data)
CREATE INDEX IF NOT EXISTS idx_trader_activity_timestamp_brin ON trader_activity 
    USING BRIN (timestamp);
CREATE INDEX IF NOT EXISTS idx_trader_activity_wallet_timestamp ON trader_activity(wallet_address, timestamp DESC);

-- ============================================================================
-- onchain_trades: All on-chain trades with market mapping
-- ============================================================================
CREATE TABLE IF NOT EXISTS onchain_trades (
    transaction_hash VARCHAR(66) PRIMARY KEY,
    wallet_address VARCHAR(42) NOT NULL REFERENCES tracked_traders(wallet_address) ON DELETE CASCADE,
    
    -- Market information
    market_slug VARCHAR(255),
    token_address VARCHAR(42),
    token_id VARCHAR(255),
    
    -- Trade details
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    size_usd DECIMAL(20, 8) DEFAULT 0.0,
    price DECIMAL(10, 8) DEFAULT 0.0,
    
    -- Blockchain metadata
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    block_number BIGINT,
    
    -- Market creation time (for early bet detection)
    market_created_at TIMESTAMP WITH TIME ZONE,
    
    -- Additional metadata
    raw_event_data JSONB DEFAULT '{}'::jsonb,
    
    -- Indexes
    CONSTRAINT onchain_trades_side_check CHECK (side IN ('BUY', 'SELL'))
);

CREATE INDEX IF NOT EXISTS idx_onchain_trades_wallet ON onchain_trades(wallet_address);
CREATE INDEX IF NOT EXISTS idx_onchain_trades_market ON onchain_trades(market_slug);
CREATE INDEX IF NOT EXISTS idx_onchain_trades_timestamp ON onchain_trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_onchain_trades_wallet_timestamp ON onchain_trades(wallet_address, timestamp DESC);

-- ============================================================================
-- Functions and Triggers
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at on tracked_traders
CREATE TRIGGER update_tracked_traders_updated_at
    BEFORE UPDATE ON tracked_traders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Comments for documentation
-- ============================================================================
COMMENT ON TABLE tracked_traders IS 'Main table for tracked traders with composite scores and metrics';
COMMENT ON TABLE trader_activity IS 'Time-series snapshots of trader metrics for historical analysis';
COMMENT ON TABLE onchain_trades IS 'All on-chain trades with market mapping for analysis';
COMMENT ON COLUMN tracked_traders.composite_score IS 'Overall trader score (0-100)';
COMMENT ON COLUMN trader_activity.timestamp IS 'Snapshot timestamp - use BRIN index for efficient queries';
COMMENT ON COLUMN onchain_trades.market_created_at IS 'Market creation time for early bet detection';


