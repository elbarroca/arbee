-- Migration 003: Create Paper Trading Tables
-- Creates tables for paper trading logs and daily summaries

-- ============================================================================
-- paper_trading_logs: All simulated trades
-- ============================================================================
CREATE TABLE IF NOT EXISTS paper_trading_logs (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(255) NOT NULL,
    wallet_address VARCHAR(42) NOT NULL,
    market_slug VARCHAR(255) NOT NULL,
    
    -- Trade details
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    size_usd DECIMAL(20, 8) NOT NULL,
    expected_price DECIMAL(10, 8) NOT NULL,
    fill_price DECIMAL(10, 8) NOT NULL,
    slippage_bps INTEGER DEFAULT 0,
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'filled' CHECK (status IN ('filled', 'rejected')),
    
    -- P&L tracking
    pnl_realized DECIMAL(20, 8) DEFAULT NULL,  -- Set after market resolution
    pnl_unrealized DECIMAL(20, 8) DEFAULT 0.0,  -- Updated via mark-to-market
    
    -- Timestamps
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT paper_trading_logs_side_check CHECK (side IN ('BUY', 'SELL')),
    CONSTRAINT paper_trading_logs_status_check CHECK (status IN ('filled', 'rejected'))
);

CREATE INDEX IF NOT EXISTS idx_paper_trading_logs_wallet ON paper_trading_logs(wallet_address);
CREATE INDEX IF NOT EXISTS idx_paper_trading_logs_market ON paper_trading_logs(market_slug);
CREATE INDEX IF NOT EXISTS idx_paper_trading_logs_timestamp ON paper_trading_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_paper_trading_logs_signal_id ON paper_trading_logs(signal_id);

-- ============================================================================
-- paper_trading_summary: Daily aggregates
-- ============================================================================
CREATE TABLE IF NOT EXISTS paper_trading_summary (
    date DATE PRIMARY KEY,
    
    -- Trade counts
    total_trades INTEGER DEFAULT 0,
    filled_trades INTEGER DEFAULT 0,
    rejected_trades INTEGER DEFAULT 0,
    
    -- P&L metrics
    total_pnl DECIMAL(20, 8) DEFAULT 0.0,
    realized_pnl DECIMAL(20, 8) DEFAULT 0.0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0.0,
    
    -- Performance metrics
    sharpe_ratio DECIMAL(10, 4) DEFAULT NULL,
    sortino_ratio DECIMAL(10, 4) DEFAULT NULL,
    max_drawdown DECIMAL(10, 4) DEFAULT NULL,
    win_rate DECIMAL(5, 4) DEFAULT 0.0,
    
    -- Volume metrics
    total_volume_usd DECIMAL(20, 8) DEFAULT 0.0,
    avg_trade_size_usd DECIMAL(20, 8) DEFAULT 0.0,
    
    -- Slippage metrics
    avg_slippage_bps DECIMAL(10, 2) DEFAULT 0.0,
    max_slippage_bps INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_paper_trading_summary_date ON paper_trading_summary(date DESC);

-- ============================================================================
-- Functions and Triggers
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_paper_trading_summary_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at on paper_trading_summary
CREATE TRIGGER update_paper_trading_summary_updated_at
    BEFORE UPDATE ON paper_trading_summary
    FOR EACH ROW
    EXECUTE FUNCTION update_paper_trading_summary_updated_at();

-- ============================================================================
-- Comments for documentation
-- ============================================================================
COMMENT ON TABLE paper_trading_logs IS 'All simulated trades from paper trading mode';
COMMENT ON TABLE paper_trading_summary IS 'Daily aggregates of paper trading performance';
COMMENT ON COLUMN paper_trading_logs.pnl_realized IS 'Realized P&L after market resolution (NULL if unresolved)';
COMMENT ON COLUMN paper_trading_logs.pnl_unrealized IS 'Unrealized P&L updated via mark-to-market';
COMMENT ON COLUMN paper_trading_summary.sharpe_ratio IS 'Sharpe ratio calculated from daily returns';
COMMENT ON COLUMN paper_trading_summary.max_drawdown IS 'Maximum drawdown percentage';


