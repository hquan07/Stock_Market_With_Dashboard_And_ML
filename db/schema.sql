CREATE DATABASE IF NOT EXISTS stock_db;
USE stock_db;

CREATE TABLE IF NOT EXISTS prices (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  ticker VARCHAR(16) NOT NULL,
  trade_date DATE NOT NULL,
  open DECIMAL(18,6),
  high DECIMAL(18,6),
  low DECIMAL(18,6),
  close DECIMAL(18,6),
  adj_close DECIMAL(18,6),
  volume BIGINT,
  daily_return DOUBLE,
  sector VARCHAR(100),
  market_cap BIGINT,
  INDEX idx_ticker_date (ticker, trade_date),
  INDEX idx_date (trade_date),
  INDEX idx_sector (sector)
) ENGINE=InnoDB;