#!/usr/bin/env python3
"""
Generate sample CSV files for testing the IrisTrading data import.

This creates realistic-looking OHLCV candle data for multiple symbols.
Use this for testing or when you don't have access to real historical data.

Usage:
    python scripts/generate_sample_csv.py --output data/ --count 5000
"""

import argparse
import csv
import random
from datetime import datetime, timedelta


def generate_candles(symbol: str, count: int, start_price: float, start_time: int) -> list:
    """Generate realistic OHLCV candle data."""
    candles = []
    price = start_price
    timestamp = start_time
    
    # 5-minute intervals
    interval_ms = 5 * 60 * 1000
    
    for i in range(count):
        # Simulate price movement with random walk
        volatility = 0.002  # 0.2% volatility
        change_pct = random.gauss(0, volatility)
        
        open_price = price
        price_change = price * change_pct
        close_price = price + price_change
        
        # Generate high and low
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.001))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.001))
        
        # Generate volume (randomized)
        volume = random.uniform(50, 500)
        
        candles.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2),
        })
        
        price = close_price
        timestamp += interval_ms
    
    return candles


def save_to_csv(symbol: str, candles: list, output_dir: str):
    """Save candles to CSV file."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"{symbol}.csv")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        writer.writeheader()
        writer.writerows(candles)
    
    print(f"✓ Generated {len(candles)} candles for {symbol} -> {filename}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample CSV data for IrisTrading')
    parser.add_argument('--output', '-o', default='data', help='Output directory for CSV files')
    parser.add_argument('--count', '-c', type=int, default=5000, help='Number of candles per symbol')
    parser.add_argument('--symbols', nargs='+', 
                        default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'],
                        help='Symbols to generate')
    
    args = parser.parse_args()
    
    # Start from 30 days ago
    start_datetime = datetime.now() - timedelta(days=30)
    start_timestamp = int(start_datetime.timestamp() * 1000)
    
    # Base prices for different symbols
    base_prices = {
        'BTCUSDT': 42000.0,
        'ETHUSDT': 2200.0,
        'SOLUSDT': 100.0,
        'BNBUSDT': 300.0,
        'XRPUSDT': 0.6,
    }
    
    print(f"Generating {args.count} candles for {len(args.symbols)} symbols...")
    print(f"Output directory: {args.output}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()
    
    for symbol in args.symbols:
        base_price = base_prices.get(symbol, 100.0)
        candles = generate_candles(symbol, args.count, base_price, start_timestamp)
        save_to_csv(symbol, candles, args.output)
    
    print()
    print(f"Done! Generated CSV files in '{args.output}/' directory.")
    print()
    print("To import into IrisTrading database:")
    print(f"  cargo run --release -- --mode import --dir {args.output}")


if __name__ == '__main__':
    main()
