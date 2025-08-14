#!/usr/bin/env python3
"""
Test script to verify Alpaca connection and data availability
Run this to debug data issues
"""

import os
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

def test_alpaca_connection():
    """Test Alpaca API connection and data feeds"""
    
    # Load credentials
    API_KEY = os.getenv('APCA_API_KEY_ID')
    API_SECRET = os.getenv('APCA_API_SECRET_KEY')
    BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    if not API_KEY or not API_SECRET:
        print(" API credentials not found in .env file!")
        return
    
    print(" Testing Alpaca Connection...")
    print(f" Base URL: {BASE_URL}")
    
    try:
        # Initialize API
        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        
        # Test 1: Account Status
        print("\n Testing Account Access...")
        account = api.get_account()
        print(f" Account Status: {account.status}")
        print(f" Buying Power: ${float(account.buying_power):,.2f}")
        print(f" Portfolio Value: ${float(account.portfolio_value):,.2f}")
        
        # Test 2: Clock
        print("\n Testing Market Clock...")
        clock = api.get_clock()
        print(f" Current time: {clock.timestamp}")
        print(f" Market is: {'OPEN' if clock.is_open else 'CLOSED'}")
        if not clock.is_open:
            print(f" Next open: {clock.next_open}")
            print(f" Next close: {clock.next_close}")
        
        # Test 3: Historical Data
        print("\n Testing Historical Data Access...")
        symbol = 'AAPL'
        
        # Try different timeframes and feeds
        timeframes = ['1Day', '1Hour', '30Min', '15Min']
        feeds = ['iex', 'sip']
        
        for timeframe in timeframes:
            print(f"\n Testing {timeframe} bars...")
            for feed in feeds:
                try:
                    end = datetime.now()
                    start = end - timedelta(days=5)
                    
                    bars = api.get_bars(
                        symbol,
                        timeframe,
                        start=start.strftime('%Y-%m-%d'),
                        end=end.strftime('%Y-%m-%d'),
                        limit=100,
                        feed=feed
                    ).df
                    
                    if not bars.empty:
                        print(f"   {feed.upper()}: Got {len(bars)} bars")
                        print(f"     Latest: {bars.index[-1]} - Open: ${bars.iloc[-1]['open']:.2f}")
                    else:
                        print(f"   {feed.upper()}: No data")
                        
                except Exception as e:
                    print(f"   {feed.upper()}: {str(e)[:50]}...")
        
        # Test 4: Latest Price
        print(f"\n Testing Latest Price for {symbol}...")
        try:
            # Try IEX quote
            quote = api.get_latest_quote(symbol, feed='iex')
            print(f" IEX Quote: Bid=${quote.bid_price:.2f}, Ask=${quote.ask_price:.2f}")
        except Exception as e:
            print(f" IEX Quote: {e}")
        
        try:
            # Try latest trade
            trade = api.get_latest_trade(symbol, feed='iex')
            print(f" Latest Trade: ${trade.price:.2f} at {trade.timestamp}")
        except Exception as e:
            print(f" Latest Trade: {e}")
        
        # Test 5: Check if we can trade
        print(f"\n Testing Trading Capability...")
        try:
            # Check if we can get positions
            positions = api.list_positions()
            print(f" Can access positions: {len(positions)} open positions")
            
            # Check tradable status
            asset = api.get_asset(symbol)
            print(f" {symbol} is {'tradable' if asset.tradable else 'not tradable'}")
            print(f" {symbol} is {'shortable' if asset.shortable else 'not shortable'}")
            
        except Exception as e:
            print(f" Trading check: {e}")
        
        print("\n✨ Connection test complete!")
        
    except Exception as e:
        print(f"\n Connection failed: {e}")
        print(" Please check:")
        print("   1. Your API keys are correct")
        print("   2. You're using paper trading keys")
        print("   3. Your account is active")

if __name__ == "__main__":
    test_alpaca_connection()
