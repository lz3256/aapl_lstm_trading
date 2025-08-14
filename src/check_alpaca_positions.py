#!/usr/bin/env python3
"""
Check current positions and recent orders in Alpaca account
"""

import os
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

def check_account_status():
    """Check detailed account status, positions, and orders"""
    
    # Load credentials
    API_KEY = os.getenv('APCA_API_KEY_ID')
    API_SECRET = os.getenv('APCA_API_SECRET_KEY')
    BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    if not API_KEY or not API_SECRET:
        print("API credentials not found!")
        return
    
    try:
        # Initialize API
        api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
        
        print("=" * 60)
        print("   ALPACA ACCOUNT STATUS CHECK")
        print("=" * 60)
        print(f"  Time: {datetime.now()}")
        print()
        
        # 1. Account Overview
        print("  ACCOUNT OVERVIEW:")
        print("-" * 40)
        account = api.get_account()
        print(f"Status: {account.status}")
        print(f"Cash: ${float(account.cash):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"Long Market Value: ${float(account.long_market_value):,.2f}")
        print(f"Short Market Value: ${float(account.short_market_value):,.2f}")
        
        # 2. Current Positions
        print("\n  CURRENT POSITIONS:")
        print("-" * 40)
        positions = api.list_positions()
        
        if not positions:
            print("No open positions")
        else:
            for position in positions:
                print(f"\nSymbol: {position.symbol}")
                print(f"  Quantity: {position.qty} shares")
                print(f"  Side: {position.side}")
                print(f"  Entry Price: ${float(position.avg_entry_price):.2f}")
                print(f"  Current Price: ${float(position.current_price):.2f}")
                print(f"  Market Value: ${float(position.market_value):,.2f}")
                print(f"  Unrealized P&L: ${float(position.unrealized_pl):.2f}")
                print(f"  Unrealized P&L %: {float(position.unrealized_plpc)*100:.2f}%")
        
        # 3. Recent Orders (last 24 hours)
        print("\n  RECENT ORDERS (Last 24 hours):")
        print("-" * 40)
        
        # Get orders from last 24 hours
        after = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        orders = api.list_orders(status='all', after=after, limit=20)
        
        if not orders:
            print("No orders in the last 24 hours")
        else:
            for order in orders[:10]:  # Show last 10 orders
                print(f"\nOrder ID: {order.id}")
                print(f"  Symbol: {order.symbol}")
                print(f"  Side: {order.side}")
                print(f"  Quantity: {order.qty}")
                print(f"  Type: {order.order_type}")
                print(f"  Status: {order.status}")
                print(f"  Submitted: {order.submitted_at}")
                if order.filled_at:
                    print(f"  Filled: {order.filled_at}")
                    print(f"  Filled Price: ${float(order.filled_avg_price):.2f}")
        
        # 4. Today's Activity Summary
        print("\n  TODAY'S ACTIVITY:")
        print("-" * 40)
        
        # Count today's orders
        today = datetime.now().strftime('%Y-%m-%d')
        todays_orders = api.list_orders(status='all', after=today, limit=100)
        
        filled_orders = [o for o in todays_orders if o.status == 'filled']
        print(f"Orders placed today: {len(todays_orders)}")
        print(f"Orders filled today: {len(filled_orders)}")
        
        # Calculate P&L for today if positions exist
        if positions:
            total_unrealized_pl = sum(float(p.unrealized_pl) for p in positions)
            print(f"\nTotal Unrealized P&L: ${total_unrealized_pl:.2f}")
        
        print("\n  Check complete!")
        
    except Exception as e:
        print(f"\n  Error: {e}")
        print("Please check your API credentials and connection")

if __name__ == "__main__":
    check_account_status()
