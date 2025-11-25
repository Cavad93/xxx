#!/usr/bin/env python3
"""
Backtesting Crypto Portfolio Strategy with Fear & Greed Index

Strategy:
- Monthly rebalancing of TOP-20 cryptocurrencies (excluding stablecoins)
- Investment Capital = Total Capital * (1 - Fear&Greed Index)
- Equal weight: 5% per coin (20 coins total)
- Remaining capital in stablecoins
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class CryptoStrategyBacktest:
    def __init__(self, initial_capital: float = 1000, start_date: str = "2020-01-01"):
        self.initial_capital = initial_capital
        self.start_date = pd.to_datetime(start_date)
        self.current_capital = initial_capital

        # Data storage
        self.fear_greed_data = None
        self.price_data = {}
        self.portfolio_history = []
        self.rebalance_dates = []

        # Stablecoins to exclude
        self.stablecoins = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP',
                           'USDD', 'GUSD', 'FRAX', 'LUSD', 'USDK']

    def fetch_fear_greed_index(self) -> pd.DataFrame:
        """
        Fetch historical Fear & Greed Index data
        API: Alternative.me
        """
        print("📊 Fetching Fear & Greed Index data...")

        try:
            # Try with smaller limit first (API has limits)
            url = "https://api.alternative.me/fng/?limit=2000&format=json"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'data' in data:
                df = pd.DataFrame(data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                df['value'] = df['value'].astype(float) / 100  # Convert to 0-1 scale
                df = df[['timestamp', 'value']].rename(columns={'timestamp': 'date'})
                df = df.sort_values('date').reset_index(drop=True)

                # Filter by start date
                df = df[df['date'] >= self.start_date]

                print(f"✅ Loaded {len(df)} days of Fear & Greed Index data (REAL)")
                print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
                print(f"   Average F&G: {df['value'].mean():.2%}")

                self.fear_greed_data = df
                return df
            else:
                print("⚠️  No data in response, generating synthetic data...")
                return self.generate_synthetic_fear_greed()

        except Exception as e:
            print(f"⚠️  Error fetching Fear & Greed Index: {e}")
            print("   Generating synthetic data for demonstration...")
            return self.generate_synthetic_fear_greed()

    def generate_synthetic_fear_greed(self) -> pd.DataFrame:
        """
        Generate synthetic Fear & Greed Index data for demonstration
        Uses realistic patterns with cycles and random variations
        """
        print("🔄 Generating synthetic Fear & Greed Index data...")

        days_count = (datetime.now() - self.start_date).days
        dates = pd.date_range(start=self.start_date, periods=days_count, freq='D')

        # Generate realistic fear & greed cycles
        # Base sine wave for market cycles (roughly 180-day cycles)
        base_cycle = np.sin(np.linspace(0, days_count / 180 * 2 * np.pi, days_count))

        # Add shorter-term volatility
        short_cycle = np.sin(np.linspace(0, days_count / 30 * 2 * np.pi, days_count)) * 0.3

        # Random noise
        noise = np.random.normal(0, 0.15, days_count)

        # Combine and normalize to 0-1 range
        values = (base_cycle + short_cycle + noise + 1) / 2
        values = np.clip(values, 0.1, 0.9)  # Keep within reasonable bounds

        df = pd.DataFrame({
            'date': dates,
            'value': values
        })

        print(f"✅ Generated {len(df)} days of synthetic F&G data")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Average F&G: {df['value'].mean():.2%}")

        self.fear_greed_data = df
        return df

    def get_top_cryptos(self, date: datetime = None, limit: int = 50) -> List[str]:
        """
        Get top cryptocurrencies by market cap (excluding stablecoins)
        """
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            coins = response.json()

            # Filter out stablecoins
            filtered_coins = [
                coin['symbol'].upper()
                for coin in coins
                if coin['symbol'].upper() not in self.stablecoins
            ][:20]

            return filtered_coins

        except Exception as e:
            print(f"❌ Error fetching top cryptos: {e}")
            # Return hardcoded top-20 for fallback
            return ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'TRX',
                   'DOT', 'MATIC', 'LTC', 'SHIB', 'AVAX', 'UNI', 'LINK',
                   'XLM', 'ATOM', 'XMR', 'BCH', 'ALGO']

    def fetch_crypto_prices(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for cryptocurrencies
        Using CoinGecko API
        """
        print(f"\n💰 Fetching price data for {len(symbols)} cryptocurrencies...")

        # Map common symbols to CoinGecko IDs
        symbol_to_id = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'SOL': 'solana',
            'TRX': 'tron',
            'DOT': 'polkadot',
            'MATIC': 'matic-network',
            'LTC': 'litecoin',
            'SHIB': 'shiba-inu',
            'AVAX': 'avalanche-2',
            'UNI': 'uniswap',
            'LINK': 'chainlink',
            'XLM': 'stellar',
            'ATOM': 'cosmos',
            'ETC': 'ethereum-classic',
            'XMR': 'monero',
            'BCH': 'bitcoin-cash',
            'ALGO': 'algorand',
            'VET': 'vechain',
            'FIL': 'filecoin',
            'APT': 'aptos',
            'HBAR': 'hedera-hashgraph',
            'NEAR': 'near',
            'LEO': 'leo-token',
            'ICP': 'internet-computer',
            'CRO': 'crypto-com-chain',
            'QNT': 'quant-network',
            'ARB': 'arbitrum',
            'OP': 'optimism',
            'INJ': 'injective-protocol',
            'STX': 'blockstack',
            'SUI': 'sui',
            'RUNE': 'thorchain',
            'GRT': 'the-graph',
            'MKR': 'maker',
            'AAVE': 'aave',
            'IMX': 'immutable-x',
        }

        prices = {}
        days_ago = (datetime.now() - self.start_date).days

        for symbol in symbols:
            try:
                coin_id = symbol_to_id.get(symbol)
                if not coin_id:
                    print(f"⚠️  Skipping {symbol} (ID not found)")
                    continue

                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days_ago,
                    'interval': 'daily'
                }
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }

                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()

                if 'prices' in data:
                    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df[['date', 'price']].sort_values('date').reset_index(drop=True)

                    # Filter by start date
                    df = df[df['date'] >= self.start_date]

                    prices[symbol] = df
                    print(f"✅ {symbol}: {len(df)} days")

                # Rate limiting (CoinGecko free tier: 10-50 calls/min)
                time.sleep(2)

            except Exception as e:
                print(f"❌ Error fetching {symbol}: {e}")
                continue

        print(f"\n✅ Successfully loaded {len(prices)} cryptocurrencies")
        self.price_data = prices
        return prices

    def calculate_monthly_avg_fear_greed(self, month_start: datetime, month_end: datetime) -> float:
        """
        Calculate average Fear & Greed Index for previous month
        """
        if self.fear_greed_data is None or len(self.fear_greed_data) == 0:
            return 0.5  # Default to 50% if no data

        mask = (self.fear_greed_data['date'] >= month_start) & \
               (self.fear_greed_data['date'] < month_end)

        month_data = self.fear_greed_data[mask]

        if len(month_data) == 0:
            return 0.5

        return month_data['value'].mean()

    def get_price_at_date(self, symbol: str, target_date: datetime) -> float:
        """
        Get price for a symbol at a specific date
        """
        if symbol not in self.price_data:
            return None

        df = self.price_data[symbol]

        # Find closest date
        df['diff'] = abs((df['date'] - target_date).dt.total_seconds())
        closest = df.loc[df['diff'].idxmin()]

        return closest['price']

    def run_backtest(self):
        """
        Run the backtest strategy
        """
        print("\n" + "="*60)
        print("🚀 STARTING BACKTEST")
        print("="*60)

        # Get top cryptocurrencies
        print("\n📋 Getting TOP-20 cryptocurrencies...")
        top_cryptos = self.get_top_cryptos()
        print(f"TOP-20: {', '.join(top_cryptos)}")

        # Fetch price data
        if not self.fetch_crypto_prices(top_cryptos):
            print("❌ Failed to fetch price data")
            return

        # Generate monthly rebalance dates
        current_date = self.start_date
        end_date = datetime.now()

        rebalance_dates = []
        while current_date < end_date:
            rebalance_dates.append(current_date)
            # Move to first day of next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1, day=1)

        print(f"\n📅 Rebalancing {len(rebalance_dates)} times (monthly)")

        # Initialize portfolio
        total_capital = self.initial_capital
        stablecoin_balance = 0
        holdings = {}  # {symbol: quantity}

        # Track portfolio value over time
        portfolio_history = []

        for i, rebalance_date in enumerate(rebalance_dates):
            print(f"\n{'='*60}")
            print(f"📅 Rebalance #{i+1}: {rebalance_date.strftime('%Y-%m-%d')}")
            print(f"{'='*60}")

            # Calculate average F&G for previous month
            if i == 0:
                # First month: use current month data
                month_start = rebalance_date
                month_end = rebalance_date + timedelta(days=30)
            else:
                # Use previous month
                prev_date = rebalance_dates[i-1]
                month_start = prev_date
                month_end = rebalance_date

            avg_fg = self.calculate_monthly_avg_fear_greed(month_start, month_end)

            # Calculate current portfolio value
            if i > 0:
                portfolio_value = stablecoin_balance
                for symbol, qty in holdings.items():
                    price = self.get_price_at_date(symbol, rebalance_date)
                    if price:
                        portfolio_value += qty * price

                total_capital = portfolio_value

            # Calculate investment capital based on F&G
            investment_capital = total_capital * (1 - avg_fg)
            stablecoin_balance = total_capital - investment_capital

            print(f"💰 Total Capital: ${total_capital:,.2f}")
            print(f"😨 Avg Fear & Greed: {avg_fg:.1%}")
            print(f"📈 Investment Capital: ${investment_capital:,.2f}")
            print(f"💵 Stablecoin Balance: ${stablecoin_balance:,.2f}")

            # Get available coins (only those with price data)
            available_coins = [s for s in top_cryptos if s in self.price_data]
            num_coins = min(20, len(available_coins))

            if num_coins == 0:
                print("⚠️  No available coins for this period")
                continue

            # Calculate allocation per coin (5% each)
            allocation_per_coin = investment_capital / num_coins

            # Rebalance portfolio
            holdings = {}

            for symbol in available_coins[:num_coins]:
                price = self.get_price_at_date(symbol, rebalance_date)
                if price and price > 0:
                    quantity = allocation_per_coin / price
                    holdings[symbol] = quantity

            print(f"🎯 Allocated to {len(holdings)} coins: ${allocation_per_coin:,.2f} each")

            # Record portfolio state
            portfolio_history.append({
                'date': rebalance_date,
                'total_capital': total_capital,
                'investment_capital': investment_capital,
                'stablecoin_balance': stablecoin_balance,
                'fear_greed_index': avg_fg,
                'num_holdings': len(holdings),
                'holdings': holdings.copy()
            })

        # Calculate final portfolio value
        if len(rebalance_dates) > 0:
            final_date = datetime.now()
            final_value = stablecoin_balance

            for symbol, qty in holdings.items():
                price = self.get_price_at_date(symbol, final_date)
                if price:
                    final_value += qty * price

            # Add final snapshot
            portfolio_history.append({
                'date': final_date,
                'total_capital': final_value,
                'investment_capital': 0,
                'stablecoin_balance': stablecoin_balance,
                'fear_greed_index': 0,
                'num_holdings': len(holdings),
                'holdings': holdings.copy()
            })

        self.portfolio_history = portfolio_history

        # Print results
        self.print_results()
        self.plot_results()

    def print_results(self):
        """
        Print backtest results
        """
        if len(self.portfolio_history) < 2:
            print("\n❌ Not enough data to calculate results")
            return

        initial_value = self.portfolio_history[0]['total_capital']
        final_value = self.portfolio_history[-1]['total_capital']

        total_return = (final_value - initial_value) / initial_value

        # Calculate daily returns for metrics
        values = [p['total_capital'] for p in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()

        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Capital:     ${initial_value:,.2f}")
        print(f"Final Capital:       ${final_value:,.2f}")
        print(f"Total Return:        {total_return:+.2%}")
        print(f"Absolute Profit:     ${final_value - initial_value:+,.2f}")

        if len(returns) > 0:
            print(f"\n📈 Performance Metrics:")
            print(f"Average Return:      {returns.mean():.2%}")
            print(f"Volatility:          {returns.std():.2%}")
            print(f"Max Drawdown:        {(returns.min()):.2%}")
            print(f"Best Period:         {returns.max():.2%}")

            if returns.std() != 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(12)  # Annualized
                print(f"Sharpe Ratio:        {sharpe:.2f}")

        print(f"\n📅 Period:")
        print(f"Start Date:          {self.portfolio_history[0]['date'].strftime('%Y-%m-%d')}")
        print(f"End Date:            {self.portfolio_history[-1]['date'].strftime('%Y-%m-%d')}")
        print(f"Duration:            {len(self.portfolio_history)-1} months")
        print("="*60)

    def plot_results(self):
        """
        Plot backtest results
        """
        if len(self.portfolio_history) < 2:
            return

        df = pd.DataFrame(self.portfolio_history)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Portfolio Value
        ax1 = axes[0]
        ax1.plot(df['date'], df['total_capital'], linewidth=2, color='#2E86AB', label='Total Capital')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value (USD)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Plot 2: Capital Allocation
        ax2 = axes[1]
        ax2.fill_between(df['date'], 0, df['investment_capital'], alpha=0.6, color='#06A77D', label='Investment Capital')
        ax2.fill_between(df['date'], df['investment_capital'], df['total_capital'], alpha=0.6, color='#F18F01', label='Stablecoins')
        ax2.set_title('Capital Allocation', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Value (USD)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Plot 3: Fear & Greed Index
        ax3 = axes[2]
        ax3.plot(df['date'], df['fear_greed_index'] * 100, linewidth=2, color='#C73E1D', label='Fear & Greed Index')
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax3.fill_between(df['date'], 0, 25, alpha=0.2, color='red', label='Extreme Fear')
        ax3.fill_between(df['date'], 75, 100, alpha=0.2, color='green', label='Extreme Greed')
        ax3.set_title('Fear & Greed Index', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Index Value', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig('/home/user/xxx/backtest_results.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Chart saved: backtest_results.png")

        # Also save data to CSV
        df.to_csv('/home/user/xxx/backtest_data.csv', index=False)
        print(f"💾 Data saved: backtest_data.csv")


def main():
    """
    Main function to run backtest
    """
    print("="*60)
    print("🔬 CRYPTO PORTFOLIO STRATEGY BACKTEST")
    print("="*60)
    print("\n📝 Strategy Details:")
    print("   - Monthly rebalancing")
    print("   - TOP-20 cryptocurrencies (no stablecoins)")
    print("   - Equal weight: 5% per coin")
    print("   - Investment = Capital × (1 - Fear&Greed)")
    print("   - Remaining in stablecoins")
    print("="*60)

    # Initialize backtest
    backtest = CryptoStrategyBacktest(
        initial_capital=1000,
        start_date="2020-01-01"
    )

    # Fetch Fear & Greed Index
    backtest.fetch_fear_greed_index()

    if backtest.fear_greed_data is None or len(backtest.fear_greed_data) == 0:
        print("\n❌ Failed to fetch Fear & Greed Index data")
        return

    # Run backtest
    backtest.run_backtest()

    print("\n✅ Backtest completed!")


if __name__ == "__main__":
    main()
