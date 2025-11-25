#!/usr/bin/env python3
"""
Backtesting Crypto Portfolio Strategy with Fear & Greed Index
DEMO VERSION with Synthetic Data

Strategy:
- Monthly rebalancing of TOP-20 cryptocurrencies (excluding stablecoins)
- Investment Capital = Total Capital * (1 - Fear&Greed Index)
- Equal weight: 5% per coin (20 coins total)
- Remaining capital in stablecoins
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
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

        # Top-20 crypto symbols
        self.top_cryptos = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'TRX',
                           'DOT', 'MATIC', 'LTC', 'SHIB', 'AVAX', 'UNI', 'LINK',
                           'XLM', 'ATOM', 'XMR', 'BCH', 'ALGO']

    def generate_synthetic_fear_greed(self) -> pd.DataFrame:
        """
        Generate synthetic Fear & Greed Index data
        """
        print("📊 Generating Fear & Greed Index data...")

        days_count = (datetime.now() - self.start_date).days
        dates = pd.date_range(start=self.start_date, periods=days_count, freq='D')

        # Generate realistic fear & greed cycles
        # Base sine wave for market cycles (roughly 180-day cycles)
        base_cycle = np.sin(np.linspace(0, days_count / 180 * 2 * np.pi, days_count))

        # Add shorter-term volatility
        short_cycle = np.sin(np.linspace(0, days_count / 30 * 2 * np.pi, days_count)) * 0.3

        # Random noise
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, 0.15, days_count)

        # Combine and normalize to 0-1 range
        values = (base_cycle + short_cycle + noise + 1) / 2
        values = np.clip(values, 0.1, 0.9)

        df = pd.DataFrame({
            'date': dates,
            'value': values
        })

        print(f"✅ Generated {len(df)} days of F&G data")
        print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"   Average F&G: {df['value'].mean():.1%}")

        self.fear_greed_data = df
        return df

    def generate_crypto_prices(self) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic price data for cryptocurrencies
        Mimics real crypto market behavior with volatility and trends
        """
        print(f"\n💰 Generating price data for {len(self.top_cryptos)} cryptocurrencies...")

        days_count = (datetime.now() - self.start_date).days
        dates = pd.date_range(start=self.start_date, periods=days_count, freq='D')

        prices = {}
        np.random.seed(42)

        # Base parameters for different cryptos
        crypto_params = {
            'BTC': {'start_price': 7200, 'volatility': 0.03, 'trend': 0.0003},
            'ETH': {'start_price': 130, 'volatility': 0.04, 'trend': 0.0004},
            'BNB': {'start_price': 15, 'volatility': 0.045, 'trend': 0.0005},
            'XRP': {'start_price': 0.18, 'volatility': 0.05, 'trend': 0.0002},
            'ADA': {'start_price': 0.03, 'volatility': 0.05, 'trend': 0.0004},
            'DOGE': {'start_price': 0.002, 'volatility': 0.08, 'trend': 0.0006},
            'SOL': {'start_price': 0.5, 'volatility': 0.06, 'trend': 0.0008},
            'TRX': {'start_price': 0.01, 'volatility': 0.04, 'trend': 0.0002},
            'DOT': {'start_price': 2.8, 'volatility': 0.05, 'trend': 0.0003},
            'MATIC': {'start_price': 0.015, 'volatility': 0.055, 'trend': 0.0005},
            'LTC': {'start_price': 42, 'volatility': 0.04, 'trend': 0.0002},
            'SHIB': {'start_price': 0.000001, 'volatility': 0.1, 'trend': 0.001},
            'AVAX': {'start_price': 3.5, 'volatility': 0.06, 'trend': 0.0006},
            'UNI': {'start_price': 2, 'volatility': 0.05, 'trend': 0.0004},
            'LINK': {'start_price': 2, 'volatility': 0.045, 'trend': 0.0003},
            'XLM': {'start_price': 0.05, 'volatility': 0.04, 'trend': 0.0002},
            'ATOM': {'start_price': 2.2, 'volatility': 0.05, 'trend': 0.0003},
            'XMR': {'start_price': 56, 'volatility': 0.04, 'trend': 0.0002},
            'BCH': {'start_price': 220, 'volatility': 0.04, 'trend': 0.0001},
            'ALGO': {'start_price': 0.18, 'volatility': 0.05, 'trend': 0.0003},
        }

        for symbol in self.top_cryptos:
            params = crypto_params.get(symbol, {
                'start_price': 1.0,
                'volatility': 0.05,
                'trend': 0.0003
            })

            # Generate price series using geometric Brownian motion
            price = params['start_price']
            price_series = [price]

            for _ in range(days_count - 1):
                # Random daily return with trend
                daily_return = np.random.normal(params['trend'], params['volatility'])
                price = price * (1 + daily_return)
                price_series.append(max(price, 0.0001))  # Prevent negative prices

            df = pd.DataFrame({
                'date': dates,
                'price': price_series
            })

            prices[symbol] = df
            print(f"✅ {symbol}: Start ${params['start_price']:.6f} → End ${price_series[-1]:.6f}")

        print(f"\n✅ Generated data for {len(prices)} cryptocurrencies")
        self.price_data = prices
        return prices

    def calculate_monthly_avg_fear_greed(self, month_start: datetime, month_end: datetime) -> float:
        """
        Calculate average Fear & Greed Index for the period
        """
        if self.fear_greed_data is None or len(self.fear_greed_data) == 0:
            return 0.5

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
        print("\n" + "="*70)
        print("🚀 STARTING BACKTEST")
        print("="*70)

        # Generate data
        self.generate_synthetic_fear_greed()
        self.generate_crypto_prices()

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
            print(f"\n{'─'*70}")
            print(f"📅 Rebalance #{i+1}: {rebalance_date.strftime('%Y-%m-%d')}")
            print(f"{'─'*70}")

            # Calculate average F&G for previous month
            if i == 0:
                month_start = rebalance_date
                month_end = rebalance_date + timedelta(days=30)
            else:
                prev_date = rebalance_dates[i-1]
                month_start = prev_date
                month_end = rebalance_date

            avg_fg = self.calculate_monthly_avg_fear_greed(month_start, month_end)

            # Calculate current portfolio value before rebalancing
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

            print(f"💰 Total Capital:      ${total_capital:,.2f}")
            print(f"😨 Avg Fear & Greed:   {avg_fg:.1%}")
            print(f"📈 Investment Capital: ${investment_capital:,.2f}")
            print(f"💵 Stablecoin Balance: ${stablecoin_balance:,.2f}")

            # Calculate allocation per coin (5% each for 20 coins)
            num_coins = len(self.top_cryptos)
            allocation_per_coin = investment_capital / num_coins

            # Rebalance portfolio
            holdings = {}

            for symbol in self.top_cryptos:
                price = self.get_price_at_date(symbol, rebalance_date)
                if price and price > 0:
                    quantity = allocation_per_coin / price
                    holdings[symbol] = quantity

            print(f"🎯 Allocated ${allocation_per_coin:,.2f} to each of {len(holdings)} coins")

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

        # Calculate returns for each rebalance period
        values = [p['total_capital'] for p in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()

        # Calculate max drawdown
        cumulative = pd.Series(values).expanding().max()
        drawdown = (pd.Series(values) - cumulative) / cumulative
        max_drawdown = drawdown.min()

        print("\n" + "="*70)
        print("📊 BACKTEST RESULTS")
        print("="*70)
        print(f"💰 Initial Capital:     ${initial_value:,.2f}")
        print(f"💵 Final Capital:       ${final_value:,.2f}")
        print(f"📈 Total Return:        {total_return:+.2%}")
        print(f"💵 Absolute Profit:     ${final_value - initial_value:+,.2f}")

        if len(returns) > 0:
            print(f"\n📈 Performance Metrics:")
            print(f"   Average Return:      {returns.mean():.2%} per rebalance")
            print(f"   Volatility:          {returns.std():.2%}")
            print(f"   Max Drawdown:        {max_drawdown:.2%}")
            print(f"   Best Period:         {returns.max():.2%}")
            print(f"   Worst Period:        {returns.min():.2%}")

            if returns.std() != 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(12)  # Annualized
                print(f"   Sharpe Ratio:        {sharpe:.2f}")

        print(f"\n📅 Period:")
        print(f"   Start Date:          {self.portfolio_history[0]['date'].strftime('%Y-%m-%d')}")
        print(f"   End Date:            {self.portfolio_history[-1]['date'].strftime('%Y-%m-%d')}")
        print(f"   Duration:            {len(self.portfolio_history)-1} months")

        # Calculate annualized return
        years = len(self.portfolio_history) / 12
        if years > 0:
            annualized_return = (final_value / initial_value) ** (1 / years) - 1
            print(f"   Annualized Return:   {annualized_return:.2%}")

        print("="*70)

    def plot_results(self):
        """
        Plot backtest results
        """
        if len(self.portfolio_history) < 2:
            return

        df = pd.DataFrame(self.portfolio_history)

        fig, axes = plt.subplots(3, 1, figsize=(15, 11))

        # Plot 1: Portfolio Value
        ax1 = axes[0]
        ax1.plot(df['date'], df['total_capital'], linewidth=2.5, color='#2E86AB', label='Total Capital')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.fill_between(df['date'], self.initial_capital, df['total_capital'],
                         where=(df['total_capital'] >= self.initial_capital),
                         alpha=0.2, color='green', label='Profit')
        ax1.fill_between(df['date'], self.initial_capital, df['total_capital'],
                         where=(df['total_capital'] < self.initial_capital),
                         alpha=0.2, color='red', label='Loss')
        ax1.set_title('Portfolio Value Over Time', fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel('Value (USD)', fontsize=13)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.tick_params(axis='both', labelsize=11)

        # Plot 2: Capital Allocation
        ax2 = axes[1]
        ax2.fill_between(df['date'], 0, df['investment_capital'],
                         alpha=0.7, color='#06A77D', label='Crypto Investment')
        ax2.fill_between(df['date'], df['investment_capital'], df['total_capital'],
                         alpha=0.7, color='#F18F01', label='Stablecoins')
        ax2.set_title('Capital Allocation', fontsize=16, fontweight='bold', pad=15)
        ax2.set_ylabel('Value (USD)', fontsize=13)
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.tick_params(axis='both', labelsize=11)

        # Plot 3: Fear & Greed Index
        ax3 = axes[2]
        ax3.plot(df['date'], df['fear_greed_index'] * 100,
                linewidth=2.5, color='#C73E1D', label='Fear & Greed Index')
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax3.fill_between(df['date'], 0, 25, alpha=0.15, color='red', label='Extreme Fear')
        ax3.fill_between(df['date'], 75, 100, alpha=0.15, color='green', label='Extreme Greed')
        ax3.set_title('Fear & Greed Index', fontsize=16, fontweight='bold', pad=15)
        ax3.set_ylabel('Index Value', fontsize=13)
        ax3.set_xlabel('Date', fontsize=13)
        ax3.legend(loc='best', fontsize=11)
        ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='both', labelsize=11)

        plt.tight_layout()
        plt.savefig('/home/user/xxx/backtest_results.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Chart saved: backtest_results.png")

        # Save data to CSV
        df_export = df.copy()
        df_export = df_export.drop('holdings', axis=1)
        df_export.to_csv('/home/user/xxx/backtest_data.csv', index=False)
        print(f"💾 Data saved: backtest_data.csv")


def main():
    """
    Main function to run backtest
    """
    print("="*70)
    print("🔬 CRYPTO PORTFOLIO STRATEGY BACKTEST")
    print("="*70)
    print("\n📝 Strategy Details:")
    print("   - Monthly rebalancing")
    print("   - TOP-20 cryptocurrencies (no stablecoins)")
    print("   - Equal weight: 5% per coin")
    print("   - Investment = Capital × (1 - Fear&Greed)")
    print("   - Remaining in stablecoins")
    print("\n⚠️  NOTE: Using synthetic data for demonstration")
    print("="*70)

    # Initialize backtest
    backtest = CryptoStrategyBacktest(
        initial_capital=1000,
        start_date="2020-01-01"
    )

    # Run backtest
    backtest.run_backtest()

    print("\n✅ Backtest completed!")


if __name__ == "__main__":
    main()
