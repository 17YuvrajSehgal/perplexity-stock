
# evaluation.py
# Evaluation and visualization tools for trading agents

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


class TradingEvaluator:
    """
    Evaluate and visualize trading agent performance
    """

    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance

    def calculate_metrics(self, portfolio_values: List[float], trades: List[dict] = None) -> Dict:
        """
        Calculate comprehensive trading metrics

        Args:
            portfolio_values: List of portfolio values over time
            trades: List of trade dictionaries

        Returns:
            dict: Dictionary of metrics
        """
        portfolio_values = np.array(portfolio_values, dtype=np.float64)

        # Handle edge cases: need at least 2 points to compute returns
        if portfolio_values.size < 2:
            returns = np.array([], dtype=np.float64)
            total_return = 0.0
            volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0
            max_drawdown = 0.0
        else:
            # Returns
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = (np.mean(returns) * 252) / (volatility + 1e-8)

            # Maximum drawdown
            cumulative_returns = portfolio_values / portfolio_values[0]
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)

            # Sortino ratio (downside risk)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
            sortino_ratio = (np.mean(returns) * 252) / (downside_std * np.sqrt(252))

            # Calmar ratio (return / max drawdown)
            calmar_ratio = (total_return * 252) / (abs(max_drawdown) + 1e-8)

        # Win rate (if trades provided)
        win_rate = None
        avg_win = None
        avg_loss = None
        profit_factor = None

        if trades:
            profitable_trades = []
            losing_trades = []

            for trade in trades:
                if trade['type'] == 'sell':
                    # Find corresponding buy
                    buy_trades = [t for t in trades if t['type'] == 'buy' and t['step'] < trade['step']]
                    if buy_trades:
                        last_buy = buy_trades[-1]
                        profit = trade['revenue'] - last_buy['cost']
                        if profit > 0:
                            profitable_trades.append(profit)
                        else:
                            losing_trades.append(abs(profit))

            total_trades = len(profitable_trades) + len(losing_trades)
            if total_trades > 0:
                win_rate = len(profitable_trades) / total_trades
                avg_win = np.mean(profitable_trades) if profitable_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0

                total_profit = sum(profitable_trades)
                total_loss = sum(losing_trades)
                profit_factor = total_profit / (total_loss + 1e-8)

        annualized_return_pct = total_return * 252 / len(returns) * 100 if len(returns) > 0 else 0.0

        metrics = {
            'Total Return': total_return,
            'Total Return %': total_return * 100,
            'Annualized Return %': annualized_return_pct,
            'Volatility (Annual)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'Max Drawdown %': max_drawdown * 100,
            'Final Portfolio Value': portfolio_values[-1] if len(portfolio_values) > 0 else 0.0,
            'Number of Trades': len(trades) if trades else 0,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor
        }

        return metrics

    def print_metrics(self, metrics: Dict):
        """
        Print metrics in formatted table

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("TRADING PERFORMANCE METRICS")
        print("="*60)

        for key, value in metrics.items():
            if value is None:
                continue

            if isinstance(value, float):
                if 'Return' in key or 'Drawdown' in key:
                    print(f"{key:.<35} {value:>15.2f}")
                else:
                    print(f"{key:.<35} {value:>15.4f}")
            else:
                print(f"{key:.<35} {value:>15}")

        print("="*60)

    def compare_with_baseline(self, agent_values: List[float], prices: np.array):
        """
        Compare agent performance with buy-and-hold baseline

        Args:
            agent_values: Portfolio values from agent
            prices: Stock prices over same period

        Returns:
            dict: Comparison metrics
        """
        # Calculate buy-and-hold returns
        shares_bought = self.initial_balance / prices[0]
        baseline_values = shares_bought * prices

        agent_return = (agent_values[-1] - agent_values[0]) / agent_values[0]
        baseline_return = (baseline_values[-1] - baseline_values[0]) / baseline_values[0]

        comparison = {
            'Agent Final Value': agent_values[-1],
            'Agent Total Return %': agent_return * 100,
            'Baseline Final Value': baseline_values[-1],
            'Baseline Total Return %': baseline_return * 100,
            'Outperformance %': (agent_return - baseline_return) * 100
        }

        return comparison, baseline_values

    def plot_performance(
        self,
        portfolio_values: List[float],
        baseline_values: List[float] = None,
        prices: np.array = None,
        trades: List[dict] = None,
        save_path: str = None
    ):
        """
        Create comprehensive performance visualization

        Args:
            portfolio_values: Agent portfolio values
            baseline_values: Baseline (buy-and-hold) values
            prices: Stock prices
            trades: List of trades
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(15, 10))

        # 1. Portfolio Value Over Time
        ax1 = plt.subplot(3, 2, 1)
        if len(portfolio_values) > 0:
            ax1.plot(portfolio_values, label='Agent', linewidth=2)
        if baseline_values is not None and len(baseline_values) > 0:
            ax1.plot(baseline_values, label='Buy & Hold', linewidth=2, alpha=0.7)
        ax1.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Returns Distribution
        ax2 = plt.subplot(3, 2, 2)
        returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else np.array([])
        if len(returns) > 0:
            ax2.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Returns')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative Returns
        ax3 = plt.subplot(3, 2, 3)
        if len(portfolio_values) > 0:
            cumulative_returns = (np.array(portfolio_values) / portfolio_values[0] - 1) * 100
            ax3.plot(cumulative_returns, linewidth=2, color='green')
            ax3.fill_between(range(len(cumulative_returns)), cumulative_returns, alpha=0.3, color='green')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_title('Cumulative Returns (%)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Returns (%)')
        ax3.grid(True, alpha=0.3)

        # 4. Drawdown
        ax4 = plt.subplot(3, 2, 4)
        if len(portfolio_values) > 0:
            cumulative = np.array(portfolio_values) / portfolio_values[0]
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max * 100
            ax4.fill_between(range(len(drawdown)), drawdown, alpha=0.5, color='red')
        ax4.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)

        # 5. Stock Price with Trade Markers
        if prices is not None and trades is not None:
            ax5 = plt.subplot(3, 2, 5)
            ax5.plot(prices, linewidth=1.5, color='blue', alpha=0.6)

            # Mark buy/sell points
            for trade in trades:
                if trade['type'] == 'buy':
                    ax5.scatter(trade['step'], trade['price'], color='green', s=100, 
                               marker='^', alpha=0.7, label='Buy' if 'Buy' not in ax5.get_legend_handles_labels()[1] else '')
                elif trade['type'] == 'sell':
                    ax5.scatter(trade['step'], trade['price'], color='red', s=100, 
                               marker='v', alpha=0.7, label='Sell' if 'Sell' not in ax5.get_legend_handles_labels()[1] else '')

            ax5.set_title('Stock Price with Trades', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Time Steps')
            ax5.set_ylabel('Price ($)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. Rolling Sharpe Ratio
        ax6 = plt.subplot(3, 2, 6)
        window = 30
        rolling_sharpe = []
        if len(returns) >= window:
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                sharpe = (np.mean(window_returns) * 252) / (np.std(window_returns) * np.sqrt(252) + 1e-8)
                rolling_sharpe.append(sharpe)
            ax6.plot(range(window, len(returns)), rolling_sharpe, linewidth=2, color='purple')
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax6.set_title(f'Rolling Sharpe Ratio (Window={window})', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Sharpe Ratio')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")

        plt.show()

    def export_results(self, metrics: Dict, comparison: Dict, filename='results.csv'):
        """
        Export results to CSV

        Args:
            metrics: Performance metrics
            comparison: Baseline comparison
            filename: Output filename
        """
        results = {**metrics, **comparison}
        df = pd.DataFrame([results])
        df.to_csv(filename, index=False)
        print(f"\nResults exported to {filename}")


# Example usage
if __name__ == "__main__":
    # Simulate some data
    np.random.seed(42)
    n_steps = 252
    returns = np.random.normal(0.001, 0.02, n_steps)
    portfolio_values = [10000]

    for r in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))

    # Create evaluator
    evaluator = TradingEvaluator(initial_balance=10000)

    # Calculate metrics
    metrics = evaluator.calculate_metrics(portfolio_values)
    evaluator.print_metrics(metrics)

    # Create baseline
    prices = np.cumsum(np.random.normal(0.001, 0.02, len(portfolio_values))) + 100
    comparison, baseline_values = evaluator.compare_with_baseline(portfolio_values, prices)

    print("\nComparison with Baseline:")
    for key, value in comparison.items():
        print(f"{key}: {value:.2f}")

    # Plot
    evaluator.plot_performance(portfolio_values, baseline_values, prices)
