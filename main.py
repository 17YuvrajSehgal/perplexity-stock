
# main.py
# Main script to train and evaluate trading agent

import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime

from data_collection import DataCollector
from trading_environment import TradingEnvironment
from ppo_agent import PPOTradingAgent
from evaluation import TradingEvaluator


def main(args):
    """
    Main training and evaluation pipeline
    """
    print("="*60)
    print("DEEP RL TRADING SYSTEM")
    print("="*60)
    
    # Create unified output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.ticker.lower()}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    models_dir = os.path.join(output_dir, "models")
    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    logs_dir = os.path.join(output_dir, "logs")
    results_dir = os.path.join(output_dir, "results")
    
    for dir_path in [models_dir, data_dir, plots_dir, logs_dir, results_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print("-"*60)

    # 1. Data Collection
    print("\n[1/5] Collecting Data...")
    print("-"*60)

    if args.data_file and os.path.exists(args.data_file):
        print(f"Loading data from {args.data_file}")
        df = pd.read_csv(args.data_file, index_col=0)
    else:
        print(f"Downloading data for {args.ticker}")
        collector = DataCollector(
            ticker_symbol=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date
        )
        collector.download_data(interval=args.interval)
        collector.add_technical_indicators()
        df = collector.prepare_data(normalize=True)

        # Save processed data to output directory
        data_filename = os.path.join(data_dir, f"{args.ticker.lower()}_processed.csv")
        collector.save_data(data_filename)
        print(f"Processed data saved to {data_filename}")

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # 2. Train/Test Split
    print("\n[2/5] Splitting Data...")
    print("-"*60)

    train_size = int(len(df) * args.train_split)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # 3. Create Environments
    print("\n[3/5] Creating Environments...")
    print("-"*60)

    train_env = TradingEnvironment(
        df=train_df,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        window_size=args.window_size,
        action_space_type=args.action_space
    )

    test_env = TradingEnvironment(
        df=test_df,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        window_size=args.window_size,
        action_space_type=args.action_space
    )

    print(f"Action space: {train_env.action_space}")
    print(f"Observation space: {train_env.observation_space}")

    # 4. Train Agent
    print("\n[4/5] Training Agent...")
    print("-"*60)
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU")
    except ImportError:
        print("PyTorch not installed, using default device")

    agent = PPOTradingAgent(
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        verbose=1,
        device=args.device,
        tensorboard_log=logs_dir
    )

    model_path = os.path.join(models_dir, f"{args.ticker.lower()}_ppo_agent")
    agent.train(
        total_timesteps=args.total_timesteps,
        save_path=model_path
    )

    # 5. Evaluate on Test Set
    print("\n[5/5] Evaluating on Test Set...")
    print("-"*60)

    results = agent.backtest(test_env, n_episodes=1, render=False)

    # Calculate metrics
    evaluator = TradingEvaluator(initial_balance=args.initial_balance)
    portfolio_values = results['portfolio_values'][0]
    trades = test_env.trades

    metrics = evaluator.calculate_metrics(portfolio_values, trades)
    evaluator.print_metrics(metrics)

    # Compare with baseline
    test_prices = test_df['Close'].values
    comparison, baseline_values = evaluator.compare_with_baseline(portfolio_values, test_prices)

    print("\nComparison with Buy-and-Hold:")
    print("-"*60)
    for key, value in comparison.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:>15.2f}")
        else:
            print(f"{key:.<40} {value:>15}")

    # Visualize results
    plot_filename = os.path.join(plots_dir, f"{args.ticker.lower()}_performance.png")
    evaluator.plot_performance(
        portfolio_values=portfolio_values,
        baseline_values=baseline_values,
        prices=test_prices,
        trades=trades,
        save_path=plot_filename
    )

    # Export results
    results_filename = os.path.join(results_dir, f"{args.ticker.lower()}_results.csv")
    evaluator.export_results(metrics, comparison, results_filename)
    
    # Save training configuration
    config_filename = os.path.join(output_dir, "config.txt")
    with open(config_filename, 'w') as f:
        f.write("Training Configuration\n")
        f.write("="*60 + "\n")
        f.write(f"Ticker: {args.ticker}\n")
        f.write(f"Start Date: {args.start_date}\n")
        f.write(f"End Date: {args.end_date}\n")
        f.write(f"Total Timesteps: {args.total_timesteps}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"N Steps: {args.n_steps}\n")
        f.write(f"N Epochs: {args.n_epochs}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Action Space: {args.action_space}\n")
        f.write(f"Initial Balance: {args.initial_balance}\n")
        f.write(f"Transaction Cost: {args.transaction_cost}\n")
        f.write(f"Train Split: {args.train_split}\n")

    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"All outputs saved to: {output_dir}")
    print(f"  - Models: {models_dir}")
    print(f"  - Data: {data_dir}")
    print(f"  - Plots: {plots_dir}")
    print(f"  - Results: {results_dir}")
    print(f"  - Logs: {logs_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep RL Trading System')

    # Data arguments
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (1d, 1h, etc.)')
    parser.add_argument('--data_file', type=str, default=None, help='Path to existing processed data file')

    # Environment arguments
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='Initial cash balance')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost (0.001 = 0.1%)')
    parser.add_argument('--window_size', type=int, default=20, help='Observation window size')
    parser.add_argument('--action_space', type=str, default='discrete', choices=['discrete', 'continuous'],
                        help='Action space type')

    # Training arguments
    parser.add_argument('--train_split', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=2048, help='Steps per update')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training (auto, cuda, or cpu)')
    parser.add_argument('--output_dir', type=str, default='outputs', 
                        help='Base directory for all outputs (creates timestamped subdirectory)')

    args = parser.parse_args()

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run main pipeline
    main(args)
