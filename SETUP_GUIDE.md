# QUICK START SETUP GUIDE

## Step-by-Step Installation and First Run

### 1. System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space
- Internet connection for data download

### 2. Installation (5 minutes)

#### Option A: Using pip (Recommended)
```bash
# Navigate to project directory
cd deep-rl-trading

# Create virtual environment
python -m venv trading_env

# Activate environment
# On Linux/Mac:
source trading_env/bin/activate
# On Windows:
trading_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n trading_env python=3.9
conda activate trading_env

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import gymnasium; import torch; import pandas; print('✓ All packages installed successfully!')"
```

### 4. First Training Run (10-15 minutes)

#### Quick Test (5 minutes)
```bash
# Train on Apple stock with minimal timesteps
python main.py --ticker AAPL --total_timesteps 10000
```

#### Full Training (15 minutes)
```bash
# Full training run with 100k timesteps
python main.py --ticker AAPL --total_timesteps 100000
```

### 5. Expected Output

The system will:
1. ✓ Download AAPL data (2020-2024)
2. ✓ Calculate technical indicators
3. ✓ Split into train/test sets
4. ✓ Train PPO agent
5. ✓ Backtest on test data
6. ✓ Generate performance plot
7. ✓ Save results to CSV

### 6. Files Generated
```
deep-rl-trading/
├── models/
│   └── aapl_ppo_agent.zip          # Trained model
├── aapl_processed.csv               # Processed data
├── aapl_performance.png             # Performance visualization
└── aapl_results.csv                 # Metrics CSV
```

### 7. View Results

**Performance Plot:**
- Open `aapl_performance.png`
- Shows portfolio value, returns, drawdown, trades

**Metrics:**
- Open `aapl_results.csv` in Excel/Google Sheets
- Check Sharpe ratio, total return, max drawdown

**Console Output:**
```
TRADING PERFORMANCE METRICS
============================================================
Total Return %........................        15.23
Sharpe Ratio..........................         1.24
Max Drawdown %.........................        -8.45
Win Rate..............................         0.62
============================================================
```

### 8. Next Steps

#### Experiment with Different Stocks
```bash
python main.py --ticker TSLA
python main.py --ticker MSFT
python main.py --ticker SPY
```

#### Try Continuous Actions
```bash
python main.py --ticker AAPL --action_space continuous
```

#### Longer Training
```bash
python main.py --ticker AAPL --total_timesteps 500000
```

#### Adjust Hyperparameters
```bash
python main.py --ticker AAPL \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --n_epochs 20
```

### 9. Troubleshooting

**Error: "No module named 'gymnasium'"**
```bash
pip install gymnasium==0.29.0
```

**Error: "yfinance download failed"**
- Check internet connection
- Try different date range
- Use different ticker

**Error: "CUDA out of memory"**
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
python main.py --ticker AAPL
```

**Training is very slow**
- Reduce `--total_timesteps 50000`
- Reduce `--n_steps 1024`
- Close other applications

### 10. Understanding the Output

#### Terminal Output Sections:

**[1/5] Collecting Data**
- Downloads OHLCV data from Yahoo Finance
- Adds 15+ technical indicators
- Shows data shape and date range

**[2/5] Splitting Data**
- 80% training, 20% testing by default
- Ensures no look-ahead bias

**[3/5] Creating Environments**
- Sets up Gymnasium trading environment
- Defines state/action spaces

**[4/5] Training Agent**
- PPO algorithm trains on historical data
- Progress bar shows completion
- Saves model checkpoints

**[5/5] Evaluating**
- Tests on unseen data
- Compares to buy-and-hold
- Generates visualizations

### 11. Interpreting Metrics

**Good Performance Indicators:**
- Sharpe Ratio > 1.0 (excellent > 2.0)
- Positive total return
- Max drawdown < 20%
- Win rate > 50%
- Outperforms buy-and-hold

**Warning Signs:**
- Sharpe Ratio < 0.5
- Max drawdown > 30%
- Win rate < 40%
- Significantly underperforms buy-and-hold
- May indicate overfitting or poor hyperparameters

### 12. Production Considerations

Before live trading:
1. ✓ Paper trade for at least 3 months
2. ✓ Test across multiple market regimes (bull, bear, sideways)
3. ✓ Implement stop-loss mechanisms
4. ✓ Set position size limits
5. ✓ Monitor daily performance
6. ✓ Regular retraining (weekly/monthly)

### 13. Getting Help

**Check logs:**
```bash
# Training logs
tensorboard --logdir tensorboard_logs/
```

**Test individual components:**
```python
# Test data collection
from data_collection import DataCollector
collector = DataCollector('AAPL')
data = collector.download_data()
print(data.head())

# Test environment
from trading_environment import TradingEnvironment
env = TradingEnvironment(df=data)
env.reset()
print("Environment OK")
```

### 14. Common Questions

**Q: How long should I train?**
A: Start with 100k steps. Increase to 200k-500k for better results.

**Q: Which action space is better?**
A: Discrete is simpler and more stable. Continuous allows finer control.

**Q: Can I trade multiple stocks?**
A: Current version is single-stock. Extend environment for portfolios.

**Q: Is this profitable?**
A: No guarantees. This is educational. Markets are unpredictable.

**Q: How often should I retrain?**
A: Weekly or monthly with recent data to adapt to market changes.

### 15. Resources

- Documentation: README.md
- Code examples: main.py
- Research papers: See README references
- Community: GitHub Discussions

---

**You're all set! Start training your first agent:**
```bash
python main.py --ticker AAPL --total_timesteps 100000
```

Good luck! 🚀📈
