
# ppo_agent.py
# PPO Agent implementation using Stable-Baselines3

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import pandas as pd
import os
import torch
from typing import Optional


class TradingCallback(BaseCallback):
    """
    Custom callback for logging trading metrics during training
    """

    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []

    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    # Log episode metrics
                    info = self.locals['infos'][idx]
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])

                        # Log portfolio value if available
                        if 'portfolio_value' in info:
                            self.portfolio_values.append(info['portfolio_value'])

        return True

    def _on_training_end(self) -> None:
        """
        Log final statistics
        """
        if len(self.episode_rewards) > 0:
            print("\n" + "="*50)
            print("TRAINING COMPLETED")
            print("="*50)
            print(f"Total Episodes: {len(self.episode_rewards)}")
            print(f"Mean Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"Mean Episode Length: {np.mean(self.episode_lengths):.2f}")
            if len(self.portfolio_values) > 0:
                print(f"Mean Portfolio Value: ${np.mean(self.portfolio_values):.2f}")


def get_device(device='auto'):
    """
    Get the appropriate device (CPU or GPU) for training
    
    Args:
        device: 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'
    
    Returns:
        torch.device: The device to use
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            print("⚠ GPU not available, using CPU")
    else:
        if device.startswith('cuda') and not torch.cuda.is_available():
            print(f"⚠ {device} requested but CUDA not available, falling back to CPU")
            device = 'cpu'
    
    return torch.device(device)


class PPOTradingAgent:
    """
    PPO Agent wrapper for stock trading
    """

    def __init__(
        self,
        env,
        policy='MlpPolicy',
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device='auto',
        tensorboard_log='./tensorboard_logs/'
    ):
        """
        Initialize PPO agent

        Args:
            env: Trading environment
            policy: Policy network architecture
            learning_rate: Learning rate for optimizer
            n_steps: Number of steps to collect before update
            batch_size: Mini-batch size for updates
            n_epochs: Number of epochs for policy update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient for exploration
            verbose: Verbosity level
            device: Device to use ('auto', 'cuda', 'cpu', or 'cuda:0')
        """
        self.env = env
        
        # Get device
        self.device = get_device(device)
        if verbose:
            print(f"Using device: {self.device}")

        # Wrap environment
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)

        # Initialize PPO model with device
        self.model = PPO(
            policy=policy,
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device=self.device
        )

    def train(
        self,
        total_timesteps=100000,
        callback=None,
        save_path='models/ppo_trading_agent'
    ):
        """
        Train the agent

        Args:
            total_timesteps: Total number of steps to train
            callback: Training callback
            save_path: Path to save model
        """
        if callback is None:
            callback = TradingCallback(verbose=1)

        print("\nStarting training...")
        print(f"Total timesteps: {total_timesteps:,}")
        print("-" * 50)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        self.vec_env.save(f"{save_path}_vecnormalize.pkl")

        print(f"\nModel saved to {save_path}")

    def predict(self, observation, deterministic=True):
        """
        Predict action for given observation

        Args:
            observation: Current state
            deterministic: Whether to use deterministic policy

        Returns:
            action: Predicted action
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def load(self, path, load_vec_normalize=True):
        """
        Load trained model

        Args:
            path: Path to saved model
            load_vec_normalize: Whether to load VecNormalize stats
        """
        self.model = PPO.load(path, env=self.vec_env)

        if load_vec_normalize and os.path.exists(f"{path}_vecnormalize.pkl"):
            self.vec_env = VecNormalize.load(f"{path}_vecnormalize.pkl", self.vec_env)
            # Don't update stats during evaluation
            self.vec_env.training = False
            self.vec_env.norm_reward = False

        print(f"Model loaded from {path}")

    def backtest(self, test_env, n_episodes=1, render=False):
        """
        Backtest the agent on test data

        Args:
            test_env: Test environment
            n_episodes: Number of episodes to run
            render: Whether to render environment

        Returns:
            dict: Backtest results
        """
        # Wrap test environment
        test_vec_env = DummyVecEnv([lambda: test_env])
        test_vec_env = VecNormalize(test_vec_env, training=False, norm_reward=False)

        # Copy normalization stats from training
        if hasattr(self.vec_env, 'obs_rms'):
            test_vec_env.obs_rms = self.vec_env.obs_rms
            test_vec_env.ret_rms = self.vec_env.ret_rms

        results = {
            'episode_rewards': [],
            'portfolio_values': [],
            'final_balances': [],
            'total_returns': [],
            'actions': []
        }

        for episode in range(n_episodes):
            obs = test_vec_env.reset()
            done = False
            episode_reward = 0
            actions_taken = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                actions_taken.append(action[0])
                obs, reward, done, info = test_vec_env.step(action)
                episode_reward += reward[0]

                if render:
                    test_env.render()

            # Get final metrics
            final_value = test_env.portfolio_values[-1]
            initial_value = test_env.initial_balance
            total_return = (final_value - initial_value) / initial_value

            results['episode_rewards'].append(episode_reward)
            results['portfolio_values'].append(test_env.portfolio_values)
            results['final_balances'].append(final_value)
            results['total_returns'].append(total_return)
            results['actions'].append(actions_taken)

            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"Final Portfolio Value: ${final_value:,.2f}")
            print(f"Total Return: {total_return*100:.2f}%")

        return results


# Example usage
if __name__ == "__main__":
    from trading_environment import TradingEnvironment
    import pandas as pd

    # Load data
    df = pd.read_csv('aapl_processed.csv', index_col=0)

    # Split into train/test
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Create environment
    env = TradingEnvironment(
        df=train_df,
        initial_balance=10000,
        transaction_cost=0.001
    )

    # Initialize agent
    agent = PPOTradingAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1
    )

    # Train agent
    agent.train(total_timesteps=100000, save_path='models/ppo_trading_agent')

    # Backtest on test data
    test_env = TradingEnvironment(
        df=test_df,
        initial_balance=10000,
        transaction_cost=0.001
    )

    results = agent.backtest(test_env, n_episodes=1, render=False)
    print("\nBacktest Results:")
    print(f"Mean Return: {np.mean(results['total_returns'])*100:.2f}%")
