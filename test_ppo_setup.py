from finance_rl import environ, data_yf
from finance_rl.ppo_models import ActorCriticMLP
from finance_rl.ppo_validation import validation_run_ppo
import torch

prices = data_yf.load_many_from_dir("yf_data")
env = environ.StocksEnv(prices, bars_count=10, volumes=True, extra_features=True, reward_mode="step_logret")

obs, info = env.reset()
print("obs shape:", obs.shape, "instrument:", info["instrument"])

model = ActorCriticMLP(env.observation_space.shape[0], env.action_space.n)
model.eval()

res = validation_run_ppo(env, model, episodes=3, device="cpu", greedy=True)
print("validation:", res)
