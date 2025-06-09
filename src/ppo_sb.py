import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Environment setup
env_id = "Walker2d-v5"
n_envs = 1

# Create vectorized environment with normalization
env = make_vec_env(env_id, n_envs=n_envs)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Hyperparameters
hyperparams = {
    "policy": 'MlpPolicy',
    "batch_size": 32,
    "n_steps": 512,
    "gamma": 0.99,
    "learning_rate": 5.05041e-05,
    "ent_coef": 0.000585045,
    "clip_range": 0.1,
    "n_epochs": 20,
    "gae_lambda": 0.95,
    "max_grad_norm": 1,
    "vf_coef": 0.871923,
    "verbose": 1,
    "device": "auto"
}

# Create model
model = PPO(
    env=env,
    **hyperparams
)

# Training
total_timesteps = int(1e6)
model.learn(total_timesteps=total_timesteps)

# Save model and normalization stats
model.save("ppo_walker2d")
env.save("vec_normalize.pkl")
print("Training completed. Model and normalization saved.")

# ===== Evaluation =====
print("\nStarting evaluation...")

# Create new environment for evaluation
eval_env = make_vec_env(env_id, n_envs=1)

# Load normalization stats into evaluation environment
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

# Important: Disable reward normalization and training mode for evaluation
eval_env.training = False
eval_env.norm_reward = False

# Load the trained model
model = PPO.load("ppo_walker2d", env=eval_env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True
)

print(f"\nEvaluation results after {total_timesteps} timesteps:")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print(f"Equivalent to approximately {mean_reward/10:.1f} meters walked")

# Render one episode
print("\nRendering one episode...")
obs = eval_env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = eval_env.step(action)
    eval_env.render()
    if done:
        break

eval_env.close()