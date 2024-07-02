import numpy as np
from stable_baselines3 import PPO
from drone_env import DroneEnv

if __name__ == "__main__":
    env = DroneEnv()
    
    model = PPO.load("ppo_drone")
    
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            obs = env.reset()
    
    env.close()
