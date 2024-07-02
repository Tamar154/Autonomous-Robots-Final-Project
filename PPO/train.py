import numpy as np
import gym
from stable_baselines3 import PPO
from drone_env import DroneEnv

if __name__ == "__main__":
    env = DroneEnv()
    
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
    
    model.save("ppo_drone")

    env.close()
