import gym
from gym import spaces
import numpy as np
import airsim
import time

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        
        # Actions: 0=forward, 1=backward, 2=left, 3=right, 4=up, 5=down
        self.action_space = spaces.Discrete(6)
        
        self.observation_space = spaces.Box(low=np.array([-100, -100, -100, 0, 0, 0, 0, 0, 0]),
                                            high=np.array([100, 100, 100, 100, 100, 100, 100, 100, 100]), dtype=np.float32)
        
        self.starting_position = self.client.getMultirotorState().kinematics_estimated.position
    
    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        
        self.client.moveToPositionAsync(self.starting_position.x_val, 
                                        self.starting_position.y_val, 
                                        self.starting_position.z_val, 5).join()
        return self._get_obs()
    
    def step(self, action):
        if action == 0:
            self.client.moveByVelocityAsync(5, 0, 0, 1).join()
        elif action == 1:
            self.client.moveByVelocityAsync(-5, 0, 0, 1).join()
        elif action == 2:
            self.client.moveByVelocityAsync(0, 5, 0, 1).join()
        elif action == 3:
            self.client.moveByVelocityAsync(0, -5, 0, 1).join()
        elif action == 4:
            self.client.moveByVelocityAsync(0, 0, -5, 1).join()
        elif action == 5:
            self.client.moveByVelocityAsync(0, 0, 5, 1).join()
        
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._is_done(obs)
        
        return obs, reward, done, {}
    
    def _get_obs(self):
        position = self.client.getMultirotorState().kinematics_estimated.position
        distance_to_obstacles = self._get_distance_to_obstacles()
        return np.array([position.x_val, position.y_val, position.z_val] + distance_to_obstacles, dtype=np.float32)
    
    def _get_distance_to_obstacles(self):
        distances = [100, 100, 100, 100, 100, 100]
        return distances
    
    def _compute_reward(self, obs):
        distance_to_start = np.linalg.norm(obs[:3] - np.array([self.starting_position.x_val, 
                                                               self.starting_position.y_val, 
                                                               self.starting_position.z_val]))
        collision_penalty = self._collision_penalty(obs[3:])
        return -distance_to_start + collision_penalty
    
    def _collision_penalty(self, distances):
        penalty = 0
        for distance in distances:
            if distance < 5:
                penalty -= 10
        return penalty
    
    def _is_done(self, obs):
        distance_to_start = np.linalg.norm(obs[:3] - np.array([self.starting_position.x_val, 
                                                               self.starting_position.y_val, 
                                                               self.starting_position.z_val]))
        return distance_to_start < 1.0

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
