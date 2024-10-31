import gym
from gym import spaces
import numpy as np
import pandas as pd

class TrafficEnvironment(gym.Env):
    def __init__(self, data_file):
        super(TrafficEnvironment, self).__init__()
        self.data = pd.read_csv(data_file)
        self.current_step = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        observation = self._get_observation()
        reward = self._calculate_reward(action)
        return observation, reward, done, {}

    def _get_observation(self):
        return self.data.iloc[self.current_step].values

    def _calculate_reward(self):
        current_traffic = self.data.iloc[self.current_step]['vehicle_count']
        return 100 - current_traffic if current_traffic < 10 else -current_traffic