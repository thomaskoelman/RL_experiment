import numpy as np
import random

class Error_Estimator:
    def __init__(self, j_bins, k_bins, err_lvls, alpha: float, gamma: float):
        j_size, k_size, err_size = len(j_bins), len(k_bins), len(err_lvls)
        self.nb_of_errors = err_size
        self.Q = np.zeros((j_size, k_size, err_size))
        self.alpha = alpha
        self.gamma = gamma

    def greedy_action(self, j: int, k: int):
        q = self.Q
        action = np.argmax(q[j, k, :])
        return action

    def act(self, j: int, k: int, training: bool = True):
        if training:
            action = random.randrange(0, self.nb_of_errors)
        else:
            action = self.greedy_action(j, k)
        return action

    def learn(self, j: int, k: int, action: int, reward, next_j: int, next_k: int):
        q = self.Q
        q[j, k, action] += self.alpha * (reward + self.gamma * np.max(q[next_j, next_k, :]) - q[j, k, action])