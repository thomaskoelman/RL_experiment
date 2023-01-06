from Error_Estimator import Error_Estimator as Estimator
import numpy as np
from Train_Env import Train_Env


def run_episode(self, env, agent: Estimator, training: bool, gamma):
    done = False
    j, k = env.reset()
    cum_reward = 0
    while not done:
        action = agent.act(j, k)
        new_j, new_k, reward, done = env.step(action)
        if training:
            agent.learn(j, k, action, reward, new_j, new_k)
        j = new_j
        k = new_k
        cum_reward += reward
    raise NotImplementedError

def train(self, env, gamma: float):
    error_lvls = [0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.0]
    j_bins = np.arange(0, 1, 0.1)
    k_bins = np.arange(-1, 1, 0.2)
    agent = Estimator(j_bins, k_bins, error_lvls, 0.81, 0.99)
    for episode in range(1000):
        self.run_episode(env, agent, True, gamma)


l = np.array([1/10] * 10)
def reweight(l: np.ndarray):
    rand = random()
    l[rand] *= 1.1
    return norm(l)
def random() : return np.random.randint(10)
def norm(l: np.ndarray): return l/sum(l)
