from Agent import Agent
from Game import Game
import numpy as np
from Error_Estimator import Error_Estimator

class Train_Env():
    def __init__(self):
        print('start environment')

    def reset(self):
        self.agent_1 = Agent()
        self.agent_2 = Agent()
        p_1, p_2 = np.random.standard_normal(size=(16, 16)), np.random.standard_normal(
            size=(16, 16))
        g = Game(p_1, p_2)
        self.game = g
        move_1, _ = self.agent_1.pick_move(g)
        move_2, _ = self.agent_2.pick_move(g)
        self.move_1 = move_1
        self.move_2 = move_2
        r_1, r_2 = g.payoffs(move_1, move_2)
        self.agent_1.accept_reward(r_1)
        self.agent_2.accept_reward(r_2)
        j = self.agent_1.estimate_prob_of_move(g, move_2)
        k = self.agent_1.get_cooperation_value()
        return j, k

    def step(self, error_id):
        g = self.game
        error_lvls = np.array([0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.0])
        self.agent_1.distr[error_id] *= 1.1
        self.agent_1.distr = self.agent_1.distr / np.sum(self.agent_1.distr)
        error = np.average(error_lvls, weights=self.agent_1.distr)
        self.agent_1.error = error
        move_1 = self.move_1
        move_2 = self.move_2
        self.agent_1.observe_move(g, move_2)
        self.agent_2.observe_move(g, move_1)

        p_1, p_2 = np.random.standard_normal(size=(16, 16)), np.random.standard_normal(
            size=(16, 16))
        g = Game(p_1, p_2)
        self.game = g
        move_1, _ = self.agent_1.pick_move(g)
        move_2, _ = self.agent_2.pick_move(g)
        self.move_1 = move_1
        self.move_2 = move_2
        r_1, r_2 = g.payoffs(move_1, move_2)
        self.agent_1.accept_reward(r_1)
        self.agent_2.accept_reward(r_2)
        j = self.agent_1.estimate_prob_of_move(g, move_2)
        k = self.agent_1.get_cooperation_value()
        return j, k


t = Train_Env()
j, k = t.reset()
j = np.digitize(j, np.arange(0, 1, 0.1))
k = np.digitize(k, np.arange(-1, 1, 0.2))
e = Error_Estimator(j_bins=np.arange(0, 1, 0.1), k_bins=np.arange(-1, 1, 0.2), err_lvls = [0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.0], alpha=0.81, gamma=0.99)
action = e.act(j, k)
j, k = t.step(action)
print(j, k)

