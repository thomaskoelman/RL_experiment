import numpy as np
import math
from Particles import Particles
from Game import Game

class Agent:
    def __init__(self, reciprocation = 0.1, f_ab = 0.1, f_nash = 0.1):
        self.particles = Particles(nb=100, shape=(8,8))
        self.reciprocation = reciprocation
        self.f_ab = f_ab
        self.f_nash = f_nash
        self.__rewards = []
        self.error = 1
        self.distr = np.array([1/12] * 12)

    def get_cooperation_value(self):
        # assess how cooperative your opponent is
        belief = self.particles.get_average_attitude()
        opp_belief = self.particles.get_average_belief()
        return (belief + opp_belief) / (math.sqrt(belief ** 2 + 1) * math.sqrt(opp_belief ** 2 + 1))

    def pick_move(self, game: Game):
        belief = np.clip(self.particles.get_average_attitude(), -1, 1)
        #print("belief: ",belief)
        pivot = self.particles.most_frequent_pivot()
        attitude = np.clip(belief + self.reciprocation, -1, 1)
        g = game.modify(attitude, belief)
        strategy, _ = g.lemke_howson(pivot)
        move = np.random.choice(len(strategy), p=strategy)
        return move, strategy[move]

    def estimate_prob_of_move(self, game: Game, move: int):
        belief = np.clip(self.particles.get_average_attitude(), -1, 1)
        opp_bel = np.clip(self.particles.get_average_belief(), -1 , 1)
        pivot = self.particles.most_frequent_pivot()
        g = game.modify(opp_bel, belief)
        _, strategy = g.lemke_howson(pivot)
        return strategy[move]

    def observe_move(self, game: Game, move: int):
        err = self.error
        self.particles.resample(game, move)
        self.particles.perturb_all(err, self.f_ab, self.f_nash)

    def accept_reward(self, reward: float):
        self.__rewards.append(reward)

    def average_reward(self):
        return np.average(self.__rewards)

# g = Game(np.array([[-2,-10],[0,-5]]), np.array([[-2,-10],[0,-5]]))
# a = Agent()
# for i in range(100):
#     print(a.pick_move(g))