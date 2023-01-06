import numpy as np
import quantecon.game_theory as gt

class Game:
    def __init__(self, payoffs_1: np.ndarray, payoffs_2: np.ndarray):
        self.__payoffs_1 = payoffs_1
        self.__payoffs_2 = payoffs_2


    def modify(self, att, bel):
        newAgent = self.__payoffs_1 + att * self.__payoffs_2.T
        newOpp = self.__payoffs_2 + bel * self.__payoffs_1.T
        return Game(newAgent, newOpp)

    def lemke_howson(self, label=0):
        player_1 = gt.Player(self.__payoffs_1)
        player_2 = gt.Player(self.__payoffs_2)
        game = gt.NormalFormGame((player_1, player_2))
        return gt.lemke_howson(game, init_pivot=label)

    def payoffs(self, action_1, action_2):
            return self.__payoffs_1[action_1, action_2], self.__payoffs_2[action_2, action_1]

    def __repr__(self):
        return "Game(" + str(self.__payoffs_1.tolist()) + ", " + str(self.__payoffs_2.tolist()) + ")"

    def __str__(self):
        return "payoffs player 1: " + str(self.__payoffs_1.tolist()) + "\npayoffs player 2: " +str(self.__payoffs_2.tolist())

g = Game(np.array([[-2,-10],[0,-5]]), np.array([[-2,-10],[0,-5]]))

