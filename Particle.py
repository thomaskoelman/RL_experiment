import numpy as np
from Game import Game

class Particle:
    def __init__(self, random=True, att = 0.0, bel = 0.0, pivot=0, pivots=4):
        self.pivots = pivots
        if random:
            self.att = np.random.standard_normal()
            self.att = np.random.standard_normal()
            self.bel = np.random.standard_normal()
            self.pivot = np.random.randint(pivots)
        else:
            self.att = att
            self.bel = bel
            self.pivot = pivot

    def assess_performance(self, game: Game, move: int):
        # rate how well a move performs in this hypothesis when the matching game is given
        g = game.modify(self.bel, self.att)
        _, strategy = g.lemke_howson(self.pivot)
        return strategy[move]

    def perturb(self, err, f_ab, f_nash):
        self.att = np.random.normal(self.att, err * f_ab)
        self.bel = np.random.normal(self.bel, err * f_ab)
        if np.random.random() < err * f_nash:
            self.pivot = np.random.randint(self.pivots)

    def __str__(self):
        return "Particle(" + str(self.att) + ", " + str(self.bel) + ", " + str(self.pivot) + ")"

# g = Game(np.array([[-2,-10],[0,-5]]), np.array([[-2,-10],[0,-5]]))
# p = Particle()
# print(p)
# print(p.assess_performance(g, 0))

