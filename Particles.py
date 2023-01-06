import numpy as np
from Particle import Particle
from Game import Game
from copy import copy

class Particles:
    def __init__(self, nb=10, shape=(2,2)):
        self.__particles = []
        for _ in range(nb):
            self.__particles.append(Particle(pivots=sum(shape)))

    def get_all(self):
        return self.__particles

    def add_particle(self, p: Particle):
        self.__particles.append(p)

    def add_random_particle(self):
        self.__particles.append(Particle())

    def get_average_attitude(self):
        return np.mean([p.att for p in self.__particles])

    def get_average_belief(self):
        return np.mean([p.bel for p in self.__particles])

    def most_frequent_pivot(self):
        pivots = [p.pivot for p in self.__particles]
        count = np.bincount(pivots)
        return np.argmax(count)

    def resample(self, game: Game, move: int):
        # rate how likely particles are for observed move and sample them
        weights = np.array([p.assess_performance(game, move) for p in self.__particles])
        if np.sum(weights) == 0:
            weights = np.array([1/len(weights)] * len(weights))
        else:
            weights = weights / np.sum(weights)
        sample = np.random.choice(self.__particles, size=len(self.__particles), p=weights)
        self.__particles = [copy(p) for p in sample]

    def perturb_all(self, err, f_ab, f_nash):
        for p in self.__particles:
            p.perturb(err, f_ab, f_nash)

    def __str__(self):
        s = ""
        for p in self.__particles:
            s = s + str(p) + "\n"
        return s

# g = Game(np.array([[-2,-10],[0,-5]]), np.array([[-2,-10],[0,-5]]))
# db = Particles()
# print(db)
# db.resample(g, 1)
# db.resample(g, 1)
# db.resample(g, 1)
# print("-------------------------------------------")
# print(db)
# print("--------------------------------------------")
# db.perturb_all(0.5, 0.1, 0.2)
# print(db)
# print(db.most_frequent_pivot())

# g = Game(np.array([[-2,-10],[0,-5]]), np.array([[-2,-10],[0,-5]]))
# db = Particles()
# print(db)
# db.resample(g, 0)
# print("---------")
# db.perturb_all(0.5, 0.1, 1)
# print(db)