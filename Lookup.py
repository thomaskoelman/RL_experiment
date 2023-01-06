import numpy as np
from Game import Game
import math

class Lookup:
    def __init__(self, shape=(2,2), r=0.1):
        self.j_bins = np.arange(1., step=0.2)
        x = len(self.j_bins)
        self.tbl = np.zeros((x,))
        for true_belief in np.arange(-1, 1, 0.0001):
            true_attitude = min(true_belief + r, 1)
            #print("belief: ", true_belief)
            for est_belief in np.arange(-1, 1, 0.0001):
                est_attitude = min(est_belief + r, 1)
                true_pivot = np.random.randint(sum(shape))
                p_1, p_2 = np.random.standard_normal(size=shape), np.random.standard_normal(size=shape)
                g = Game(p_1, p_2)
                true_strategy, _ = g.modify(true_attitude, true_belief).lemke_howson(true_pivot)
                true_move = np.random.choice(len(true_strategy),p=true_strategy)



                error = math.sqrt((true_attitude-est_attitude) ** 2 + (true_belief - est_belief) ** 2)
                if error < 0.004:
                    print(error)




    def get_table(self):
        return self.tbl

t = Lookup()
print(t.get_table())

#(j, k , l)

#t(j=0.3) = 0.5

#j_0.0 = {all moves scored by particles between 0.0 and 0.2}
#j_0.2 = {all moves that particles assessed between 0.2 and 0.4}
#j_0.4 = {all moves that particles assessed between 0.4 and 0.6}
#j_0.6 = {all moves that particles assessed between 0.6 and 0.8}
#j_0.8 = {all moves that particles assessed between 0.8 and 1.0}

#t(j=0.3,err=1.0) = 1.0

#(j=0.3, k=1.0, err=1.0) = 0.5
#(j=0.8, k=0.8, err=1.0) = 0.9
#(j=0.1, k=-0.7, err=0.004) = 0.9

#sum(t(j_i, k=1.0, err=1.0)) = 1.0

#t(j=0.0) = 0.0
#t(j=0.2) = 0.0
#t(j=0.4) = 0.3
#t(j=0.6) = 0.1
#t(j=0.8) = 0.6

# 0.0 * 0.0 + 0.2 * 0.0 + 0.4 * 0.3 + 0.6 * 0.1 + 0.8 * 0.6 = 0.6599999999999999



#print(np.arange(-1, 1, 0.001))