from Agent import Agent
from Game import Game
import numpy as np
import matplotlib.pyplot as plt

class Env():
    def __init__(self, nb_plays=100, nb_moves=2):
        n = 10
        f_ab = 0.1
        f_nash = 0.1
        r = 0.1
        self.nb_plays = nb_plays
        self.nb_moves = nb_moves

    def run(self):
        agent_1 = Agent()
        agent_2 = Agent()
        for _ in range(self.nb):
            self.__play(agent_1, agent_2)

    def __play(self, agent_1: Agent, agent_2: Agent):
        p_1, p_2 = np.random.standard_normal(size=(self.nb_moves, self.nb_moves)), np.random.standard_normal(size=(self.nb_moves, self.nb_moves))
        #g = Game(np.array([[-2,-10],[0,-5]]), np.array([[-2,-10],[0,-5]]))
        g = Game(p_1, p_2)
        move_1, _ = agent_1.pick_move(g)
        move_2, _ = agent_2.pick_move(g)
        r_1, r_2 = g.payoffs(move_1, move_2)
        agent_1.accept_reward(r_1)
        agent_2.accept_reward(r_2)
        agent_1.observe_move(g, move_2)
        agent_2.observe_move(g, move_1)

    def __play_with_static_agent(self, agent_1: Agent, agent_2: Agent):
        p_1, p_2 = np.random.standard_normal(size=(self.nb_moves, self.nb_moves)), np.random.standard_normal(
            size=(self.nb_moves, self.nb_moves))
        g = Game(p_1, p_2)
        move_1, _ = agent_1.pick_move(g)
        move_2, _ = agent_2.pick_move(g)
        r_1, r_2 = g.payoffs(move_1, move_2)
        agent_1.accept_reward(r_1)
        agent_2.accept_reward(r_2)
        agent_1.observe_move(g, move_2)


        #print("reward: ", self.agent_1.average_reward())
        #print("cooperation value: ", self.agent_1.get_cooperation_value())

    def plot_avg_reward(self):
        results_1 = []
        results_2 = []
        x_axis = list(range(0, self.nb_plays, 1))
        for i in range(20):
            print(i)
            agent_1 = Agent()
            agent_2 = Agent()
            y_axis_1 = []
            y_axis_2 = []
            for _ in range(self.nb_plays):
                self.__play(agent_1, agent_2)
                y_axis_1.append(agent_1.average_reward())
                y_axis_2.append(agent_2.average_reward())
            results_1.append(y_axis_1)
            results_2.append(y_axis_2)
        results_1 = np.array(results_1)
        results_2 = np.array(results_2)
        results_1 = np.mean(results_1, axis=0)
        results_2 = np.mean(results_2, axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.plot(x_axis, results_1, label = "agent 1")
        plt.plot(x_axis, results_2, label = "agent 2")
        plt.legend()
        plt.title("average payoff of agents")
        plt.xlabel("games played")
        plt.ylabel("average reward")
        plt.show()

    def plot_avg_reward_static(self):
        results_1 = []
        results_2 = []
        x_axis = list(range(0, self.nb_plays, 1))
        for i in range(10):
            print(i)
            agent_1 = Agent()
            agent_2 = Agent()
            y_axis_1 = []
            y_axis_2 = []
            for _ in range(self.nb_plays):
                self.__play_with_static_agent(agent_1, agent_2)
                y_axis_1.append(agent_1.average_reward())
                y_axis_2.append(agent_2.average_reward())
            results_1.append(y_axis_1)
            results_2.append(y_axis_2)
        results_1 = np.array(results_1)
        results_2 = np.array(results_2)
        results_1 = np.mean(results_1, axis=0)
        results_2 = np.mean(results_2, axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.plot(x_axis, results_1, label = "agent 1")
        plt.plot(x_axis, results_2, label = "agent 2 (static)")
        plt.legend()
        plt.title("average payoff of agents")
        plt.xlabel("games played")
        plt.ylabel("average reward")
        plt.show()

    def plot_cooperation(self):
        results_1 = []
        results_2 = []
        x_axis = list(range(0, self.nb_plays, 1))
        for i in range(10):
           print(i)
           agent_1 = Agent()
           agent_2 = Agent()
           y_axis_1 = []
           y_axis_2 = []
           for _ in range(self.nb_plays):
               self.__play(agent_1, agent_2)
               y_axis_1.append(agent_1.get_cooperation_value())
               y_axis_2.append(agent_2.get_cooperation_value())
           results_1.append(y_axis_1)
           results_2.append(y_axis_2)
        results_1 = np.array(results_1)
        results_2 = np.array(results_2)
        results_1 = np.mean(results_1, axis=0)
        results_2 = np.mean(results_2, axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.plot(x_axis, results_1, label="agent 1")
        plt.plot(x_axis, results_2, label="agent 2")
        plt.legend()
        plt.ylim(-1, 1)
        plt.title("cooperation of agents")
        plt.xlabel("games played")
        plt.ylabel("cooperation")
        plt.show()

    def plot_cooperation_static_agent(self):
        results_1 = []
        results_2 = []
        x_axis = list(range(0, self.nb_plays, 1))
        for i in range(10):
           print(i)
           agent_1 = Agent()
           agent_2 = Agent()
           y_axis_1 = []
           y_axis_2 = []
           for _ in range(self.nb_plays):
               self.__play_with_static_agent(agent_1, agent_2)
               y_axis_1.append(agent_1.get_cooperation_value())
               y_axis_2.append(agent_2.get_cooperation_value())
           results_1.append(y_axis_1)
           results_2.append(y_axis_2)
        results_1 = np.array(results_1)
        results_2 = np.array(results_2)
        results_1 = np.mean(results_1, axis=0)
        results_2 = np.mean(results_2, axis=0)

        fig, ax = plt.subplots(figsize=(8, 8))
        plt.plot(x_axis, results_1, label="agent 1")
        plt.plot(x_axis, results_2, label="agent 2 (static)")
        plt.legend()
        plt.ylim(-1, 1)
        plt.title("cooperation of agents")
        plt.xlabel("games played")
        plt.ylabel("cooperation")
        plt.show()

    def plot_predictive_accuracy(self):
        accuracies = []
        for i in range(10):
            print(i)
            predictive_accuracy = []
            agent_1 = Agent()
            agent_2 = Agent()
            for i in range(self.nb_plays):
                p_1, p_2 = np.random.standard_normal(size=(self.nb_moves, self.nb_moves)), np.random.standard_normal(
                    size=(self.nb_moves, self.nb_moves))
                g = Game(p_1, p_2)
                move_2, prob_2 = agent_2.pick_move(g)
                agent_1.observe_move(g, move_2)
                est_prob = agent_1.estimate_prob_of_move(g, move_2)
                predictive_accuracy.append(1 - abs(est_prob - prob_2))
            accuracies.append(predictive_accuracy)
        results = np.mean(np.array(accuracies), axis=0)
        fig, ax = plt.subplots(figsize=(8, 8))
        x_axis = list(range(0, self.nb_plays, 1))
        plt.scatter(x_axis, results, label="predictive accuracy")
        plt.legend()
        plt.ylim(0, 1)
        plt.title("learning performance")
        plt.xlabel("games played")
        plt.show()







Env(nb_plays=1000, nb_moves=16).plot_predictive_accuracy()

