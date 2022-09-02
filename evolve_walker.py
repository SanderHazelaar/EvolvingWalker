import numpy as np
import pickle
import datetime
import time
from matplotlib import pyplot as plt
from mutate import mutate
from ExpandingAgent import ExpandingAgent
from my_walker import run_bipedal_walker
from copy import deepcopy
from graphviz import Digraph



class Expanding_EA():
    add_node_trys = 20
    add_link_trys = 20
    alter_weight_variance = 0.1

    def __init__(self):
        node = 0
        self.n_observations = 24
        self.n_actions = 4
        self.n_individuals = 80
        self.n_generations = 50
        self.n_best = 3
        self.neurons = []
        self.input_nodes = []
        self.output_nodes = []
        self.reward = np.zeros([self.n_individuals, ])
        self.max_fitness = np.zeros([self.n_generations, ])
        self.mean_fitness = np.zeros([self.n_generations, ])
        self.best = []
        self.fitness_best = []
        self.population = [0 for i in range(self.n_individuals)]

        for i in range(self.n_observations):
            self.neurons.append(
                {"Node": node,
                 "Type": "Input",
                 "Bias": 0,
                 "Loc": np.array([i*10/(self.n_observations - 1), 0.0])})
            self.input_nodes.append(node)
            node += 1

        while node < self.n_observations+self.n_actions:
            self.neurons.append(
                {"Node": node,
                 "Type": "Output",
                 "Bias": 0,
                 "Loc": np.array([(node-self.n_observations)*10/(self.n_actions - 1), 10.0])})
            self.output_nodes.append(node)
            node += 1

        for i in range(self.n_individuals):
            self.links = []
            initial_id = 0
            for input in range(self.n_observations):
                for output in range(self.n_actions):
                    self.links.append({"ID": initial_id,
                                       "Weight": np.random.rand() * 2 - 1,
                                       "From": input,
                                       "To": output + self.n_observations,
                                       "Enabled": True})
                    initial_id += 1
            individual = {"neurons": self.neurons,
                          "links": self.links,
                          "n_visible": 0,
                          "input_nodes": self.input_nodes,
                          "output_nodes": self.output_nodes}
            self.population[i] = individual

    def run(self):
        for g in range(self.n_generations):
            seed = np.random.randint(0, 100)
            # evaluate:
            # start = time.time()
            for i in range(self.n_individuals):
                graphics = False
                if g > 60:
                    graphics = True
                    self.visualise(self.population[i])
                self.reward[i] = self.evaluate(self.population[i], graphics=graphics, seed=seed)
            self.mean_fitness[g] = np.mean(self.reward)
            self.max_fitness[g] = np.max(self.reward)
            if g > 10 and self.max_fitness[g] < 0:
                break
            print('Generation {}, mean = {} max = {}'.format(g, self.mean_fitness[g], self.max_fitness[g]))
            # end = time.time()
            # print("Evaluation took:", end-start)
            # select:
            inds = np.argsort(self.reward)
            inds = inds[-self.n_best:]
            if len(self.best) == 0 or self.reward[inds[-1]] > self.fitness_best:
                self.best = self.population[inds[-1]]
                self.fitness_best = self.reward[inds[-1]]
                # self.visualise(self.best)
                # self.evaluate(self.best, max_steps=20*g, graphics=True, seed=seed)
            # vary:
            # start = time.time()
            new_population = [0 for i in range(self.n_individuals)]
            for i in range(self.n_individuals):
                if i < self.n_best:
                    ind = inds[i % self.n_best]
                    new_population[i] = deepcopy(self.population[ind])
                else:
                    ind = inds[i % self.n_best]
                    new_population[i] = deepcopy(self.population[ind])
                    mutate(new_population[i])
                    mutate(new_population[i])
            self.population = new_population
            # end = time.time()
            # print("Variation took:", end - start)
        self.plot_gen_fitness()
        final_reward = self.evaluate(self.best, max_steps=10000, graphics=True)
        print("Best reward:", final_reward)
        self.visualise(self.best)
        self.test_best()
        self.save_best()

    def evaluate(self, individual, max_steps=800, seed=0, graphics=False, original_reward=True):
        # initialise agent:
        agent = ExpandingAgent(individual)
        # run the agent:
        reward = run_bipedal_walker(agent,  simulation_seed=seed, max_steps=max_steps, graphics=graphics)
        # print('Reward = ' + str(reward))
        return reward

    def plot_gen_fitness(self):
        plt.figure()
        plt.plot(range(self.n_generations), self.mean_fitness)
        plt.plot(range(self.n_generations), self.max_fitness)
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.legend(['Mean fitness', 'Max fitness'])
        plt.show()

    def test_best(self, original_reward=True):
        n_tests = 30
        fit = np.zeros([n_tests, ])
        for t in range(n_tests):
            fit[t] = self.evaluate(self.best, seed=100 + t, graphics=False, original_reward=True)

        plt.figure()
        plt.boxplot(fit)
        plt.ylabel('Fitness')
        plt.xticks([1], ['Fitness best individual'])
        plt.show()

    def visualise(self, individual):
        g = Digraph(engine='neato')
        for neuron in individual["neurons"]:
            location = str(neuron["Loc"][0]) + "," + str(neuron["Loc"][1]) + "!"
            g.node(str(neuron["Node"]), label="", shape="diamond", width="0.1", height="0.1", style="filled",
                   color="red", pos=location)

        for link in individual["links"]:
            if link["Enabled"]:
                g.edge(str(link["From"]), str(link["To"]),
                       color="0.6 1 {l}".format(h=link["Weight"]/-3, l=link["Weight"]/-1.5))
        g.render('walker_network.gv')
        g.view()

    def save_best(self):
        filename = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        walker1 = open(filename, "wb")
        pickle.dump(self.best, walker1)
        walker1.close()


experiment = Expanding_EA()
experiment.run()





