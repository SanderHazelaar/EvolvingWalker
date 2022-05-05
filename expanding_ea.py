# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:04:38 2020

Evolve CTRNNs for the mountain car task

@author: guido
"""

from matplotlib import pyplot as plt
from CTRNN import CTRNN
from scipy.sparse import csr_matrix
import cart_pole
import gym
import numpy as np


# added unpacking of genome:
class genome(object):
    """ Continuous Time Recurrent Neural Network agent. """

    n_observations = 4
    n_actions = 2

    d    def __init__(self, n_observations, n_actions, links=[]):
        self.links = links
        self.neurons = {}
        for i in range(n_observations+n_actions):
            self.neurons[i:n_observations] = "input"
            self.neurons[n_observations:] = "output"

    def act(self, observation, reward, done):  # return the output of the network for the given inputs
        external_inputs = np.asarray([0.0] * self.network_size)
        external_inputs[0:self.n_observations] = observation
        self.cns.euler_step(external_inputs)  # perform euler step through the network
        output = 2.0 * (self.cns.outputs[-self.n_actions:] - 0.5)
        return output


n_neurons = 10


def evaluate(genome, seed=0, graphics=False, original_reward=True):
    # create the phenotype from the genotype:
    agent = CTRNN_agent(n_neurons, genome=genome)
    # run the agent:
    if (original_reward):
        reward = cart_pole.run_cart_continuous(agent, simulation_seed=seed, graphics=graphics)
    else:
        reward = cart_pole.run_cart_continuous(agent, simulation_seed=seed, graphics=graphics)
    # print('Reward = ' + str(reward))
    return reward


def test_best(Best, original_reward=True):
    n_tests = 30
    fit = np.zeros([n_tests, ])
    for t in range(n_tests):
        fit[t] = evaluate(Best, seed=100 + t, graphics=False, original_reward=True)

    plt.figure()
    plt.boxplot(fit)
    plt.ylabel('Fitness')
    plt.xticks([1], ['Fitness best individual'])


# Parameters CTRNN:
network_size = 10
genome_size = (network_size + 3) * network_size

# Evolutionary algorithm:
n_individuals = 100
n_generations = 30
p_mut = 0.05
n_best = 3

np.random.seed(7)  # 0-5 do not work
original_reward = False
Population = np.random.rand(n_individuals, genome_size)
Reward = np.zeros([n_individuals, ])
max_fitness = np.zeros([n_generations, ])
mean_fitness = np.zeros([n_generations, ])
Best = []
fitness_best = []
for g in range(n_generations):

    # evaluate:
    for i in range(n_individuals):
        Reward[i] = evaluate(Population[i, :], original_reward=original_reward)
    mean_fitness[g] = np.mean(Reward)
    max_fitness[g] = np.max(Reward)
    print('Generation {}, mean = {} max = {}'.format(g, mean_fitness[g], max_fitness[g]))
    # select:
    inds = np.argsort(Reward)
    inds = inds[-n_best:]
    if (len(Best) == 0 or Reward[-1] > fitness_best):
        Best = Population[inds[-1], :]
        fitness_best = Reward[-1]
    # vary:
    NewPopulation = np.zeros([n_individuals, genome_size])
    for i in range(n_individuals):
        ind = inds[i % n_best]
        NewPopulation[i, :] = Population[ind, :]
        for gene in range(genome_size):
            if (np.random.rand() <= p_mut):
                NewPopulation[i, gene] = np.random.rand()
    Population = NewPopulation

print('Best fitness ' + str(fitness_best))
print('Genome = ')
for gene in range(len(Best)):
    if (gene == 0):
        print('[' + str(Best[gene]) + ', ', end='');
    elif (gene == len(Best) - 1):
        print(str(Best[gene]) + ']');
    else:
        print(str(Best[gene]) + ', ', end='');

plt.figure();
plt.plot(range(n_generations), mean_fitness)
plt.plot(range(n_generations), max_fitness)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend(['Mean fitness', 'Max fitness'])

evaluate(Best, graphics=True)
test_best(Best)

