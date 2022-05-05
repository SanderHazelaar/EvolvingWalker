# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:18:05 2020

User can pass an agent to this script, so that it is called every time.

@author: guido
"""

import gym
import math
import numpy as np
import gym.envs.classic_control as cc
from CTRNN import CTRNN
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import time
from IPython import display
import os
from gym.utils import seeding


class random_agent(object):
    """Random agent"""

    def act(self, observation, reward, done):
        return [2.0 * np.random.rand() - 1.0]


class CTRNN_agent(object):
    """ Continuous Time Recurrent Neural Network agent. """

    n_observations = 2;
    n_actions = 1;

    def __init__(self, network_size, genome=[], weights=[], taus=[], gains=[], biases=[]):

        self.network_size = network_size;
        if (self.network_size < self.n_observations + self.n_actions):
            self.network_size = self.n_observations + self.n_actions;
        self.cns = CTRNN(self.network_size, step_size=0.1)

        if (len(genome) == self.network_size * self.network_size + 3 * self.network_size):
            # Get the network parameters from the genome:
            ind = self.network_size * self.network_size
            w = genome[:ind]
            weights = np.reshape(w, [self.network_size, self.network_size])
            biases = genome[ind:ind + self.network_size]
            ind += self.network_size
            taus = genome[ind:ind + self.network_size]
            ind += self.network_size
            gains = genome[ind:ind + self.network_size]

        if (len(weights) > 0):
            # weights must be a matrix size: network_size x network_size
            self.cns.weights = csr_matrix(weights)
        if (len(biases) > 0):
            self.cns.biases = biases
        if (len(taus) > 0):
            self.cns.taus = taus
        if (len(gains) > 0):
            self.gains = gains

    def act(self, observation, reward, done):
        external_inputs = np.asarray([0.0] * self.network_size)
        external_inputs[0:self.n_observations] = observation
        self.cns.euler_step(external_inputs)
        output = 2.0 * (self.cns.outputs[-self.n_actions:] - 0.5)
        return output



def run_cart_continuous(agent, simulation_seed=0, n_episodes=1, env=cc.CartPoleEnv(), max_steps=1000,
                        graphics=False):
    """ Runs the continous cart problem, with the agent mapping observations to actions
        - agent: should implement a method act(observation, reward, done)
        - simulation_seed: used to set the random seed for simulation
        - n_episodes: how many times the task is run for evaluation
        - env: the environment to be used. Standard is the standard continuous mountain car
        - graphics: If True, render() will be called.
    """

    # gym.make('MountainCarContinuous-v0') # CMC() # cc.Continuous_MountainCarEnv()
    env.seed(simulation_seed)

    reward = 0
    cumulative_reward = 0
    done = False
    step = 0

    for i in range(n_episodes):
        ob = env.reset()
        while True:
            action = np.argmax(agent.act(ob, reward, done))  # perform action by agent
            ob, reward, done, _ = env.step(action)  # do the action and retrieve next observation and reward
            cumulative_reward += reward
            step += 1
            if (step >= max_steps):
                done = True
            if (graphics):
                env.render()
            if done:
                break

    env.close()

    return cumulative_reward;


if __name__ == '__main__':
    n_neurons = 10;
    weights = np.zeros([n_neurons, n_neurons])
    taus = np.asarray([0.1] * n_neurons)
    gains = np.ones([n_neurons, ])
    biases = np.zeros([n_neurons, ])
    agent = CTRNN_agent(n_neurons, weights=weights, taus=taus, gains=gains, biases=biases)
    reward = run_cart_continuous(agent, simulation_seed=0, env=CMC_original(), graphics=True)