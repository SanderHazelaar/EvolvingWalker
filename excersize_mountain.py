from matplotlib import pyplot as plt
from CTRNN import CTRNN
from scipy.sparse import csr_matrix
import run_cart
import gym
import numpy as np


# added unpacking of genome:
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
            weight_range = 3
            ind = self.network_size * self.network_size
            w = weight_range * (2.0 * (genome[:ind] - 0.5))
            weights = np.reshape(w, [self.network_size, self.network_size])
            biases = weight_range * (2.0 * (genome[ind:ind + self.network_size] - 0.5))
            ind += self.network_size
            taus = 0.9 * genome[ind:ind + self.network_size] + 0.05
            ind += self.network_size
            gains = 2.0 * (genome[ind:ind + self.network_size] - 0.5)

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

# set up a CTRNN agent:
n_neurons = 10;


def evaluate(genome, seed=0, n_episodes=1, graphics=False):
    # create the phenotype from the genotype:
    agent = CTRNN_agent(n_neurons, genome=genome)
    # run the agent:
    reward = run_cart.run_cart_continuous(agent, env=run_cart.CMC_original(), simulation_seed=seed,
                                          n_episodes=n_episodes, graphics=graphics)
    # print('Reward = ' + str(reward))
    return reward