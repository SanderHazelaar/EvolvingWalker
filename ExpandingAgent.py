import numpy as np


class Expanding_Agent():
    # set all neurons to 0 and put observations in input neurons
    def __init__(self, individual):
        neurons_sum = np.zeros(len(individual["neurons"]))


    def act(self, observation, reward, done):  # return the output of the network for the given inputs
        external_inputs = np.asarray([0.0] * self.network_size)
        external_inputs[0:self.n_observations] = observation
        self.cns.euler_step(external_inputs)  # perform euler step through the network
        output = 2.0 * (self.cns.outputs[-self.n_actions:] - 0.5)
        return output