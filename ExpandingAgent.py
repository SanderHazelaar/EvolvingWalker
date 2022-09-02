import numpy as np

class ExpandingAgent():
    # set all neurons to 0 and put observations in input neurons
    def __init__(self, individual):
        self.neurons = individual["neurons"]
        self.links = individual["links"]
        self.input_nodes = individual["input_nodes"]
        self.output_nodes = individual["output_nodes"]
        self.neuron_values = np.zeros(len(self.neurons))

    def update(self, inputs):
        # Order neurons by y coordinate
        sorted_neurons = sorted(self.neurons, key=lambda d: d["Loc"][1])
        # Step through the network a neuron at a time
        for neuron in sorted_neurons:
            node = neuron["Node"]
            incoming_links = [x for x in self.links if x["To"] == node and x["Enabled"]]
            # Sum all inputs to the neuron by looping through the links
            input_sum = 0
            for link in incoming_links:
                input_sum = input_sum + self.neuron_values[link["From"]]*link["Weight"]
            if neuron["Type"] == 'Input':
                self.neuron_values[node] = input_sum + inputs[node]
            else:
                activation_input = input_sum + neuron["Bias"]
                self.neuron_values[node] = max(-1.0, min(1.0, activation_input))  # clamped activation

    def act(self, observation, reward, done):  # return the output of the network for the given inputs
        n_observations = len(self.input_nodes)
        external_inputs = np.asarray([0.0] * n_observations)
        external_inputs[0:n_observations] = observation
        self.update(external_inputs)  # perform step through the network
        output = self.neuron_values[self.output_nodes]
        #print('action:', output)
        return output

