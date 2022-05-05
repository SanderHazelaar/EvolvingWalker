import numpy as np
from graphviz import Digraph
import os

class Expanding_ea_agent(object):
    add_node_trys = 20
    add_link_trys = 20
    alter_weight_variance = 0.1
    def __init__(self, neurons = [], links = []):
        node = 0
        n_observations = 4
        n_actions = 2
        for i in range(n_observations):
            neurons.append(
                {"Node": node, "Type": "Input", "ActResponse": 0, "Loc": np.array([i*10/(n_observations - 1), 0.0])})
            node += 1
        while node < n_observations+n_actions:
            neurons.append(
                {"Node": node, "Type": "Output", "ActResponse": 0, "Loc": np.array([(node-n_observations)*10/(n_actions - 1), 10.0])})
            node += 1

        initial_id = 0
        for i in range(n_observations):
            for j in range(n_actions):
                links.append({"ID": initial_id, "Weight": np.random.rand() * 2 - 2, "From": i, "To": j + n_observations, "Enabled": True})
                initial_id += 1


# Def Mutate add node - need links, neurons
def add_node(add_node_trys = 20, add_link_trys = 20)
    for j in range(add_node_trys):
        idx = np.random.choice(range(len(links)))
        if not links[idx]["Enabled"]:
            continue
        links[idx]["Enabled"] = False
        node1 = links[idx]["From"]
        node2 = links[idx]["To"]
        new_node = len(neurons)
        split_x, split_y = (neurons[node1]["Loc"] + neurons[node2]["Loc"])/2

        neurons.append(
                {"Node": new_node, "Type": "Hidden", "ActResponse": 0, "Loc": np.array([split_x, split_y])})
        # make 2 new links
        links.append({"ID": len(links), "Weight": 1, "From": node1, "To": new_node, "Enabled": True})
        links.append({"ID": len(links), "Weight": links[idx]['Weight'], "From": new_node, "To": node2, "Enabled": True})
        break

# Def Mutate add link
for k in range(add_link_trys):
    neuron1 = np.random.choice(range(n_observations+n_actions, len(neurons)))
    other_neurons = [node for node in range(len(neurons)) if node != neuron1]
    neuron2 = np.random.choice(other_neurons)
    if len([x for x in links if x['From'] == neuron1 and x['To'] == neuron2]) > 0:
        continue
    if len([x for x in links if x['From'] == neuron2 and x['To'] == neuron1]) > 0:
        continue
    if neurons[neuron2]["Loc"][1] >= neurons[neuron1]["Loc"][1]:
        links.append({"ID": len(links), "Weight": np.random.rand() * 2 - 2, "From": neuron1, "To": neuron2, "Enabled": True})
    else:
        links.append({"ID": len(links), "Weight": np.random.rand() * 2 - 2, "From": neuron2, "To": neuron1, "Enabled": True})
    break

# Def Mutate alter weights
altered_link = np.random.choice(range(len(links)))
new_weight = np.random.normal(0, alter_weight_variance) + links[altered_link]["Weight"]
links[altered_link]["Weight"] = new_weight

# Def Mutate
def mutate()


# Def visualise
# Create phenotype from genotype
g = Digraph(engine='neato')
for neuron in neurons:
    location = str(neuron["Loc"][0]) + "," + str(neuron["Loc"][1]) + "!"
    g.node(str(neuron["Node"]), pos=location)

for link in links:
    if link["Enabled"]:
        g.edge(str(link["From"]), str(link["To"]), str(round(link["Weight"], 1)), color="0.6 1 {l}".format(h=link["Weight"]/-3, l=link["Weight"]/-1.5))

g.render('testrender.gv')
g.view()







