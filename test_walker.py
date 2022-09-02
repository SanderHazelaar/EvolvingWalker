import pickle
from ExpandingAgent import ExpandingAgent
from my_walker import run_bipedal_walker
from graphviz import Digraph


def evaluate(individual, max_steps=1000, seed=0, graphics=False, original_reward=True):
    # initialise agent:
    agent = ExpandingAgent(individual)
    # run the agent:
    reward = run_bipedal_walker(agent, simulation_seed=seed, max_steps=max_steps, graphics=graphics)
    # print('Reward = ' + str(reward))

    return reward


def visualise(individual):
    g = Digraph(engine='neato')
    for neuron in individual["neurons"]:
        location = str(neuron["Loc"][0]) + "," + str(neuron["Loc"][1]) + "!"
        g.node(str(neuron["Node"]), label="", shape="diamond", width="0.1", height="0.1", style="filled", color="red", pos=location)

    for link in individual["links"]:
        if link["Enabled"]:
            g.edge(str(link["From"]), str(link["To"]),
                   color="0.6 1 {l}".format(h=link["Weight"]/-3, l=link["Weight"]/-1.5))

    g.render('testrender.gv')
    g.view()


with (open("walker1.pickle", "rb")) as openfile:
    walker = pickle.load(openfile)
reward = evaluate(walker, max_steps=1600, graphics=True)
visualise(walker)
print("Reward:", reward)
