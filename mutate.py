import numpy as np


def mutate(neural_net):
    n_nodes = len(neural_net["neurons"])
    n_hidden = n_nodes - len(neural_net["input_nodes"]) - len(neural_net["output_nodes"])
    if n_hidden < 35:
        add_node_chance = 0.4 - n_hidden*0.009
        add_link_chance = 0.0 + n_hidden*0.009
    else:
        add_node_chance = 0.1
        add_link_chance = 0.3
    alter_weights_chance = 0.3
    remove_link_chance = 0.3
    mutation_number = np.random.rand()
    if mutation_number < add_node_chance:
        new_individual = add_node(neural_net)
    elif mutation_number < (add_node_chance + add_link_chance):
        new_individual = add_link(neural_net)
    elif mutation_number < (add_node_chance + add_link_chance + alter_weights_chance):
        new_individual = alter_weights(neural_net)
    elif mutation_number < (add_node_chance + add_link_chance + alter_weights_chance + remove_link_chance):
        new_individual = remove_link(neural_net)
    else:
        new_individual = 0

    return new_individual


def add_node(neural_net, add_node_trys=20):
    for j in range(add_node_trys):
        idx = np.random.choice(range(len(neural_net["links"])))
        if not neural_net["links"][idx]["Enabled"]:
            continue
        neural_net["links"][idx]["Enabled"] = False
        node1 = neural_net["links"][idx]["From"]
        node2 = neural_net["links"][idx]["To"]
        new_node = len(neural_net["neurons"])
        split_x, split_y = (neural_net["neurons"][node1]["Loc"] + neural_net["neurons"][node2]["Loc"])/2

        neural_net["neurons"].append(
                {"Node": new_node,
                 "Type": "Hidden",
                 "Bias": 0,
                 "Loc": np.array([split_x, split_y])})
        # make 2 new links
        neural_net["links"].append({"ID": len(neural_net["links"]),
                                    "Weight": 1,
                                    "From": node1,
                                    "To": new_node,
                                    "Enabled": True})
        neural_net["links"].append({"ID": len(neural_net["links"]),
                                    "Weight": neural_net["links"][idx]['Weight'],
                                    "From": new_node,
                                    "To": node2,
                                    "Enabled": True})
        neural_net["n_visible"] += 1

        return neural_net


def add_link(individual, add_link_trys=30):
    for k in range(add_link_trys):
        neuron1 = np.random.choice(range(len(individual["neurons"])))
        # other_neurons = [node for node in range(len(individual["neurons"])) if node != neuron1]
        neuron2 = np.random.choice(range(len(individual["neurons"])))
        existing_link = [x for x in individual["links"] if x['From'] == neuron1 and x['To'] == neuron2
                         or x['From'] == neuron2 and x['To'] == neuron1]

        if len(existing_link) > 0:
            if existing_link[0]["Enabled"]:
                continue
            else:
                individual["links"][existing_link[0]["ID"]]["Enabled"] = True
        else:
            individual["links"].append({"ID": len(individual["links"]),
                                        "Weight": np.random.rand() * 2 - 1,
                                        "From": neuron1,
                                        "To": neuron2,
                                        "Enabled": True})

        return individual


def alter_weights(individual, alter_weight_st_deviation=0.1, alter_bias_st_deviation=0.1):
    # alter weights
    for ind in range(len(individual["links"])):
        if np.random.rand() < 0.95:
            new_weight = np.random.normal(0, alter_weight_st_deviation) + individual["links"][ind]["Weight"]
        else:
            new_weight = np.random.rand() * 2 - 1
        individual["links"][ind]["Weight"] = new_weight
    # alter bias
    for ind in range(len(individual["neurons"])):
        if np.random.rand() < 0.95:
            new_bias = np.random.normal(0, alter_bias_st_deviation) + individual["neurons"][ind]["Bias"]
        else:
            new_bias = np.random.rand() * 2 - 1
        individual["neurons"][ind]["Bias"] = new_bias

    return individual


def remove_link(individual):
    active_links = [x for x in individual["links"] if x["Enabled"]]
    ind = np.random.choice(range(len(active_links)))
    individual["links"][ind]["Enabled"] = False

    return individual
