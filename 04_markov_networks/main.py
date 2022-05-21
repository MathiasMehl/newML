import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx import read_edgelist
import random

POTENTIAL_1 = False
# TODO!
POTENTIAL_2 = False
POTENTIAL_2_W1 = -2
POTENTIAL_2_W2 = 3
POTENTIAL_FROM_SLIDE = True
ITERATIONS = 2000
BURN_IN = 2000

SILENT_FLAG = True
PRINT_PREDICTION_ARRAY = False
PRINT_AVG_DIFFERENCE = False


def print_prediction_array(prediction_array):
    for i, row in enumerate(prediction_array):
        if row == [0, 0]:
            continue
        print(f"Node: {i}   1's: {row[0]}     2's: {row[1]}")


def n_log_potential_1(n, w1, w2):
    if n['gender'] == 1:
        return w1, w2
    else:
        return w2, w1


def n_log_potential_2(n1, n2, w1, w2):
    if n1['predicted_practice'] == n2['predicted_practice']:
        return w1, w2
    else:
        return w2, w1


def get_att_array(G, att_name):
    ret_array = np.zeros(nx.number_of_nodes(G))
    for i, n in enumerate(G.nodes()):
        ret_array[i] = G.nodes[n][att_name]
    return (ret_array)


# Task 1
def init_practice(lazega):
    for n in lazega.nodes():
        node = lazega.nodes[n]
        if np.isnan(node['observed_practice']):
            value = random.randint(1, 2)
        else:
            value = node['observed_practice']

        lazega.add_node(n, predicted_practice=value)

    return lazega


# Task 2
# Write a Gibbs-sampling function for re-sampling the 'predicted_practice' attribute value for a node n.
# At this point we need not worry whether for n the actual 'practice' value is known or not.
def potential_from_slide(lazega, n):
    node_neighbours = nx.all_neighbors(lazega, n)

    ones = 0
    twos = 0
    for n_prime in node_neighbours:
        if lazega.nodes[n_prime]['predicted_practice'] == 1:
            ones += 1
        elif lazega.nodes[n_prime]['predicted_practice'] == 2:
            twos += 1

    # one_minus_two_term = math.exp(WEIGHT_SLIDE * (ones - twos))
    # two_minus_one_term = math.exp(WEIGHT_SLIDE * (twos - ones))

    one_res = weight_slide * ones - twos
    two_res = weight_slide * twos - ones

    return one_res, two_res


def gibbs_sample(lazega, n):
    one_term = 0
    two_term = 0
    if POTENTIAL_FROM_SLIDE:
        one_res, two_res = potential_from_slide(lazega, n)
        one_term += one_res
        two_term += two_res
    if POTENTIAL_1:
        one_res, two_res = n_log_potential_1(lazega.nodes[n], potential_1_w1, potential_1_w2)
        one_term += one_res
        two_term += two_res
    if POTENTIAL_2:
        # TODO!
        pass
        # one_res, two_res = n_log_potential_2(lazega.nodes[n], lazega.nodes[n], POTENTIAL_2_W1, POTENTIAL_2_W2)
        # one_term += one_res
        # two_term += two_res

    random_number = random.random()

    numerator = math.exp(one_term)
    denominator = math.exp(one_term) + math.exp(two_term)
    probability_for_1 = numerator / denominator

    if probability_for_1 < random_number:
        new_value = 2
    else:
        new_value = 1

    return {n: new_value}


def gibbs_one_round(G):
    updates = {}
    for n in G.nodes:
        if np.isnan(G.nodes[n]['observed_practice']):
            updates.update(gibbs_sample(G, n))
    nx.set_node_attributes(G, updates, 'predicted_practice')
    return G


def update_prediction_array(lazega, prediction_array):
    for n in lazega.nodes():
        node = lazega.nodes[n]
        if np.isnan(node['observed_practice']):
            value_of_predicted_practice = node['predicted_practice']
            if value_of_predicted_practice == 1:
                prediction_array[n - 1][0] = prediction_array[n - 1][0] + 1
            elif value_of_predicted_practice == 2:
                prediction_array[n - 1][1] = prediction_array[n - 1][1] + 1
            else:
                print("shits fucked, yo")
    return prediction_array


def find_most_likely(lazega, prediction_array):
    prediction_dict = {}
    for n in lazega.nodes():
        node = lazega.nodes[n]
        if np.isnan(node['observed_practice']):
            if prediction_array[n - 1][0] < prediction_array[n - 1][1]:
                most_likely_value = 2
            elif prediction_array[n - 1][0] > prediction_array[n - 1][1]:
                most_likely_value = 1
            else:
                if not SILENT_FLAG:
                    print("Both values equally likely")
                most_likely_value = 0
            prediction_dict.update({n: most_likely_value})

    nx.set_node_attributes(lazega, prediction_dict, 'predicted_practice')
    return lazega


def calculate_accuracy(lazega, nans):
    predictions = get_att_array(lazega, 'predicted_practice')
    actual = get_att_array(lazega, 'observed_practice')
    correct_predictions = 0
    for n in lazega.nodes():
        if np.isnan(lazega.nodes[n]['observed_practice']):
            if predictions.flatten()[n - 1] == actual.flatten()[n - 1]:
                correct_predictions += 1

    percentage = correct_predictions / nans
    if not SILENT_FLAG:
        print(f"Correct prediction percentage = {percentage * 100}%")
    return percentage * 100


def print_avg_difference(prediction_array):
    number_of_nans = 0
    sum = 0
    for row in prediction_array:
        if row == [0, 0]:
            continue
        number_of_nans += 1
        sum += (math.sqrt(math.pow(row[0] - row[1], 2)))

    print(
        f"The average node had a difference of {sum / number_of_nans} between the number of times it was 1 and 2")


def main(potential_1_w1, potential_1_w2, weight_slide):
    if not (POTENTIAL_FROM_SLIDE | POTENTIAL_1 | POTENTIAL_2):
        print("Select potential function(s) to continue")
        return
    lazega = read_edgelist('lazega-friends.edges', nodetype=int)
    node_atts = pd.read_csv("lazega-attributes.txt", sep=' ')
    nans = 0
    for i in range(node_atts.shape[0]):
        lazega.add_node(node_atts.loc[i, 'nodeID'], gender=node_atts.loc[i, 'nodeGender'])
        lazega.add_node(node_atts.loc[i, 'nodeID'], office=node_atts.loc[i, 'nodeOffice'])
        lazega.add_node(node_atts.loc[i, 'nodeID'], true_practice=node_atts.loc[i, 'nodePractice'])
        if random.random() > 0.4:
            lazega.add_node(node_atts.loc[i, 'nodeID'], observed_practice=node_atts.loc[i, 'nodePractice'])
        else:
            lazega.add_node(node_atts.loc[i, 'nodeID'], observed_practice=np.nan)
            nans += 1

    lazega = init_practice(lazega)

    # Burn in
    if not SILENT_FLAG:
        print(f"Doing a burn in of {BURN_IN} iterations")
    for i in range(0, BURN_IN):
        lazega = gibbs_one_round(lazega)

    prediction_array = [[0] * 2 for i in range(lazega.number_of_nodes())]
    if not SILENT_FLAG:
        print(f"Doing {ITERATIONS} iterations of gibbs sampling")
    for i in range(0, ITERATIONS):
        lazega = gibbs_one_round(lazega)

        prediction_array = update_prediction_array(lazega, prediction_array)

    if PRINT_PREDICTION_ARRAY:
        print_prediction_array(prediction_array)

    if PRINT_AVG_DIFFERENCE:
        print_avg_difference(prediction_array)

    lazega = find_most_likely(lazega, prediction_array)

    return calculate_accuracy(lazega, nans)


if __name__ == "__main__":
    epochs = 3
    weight_settings = 6

    scores = [[[0] * 4 for i in range(weight_settings * weight_settings * weight_settings)] for j in range(epochs)]
    index = 0
    reported_yet = False
    for i in range(epochs):
        random.seed(i)
        scores_this_epoch = scores[i]
        epoch_counter = 0
        for j in range(0, 25 * weight_settings, 25):
            potential_1_w1 = j / 100
            for k in range(0, 25 * weight_settings, 25):
                potential_1_w2 = k / 100
                for l in range(0, 25 * weight_settings, 25):
                    weight_slide = l / 100
                    if not SILENT_FLAG:
                        print(
                            f"Testing with parameters: w1={potential_1_w1}, w2={potential_1_w2}, w_slide={weight_slide}")
                    res = main(potential_1_w1, potential_1_w2, weight_slide)

                    scores_this_epoch[epoch_counter] = [j / 100, k / 100, l / 100, res]

                    index += 1
                    epoch_counter += 1
                    if not SILENT_FLAG:
                        print(f"Configurations to go: {int((math.pow(weight_settings, 3) * epochs) - index)}")

        scores[i] = scores_this_epoch
        print(f"Done with epoch {i}, still {epochs - (i + 1)} to go")

    avg_scores = [0] * (weight_settings * weight_settings * weight_settings)
    for i in range(epochs):
        epoch_scores = scores[i]
        for j in range(weight_settings * weight_settings * weight_settings):
            avg_scores[j] += epoch_scores[j][3]

    new_avg_scores = list(map(lambda x: x / epochs, avg_scores))
    best_score = 0
    best_configuration = [0, 0, 0, 0]
    for i, score in enumerate(new_avg_scores):
        if best_score < score:
            best_score = score
            best_configuration = scores[0][i]
    print(f"Best score was {best_score} found with configuration {best_configuration}")
    print(f"The total avg scores over {epochs} different seeds: {new_avg_scores}")
    # Best score was 68.6547751765143 found with configuration [0.75, 0.5, 0.0, 65.38461538461539]
