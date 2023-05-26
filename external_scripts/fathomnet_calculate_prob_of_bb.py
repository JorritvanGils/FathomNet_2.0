import numpy as np


def load_probabilities():
    array = np.loadtxt('distribution.out', delimiter=",")
    return array


def probability_of_sample(labels, scaling = 2073, array = None):
    if array == None:
        array = load_probabilities()

    prob = []
    for label in labels:
        label = label.split(" ")
        surface =  int(float(label[3]) * float(label[4]) * scaling)
        if surface > 99:
            surface = 99

        prob.append((array[surface]/np.sum(array))/(np.max(array)/np.sum(array)))

    return prob