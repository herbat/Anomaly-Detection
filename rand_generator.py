import numpy as np
import pickle
import matplotlib.pyplot as plt

def gen(N, s1, s2, m1, m2):
    set = [None] * N
    for i in range(N):
        ampl = np.random.normal(m1, s1)
        freq = np.random.normal(m2, s2)
        xi = np.sin(np.linspace(1, 100, num=1000) * freq) * ampl
        set[i] = (xi)
    return set


def genTraining():
    return gen(10000, 0.07, 2.5, 1, 10), "trainingset.pkl"


def genTesting():
    return gen(1000, 0.07, 2.5, 1, 10), "testingset.pkl"


def genAnomaly():
    return gen(500, 0.07, 5, 0.5, 20), "anomalyset.pkl"


set, filename = genAnomaly()

print(np.size(set))
pickle.dump(set, open(filename, "wb"))