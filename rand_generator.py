import numpy as np
import pickle

N = 5000
Pa = 0.003 #probability of anomaly happening
set = []

for i in range(N):
    # values of sigma
    s1 = 0.07
    s2 = 2.5

    if(np.random.uniform() < Pa):
        # and in case of an anomaly
        s1 = 0.7
        s2 = 10

    ampl = np.random.normal(1, s1)
    freq = np.random.normal(10, s2)
    xi = np.sin(np.linspace(1, 0.0001, 2) * freq) * ampl
    set.append(xi)


with open('trainingset.pkl', 'rb') as f:
 pickle.dump(set, f)