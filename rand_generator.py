import numpy as np
import pickle
import matplotlib.pyplot as plt

N = 5000
Pa = 0 #probability of anomaly happening
set = [None]*N

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
    xi = np.sin(np.linspace(1, 100, num=1000) * freq) * ampl
    set[i] = (xi)

print(np.size(set))
pickle.dump(set, open('trainingset.pkl', "wb"))