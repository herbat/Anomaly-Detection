import numpy as np
import pickle

N = 5000
set = []

for i in range(N):
    A = np.random.normal(1, 0.07)
    w = np.random.normal(10, 2.5)
    xi = np.sin(np.linspace(1, 0.0001, 2) * w) * A
    set.append(xi)

with open('trainingset.pkl', 'rb') as f:
 pickle.dump(set, f)