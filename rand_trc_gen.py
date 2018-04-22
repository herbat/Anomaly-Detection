import pickle
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
N = 10000  # number of trajectories
D = 150  # expected value of trajectory data points
SIZE = (200, 200)
anomalies = False
anomaly_p = 0.1  # probability of anomaly happening

# updating the vector will be a perfect anomaly(basically random walk)


def plot_trajectory(_c):
    for t in _c:
        x = []
        y = []
        for i, j in t:
            x.append(i)
            y.append(j)
        plt.plot(x, y)

    plt.axis([0, SIZE[1], 0, SIZE[0]])
    plt.show()


collect = []

for i in range(N):
    # INIT

    d = rnd.randint(D-10, D+10)
    s_vec = [rnd.normal(0, 0.4), rnd.normal(0, 0.4)]
    s_pos = [rnd.uniform(0, SIZE[0]), rnd.uniform(0, SIZE[1])]
    prev_pos = list(map(sum,zip(s_pos, s_vec)))
    tmp = [s_pos, prev_pos]
    # decide if anomaly or not
    a = True if anomalies and rnd.uniform() < 0.1 else False
    # generate the trajectory
    for j in range(d):
        c_vec = [rnd.normal(s_vec[0], 0.1), rnd.normal(s_vec[1], 0.1)]  # current vector
        c_pos = list(map(sum,zip(prev_pos, c_vec)))  # current position
        prev_pos = c_pos
        if a:  # if anomaly, update the base vector on every iteration
            s_vec = c_vec
        tmp.append(c_pos)
    collect.append(tmp)

print(np.shape(collect[2]))
plot_trajectory([collect[453], collect[9832]])
# pickle.dump(collect, open('trajectory_data/training_rand1.p', 'wb'))
