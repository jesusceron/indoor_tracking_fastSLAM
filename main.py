import numpy as np
# import scipy.stats
from numpy.random import uniform, seed
import pandas as pd
import time

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

#seed(20)


class Particle:

    def __init__(self, N_LM):
        self.w = 1.0 / N_PARTICLES
        self.x = 0.0
        self.y = 0.
        # landmark x-y positions
        self.lm = np.zeros((N_LM, LM_SIZE))
        # landmark position covariance
        self.lmP = np.zeros((N_LM * LM_SIZE, LM_SIZE))
        # trajectory
        self.t_x = [0.0]
        self.t_y = [6.]


# Predict step
def predict(particles, dist_traveled, STD_PERSON_POSITION):
    for i in range(N_PARTICLES):
        # move in the (noisy) commanded direction
        dist = np.empty(2)
        dist[0] = dist_traveled[0] + (np.random.randn(1) * STD_PERSON_POSITION)
        dist[1] = dist_traveled[1] + (np.random.randn(1) * STD_PERSON_POSITION)
        particles[i].x += dist[0]
        particles[i].y += dist[1]

        particles[i].t_x.append(particles[i].x)
        particles[i].t_y.append(particles[i].y)
    return particles


# ## -------------------------------------------- MAIN LOOP ---------------------------------------------------------###

# Load IMU and beacons data
data = pd.read_csv("pos_rssi2.csv")
X = data['x']
Y = data['y']
# Z = data['z']
RSSI = data['door']

dataset_length = len(data)
first_zs = True
x_prev = 0.
y_prev = 0.
beacon_position = []
beacon_position_var = []
minimun_var = 1000

N_PARTICLES = 3
N_LM = 1
LM_SIZE = 2  # position in (x,y)
STD_PERSON_POSITION = 0.001
particles = [Particle(N_LM) for i in range(N_PARTICLES)]

# Plot initialization
plt.style.use('ggplot')
plt.ion()
fig = plt.figure()
ax = fig.gca()
plt.xlim(-3, 15)
plt.ylim(-2, 9)
# plt.axis('equal')
plt.title("Initializing beacons position")
plt.xlabel("x(m)")
plt.ylabel("y(m)")
# dynamic plotting
trajectory_lines = []
for i_line in range(N_PARTICLES):

    trajectory_lines.append(ax.plot([], [], color='#7f7f7f'))
    # particles_scattered, = ax.plot([], [], 'ok', markersize=1)  # Plot particles
    # beacons, = ax.plot([], [], 'D', markersize=5)  # Plot beacon position

# current sample (sample '0' is (0,6,0), I don't have any uncertainty about it. I create the particles from sample '1')
c_sample = 1
start_time = time.time()
for c_sample in range(dataset_length):
    x = X[c_sample]
    y = Y[c_sample]
    # z = Z[c_sample]
    # rssi = RSSI[c_sample]

    predict(particles, [x - x_prev, y - y_prev], STD_PERSON_POSITION)
    x_prev = x
    y_prev = y

    # Plot person trajectory
    for i_line in range(N_PARTICLES):
        trajectory_lines[i_line][0].set_xdata(np.append(trajectory_lines[i_line][0].get_xdata(), particles[i_line].x))
        trajectory_lines[i_line][0].set_ydata(np.append(trajectory_lines[i_line][0].get_ydata(), particles[i_line].y))
        plt.pause(0.00000000000000000001)

print("--- % execution time ---" % (time.time() - start_time))
plt.pause(1)