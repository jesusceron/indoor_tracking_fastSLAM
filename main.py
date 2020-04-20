import numpy as np
# import scipy.stats
import scipy.stats
from numpy.random import uniform, seed
import pandas as pd
import time
from filterpy.monte_carlo import multinomial_resample
from filterpy.monte_carlo import residual_resample
from filterpy.monte_carlo import stratified_resample
from filterpy.monte_carlo import systematic_resample

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# seed(20)


class Particle:

    def __init__(self, N_LM):
        # weight of the particle
        self.w = 1.0 / N_PARTICLES
        # current person position
        self.x = 0.0
        self.y = 0.
        # landmark x-y positions
        self.lm = np.zeros((N_LM, LM_SIZE))
        # landmark position covariance's matrices
        self.lmP = [np.zeros((LM_SIZE, LM_SIZE)), np.zeros((LM_SIZE, LM_SIZE))]
        # person trajectory (historical of particles positions)
        self.t_x = [0.0]
        self.t_y = [6.]


def predict(particles, dist_traveled, STD_PERSON_POSITION):
    for i_particle in range(N_PARTICLES):
        # move in the (noisy) commanded direction
        dist = np.empty(2)
        dist[0] = dist_traveled[0] + (np.random.randn(1) * STD_PERSON_POSITION)
        dist[1] = dist_traveled[1] + (np.random.randn(1) * STD_PERSON_POSITION)
        particles[i_particle].x += dist[0]
        particles[i_particle].y += dist[1]

        particles[i_particle].t_x.append(particles[i_particle].x)
        particles[i_particle].t_y.append(particles[i_particle].y)


def update(particles, rssi, R, lm_id, particles_weights):
    distance_rssi = 10 ** ((rssi + 80) / -10 * 0.4)  # Calculate distance to beacon

    for i_particle in range(N_PARTICLES):
        # Get the distance between the person and the landmark
        dx = particles[i_particle].x - particles[i_particle].lm[lm_id][0]
        dy = particles[i_particle].y - particles[i_particle].lm[lm_id][1]
        dist = np.sqrt([(dx * dx) + (dy * dy)])
        residual = distance_rssi - dist

        # Compute Jacobians
        H = [-dx / dist, -dy / dist]

        # Compute covariance of the residual
        # covV = H * Cov_s * H^T + error
        HxCov = [particles[i_particle].lmP[lm_id][0, 0] * H[0] + particles[i_particle].lmP[lm_id][0, 1] * H[1],
                 particles[i_particle].lmP[lm_id][1, 0] * H[0] + particles[i_particle].lmP[lm_id][1, 1] * H[1]]

        covV = (HxCov[0] * H[0]) + (HxCov[1] * H[1]) + R

        # Calculate Kalman gain
        K_gain = [HxCov[0] * (1 / covV), HxCov[1] * (1.0 / covV)]

        # Calculate the new landmark position
        lm_x = particles[i_particle].lm[lm_id][0] + (K_gain[0] * residual)
        lm_y = particles[i_particle].lm[lm_id][1] + (K_gain[1] * residual)

        # Calculate the new covariance matrix of the landmark
        # cov_t = cov_t-1 - K * covV * K^T
        lm_P_aux = [[K_gain[0] * K_gain[0] * covV, K_gain[0] * K_gain[1] * covV],
                    [K_gain[1] * K_gain[0] * covV, K_gain[1] * K_gain[1] * covV]]

        lm_P = [[particles[i_particle].lmP[lm_id][0, 0] - lm_P_aux[0][0],
                 particles[i_particle].lmP[lm_id][0, 1] - lm_P_aux[0][1]],
                [particles[i_particle].lmP[lm_id][1, 0] - lm_P_aux[1][0],
                 particles[i_particle].lmP[lm_id][1, 1] - lm_P_aux[1][1]]]

        # Update particle weight
        particles[i_particle].w *= scipy.stats.norm(dist, covV).pdf(distance_rssi)
        particles_weights[i_particle] = particles[i_particle].w + 1.e-300

        # Update landmark in particle
        particles[i_particle].lm[lm_id][0] = lm_x
        particles[i_particle].lm[lm_id][1] = lm_y
        particles[i_particle].lmP[lm_id] = np.array(lm_P)

    particles_weights /= sum(particles_weights)
    for i_particle in range(N_PARTICLES):
        particles[i_particle].w = particles_weights[i_particle]

    return particles, particles_weights


def neff(weights):
    return 1. / np.sum(np.square(weights))


def resample_from_index(particles, particles_weights, indexes):

    for i_particle in range(N_PARTICLES):
        particles[i_particle].w = particles[indexes[i_particle]].w
        particles_weights[i_particle] = particles_weights[indexes[i_particle]]

    particles_weights.fill(1.0 / len(particles_weights))


# ## -------------------------------------------- MAIN LOOP ---------------------------------------------------------###

# Load IMU and beacons data
data = pd.read_csv("pos_rssi2.csv")
X = data['x']
Y = data['y']
# Z = data['z']
RSSI_room = data['room']
RSSI_kitchen = data['kitchen']
RSSI_bathroom = data['bathroom']
RSSI_dining = data['dining']
RSSI_living = data['living']

dataset_length = len(data)
first_zs = True
x_prev = 0.
y_prev = 0.
beacon_position = []
beacon_position_var = []
minimun_var = 1000

N_PARTICLES = 10
N_LM = 1
LM_SIZE = 2  # landmark positions in (x,y)
STD_PERSON_POSITION = 0.001
particles = [Particle(N_LM) for i in range(N_PARTICLES)]
THRESHOLD_RESAMPLE = N_PARTICLES / 3
particles_weights = np.zeros(N_PARTICLES) + (1 / N_PARTICLES)

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

# Initialize lm[0] (beacon in the door)
for i_particle in range(N_PARTICLES):
    particles[i_particle].lm[0][:] = [11, 5]
    particles[i_particle].lmP[0][:] = [[0.00287, 0], [0, 0.00061]]

# # Initialize lm[0] (beacon in the door)
# for i_particle in range(N_PARTICLES):
#     particles[i_particle].lm[1][:] = [2, 4]
#     particles[i_particle].lmP[1][:] = [[0.00287, 0], [0, 0.00061]]

for c_sample in range(dataset_length):
    x = X[c_sample]
    y = Y[c_sample]
    # z = Z[c_sample]
    rssi_room = RSSI_room[c_sample]
    rssi_kitchen = RSSI_kitchen[c_sample]
    rssi_bathroom = RSSI_bathroom[c_sample]
    rssi_dining = RSSI_dining[c_sample]
    rssi_living = RSSI_living[c_sample]

    # ------------------------- PREDICTION STEP --------------------------
    predict(particles, [x - x_prev, y - y_prev], STD_PERSON_POSITION)
    x_prev = x
    y_prev = y

    # ------------------------- UPDATE STEP ------------------------------
    if rssi_room != 0 and rssi_room > -88:
        R = 0.20 ** 2  # this value depends on the RSSI range
        update(particles, rssi_room, R, 0, particles_weights)

        # RESAMPLE.
        # resample if too few effective particles
        if neff(particles_weights) < THRESHOLD_RESAMPLE:
            # indexes = stratified_resample(weights)
            indexes = systematic_resample(particles_weights)
            resample_from_index(particles, particles_weights, indexes)
            assert np.allclose(particles_weights, 1 / N_PARTICLES)

    # if rssi_kitchen != 0 and rssi_kitchen > -88:
    #     R = 0.20 ** 2  # this value depends on the RSSI range
    #     update(particles, rssi_kitchen, R, 1)
    #
    # if rssi_bathroom != 0 and rssi_bathroom > -88:
    #     R = 0.20 ** 2  # this value depends on the RSSI range
    #     update(particles, rssi_bathroom, R, 2)
    #
    # if rssi_dining != 0 and rssi_dining > -88:
    #     R = 0.20 ** 2  # this value depends on the RSSI range
    #     update(particles, rssi_dining, R, 3)
    #
    # if rssi_living != 0 and rssi_living > -88:
    #     R = 0.20 ** 2  # this value depends on the RSSI range
    #     update(particles, rssi_living, R, 4)

    # ----------------------- RESAMPLE -----------------------------------

    # # Plot person trajectory
    # for i_line in range(N_PARTICLES):
    #     trajectory_lines[i_line][0].set_xdata(np.append(trajectory_lines[i_line][0].get_xdata(), particles[i_line].x))
    #     trajectory_lines[i_line][0].set_ydata(np.append(trajectory_lines[i_line][0].get_ydata(), particles[i_line].y))
    #     # plt.pause(0.00000000000000000001)

print("--- % execution time ---" % (time.time() - start_time))
plt.pause(1)
