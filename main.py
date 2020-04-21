import numpy as np
# import scipy.stats
import scipy.stats
from numpy.random import uniform, seed
import pandas as pd
import time
import copy
from filterpy.monte_carlo import stratified_resample
from filterpy.monte_carlo import systematic_resample

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

seed(20)
N_PARTICLES = 10
N_LM = 5
LM_SIZE = 2  # landmark positions in (x,y)
STD_PERSON_POSITION = 0.005
THRESHOLD_RESAMPLE = N_PARTICLES / 2


class Particle:

    def __init__(self):
        # weight of the particle
        self.w = 1.0 / N_PARTICLES
        # current person position
        self.x = 0.0
        self.y = 0.
        # landmark x-y positions
        self.lm = np.zeros((N_LM, LM_SIZE))
        # landmark position covariance's matrices
        self.lmP = []
        for _ in range(N_LM):
            self.lmP.append(np.zeros((LM_SIZE, LM_SIZE)))
        # person trajectory (historical of particles positions)
        self.t_x = [0.0]
        self.t_y = [6.]


def predict(particles, dist_traveled):
    for i_particle in range(N_PARTICLES):
        # move in the (noisy) commanded direction
        dist = np.empty(2)
        dist[0] = dist_traveled[0] + (np.random.randn(1) * STD_PERSON_POSITION)
        dist[1] = dist_traveled[1] + (np.random.randn(1) * STD_PERSON_POSITION)
        particles[i_particle].x += dist[0]
        particles[i_particle].y += dist[1]

        particles[i_particle].t_x.append(particles[i_particle].x)
        particles[i_particle].t_y.append(particles[i_particle].y)
    return particles


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

        # Update landmark in particle
        particles[i_particle].lm[lm_id][0] = lm_x
        particles[i_particle].lm[lm_id][1] = lm_y
        particles[i_particle].lmP[lm_id] = np.array(lm_P)

        # Update particles weights
        particles_weights[i_particle] *= (scipy.stats.norm(dist, covV).pdf(distance_rssi) + 1.e-300)

    particles_weights /= sum(particles_weights)
    for i_particle in range(N_PARTICLES):
        particles[i_particle].w = particles_weights[i_particle]

    return particles, particles_weights


def neff(weights):
    return 1. / np.sum(np.square(weights))


def resample_from_index(particles, indexes):

    # particles_weights.fill(1.0 / N_PARTICLES)
    for i_particle in reversed(np.arange(1, N_PARTICLES)):
        # particles[i_particle] = copy.deepcopy(particles[indexes[i_particle]])
        particles[i_particle].w = particles[indexes[i_particle]].w
        particles[i_particle].x = particles[indexes[i_particle]].x
        particles[i_particle].y = particles[indexes[i_particle]].y
        particles[i_particle].lm = particles[indexes[i_particle]].lm
        particles[i_particle].lmP = particles[indexes[i_particle]].lmP
        particles[i_particle].t_x = particles[indexes[i_particle]].t_x
        particles[i_particle].t_y = particles[indexes[i_particle]].t_y

    return particles


# ## -------------------------------------------- MAIN LOOP ---------------------------------------------------------###
def main():

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

    particles = [Particle() for _ in range(N_PARTICLES)]
    particles_weights = np.ones(N_PARTICLES) / N_PARTICLES

    # Plot initialization
    plt.style.use('ggplot')
    plt.ion()
    fig = plt.figure()
    ax = fig.gca()
    plt.xlim(-3, 15)
    plt.ylim(-2, 9)
    plt.title("Initializing beacons position")
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    # dynamic plotting
    trajectory_lines = []
    for i_line in range(N_PARTICLES):
        trajectory_lines.append(ax.plot([], [], color='#7f7f7f'))
        # particles_scattered, = ax.plot([], [], 'ok', markersize=1)  # Plot particles
        # beacons, = ax.plot([], [], 'D', markersize=5)  # Plot beacon position

    # Initialize lm[0] (beacon in the door)
    for i_particle in range(N_PARTICLES):
        particles[i_particle].lm[0][:] = [11, 5]
        particles[i_particle].lmP[0][:] = [[0.00287, 0], [0, 0.00061]]

    # Initialize lm[0] (beacon in the door)
    for i_particle in range(N_PARTICLES):
        particles[i_particle].lm[1][:] = [2, 4]
        particles[i_particle].lmP[1][:] = [[0.00287, 0], [0, 0.00061]]
        # Initialize lm[0] (beacon in the door)

    for i_particle in range(N_PARTICLES):
        particles[i_particle].lm[2][:] = [11, 5]
        particles[i_particle].lmP[2][:] = [[0.00287, 0], [0, 0.00061]]

    # Initialize lm[0] (beacon in the door)
    for i_particle in range(N_PARTICLES):
        particles[i_particle].lm[3][:] = [2, 4]
        particles[i_particle].lmP[3][:] = [[0.00287, 0], [0, 0.00061]]
        # Initialize lm[0] (beacon in the door)

    for i_particle in range(N_PARTICLES):
        particles[i_particle].lm[4][:] = [11, 5]
        particles[i_particle].lmP[4][:] = [[0.00287, 0], [0, 0.00061]]

    start_time = time.time()
    flag_update = 0  # if the filter was updated, go to resample
    last_sample_plotted = 0
    # sample '0' is (0,6,0), I don't have any uncertainty about it. I create the particles from sample '1')
    for c_sample in range(1, dataset_length):
        x = X[c_sample]
        y = Y[c_sample]
        # z = Z[c_sample]
        rssi_room = RSSI_room[c_sample]
        rssi_kitchen = RSSI_kitchen[c_sample]
        rssi_bathroom = RSSI_bathroom[c_sample]
        rssi_dining = RSSI_dining[c_sample]
        rssi_living = RSSI_living[c_sample]

        # ------------------------- PREDICTION STEP --------------------------
        particles = predict(particles, [x - x_prev, y - y_prev])
        x_prev = x
        y_prev = y

        # ------------------------- UPDATE STEP ------------------------------
        if rssi_room != 0 and rssi_room > -88:
            R = 0.20 ** 2  # this value depends on the RSSI range
            particles, particles_weights = update(particles, rssi_room, R, 0, particles_weights)
            flag_update = 1

        if rssi_kitchen != 0 and rssi_kitchen > -88:
            R = 0.20 ** 2  # this value depends on the RSSI range
            particles, particles_weights = update(particles, rssi_kitchen, R, 1, particles_weights)
            flag_update = 1

        if rssi_bathroom != 0 and rssi_bathroom > -88:
            R = 0.20 ** 2  # this value depends on the RSSI range
            particles, particles_weights = update(particles, rssi_bathroom, R, 2, particles_weights)
            flag_update = 1

        if rssi_dining != 0 and rssi_dining > -88:
            R = 0.20 ** 2  # this value depends on the RSSI range
            particles, particles_weights = update(particles, rssi_dining, R, 3, particles_weights)
            flag_update = 1

        if rssi_living != 0 and rssi_living > -88:
            R = 0.20 ** 2  # this value depends on the RSSI range
            particles, particles_weights = update(particles, rssi_living, R, 4, particles_weights)
            flag_update = 1

        # ----------------------- RESAMPLE -----------------------------------
        if flag_update:
            # resample if too few effective particles
            if neff(particles_weights) < THRESHOLD_RESAMPLE:
                # indexes = stratified_resample(weights)
                indexes = systematic_resample(particles_weights)
                particles = resample_from_index(particles, indexes)
                flag_update = 0

                # Plot person trajectory
                for i_particle in reversed(np.arange(0, N_PARTICLES)):

                    if i_particle != indexes[i_particle]:
                        trajectory_lines[i_particle][0].remove()
                        trajectory_lines[i_particle] = ax.plot(trajectory_lines[indexes[i_particle]][0].get_xdata(),
                                                               trajectory_lines[indexes[i_particle]][0].get_ydata())

        for i_particle in range(N_PARTICLES):
            trajectory_lines[i_particle][0].set_xdata(np.append(trajectory_lines[i_particle][0].get_xdata(), particles[i_particle].x))
            trajectory_lines[i_particle][0].set_ydata(np.append(trajectory_lines[i_particle][0].get_ydata(), particles[i_particle].y))
            # plt.pause(0.00000000000000000001)

    print("--- % execution time ---" % (time.time() - start_time))
    plt.pause(1)


main()
