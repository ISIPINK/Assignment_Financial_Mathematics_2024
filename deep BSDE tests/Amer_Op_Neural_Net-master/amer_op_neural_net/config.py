import multiprocessing
from tensorflow.python.training.moving_averages import assign_moving_average
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
import _pickle
import time
import os
import sys
import errno
import math
import random
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import pairwise_distances_argmin
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

savefig_mode = False
matplotlib.use('Agg')


np_floattype = np.float64
tf_floattype = tf.float64
np.set_printoptions(precision=6, threshold=np.inf,
                    linewidth=np.inf, suppress=True)
opt_seed = True
global_seed = 347161
if opt_seed:
    random.seed(global_seed)
    np.random.seed(random.randint(0, 2 ** 31 - 1))
    tf.reset_default_graph()
    # tf.set_random_seed should be applied on the graph level!
    tf.set_random_seed(random.randint(0, 2 ** 31 - 1))
num_cpus = multiprocessing.cpu_count()
print(" Number of CPUs: ", num_cpus)
session_config = tf.ConfigProto(device_count={"CPU": num_cpus},
                                intra_op_parallelism_threads=num_cpus,
                                inter_op_parallelism_threads=num_cpus,
                                allow_soft_placement=True,
                                log_device_placement=False)


option_type = ['put', 'geometric', 'vanilla']
d = 1
r = 0.02
qq = np.array([[0.02] * d], dtype=np_floattype)
# mu = r - qq
mu = np.array([[0.02] * d], dtype=np_floattype)

sigma = np.array([[0.3] * d], dtype=np_floattype)
rho = np.eye(d, dtype=np_floattype) + 0.75 * (
    np.ones((d, d), dtype=np_floattype) - np.eye(d, dtype=np_floattype))

rhoL = np.linalg.cholesky(rho).T
# K = 1.0
K = 110
# x0 = K
x0 = 100
X0 = np.array([[x0] * d], dtype=np_floattype)
T = 1.0
N = 10

dim_h = 6
num_layers = 6
dim_layers = num_layers*[d+dim_h]

dim_layers = [None] + dim_layers + [1]

###############################################
# directory = "Results/option=" + option_type[0] + "&" + option_type[1] + "&" + option_type[2] + "_d=" + str(
#     d) + "_X0=" + str(x0).replace(".", "p") + "_N=" + str(N) + "_seed=" + str(global_seed) + "/"
# print("\n directory: ", directory, "\n")
# try:
#     os.makedirs(directory)
# except OSError as e:
#     if e.errno != errno.EEXIST:
#         raise

control_tol = 1e-5
payoff_tol = 1e-8
def opregion_function(L): return L > payoff_tol


dt = T / N
sharpness = 1 / dt
updaten = 2

num_channels = 1
batch_size = 100
n_relaxstep = 5  # Monitor the first few steps closely
n_decaystep = 50
n_totalstep = 100
simulation_size = n_totalstep * num_channels * batch_size
simulation_index = np.arange(simulation_size, dtype=int)
print(" Simulation size: ", simulation_size)
if opt_seed:
    AmerOp_seeds = np.random.randint(2 ** 31 - 1, size=N + 1)
else:
    AmerOp_seeds = None
