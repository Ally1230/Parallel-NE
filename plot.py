import matplotlib.pyplot as plt
import numpy as np


# On GHC Machine
nthread = np.array([1,2,4,8])

# 2x18_OpenMp
total_time = np.array([10.6977, 5.73303, 3.2932, 2.155])
# init_time = np.array([ 0.671051, 0.669744, 0.679091, 0.671128])
# plt.plot(nthread,total_time[0]/total_time, label = "2x18")
total_time = np.array([11.2192, 6.70999, 4.26146, 3.18391])
total_time = np.array([10.8771, 7.80326, 4.94775, 2.65035])
plt.plot(nthread,total_time[0]/total_time, label = "2x18")

# 5x15_OpenMP
total_time = np.array([21.0126, 10.8311,  5.78805, 3.46685])
init_time = np.array([0.465993 , 0.461292, 0.463605, 0.463605])
# plt.plot(nthread,total_time[0]/total_time, label = "5x15")
#5x15_MPI
total_time = np.array([21.5535, 11.9045,  7.2029,  4.64817])
total_time = np.array([21.4647, 12.7571, 7.6272, 4.6765])
plt.plot(nthread,total_time[0]/total_time, label = "5x15")

# 10x10 OpenMp
total_time = np.array([34.4611, 17.6177, 9.2758, 5.43171, ])
init_time = np.array([0.432944, .430982, 0.430894, 0.436202])
# plt.plot(nthread,total_time[0]/total_time, label = "10x10")
# 10x10_MPI
total_time = np.array([34.8654, 19.4925, 11.1256,7.03265])
total_time = np.array([ 34.9989, 19.1921, 10.4743, 6.13452])
plt.plot(nthread,total_time[0]/total_time, label = "10x10")

# 15x5_OpenMp
total_time = np.array([26.562, 13.6687, 7.42719, 4.30495])
# plt.plot(nthread,total_time[0]/total_time, label = "15x5")
# MPI
total_time = np.array([27.1871, 16.6349, 10.1362, 6.74461])
total_time = np.array([21.577, 12.8072, 7.5118, 4.62007])
plt.plot(nthread,total_time[0]/total_time, label = "15x5")

# 18x2_OpenMp
total_time = np.array([15.4538, 8.08319, 4.44557, 2.79748])
init_time = np.array([0.542693, 0.541598, 0.544733, 0.542949])
# plt.plot(nthread,total_time[0]/total_time, label = "18x2")
# MPI
total_time = np.array([15.6988, 11.9638, 8.08115, 5.17659])
total_time = np.array([10.8721, 7.78204, 4.98377, 2.66225]) # improved version
plt.plot(nthread,total_time[0]/total_time, label = "18x2")

plt.ylabel("Speedup (x)")
plt.xlabel("Number of Threads")
plt.legend()
plt.savefig("OpenMP_improved")
# plt.show()