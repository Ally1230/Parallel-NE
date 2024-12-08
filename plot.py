import matplotlib.pyplot as plt
import numpy as np

nthread = np.array([1,2,4,8])
total_time = np.array([65.4124, 34.1314, 19.1601, 12.6315])
init_time = np.array([2.52383, 2.52452, 2.51891, 2.53156])

plt.plot(nthread,total_time[0]/total_time)
plt.plot(nthread, (total_time[0]-init_time[0])/ (total_time - init_time))
plt.savefig("plot")