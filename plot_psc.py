import numpy as np
import matplotlib.pyplot as plt

#MPI:
#2x18:
nthread = np.array([1,2,4,8, 16, 32, 64, 128])
t = np.array([0.238362, 0.437532, 0.789543, 1.53709, 3.00769, 5.91207, 9.56609, 13.5249][::-1])
plt.plot(nthread,t[0]/t, label = "2x18")
t = np.array([0.406752, 0.745114, 1.32813, 2.55799, 4.88068, 9.46662, 18.143, 35.1052][::-1])
plt.plot(nthread,t[0]/t, label = "5x15")
t = np.array([0.577986, 1.06578, 1.8932, 3.61338, 6.90038, 13.0932, 24.4509, 45.6354][::-1])
plt.plot(nthread,t[0]/t, label = "10x10")
t = np.array([0.406297, 0.746866, 1.32779, 2.55162, 4.87873, 9.43029, 18.2228, 35.0362][::-1])
plt.plot(nthread,t[0]/t, label = "15x5")
t = np.array([0.225604, 0.403418, 0.7212, 1.40104, 2.7193, 5.28769, 10.1416, 19.7802][::-1])
plt.plot(nthread,t[0]/t, label = "18x2")

plt.ylabel("Speedup (x)")
plt.xlabel("Number of Threads")
plt.legend()
plt.savefig("plot/PSC_MPI")

#OpenMP:
t = np.array([0.799786, 0.918196, 1.14937, 1.55169, 2.36393, 4.02306, 7.11843, 13.5525][::-1])
plt.plot(nthread,t[0]/t, label = "2x18")
t = np.array([0.753756, 1.01636, 1.57121, 2.67613, 4.83074, 9.1563, 17.6904, 34.8556][::-1])
plt.plot(nthread,t[0]/t, label = "5x15")
t = np.array([0.811357, 1.14293, 1.88808, 3.28696, 6.09743, 11.7589, 22.7908, 45.4175][::-1])
plt.plot(nthread,t[0]/t, label = "10x10")
t = np.array([0.751352, 1.01337, 1.56729, 2.67665, 4.8459, 9.15225, 17.7428, 35.0588][::-1])
plt.plot(nthread,t[0]/t, label = "15x5")
t = np.array([0.816701, 0.956203, 1.25908, 1.88895, 3.117, 5.43628, 10.1861, 19.7634][::-1])
plt.plot(nthread,t[0]/t, label = "18x2")

plt.ylabel("Speedup (x)")
plt.xlabel("Number of Threads")
plt.legend()
plt.savefig("plot/PSC_OpenMP")