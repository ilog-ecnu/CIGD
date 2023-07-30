import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


class ZDT1:
    def le():
        return np.sqrt(5)/2+np.log(2+np.sqrt(5))/4

    def y(x):
        return (1-x)*(1-x)

    def ref(n):
        ret = np.zeros((n+1, 2))
        #ret[:,0] = np.arange(0, n+1, 1) / n
        #ret[:,1] = 1 - np.sqrt(ret[:,0])

        #ret[:, 1] = np.arange(0, n+1, 1) / n
        #ret[:, 0] = np.power((1 - ret[:, 1]), 2)

        x0 = np.linspace(0, 0.99999, n+1)
        y0 = 1 - x0

        ret[:, 0] = (-x0 + np.sqrt(x0**2 + 4*x0*y0)) / (2*y0)
        ret[:, 1] = 1 - np.sqrt(ret[:, 0])

        return ret

def eigd(sol, k):
    return integrate.romberg(lambda x: np.min(np.linalg.norm(
        sol - np.array([ZDT1.y(x), x]), axis=1))*np.sqrt(1+(2*x-2)**2), 0, 1, divmax=k, show=False) / ZDT1.le()

def igd(sol, ref):
    mind = np.zeros(ref.shape[0])
    for i in range(ref.shape[0]):
        mind[i] = np.min(np.linalg.norm(sol - ref[i,:], axis=1))
    return np.mean(mind)

def draw_pop(pop):
    plt.rc('font', family='Helvetica')
    x=np.linspace(0,1,100)
    y=1-np.sqrt(x)
    plt.figure(figsize=(4,4))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.plot(x, y, label="PF", color="grey", linewidth=0.5) 
    plt.scatter(x= pop[:,0], y=pop[:,1], color=[1,1,1,0], label='NSGA3', marker='o', s=30, edgecolors='r', linewidths=0.5)
    plt.legend()
    plt.show()


pop_lssa = np.loadtxt("./lssa.txt")
pop_nsga3 = np.loadtxt("./nsga3.txt")
print(ZDT1.ref(100))
# draw_pop(ZDT1.ref(100))
# draw_pop(pop_lssa)
# draw_pop(pop_nsga3)
nn = [10, 50, 100, 1000, 10000, 100000]
kk = [4, 6, 8, 10, 12, 14]
for i in range(6):
    print(nn[i], igd(pop_nsga3, ZDT1.ref(nn[i])), igd(pop_lssa, ZDT1.ref(nn[i])),
          kk[i], eigd(pop_nsga3, kk[i]), eigd(pop_lssa, kk[i]), sep='\t')
