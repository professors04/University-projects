

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

plt.rcParams['text.usetex'] = False


#do the calculations for glued overhanging bricks
N_bricks = 20
N_l = 1 #number of layers per brick

l = 100*1e-9 #brick length
w = 50*1e-9 #brick width
d = 3.37*1e-10 #layer distance

m = l*w * (2630*1e-3) * N_l #weight for N_l layers
g = 9.81

F_g = m*g #grav force on one brick
P_stick = 1.1*1e9 #pressure by glue | in 1D force per length




def T_left(x):
    return 1/2*P_stick*x**2 * w + x**2/(2*l)*F_g

def T_right(x):
    return (l-x)**2/(2*l)*F_g


x_bal = [] #save all the balance points

x_overlap = [0]
bricks = [1]


for i in range(2, N_bricks+1):
    bricks.append(i)
    
    def torque_add(x):        
        if i>2:
            d = []
            for k,j in enumerate(x_bal):
                d.append((k+1)*j)
            
            return (np.sum([j for j in range(1,i-1)]) * l - (i-2)*x + (i-2)*l/2 - np.sum(d)) * F_g
        else:
            return 0
        
        
    c = fsolve(lambda x: T_left(x) - ( T_right(x) + torque_add(x) ), x0=1)[0]
    x_bal.append(c)
            
    x_overlap.append(l-c + x_overlap[-1])

    
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.rcParams['figure.figsize'] = [18, 12]

ax.scatter(bricks, np.array(x_overlap)/l, label="adhesive stacking")
ax.plot(bricks, np.array(x_overlap)/l)



# compare with non sticking blocks
x = [1]
y = [0]
for i in range(2,N_bricks+1):
    x.append(i)
    y.append(l * 1/(2*(i-1)) + y[-1])

ax.scatter(x,np.array(y)/l, label = "stacking without vdw force")
ax.plot(x,np.array(y)/l)

ax.grid()
ax.legend()
ax.set_title("relative overhang of stacked graphene layers")
ax.set_xlabel("N")
ax.set_ylabel(r"$x_{overhang}~/~l$")

plt.tight_layout()
