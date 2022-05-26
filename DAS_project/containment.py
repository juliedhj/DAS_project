#
# Containment Algorithm
# Ivano Notarnicola, Lorenzo Pichierri
# Bologna, 30/03/2022
#
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import networkx as nx

def waves(amp, omega, phi, t, n_x, n_agents):
    """
    This function generates a sinusoidal input trajectory

    Input:
        - amp, omega, phi = sine parameter u = amp*sin(omega*t + phi)
        - n_agents = number of agents
        - n_x = agent dimension
        - t = time variable

    Output:
        - u = input trajectory

    """
    u = []
    for ii in range(n_x*n_agents):
        u_i = amp*np.sin(omega*t+phi)
        u.append(u_i)
    return np.array(u)


def animation(xx, NN, n_x, n_leaders, horizon, dt, animate=True):
    """
    This function generates the containment animation

    """

    TT = np.size(horizon,0)
    # for tt in range(0,TT,int(TT/200)):
    for tt in range(0,TT,dt):
        xx_tt = xx[:,tt].T

        # Plot trajectories
        if tt>dt and tt<TT-1:
            plt.plot(xx[0:n_x*(NN-n_leaders):n_x,tt-dt:tt+1].T,xx[1:n_x*(NN-n_leaders):n_x,tt-dt:tt+1].T, linewidth = 2, color = 'tab:blue')
            plt.plot(xx[n_x*(NN-n_leaders):n_x*NN:n_x,tt-dt:tt+1].T,xx[n_x*(NN-n_leaders)+1:n_x*NN:n_x,tt-dt:tt+1].T, linewidth = 3, color = 'tab:red')

        # Plot convex hull
        leaders_pos = np.reshape(xx[n_x*(NN-n_leaders):n_x*NN,tt],(n_leaders,n_x))
        hull = ConvexHull(leaders_pos)
        plt.fill(leaders_pos[hull.vertices,0], leaders_pos[hull.vertices,1], 'darkred', alpha=0.3)
        vertices = np.hstack((hull.vertices,hull.vertices[0])) # add the firt in the last position to draw the last line
        plt.plot(leaders_pos[vertices,0], leaders_pos[vertices,1], linewidth = 2, color = 'darkred', alpha=0.7)


        # Plot agent position
        for ii in range(NN):
            index_ii =  ii*n_x + np.array(range(n_x))
            p_prev = xx_tt[index_ii]
            agent_color = 'blue' if ii < NN-n_leaders else 'red'
            plt.plot(p_prev[0],p_prev[1], marker='o', markersize=10, fillstyle='full', color = agent_color)
    
    
        x_lim = (np.min(leaders_pos[hull.vertices,0])-1,np.max(leaders_pos[hull.vertices,0])+1)
        y_lim = (np.min(leaders_pos[hull.vertices,1])-1,np.max(leaders_pos[hull.vertices,1])+1)
        # axes_lim = (0,0)
        # plt.axis('equal')
        plt.title("Agents position in $\mathbb{R}^2$")
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.show(block=False)
        plt.pause(0.1)

        plt.clf()

np.random.seed(10)

# Bool vars
ANIMATION = True

#################################################################

NN = 20 # number of agents
n_x = 2 # dimension of x_i 
n_leaders = 5

p_ER = 0.5

I_NN = np.identity(NN, dtype=int)
I_nx = np.identity(n_x, dtype=int)
I_NN_nx = np.identity(n_x*NN, dtype=int)
O_NN = np.ones((NN,1), dtype=int)


# Generate a Connected graph
while 1:
	G = nx.binomial_graph(NN,p_ER)
	Adj = nx.adjacency_matrix(G)
	Adj = Adj.toarray()
  
	# test connectivity
	test = np.linalg.matrix_power((I_NN+Adj),NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break
	else:
		print("the graph is NOT connected\n")


DEGREE = np.sum(Adj,axis=0) 
D_IN = np.diag(DEGREE)
L_IN = D_IN - Adj.T

L_f = L_IN[0:NN-n_leaders,0:NN-n_leaders]
L_fl = L_IN[0:NN-n_leaders,NN-n_leaders:]

# followers dynamics
LL = np.concatenate((L_f, L_fl), axis = 1)

# leaders dynamics
LL = np.concatenate((LL, np.zeros((n_leaders,NN))), axis = 0)

# replicate for each dimension -> kronecker product
LL_kron = np.kron(LL,I_nx)

# Initiate the agents' state
XX_init = np.vstack((np.ones((n_x*n_leaders,1)),np.zeros((n_x*(NN-n_leaders),1))))
XX_init += 5*np.random.rand(n_x*NN,1)

# Consider only the leaders in the B matrix
BB_kron = np.zeros((NN*n_x,n_leaders*n_x))
BB_kron[(NN-n_leaders)*n_x:,:] = np.identity(n_x*n_leaders, dtype=int)

################################################
## followers integral Action

k_i = 2 #2
K_I = - k_i*I_NN_nx

# Setup the extended dynamics
LL_ext_up = np.concatenate((LL_kron, K_I), axis = 1)
LL_ext_low = np.concatenate((LL_kron, np.zeros(LL_kron.shape)), axis = 1)
LL_ext = np.concatenate((LL_ext_up, LL_ext_low), axis = 0)

# extende the initial state with the integral state
XX_init = np.concatenate((XX_init,np.zeros((n_x*NN,1))))
BB_kron = np.concatenate((BB_kron, np.zeros((NN*n_x,n_leaders*n_x))), axis = 0)

A = -LL_ext
B = BB_kron

################################################
# CONTAINMENT Dynamics

dt = 0.005 	# Sampling time
TT = 30.0	# Simulation time
horizon = np.arange(0.0, TT, dt)

XX = np.zeros((A.shape[1],len(horizon)))
XX[:,0] = XX_init.T

# Leaders input: null, sinusoidal
(amp, omega, phi) = (5, 0.5, 0)
UU = waves(amp, omega, phi, horizon, n_x, n_leaders)
# UU = np.zeros((n_leaders*n_x, len(horizon)))

for tt in range(len(horizon)-1):
	XX[:, tt + 1] = XX[:, tt] + dt*(A @ XX[:, tt] + B @ UU[:, tt])


################################################
# Drawings

plt.figure()
label = []
for ii in range(n_x*NN):
  plt.plot(horizon, XX[ii,:])
  label.append('$x_{},{}$'.format(int(ii/2),ii%2))

plt.legend(label)

plt.title("Evolution of the local estimates")
plt.xlabel("$t$")
plt.ylabel("$x_i^t$")

if ANIMATION: 
  if n_x == 2: 
    plt.figure()
    animation(XX, NN, n_x, n_leaders, horizon, dt = 50)
  else:
    print('Animation allowed only for bi-dimensional agent')

plt.show()