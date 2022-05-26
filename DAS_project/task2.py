import numpy as np 
import matplotlib.pyplot as mpl
import networkx as nx
#Formation control

#1 Discrete-time version model
n = 4 #Agents
d = 4
V =[ *range(1,n+1,1)]
n_l = 2
leaders = V[0:n_l]
followers = V[n_l:]

p_t = [(1,2),(3,4),(3,2),(2,5)] #Position
v_t = [1,5,2,5] #Velocity
u_t = [3,4,2,1] #Acceleration

p_t_prime = v_t 
v_t_prime = u_t  

I_N = np.identity(d, dtype=int)

#E = edges
#Graph
#G = (V, E)
p_ER = 0.5

# Generate a Connected graph
while 1:
	G = nx.binomial_graph(n,p_ER)
	Adj = nx.adjacency_matrix(G)
	Adj = Adj.toarray()
  
	# test connectivity
	test = np.linalg.matrix_power((I_N+Adj),n)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break
	else:
		print("the graph is NOT connected\n")
x_t = (p_t, v_t) 

print(G)

#Bearing of agent j relative to agent i
g_ij = []

for i in p_t:
    for j in p_t:
        g_ij[i][j] = (p_t[j]-p_t[i])/len(p_t[j]-p_t[i])
        

#P_g = I_N - g_ij*np.transpose(g_ij)

#2 ROS 2 

