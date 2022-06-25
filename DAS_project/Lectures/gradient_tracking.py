#
# Gradient Tracking Scalar Case
# Ivano Notarnicola
# Bologna, 18/05/2022
#
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Quadratic Function
def quadratic_fn(x,Q,r):
	if not np.isscalar(Q) or not np.isscalar(r):
		print('Error')
		exit(0)

	fval = 0.5*x*Q*x+r*x
	fgrad = Q*x+r
	return fval, fgrad


###############################################################################
# Useful constants
MAXITERS = np.int(1e3) # Explicit Casting
NN = 10

###############################################################################
# Generate Network Bynomial Graph
I_NN = np.identity(NN, dtype=int)
p_ER = 0.3
while 1:
	Adj = np.random.binomial(1, p_ER, (NN,NN))
	Adj = np.logical_or(Adj,Adj.T)
	Adj = np.multiply(Adj,np.logical_not(I_NN)).astype(int)

	test = np.linalg.matrix_power((I_NN+Adj),NN)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break 
	else:
		print("the graph is NOT connected\n")
		quit()

###############################################################################
# Compute mixing matrices
# Metropolis-Hastings method to obtain a doubly-stochastic matrix
WW = np.zeros((NN,NN))

for i in range(NN):
	N_i = np.nonzero(Adj[i])[0] # In-Neighbors of node i
	deg_i = len(N_i)

	for jj in N_i:
		N_jj = np.nonzero(Adj[jj])[0] # In-Neighbors of node j
		# deg_jj = len(N_jj)
		deg_jj = N_jj.shape[0]

		WW[i,jj] = 1/(1+max( [deg_i,deg_jj] ))
		# WW[i,jj] = 1/(1+np.max(np.stack((deg_i,deg_jj)) ))

WW += I_NN - np.diag(np.sum(WW,axis=0))
	
with np.printoptions(precision=4, suppress=True):
	print('Check Stochasticity\n row:    {} \n column: {}'.format(
  	np.sum(WW,axis=1),
		np.sum(WW,axis=0)
	))

# quit(0)
###############################################################################
# Declare Cost Variables
Q = 10*np.random.rand(NN)
# Q = Q + Q.T
R = 10*(np.random.rand(NN)-1)

xopt = -np.sum(R)/np.sum(Q)
fopt = 0.5*xopt*np.sum(Q)*xopt+np.sum(R)*xopt
print('The optimal cost is: {:.4f}'.format(fopt))

# Declare Algorithmic Variables
XX = np.zeros((NN,MAXITERS))
XX_dg = np.zeros((NN,MAXITERS)) # distributed gradient
YY = np.zeros((NN,MAXITERS))

XX_init = 10*np.random.rand(NN)
XX[:,0] = XX_init
XX_dg[:,0] = XX_init

for i in range (NN):
	_, YY[i,0] = quadratic_fn(XX[i,0],Q[i],R[i])

FF = np.zeros((MAXITERS))
FF_dg = np.zeros((MAXITERS))

###############################################################################
# GO!
ss = 1e-2 # stepsize

for t in range (MAXITERS-1):
	if (t % 50) == 0:
		print("Iteration {:3d}".format(t), end="\n")
	
	for i in range (NN):
		Ni = np.nonzero(Adj[i])[0]
		
		XX[i,t+1] = WW[i,i]*XX[i,t] - ss*YY[i,t]
		for jj in Ni:
			XX[i,t+1] += WW[i,jj]*XX[jj,t]

		f_i, grad_fi = quadratic_fn(XX[i,t],Q[i],R[i])
		_, grad_fi_p = quadratic_fn(XX[i,t+1],Q[i],R[i])
		YY[i,t+1] = WW[i,i]*YY[i,t] +(grad_fi_p - grad_fi)
		for jj in Ni:
			YY[i,t+1] += WW[i,jj]*YY[jj,t]

		FF[t] +=f_i

		# Distributed Gradient for Comparison
		f_i_dg, grad_fi_dg = quadratic_fn(XX_dg[i,t],Q[i],R[i])
		FF_dg[t] +=f_i_dg
		XX_dg[i,t+1] = WW[i,i]*XX_dg[i,t] - ss*grad_fi_dg
		# XX_dg[i,t+1] = WW[i,i]*XX_dg[i,t] - ss*grad_fi_dg/(t+1)*10 # diminishing
		for jj in Ni:
			XX_dg[i,t+1] += WW[i,jj]*XX_dg[jj,t]

# Terminal iteration
for i in range (NN):
	f_i, _ = quadratic_fn(XX[i,-1],Q[i],R[i])
	FF[-1] += f_i
	f_i_dg, _ = quadratic_fn(XX_dg[i,-1],Q[i],R[i])
	FF_dg[-1] += f_i_dg


###############################################################################
# generate N random colors
colors = {}
for i in range(NN):
	colors[i] = np.random.rand(3)

###############################################################################
# Figure 1 : Evolution of the local estimates
if 0:
	plt.figure()
	plt.plot(np.arange(MAXITERS), np.repeat(xopt,MAXITERS), '--', linewidth=3)
	for i in range(NN):
		plt.plot(np.arange(MAXITERS), XX[i], color=colors[i])
		
	plt.xlabel(r"iterations $t$")
	plt.ylabel(r"$x_i^t$")
	plt.title("Evolution of the local estimates")
	plt.grid()

###############################################################################
# Figure 2 : Cost Evolution
if 0:
	plt.figure()
	plt.plot(np.arange(MAXITERS), np.repeat(fopt,MAXITERS), '--', linewidth=3)
	plt.plot(np.arange(MAXITERS), FF)
	plt.xlabel(r"iterations $t$")
	plt.ylabel(r"$x_i^t$")
	plt.title("Evolution of the cost")
	plt.grid()

###############################################################################
# Figure 3 : Cost Error Evolution
if 1:
	plt.figure()
	plt.semilogy(np.arange(MAXITERS), np.abs(FF-np.repeat(fopt,MAXITERS)), '--', linewidth=3)
	plt.semilogy(np.arange(MAXITERS), np.abs(FF_dg-np.repeat(fopt,MAXITERS)), '--', linewidth=3)
	plt.xlabel(r"iterations $t$")
	plt.ylabel(r"$|\sum_{i=1}^N f_i(x_i^t) - f^\star|$")
	plt.title("Evolution of the cost error")
	plt.grid()

plt.show()