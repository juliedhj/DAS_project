import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


dd = 1

# Quadratic Function
def quadratic_fn(x,q,r):

	fval = 0.5*q*x**2+r*x
	fgrad = q*x + r
	return fval,fgrad
	


###############################################################################
# Useful constants
MAXITERS = int(1e4) # Explicit Casting
NN = 5

###############################################################################
# Generate Network Binomial Graph
p_ER = 0.3
I_NN = np.eye(NN) # np.identity
while 1:
	Adj = np.random.binomial(1,p_ER, (NN,NN))
	Adj = np.logical_or(Adj,Adj.T)
	Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)

	test = np.linalg.matrix_power(I_NN+Adj, NN)
	if np.all(test>0):
		break

###############################################################################
# Compute mixing matrix
WW = 1.5*I_NN + 0.5*Adj

ONES = np.ones((NN,NN))
ZEROS = np.zeros((NN,NN))

while any(abs( np.sum(WW,axis=1)-1) )>10e-10:

	WW = WW/(WW@ONES)
	WW = WW/(ONES@WW)
	WW = np.maximum(WW,0)


###############################################################################
# Declare Cost Variables
Q = 10*np.random.rand(NN)
R = 10*(np.random.rand(NN)-0.5)

# Compute Optimal Solution
xopt = -np.sum(R)/np.sum(Q)
fopt,_ = quadratic_fn(xopt,np.sum(Q),np.sum(R))
print(fopt)

# Declare Algorithmic Variables
XX = np.zeros((NN,MAXITERS))

XX_init = np.random.rand(NN)
XX[:,0] = XX_init # print('Shape:\n {}'.format(XX[:,0].shape))
FF = np.zeros((MAXITERS))

###############################################################################
# GO!
stepsize = 1e-3 # Constant Stepsize

for kk in range (MAXITERS-1):
	# stepsize = 1/(kk+1) # Diminishing Stepsize

	if (kk % 10) == 0:
		print("Iteration {:3d}".format(kk), end="\n")
	
	for ii in range (NN):
		Nii = np.nonzero(Adj[ii])[0]

		for jj in Nii:
			XX[ii,kk+1] += WW[ii,jj]*XX[jj,kk]
		XX[ii,kk+1] += WW[ii,ii]*XX[ii,kk]

		_,grad_f_i = quadratic_fn( XX[ii,kk+1], Q[ii], R[ii] )
		XX[ii,kk+1] += -stepsize*grad_f_i

		f_ii,_ = quadratic_fn( XX[ii,kk+1], Q[ii], R[ii] )
		FF[kk+1] += f_ii

		
		

###############################################################################
# generate N random colors
colors = {}
for ii in range(NN):
	colors[ii] = np.random.rand(3)

###############################################################################
# Figure 1: Evolution of the local estimates
plt.figure()
plt.plot(np.arange(MAXITERS), np.repeat(xopt,MAXITERS), '--', linewidth=3)
for ii in range(NN):
	plt.plot(np.arange(MAXITERS), XX[ii], color=colors[ii])
	
plt.xlabel(r"iterations $k$")
plt.ylabel(r"$x_i^k$")
plt.title(r"Evolution of the local estimates")
plt.grid()


###############################################################################
# Figure 2: Evolution of the cost
plt.figure()
plt.semilogy(np.arange(MAXITERS), np.abs(FF-fopt), '--', linewidth=3)	
plt.xlabel(r"iterations $k$")
plt.ylabel(r"$\sum_{i=1}^N f_i(x_i^k)$")
plt.title(r"Evolution of the cost")
plt.grid()

plt.show()