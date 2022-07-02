from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
import tensorflow as tf

# Useful constants
MAXITERS = np.int(784) 
N = 10
data = mnist.load_data()

#Split the dataset
(X_train, y_train), (X_test, y_test) = data


#Reshape the dataset to acces every pixel
X_train = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')

#Get value of every pixel as a number between 0 and 1
X_train = X_train / 255
X_test = X_test / 255


#1
# Digit = 4
# Digit 4 -> Label 1
# All other digits -> Label -1

y_train = [1 if y==4 else 0 for y in y_train]
y_test = np.array([1 if y==4 else 0 for y in y_test], dtype='float32')

#2 
#Shuffle training set randomly
p = np.random.permutation(len(X_train)) 
y_train = np.array(y_train, dtype='float32')[p.astype(int)]
X_train = np.array(X_train, dtype='float32')[p.astype(int)]

#Split training set in N subsets
def splitSet(n):
    subsets_X = np.array_split(X_train, n)
    subsets_y = np.array_split(y_train, n)
    return subsets_X, subsets_y


#3 Gradient Tracking Algorithm -> distributed algorithm because we only have to exchange local data
#backward propagation -> gradient
# gradient from keras for tracking

def get_network() -> Sequential:
    network = Sequential()
    network.add(Input(784,))
    network.add(Dense(32, activation='relu', name="dense32"))
    network.add(Dense(16, activation='relu', name="dense16" ))
    network.add(Dense(1, activation='sigmoid', name="dense1")) #aktive = true, not active = false

    network.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
    return network


def trainSet(n):
    #n is number of agents
    (subsets_x, subsets_y) = splitSet(n)
    for i in range(0,n):
        #network.fit(np.asarray(subsets_x[i]), np.asarray(subsets_y[i]), epochs=10, batch_size=100)
        network = get_network()
        network.fit(subsets_x[i], subsets_y[i], validation_data=(X_test, y_test), epochs=5, batch_size=128)
        network.summary()


def gradient(model, x):
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        loss = model(x_tensor)
    return t.gradient(loss, x_tensor).numpy()

trainSet(N)

###############################################################################
#Gradient Tracking algorithm

np.random.seed(0)

# Quadratic Function
#not needed?
def quadratic_fn(x,Q,r):
	
	fval = 0.5*x*Q*x+r*x
	#call on the gradient() method here?
	fgrad = Q*x+r
	return fval, fgrad



# Generate Network Bynomial Graph
I_N = np.identity(N, dtype=int)
p_ER = 0.3
while 1:
	Adj = np.random.binomial(1, p_ER, (N,N))
	Adj = np.logical_or(Adj,Adj.T)
	Adj = np.multiply(Adj,np.logical_not(I_N)).astype(int)

	test = np.linalg.matrix_power((I_N+Adj),N)
	
	if np.all(test>0):
		print("the graph is connected\n")
		break 
	else:
		print("the graph is NOT connected\n")
		quit()


###############################################################################
# Compute mixing matrices
# Metropolis-Hastings method to obtain a doubly-stochastic matrix
W = np.zeros((N,N))

for i in range(N):
	N_i = np.nonzero(Adj[i])[0] # In-Neighbors of node i
	deg_i = len(N_i)

	for jj in N_i:
		N_jj = np.nonzero(Adj[jj])[0] # In-Neighbors of node j
		# deg_jj = len(N_jj)
		deg_jj = N_jj.shape[0]

		W[i,jj] = 1/(1+max( [deg_i,deg_jj] ))
		# WW[i,jj] = 1/(1+np.max(np.stack((deg_i,deg_jj)) ))

W += I_N - np.diag(np.sum(W,axis=0))
	
with np.printoptions(precision=4, suppress=True):
	print('Check Stochasticity\n row:    {} \n column: {}'.format(
  	np.sum(W,axis=1),
		np.sum(W,axis=0)
	))

###############################################################################
# Declare Cost Variables
Q = 10*np.random.rand(N)
# Q = Q + Q.T
R = 10*(np.random.rand(N)-1)

xopt = -np.sum(R)/np.sum(Q)
fopt = 0.5*xopt*np.sum(Q)*xopt+np.sum(R)*xopt
print('The optimal cost is: {:.4f}'.format(fopt))

# Declare Algorithmic Variables
X = np.zeros((N,MAXITERS))
#X_dg = np.zeros((N,MAXITERS)) # distributed gradient
Y = np.zeros((N,MAXITERS))
X_dg = gradient(get_network(), X_train) # distributed gradient


#X_init = 10*np.random.rand(N)
#X[:,0] = X_init
#X_dg[:,0] = X_init

for i in range (N):
	_, Y[i,0] = quadratic_fn(X[i,0],Q[i],R[i])

FF = np.zeros((MAXITERS))
FF_dg = np.zeros((MAXITERS))
	 
#GO
s = 1e-2 # stepsize

for t in range (MAXITERS-1): #iterate loop
	totalcost = 0 #store variable

	if (t % 50) == 0:
		print("Iteration {:3d}".format(t), end="\n")
		
	(subsets_x, subsets_y) = splitSet(N)
	for i in range (N): #iterate over nodes

		for k in range(len(subsets_x)): #iterate over each image
		#store image and label 
			label = subsets_y[i][k]
			image = subsets_x[i][k]

		#forward pass 
		#forwards av image

		#BCE from slides@		
		#totalcost += cost


		#backward propogatiom
		Ni = np.nonzero(Adj[i])[0]
		
		X[i,t+1] = W[i,i]*X[i,t] - s*Y[i,t]
		for j in Ni:
			X[i,t+1] += W[i,j]*X[j,t]

		f_i, grad_fi = quadratic_fn(X[i,t],Q[i],R[i])
		_, grad_fi_p = quadratic_fn(X[i,t+1],Q[i],R[i])
		Y[i,t+1] = W[i,i]*Y[i,t] +(grad_fi_p - grad_fi)
		for j in Ni:
			Y[i,t+1] += W[i,j]*Y[j,t]

		FF[t] +=f_i

		# Distributed Gradient for Comparison
		f_i_dg, grad_fi_dg = quadratic_fn(X_dg[i,t],Q[i],R[i])
		FF_dg[t] +=f_i_dg
		X_dg[i,t+1] = W[i,i]*X_dg[i,t] - s*grad_fi_dg
		# X_dg[i,t+1] = W[i,i]*X_dg[i,t] - s*grad_fi_dg/(t+1)*10 # diminishing
		for j in Ni:
			X_dg[i,t+1] += W[i,j]*X_dg[j,t]

#for each node, update the weights 
# Terminal iteration
	for i in range (N):
		f_i, _ = quadratic_fn(X[i,-1],Q[i],R[i])
		FF[-1] += f_i
		f_i_dg, _ = quadratic_fn(X_dg[i,-1],Q[i],R[i])
		FF_dg[-1] += f_i_dg


###############################################################################
# generate N random colors
colors = {}
for i in range(N):
	colors[i] = np.random.rand(3)

###############################################################################
# Figure 1 : Evolution of the local estimates
if 1:
	plt.figure()
	plt.plot(np.arange(MAXITERS), np.repeat(xopt,MAXITERS), '--', linewidth=3)
	for i in range(N):
		plt.plot(np.arange(MAXITERS), X[i], color=colors[i])
		
	plt.xlabel(r"iterations $t$")
	plt.ylabel(r"$x_i^t$")
	plt.title("Evolution of the local estimates")
	plt.grid()

###############################################################################
# Figure 2 : Cost Evolution
if 1:
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
