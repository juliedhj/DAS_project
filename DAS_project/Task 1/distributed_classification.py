from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Useful constants
N = 10
T = 3  # Layers
d = 784  # Number of neurons in each layer. Same numbers for all the layers
img_num = 50
max_iters = 20 # epochs
stepsize = 0.2 # learning rate
TEST_FLAG = 1 #Flag to set for testing 

data = mnist.load_data()
#Split the dataset
(X_train, y_train), (X_test, y_test) = data


#Reshape the dataset to acces every pixel
X_train = np.reshape(X_train, (-1, 28*28))
X_test = np.reshape(X_test, (-1, 28*28))

#Get value of every pixel as a number between 0 and 1
X_train = X_train / 255
X_test = X_test / 255

#1
# Digit = 4
# Digit 4 -> Label 1
# All other digits -> Label -1

y_train = y_train.copy().astype(np.int8)
y_train = [1 if y==4 else 0 for y in y_train]
y_test = y_test.copy().astype(np.int8)
y_test = [1 if y==4 else 0 for y in y_test]


#2 
#Shuffle training set randomly
p = np.random.permutation(len(X_train)) 
y_train = np.array(y_train, dtype='float32')[p.astype(int)]
X_train = np.array(X_train, dtype='float32')[p.astype(int)]


# #Split training set in N subsets
def splitSet(n):
    subsets_X = np.array_split(X_train, n)
    subsets_y = np.array_split(y_train, n)
    return np.asarray(subsets_X), np.asarray(subsets_y)

#3 
#Distributed Gradient tracking 

# Training Set
(data_arrays, label_arrays) = splitSet(N)
print(data_arrays.shape)

###############################################################################
# Activation Function
def sigmoid(xi):
    return 1 / (1 + np.exp(-xi))

# Derivative of Activation Function
def sigmoid_derivative(xi):
    return sigmoid(xi) * (1 - sigmoid(xi))

def forward_pass(uu, x0):
    xx = np.zeros((T,d))
    xx[0] = x0
    # propagate the xx from 0 to T-1 by running the inference dynamics
    for t in range(T - 2):
        for el in range(d): #el = neurons in the layers
            # update term
            temp = (xx[t] @ uu[t, el, 1:]) + uu[t, el, 0]  
            xx[t+1, el] = sigmoid(temp)  # x(t+1) = f(x(t)' * u_ell), f = sigmoid

    # output layer: only one neuron is computed (the others output zero)
    xx[T-1, 0] = sigmoid(xx[T-2] @ uu[T-2, 0, 1:] + uu[T-2, 0, 0])
    # thresholded value
    return xx
  
# Adjoint dynamics: 
#   state:    lambda_t = A.T lambda_tp
#   output: deltau_t = B.T lambda_tp
def adjoint_dynamics(ltp, xt, ut):
    
    df_dx = np.zeros((d, d))
    # df_du = np.zeros((d,(d+1)*d))
    delta_uut = np.zeros((d, d+1))

    for j in range(d):
        dsigma_j = sigmoid_derivative((xt@ut[j, 1:]) + ut[j, 0])

        df_dx[:, j]  = ut[j, 1:] * dsigma_j
        # dfu[hh, xx] = dsigma_j*np.hstack([1,xxt])

        # B'@ltp
        delta_uut[j, 0]  = ltp[j] * dsigma_j
        delta_uut[j, 1:] = xt * ltp[j] * dsigma_j

    lt = df_dx@ltp  # A'@ltp

    return lt, delta_uut

# Backward Propagation
def backward_pass(xx, uu, llambdaT):
    
    llambda = np.zeros((T, d))
    llambda[-1,0] = llambdaT
    Delta_u = np.zeros((T-1, d, d+1))

    # Compute the first value of the costate
    dfx = np.zeros((d, d))

    dsigma_T = sigmoid_derivative((xx[T-2]@uu[T-2, 0, 1:]) + uu[T-2, 0, 0])

    dfx[:, 0]  = uu[T-2, 0, 1:] * dsigma_T

    Delta_u[0, 0]  = llambdaT * dsigma_T
    Delta_u[0, 1:] = xx[T-2, 0] * llambdaT * dsigma_T
    llambda[T-2] = dfx @ llambda[-1]

    for t in reversed(range(T - 2)): # T-2 ... 1 0
        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t])
    return Delta_u

def current_cost(y_pred, y_true):
    cost_current = (y_true - y_pred)**2
    return cost_current

#Generate the network 
I_N = np.identity(N, dtype=int)
p_ER = 0.3
while 1:
    neigh = np.random.binomial(1, p_ER, (N,N))
    neigh = np.logical_or(neigh,neigh.T)
    neigh = np.multiply(neigh,np.logical_not(I_N)).astype(int)

    test = np.linalg.matrix_power((I_N+neigh),N)
    
    if np.all(test>0):
        print("the graph is connected\n")
        break 
    else:
        print("the graph is NOT connected\n")
        quit()

#Compute the mixing matrices
W = np.zeros((N,N))

for i in range(N):
    N_i = np.nonzero(neigh[i])[0] # In-Neighbors of node i
    deg_i = len(N_i)

    for j in N_i:
        N_j = np.nonzero(neigh[j])[0] # In-Neighbors of node j
        # deg_jj = len(N_jj)
        deg_j = N_j.shape[0]

        W[i,j] = 1 / (1 + max([deg_i,deg_j]))
        # WW[i,jj] = 1/(1+np.max(np.stack((deg_i,deg_jj)) ))

W += I_N - np.diag(np.sum(W,axis=0))

    
###############################################################################
# MAIN
###############################################################################

J = np.zeros((N, max_iters))                       # Cost

# Initial Weights / Initial Input Trajectory
uu = (1/d)*np.random.randn(N, T-1, d, d+1)
zz = np.zeros((N, T-1, d, d+1))
xx = np.zeros((N, T, d))
Delta_u = np.zeros((N, img_num, T-1, d, d+1))
agent_weights = np.zeros((N,max_iters, 4))
gradient_evolution = np.zeros((N, max_iters, img_num, T-1, d, d+1))

# Want 1 neuron with weights in the output layer
# for j in range(1,d):
#     uu[-1,j] = 0

# GO!
for k in range(max_iters - 1):
      #print('Cost at k={:d} is {:.4f}'.format(k,J[k-1]))
    for i in range(N): 
        # Initial State Trajectory
        print("i: ", i)
        neigh_i = np.nonzero(neigh[i])[0]
      
        for img in range(img_num):
            data_point = data_arrays[ i, img]
            label_point = label_arrays[i, img]
            
            # Forward propagation
            xx[i] = forward_pass(uu[i], data_point)
      
            xx_out = xx[i, T-1, 0]
            
            # Backward propagation
            llambdaT = 2 * (xx[i, T-1, 0] - label_point)
            Delta_u[i, img] = backward_pass(xx[i],uu[i],llambdaT) # the gradient of the loss function
      
            J[i, k] += current_cost(xx[i, T-1, 0], label_point)

        gradient_evolution[i, k] = Delta_u[i]
        print(f"\r Current cost [{k + 1}][Agent {i + 1}] -> {J[i, k]:.2f}", end=' ')
        # Update the weights
        # Want to find a common u for all agents, where u is calculated over the sum of the 
        # optimized u (with gradient) for all agents i
        #uu = uu - stepsize*Delta_u # overwriting the old value
        uu[i] = W[i, i] * uu[i] + zz[i] - stepsize * np.sum(Delta_u[i], axis=0)
        zz[i] = W[i, i] * zz[i] - stepsize * np.sum(Delta_u[i], axis=0) * (W[i,i] - 1)

        for j in neigh_i:
            uu[i] += W[i,j] * uu[j]
            zz[i] += W[i,j] * (zz[j] - stepsize * np.sum(Delta_u[j], axis=0))

        for el in range(4):
            agent_weights[i, k, el] = uu[i, T-2, el, 0]
        print("weights updated")
        #The loss function for classification problems with (0,1) classes - Binary Cross Entropy 
        # Store the Loss Value across Iterations
J[:, -1] = J[:, k]

sum_cost = np.zeros((max_iters))
for k in range(max_iters):
  for i in range(N):
    sum_cost[k] += J[i,k]


VALUE = 1
OTHER = 0
#4 
#Compute the accuracy of the solution
tot_accuracy = np.zeros(N)
value_accuracy = np.zeros(N)
other_accuracy = np.zeros(N)
count_value = 0 
count_other = 0
threshold = 0.5 
N_test = int(np.floor(len(y_test)/100))

if TEST_FLAG: 
    for el in range(N_test):
        if y_test[el] == VALUE:
            count_value += 1
        else: 
            count_other += 1 
    print(count_value, count_other)
    
    for i in range(N):
        correct_value = 0
        correct_other = 0 
        for img in range(N_test):
        #current agent's estimate of img
            xx_img = forward_pass(uu[i], X_test[img])
            if y_test[img] == VALUE: 
                if xx_img[T-1, 0] >= threshold:
                    correct_value += 1 
            elif y_test[img] == OTHER:
                if xx_img[T-1, 0] < threshold:
                    correct_other += 1
        print(correct_value, correct_other)

        #Calculate the accuracies
        value_accuracy[i] = correct_value / count_value
        other_accuracy[i] = correct_other / count_other
        tot_accuracy[i] = (correct_other + correct_value) / N_test

    print(f"AGENT ACCURACY \n"
          f"Number of test samples per agent: {N_test} \n" 
          f"----- TOTAL ------")
    for i in range(N):
        print(f"Agent [{i+1}]: {tot_accuracy[i]:.3f}")

#PLOT THE SIMULATIONS 
# _,ax = plt.subplots()
# ax.plot(range(max_iters),J)
# plt.show()

colors = {}
for i in range(N):
    colors[i] = np.random.rand(3)

#Plot the evolution of the cost 
plt.figure()
for i in range(N):
    plt.plot(np.arange(max_iters), sum_cost, color=colors[i])
plt.xlabel(r"iterations $k$")
plt.ylabel(r"$J_i^k$")
plt.yscale("log")
plt.title("Evolution of the total cost functions")
plt.grid()
plt.show()

#Plot the sum of the weights 
plt.figure()
for i in range(N):
    plt.plot(np.arange(max_iters-1), agent_weights[i, 0:max_iters-1], color=colors[i])
plt.xlabel(r"Iterations $k$")
plt.ylabel(r"$u_i$ - Weights of agents")
#plt.yscale("log")
plt.title("Evolution of the weights for 4 neurons for the agents")
plt.grid()
plt.show()

# Plot the norm of the gradient 
# plt.figure()
# for i in range(N):
#     for k in range(max_iters): 
#         plt.plot(k, np.linalg.norm(gradient_evolution[i, k]), color=colors[i])
# plt.xlabel(r"Iterations $k$")
# plt.ylabel(r"Norm of the gradients")
# #plt.yscale("log")
# plt.title("Sum of the weights for each iteration for the agents")
# plt.grid()
# plt.show()
