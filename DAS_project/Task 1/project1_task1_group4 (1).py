import random
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

###############################################################################
TT = 3   # Layers
dd = 784 # Number of neurons in each layer. Same numbers for all the layers
# Distributed network parameters
NN_train = 60000
NN = 5    # Number of agents
mm_i = 25 # Number of images for each agent
# Dataset parameters
TRUE_VALUE = 1  # Binary values of yy_hat
FALSE_VALUE = 0
digit_to_recognize = 1
# Gradient Tracking Method Parameters
max_iters = 15  # Epochs
# Heuristics obtained through experience:
# it doesn't seem to be that precise, but there is a relation between the number of images and the stepsize
# as images mm_i increase, class 2 will be preferred, as stepsize increases, class 1 will be preferred
stepsize = 0.35

# FLAGS
TEST_SAMPLES = 1

if TT < 3:
    print("Error, minimum number of layers is 3\n")
    quit()
###############################################################################

# Activation Function (Sigmoid)
def sigmoid(zz):
    return 1 / (1 + np.exp(-zz))

# Derivative of Activation Function
def dsigmoid(zz):
    return sigmoid(zz) * (1 - sigmoid(zz))

# Forward Propagation
def forward_dynamics(uu, xx0):
    """
      input:
                uu input trajectory: u[0],u[1],..., u[T-1]
                x0 initial condition
      output:
                xx state trajectory: x[1],x[2],..., x[T]
    """
    xx = np.zeros((TT,dd))
    xx[0] = xx0
    # propagate the xx from 0 to T-1 by running the inference dynamics
    for tt in range(TT - 2):
        for hh in range(dd):
            # update term
            temp = (xx[tt] @ uu[tt, hh, 1:]) + uu[tt, hh, 0]  # including the bias
            xx[tt + 1, hh] = sigmoid(temp)  # x(t+1) = f(x(t)' * u_ell), f = sigmoid

    # output layer: only one neuron is computed (the others output zero)
    xx[TT-1, 0] = sigmoid(xx[TT - 2] @ uu[TT - 2, 0, 1:] + uu[TT - 2, 0, 0])
    # thresholded value
    return xx

# Adjoint dynamics:
#   state:    lambda_t = A.T lambda_tp
#   output: deltau_t = B.T lambda_tp
def adjoint_dynamics(lltp, xxt, uut):
    """
      input:
                llambda_tp current costate
                xt current state
                ut current input
      output:
                llambda_t next costate
                delta_ut loss gradient wrt u_t
    """
    dfx = np.zeros((dd, dd))
    dfu = np.zeros((dd,(dd+1)*dd))
    # bias to be considered
    delta_uut = np.zeros((dd, dd + 1))

    for hh in range(dd):
        dsigma_j = dsigmoid((xxt@uut[hh, 1:]) + uut[hh, 0])

        dfx[:, hh]  = uut[hh, 1:] * dsigma_j
        # add first element at the beginning of xxt (horizontally)
        # dfu[hh, xx] = dsigma_j*np.hstack([1,xxt])

        # B'@ltp
        delta_uut[hh, 0]  = lltp[hh] * dsigma_j
        delta_uut[hh, 1:] = xxt * lltp[hh] * dsigma_j

    llt = dfx@lltp  # A'@ltp

    return llt, delta_uut

# Backward Propagation
def backward_pass(xx, uu, llambdaT):
    """
      input:
                xx state trajectory: x[1],x[2],..., x[T]
                uu input trajectory: u[0],u[1],..., u[T-1]
                llambdaT terminal condition
      output:
                llambda costate trajectory
                delta_u costate output, i.e., the loss gradient
    """
    llambda = np.zeros((TT, dd))
    llambda[-1,0] = llambdaT
    delta_uu = np.zeros((TT-1, dd, dd+1))

    # Compute the first value of the costate
    dfx = np.zeros((dd, dd))

    dsigma_T = dsigmoid((xx[TT-2]@uu[TT-2, 0, 1:]) + uu[TT-2, 0, 0])

    dfx[:, 0]  = uu[TT-2, 0, 1:] * dsigma_T

    delta_uu[0, 0]  = llambdaT * dsigma_T
    delta_uu[0, 1:] = xx[TT-2, 0] * llambdaT * dsigma_T
    llambda[TT-2] = dfx @ llambda[-1]

    for tt in reversed(range(TT - 2)): # T-2 ... 1 0
        llambda[tt], delta_uu[tt] = adjoint_dynamics(llambda[tt + 1], xx[tt], uu[tt])
    return delta_uu

# Computes the cost function given the output error
def get_current_cost(yy_est, yy_true):
    Cost = (yy_true - yy_est)**2
    return Cost

###############################################################################
# MAIN
###############################################################################

# Training Set: (xx_train, yy_train)
# Test set: (xx_test,yy_test)
(xx_train, yy_train), (xx_test, yy_test) = keras.datasets.mnist.load_data()
normalize_value = 1/255
xx_train = normalize_value * np.reshape(xx_train, (-1, 784))
xx_test  = normalize_value * np.reshape(xx_test , (-1, 784))
yy_test = yy_test.copy().astype(np.int8)
yy_train_task_1 = yy_train.copy().astype(np.int8)
'''for image, label in zip(xx_train,yy_train):
    cv2.imshow(f'{label}',image)
    cv2.waitKey(0)'''

# "Rework" of the training set in order to have a binary classification problem
if digit_to_recognize < 0 or digit_to_recognize > 9:
    print('Error, invalid digit to recognize')
for i, elem in enumerate(yy_train_task_1):
    if elem == digit_to_recognize:
        yy_train_task_1[i] = TRUE_VALUE
    else:
        yy_train_task_1[i] = FALSE_VALUE

for i, elem in enumerate(yy_test):
    if elem == digit_to_recognize:
        yy_test[i] = TRUE_VALUE
    else:
        yy_test[i] = FALSE_VALUE

#Task 1.2
xx_train_split = np.zeros((dd, NN, mm_i))
yy_train_split = np.zeros((NN, mm_i))
index_list = list(range(NN_train))
random.shuffle(index_list)
index_list = np.array_split(index_list, NN)

temp_xx = np.zeros((TT, dd, mm_i))
temp_yy = np.zeros((    dd, mm_i))

# Random split of the training set into N subsets
for ii in range(NN):
    agent_idx = 0
    for split_idx in index_list[ii][:mm_i]:
        temp_xx = xx_train[split_idx, :]
        temp_yy = yy_train_task_1[split_idx]
        # "x_i(t=0)" set equal to the data
        xx_train_split[:, ii, agent_idx] = temp_xx
        yy_train_split[ii, agent_idx] = temp_yy
        agent_idx = agent_idx + 1
###############################################################################
# Generate the Network of the agents
I_NN = np.identity(NN, dtype=int)
p_ER = 0.3
while 1:
    Adj = np.random.binomial(1, p_ER, (NN, NN))
    Adj = np.logical_or(Adj, Adj.T)
    Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)

    test = np.linalg.matrix_power((I_NN + Adj), NN)

    if np.all(test > 0):
        print("the graph is connected\n")
        break
    else:
        print("the graph is NOT connected\n")
        quit()
###############################################################################
# Compute mixing matrices
# Metropolis-Hastings method to obtain a doubly-stochastic matrix
WW = np.zeros((NN, NN))

for ii in range(NN):
    N_ii = np.nonzero(Adj[ii])[0]  # In-Neighbors of node i
    deg_ii = len(N_ii)

    for jj in N_ii:
        N_jj = np.nonzero(Adj[jj])[0]  # In-Neighbors of node j
        # deg_jj = len(N_jj)
        deg_jj = N_jj.shape[0]

        WW[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))
    # WW[ii,jj] = 1/(1+np.max(np.stack((deg_ii,deg_jj)) ))

WW += I_NN - np.diag(np.sum(WW, axis=0))

with np.printoptions(precision=4, suppress=True):
    print('Check Stochasticity\n row:    {} \n column: {}'.format(
        np.sum(WW, axis=1),
        np.sum(WW, axis=0)
    ))

###############################################################################
# GO!
uu = (1/dd)*np.random.randn(NN, TT-1, dd, dd+1)
zz = np.zeros((NN, TT-1, dd, dd+1))
xx = np.zeros((NN, TT, dd))
Delta_u = np.zeros((NN, mm_i, TT-1, dd, dd+1))

JJ = np.zeros((NN, max_iters))
for kk in range(max_iters - 1):
    # distributed computation (gradient tracking)
    for ii in range(NN):
        Neigh_ii = np.nonzero(Adj[ii])[0]
        # loop through the images
        for ii_d in range(mm_i):
            # Current data (analogous to a single sample test with single agent)
            xx_current = xx_train_split[:, ii, ii_d]
            yy_current = yy_train_split[ii, ii_d]
            # Initial State Trajectory (of current agent)
            xx[ii] = forward_dynamics(uu[ii], xx_current)  # T x d

            xx_out = xx[ii, TT-1, 0]

            llambdaT = 2 * (xx_out - yy_current)

            Delta_u[ii, ii_d] = backward_pass(xx[ii], uu[ii], llambdaT)  # the gradient of the loss function

            # Store the Loss Value across Iterations
            dJJ = get_current_cost(xx_out, yy_current)
            JJ[ii, kk] += dJJ # it is the cost at k+1

        print(f"\r Current cost [{kk + 1}][Agent {ii + 1}] -> {JJ[ii, kk]:.2f}", end=' ')
        # Update the weights
        uu[ii] = WW[ii, ii] * uu[ii] + zz[ii] - stepsize * np.sum(Delta_u[ii], axis=0)
        zz[ii] = WW[ii, ii] * zz[ii] - stepsize * np.sum(Delta_u[ii], axis=0) * (WW[ii,ii] - 1)

        for jj in Neigh_ii:
            uu[ii] += WW[ii,jj] * uu[jj]
            zz[ii] += WW[ii,jj] * (zz[jj] - stepsize * np.sum(Delta_u[jj], axis=0))

        # Forward propagation
        # xx[ii] = forward_dynamics(uu[ii], xx_current)

JJ[:,-1] = JJ[:,kk]


###############################################################################
# compute accuracies
c1_acc = np.zeros(NN)
c2_acc = np.zeros(NN)
total_acc  = np.zeros(NN)
c1_NN = 0
c2_NN = 0
THRESHOLD_VALUE = 0.5
NN_TEST = int(np.floor(len(yy_test)/100))

if TEST_SAMPLES:
    for hh in range(NN_TEST):
        if yy_test[hh] == TRUE_VALUE:
            c1_NN += 1
        else:
            c2_NN += 1

    for ii in range(NN):
        c1_correct = 0
        c2_correct = 0
        for hh in range(NN_TEST):
            # current agent's estimation of test image h
            xx_hh = forward_dynamics(uu[ii], xx_test[hh])
            # threshold at zero, if xx_h close to 1 -> y_hat = 1
            if yy_test[hh] == TRUE_VALUE:
                if xx_hh[TT-1, 0] >= THRESHOLD_VALUE:
                    c1_correct += 1
            elif yy_test[hh] == FALSE_VALUE:
                if xx_hh[TT-1, 0] < THRESHOLD_VALUE:
                    c2_correct += 1

        total_acc[ii] = (c1_correct + c2_correct) / NN_TEST
        c1_acc[ii] = c1_correct / c1_NN
        c2_acc[ii] = c2_correct / c2_NN

    print(f"\r########## AGENTS' ACCURACIES ##########\n"
            f"-----------C1-----C2-----TOTAL-----------")
    for ii in range(NN):
        print(f"\rAgent [{ii + 1}]: {c1_acc[ii]:.2f} - {c2_acc[ii]:.2f} - {total_acc[ii]:.2f}")

###############################################################################
# generate N random colors
colors = {}
for ii in range(NN):
    colors[ii] = np.random.rand(3)

plt.figure()
for ii in range(NN):
    plt.plot(np.arange(max_iters), JJ[ii], color=colors[ii])
plt.xlabel(r"iterations $k$")
plt.ylabel(r"$J_i^k$")
plt.title("Evolution of the local cost functions")
plt.grid()
plt.show()
###############################################################################