import numpy as np

d=4
I_N = np.identity(d, dtype=int)
P_ = np.array([
  [0, 0, 1, 0, 1, 1, 0, 1] 
])

    
def compute_P(p_ti, p_tj):

    g_ij = (p_tj-p_ti)/len(p_tj-p_ti)
    P = I_N - g_ij@g_ij.T

    return P


def formation_update(dt, x_i, neigh, data):
    kp =1
    kv =1
    """
      dt    = discretization step
      x_i   = state pf agent i
      neigh = list of neihbors
      data  = state of neighbors
      dist  = coefficient for formation control law 
    """
    xdot_i = np.zeros(x_i.shape)
    p_i = x_i[0:2]
    v_i = x_i[2:4]

    for j in neigh:
        x_j = np.array(data[j].pop(0)[1:])
        p_j = x_j[0:2]
        v_j = x_j[2:4]
        p = compute_P(p_i, p_j)
        u_ij = p * (kp (p_i - p_j) + kv( v_i - v_j))
        xdot_i += - u_ij

    # Forward Euler
    x_i += dt*xdot_i

    return x_i