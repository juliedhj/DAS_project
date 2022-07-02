from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    N = 4 #Number of agents
    MAXITERS = 500 
    COMM_TIME = 5e-2 #Communication time period
    KP = 1
    KV = 1
    leaders = 2
    
    n_x = 4 #dimension of x_i

    Adj = np.asarray([
        [0, 1, 0, 1],
        [1, 0, 1, 0], 
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

    #x_i consists of pi_x, pi_y, vi_x, vi_y
    # definite initial positions
    x_leader = np.array([0, 0, 0, 0, 1, 0, 0, 0]) #Velocity of leader is 0.
    x_follower = np.random.rand(n_x*(N-leaders),1)
    x_init = np.concatenate((x_leader, x_follower), axis=None)

    launch_description = [] #append here your nodes

    for ii in range(N):
        N_ii = np.nonzero(Adj[:, ii])[0].tolist()
        ii_index = ii*n_x + np.arange(n_x)
        x_init_ii = x_init[ii_index].flatten().tolist()

        launch_description.append(
            Node(
                package='formation_control',
                node_namespace='agent_{}'.format(ii),
                node_executable='agent_i',
                parameters=[{
                    'agent_id': ii,
                    'max_iters': MAXITERS,
                    'communication_time': COMM_TIME,
                    'x_init': x_init_ii,
                    'neigh': N_ii,
                    'kp': KP,
                    'kv' : KV
                }],
                output='screen',
                prefix='xterm -title "agent_{}" -hold -e'.format(ii)
            ))
    return LaunchDescription(launch_description)

