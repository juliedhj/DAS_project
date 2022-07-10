from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    COMM_TIME = 0.1 #0.1 #5e-2=0.05 #Communication time period #2 for six agents and 4 for 8 agents.
    MAXITERS = 10000 #Define iterations
    n_x = 4 #Dimension of x_i

    #Constant gains
    KP = 1
    KV = 1

    formation = "square"

############## TASK 2.3 #######################
    if formation == "square":
        #Formation: a square
        N = 4 #Number of agents
        leaders = 2 #Number of leaders
        desired_positions = [
            [0,0], 
            [1,0], 
            [1,1], 
            [0,1]]

    if formation == "hexagon":
        N = 6
        leaders = 2
        desired_positions = [
            [0,-1], 
            [1,-1], 
            [1,0], 
            [0,1],
            [-1,1], 
            [-1,0]]

    elif formation == "octagon":
        N = 8
        leaders = 3
        desired_positions = [
                [0,4],
                [0,-4],
                [3,3],
                [4,0],
                [3,-3],
                [-3,-3],
                [-4,0],
                [-3,3]
            ]

############ TASK 2.4 #################################
# Agents will draw the work CIAO.

    elif formation == "C":
        N = 4
        leaders = 2
        desired_positions = [
                [1,0], [0,1], [0,2], [1,3]
            ]
    elif formation == "I":
        N = 4
        leaders = 2
        desired_positions = [
                [1,-1],
                [1,0],
                [1,1],
                [1,2]
            ]
    elif formation == "A":
        N = 4
        leaders = 2
        desired_positions = [
                [0,0],
                [1,2],
                [2,0],
                [1,1]
        ]
    elif formation == "O":
        N = 4
        leaders = 2
        desired_positions = [
                [1,2],
                [1,0],
                [2,0],
                [2,2]
            ]
    elif formation == "C_big":
        N = 6
        leaders = 2
        desired_positions = [
                [1,0], [0,1], [0,2], [1,3], [0.5, 0.5], [0.5, 2.5] 
            ]
    elif formation == "I_big":
        N = 6
        leaders = 2
        desired_positions = [
                [1.1,-1],
                [1.2,0],
                [1.1,1],
                [1.2,2],
                [1.1,-2],
                [1.2,3]
            ]

    elif formation == "A_big":
        N = 6
        leaders = 2
        desired_positions = [
                [0,0],
                [1,2],
                [2,0],
                [1,1],
                [0.5,1],
                [1.5,1]
            ]
    elif formation == "O_big":
        N = 6
        leaders = 2
        desired_positions = [
                [0.8,1.5],
                [0.8,0.5],
                [2.2,0.5],
                [2.2,1.5],
                [1.5, 2.5],
                [1.5,-0.5]
            ]

    #Generate the  adjaceny matrix. Followers need to communicate with at least one leader
    Adj = np.ones((N,N), dtype=int)
    np.fill_diagonal(Adj, 0)
    Adj = np.asarray(Adj)
    #print(Adj)

    #Define inital values
    x_init = np.random.randint(2,4)*np.random.rand(n_x*N,1)
    
    launch_description = [] #append here your nodes

     ################################################################################
    # RVIZ
    ################################################################################

    # initialize launch description with rviz executable
    rviz_config_dir = get_package_share_directory('formation_control')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    launch_description.append(
        Node(
            package='rviz2', 
            executable='rviz2', 
            arguments=['-d', rviz_config_file],
            # output='screen',
            # prefix='xterm -title "rviz2" -hold -e'
            ))

    ################################################################################

    for ii in range(N):
        N_ii = np.nonzero(Adj[:, ii])[0].tolist() #neigbors
        ii_index = ii*n_x + np.arange(n_x) #Assign random x_i
        x_init_ii = x_init[ii_index].flatten().tolist()
        if ii < leaders:
            agent_type = 'leader'
            #x_i consists of pi_x, pi_y, vi_x, vi_y
            x_init_ii[0] = desired_positions[ii][0]
            x_init_ii[1] = desired_positions[ii][1]
            x_init_ii[2] = 0
            x_init_ii[3] = 0
            x_init_ii = np.asarray(x_init_ii)
            x_init_ii = x_init_ii.flatten().tolist()
        
        else:
            agent_type = 'follower'

        #print(N_ii)
        launch_description.append(
            Node(
                package='formation_control',
                namespace='agent_{}'.format(ii),
                executable='agent_i',
                parameters=[{
                    'agent_id': ii,
                    'max_iters': MAXITERS,
                    'communication_time': COMM_TIME,
                    'x_init': x_init_ii,
                    'neigh': N_ii,
                    'kp': KP,
                    'kv' : KV, 
                    'P_': np.asarray(desired_positions).flatten().tolist(),
                    'type': agent_type,
                }],
                output='screen',
                prefix='xterm -title "agent_{}" -hold -e'.format(ii)
            ))

        ################################################################################
        # RVIZ
        ################################################################################

        launch_description.append(
            Node(
                package='formation_control', 
                namespace='agent_{}'.format(ii),
                executable='visualizer', 
                parameters=[{
                                'agent_id': ii,
                                'communication_time': COMM_TIME,
                                }],
            ))

    return LaunchDescription(launch_description)

