
from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node 
from std_msgs.msg import Float32MultiArray as msg_float

#Take in agent_id, set the first two to leaders and the last two to followers
def formation_update(dt, x_i, neigh, data, kp, kv, P_, agent_id, type):
     
     # dt    = discretization step
     # x_i   = state pf agent i
     # neigh = list of neihbors
     # data  = state of neighbors
     # kv  = coefficient for formation control law 
     # kv  = coefficient for formation control law 
     # I_D  = Identity matrix

    I_D = np.identity(2) #Two because dimension is 2. 2x2 nodes which is 4.
    xdot_i = np.zeros(4) #X_i shape is 4. 
    P = np.zeros((2,2))
    u_ij = np.zeros(2) #Vector that contains the acceleration
    g_ij_vec = np.zeros(2)
    diff = np.zeros(2)

    p_i = np.array(x_i[0:2]) #Position
    v_i = np.array(x_i[2:4]) #Velocity

    for j in neigh:
        x_j = np.array(data[j].pop(0)[1:]) #Pop the first element, and take the rest. The first is just the time.
        p_j = np.array(x_j[0:2]) #to np array
        v_j = np.array(x_j[2:4]) #to np array

        g_ij_vec[0] = P_[j,0] - P_[agent_id,0]
        g_ij_vec[1] = P_[j,1] - P_[agent_id,1]
        g_ij_norm = np.linalg.norm(g_ij_vec)
        g_ij = np.array([g_ij_vec]) / g_ij_norm
     
        P = I_D - g_ij.T@g_ij #Left associate
        diff = [kp*(p_i[0] - p_j[0]) + kv*(v_i[0]-v_j[0]), 
                kp*(p_i[1] - p_j[1]) + kv*(v_i[1]-v_j[1])]
        u_ij += -np.matmul(P,diff) #Format [xp_x,p_y,v_x,v_y]. U is acceleration
        
    #For leaders, this u_ij must be zero
    if type == 'leader': #The first two are leaders
        u_ij[0] = 0
        u_ij[1] = 0        

    #Forward Euler. The control law will be applied to the agent if it is a follower. 
    xdot_i[0] = p_i[0] + dt*v_i[0]
    xdot_i[1] = p_i[1] + dt*v_i[1]
    xdot_i[2] = v_i[0] + dt*u_ij[0]
    xdot_i[3] = v_i[1] + dt*u_ij[1]

    return xdot_i


def writer(file_name, string):
    """
      inner function for logging
    """
    file = open(file_name, "a") # "a" is for append
    file.write(string)
    file.close()

class Agent(Node):

    #Constructur, private method
    def __init__(self): 
        super().__init__('agent', 
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True) #Agent number should be declared. Constructor of the father level class

        #Get parameters from launcher
        self.agent_id = self.get_parameter('agent_id').value
        self.neigh = self.get_parameter('neigh').value

        x_i = self.get_parameter('x_init').value
        self.n_x = len(x_i)
        self.x_i = np.array(x_i)

        self.max_iters = self.get_parameter('max_iters').value
        self.communication_time = self.get_parameter('communication_time').value
        
        self.kp = self.get_parameter('kp').value
        self.kv = self.get_parameter('kv').value

        P_raw = self.get_parameter('P_').value
        P_ = np.array_split(P_raw, len(P_raw)/2)
        self.P_ = np.array(P_)

        self.type = self.get_parameter('type').value

        self.tt = 0

         # create logging file
        self.file_name = "_csv_file/agent_{}.csv".format(self.agent_id)
        file = open(self.file_name, "w+") # 'w+' needs to create file and open in writing mode if doesn't exist
        file.close()

        #Initialize subscription dict
        self.subscription_list = {}

        #Create subscription to each neighbor
        for j in self.neigh:
            topic_name = '/topic_{}'.format(j)
            self.subscription_list[j] = self.create_subscription(
                msg_float,
                topic_name,
                lambda msg, node = j: self.listener_callback(msg,node), #A way to express in a compact way a function without defining the object. Need to define a callback specified for each of the neighbors.
                10
            )


        #Create the publisher
        self.publisher_ = self.create_publisher(
                    msg_float, #Sending format of message
                    '/topic_{}'.format(self.agent_id), #Same for sender and receiver
                    10 #Queue size, 10 messages
        )

    
        self.timer = self.create_timer(
            self.communication_time, #Variable
            self.timer_callback
        )

        #initialize a disctionary with the list of received messages from each neighbour j [a queue]
        self.received_data = {j: [] for j in self.neigh}

        print("Setup of agent {} coomplete".format(self.agent_id))

    def listener_callback(self, msg, node):
        self.received_data[node].append(list(msg.data))

    def timer_callback(self):
        # Perform consensus
        # Initialize a message of type float
        msg = msg_float()

        # Take messages from buffer
        if self.tt == 0:
            msg.data = [float(self.tt)]
            [msg.data.append(float(element)) for element in self.x_i]

            self.publisher_.publish(msg)
            self.tt += 1

            string_for_logger = [round(i,4) for i in msg.data.tolist()[1:]]
            self.get_logger().info("Iter = {}  Value = {}".format(int(msg.data[0]), string_for_logger))

            # 2) save on file
            data_for_csv = msg.data.tolist().copy()
            data_for_csv = [str(round(element,4)) for element in data_for_csv[1:]]
            data_for_csv = ','.join(data_for_csv)
            writer(self.file_name,data_for_csv+'\n')



        else: #Have all messages at time t-1 arrived?
             # Check if lists are nonempty
            all_received = all(self.received_data[j] for j in self.neigh) # check if all neighbors' have been received
            sync = False

            # Have all messages at time t-1 arrived?
            if all_received:
                sync = all(self.tt-1 == self.received_data[j][0][0] for j in self.neigh) # True if all True

            if sync:
                DeltaT = self.communication_time/10
                self.x_i = formation_update(DeltaT, self.x_i, self.neigh, self.received_data, self.kp, self.kv, self.P_, self.agent_id, self.type)
                
                # publish the updated message
                msg.data = [float(self.tt)]
                [msg.data.append(float(element)) for element in self.x_i]
                self.publisher_.publish(msg)

                # save data on csv file
                data_for_csv = msg.data.tolist().copy()
                data_for_csv = [str(round(element,4)) for element in data_for_csv[1:]]
                data_for_csv = ','.join(data_for_csv)
                writer(self.file_name,data_for_csv+'\n')

                string_for_logger = [round(i,4) for i in msg.data.tolist()[1:]]
                print("Iter = {} \t Value = {}".format(int(msg.data[0]), string_for_logger))

                # Stop the node if tt exceeds MAXITERS
                if self.tt > self.max_iters:
                    print("\nMAXITERS reached")
                    sleep(3) # [seconds]
                    self.destroy_node()

                # update iteration counter
                self.tt += 1


def main(args=None):
    rclpy.init(args=args)
    agent = Agent()

    print("Agent {:d} -- Waiting for sync.".format(agent.agent_id))
    sleep(0.5)
    print("GO!")

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        print("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 


if __name__ == "__main__":
    main()