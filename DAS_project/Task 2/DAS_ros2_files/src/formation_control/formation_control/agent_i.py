
from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node 
from std_msgs.msg import Float32MultiArray as msg_float

#The goal. Define the desired position in g_ij
P_ = np.array([0, 0, 1, 0, 1, 1, 0, 1]) #Square with four nodes

#Take in agent_id, set the first two to leaders and the last two to followers
def formation_update(dt, x_i, neigh, data, kp, kv, agent_id):
     
     # dt    = discretization step
     # x_i   = state pf agent i
     # neigh = list of neihbors
     # data  = state of neighbors
     # kv  = coefficient for formation control law 
     # kv  = coefficient for formation control law 
     # I_N  = Identity matrix
    I_D = np.identity(2, dtype=int) #Two because dimension is 2. 2x2 nodes which is 4.
    xdot_i = np.zeros(x_i.shape)
    p_i = np.array(x_i[0:2])
    v_i = np.array(x_i[2:4]) 

    P_elements = P_.size

    p_index = 0
    
    for i in range(agent_id):
        p_index += 2

    for j in neigh:
        x_j = np.array(data[j].pop(0)[1:]) #Pop the first element, and take the rest. The first is just the time.
        p_j = np.array(x_j[0:2]) #to np array
        v_j = np.array(x_j[2:4]) #to np array

        #Compute p, this is P_ for agent 0, must do it iterative
        #g_ij = (P_[2:4] - P_[0:2])/np.linalg.norm(P_[2:4] - P_[0:2])
        
        if P_elements - 2 == p_index: 
            g_ij = np.array((P_[0:2] - P_[p_index:p_index+2])/np.linalg.norm(P_[0:2] - P_[p_index:p_index+2]))
        else:
            g_ij = np.array((P_[p_index+2:p_index+4] - P_[p_index:p_index+2])/np.linalg.norm(P_[p_index+2:p_index+4] - P_[p_index:p_index+2]))

        #if agent_id == 2:
        #g_ij = (P_[2:4] - P_[0:2])/np.linalg.norm(P_[2:4] - P_[0:2])
        #g_ij = (P_[4:6] - P_[2:4])/np.linalg.norm(P_[2:4] - P_[0:2])
           # g_ij = (P_[6:] - P_[4:6])/np.linalg.norm(P_[2:4] - P_[0:2])
        #if agent_id == 3:
         #   g_ij = (P_[0:2] - P_[6:])/np.linalg.norm(P_[2:4] - P_[0:2])
    

        #For leaders, this u_ij must be zero
        if agent_id == 0 or agent_id == 1: #The first two are leaders
            u_ij = np.array(0)

        else:
            P = I_D - g_ij@g_ij.T
            p_combined_v = np.array((kp*(p_i - p_j), kv*(v_i - v_j))) #Format [[p_x,p_y], [v_x,v_y]]
            u_ij = np.concatenate(P@p_combined_v) #Format [xp_x,p_y,v_x,v_y]. U is acceleration
        
        xdot_i += - u_ij

    #Forward Euler
    x_i += dt*xdot_i
    print(x_i, agent_id)

    return x_i

class Agent(Node):

    #Constructur, private method
    def __init__(self): 
        super().__init__('agent', 
                        allow_undeclared_parameters=True,
                        automatically_declare_parameters_from_overrides=True) #Agent number should be declared. Constructor of the father level class

        #Get parameters from launcher
        self.agent_id = self.get_parameter('agent_id').value
        print("agent_id", self.agent_id)
        self.neigh = self.get_parameter('neigh').value
        print("neigh", self.neigh)

        x_i = self.get_parameter('x_init').value
        print(x_i, "x_i") #Prints None, x_i...
        self.n_x = len(x_i)
        self.x_i = np.array(x_i)

        self.max_iters = self.get_parameter('max_iters').value
        self.communication_time = self.get_parameter('communication_time').value
        
        self.kp = self.get_parameter('kp').value
        self.kv = self.get_parameter('kv').value

        self.tt = 0

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

            string_for_logger = [round(i,4) for i in msg.data.tolist()[1:]]
            self.get_logger().info("Iter = {}  Value = {}".format(int(msg.data[0]), string_for_logger))

            self.tt += 1

        else: #Have all messages at time t-1 arrived?
             # Check if lists are nonempty
            all_received = all(self.received_data[j] for j in self.neigh) # check if all neighbors' have been received

            sync = False
            # Have all messages at time t-1 arrived?
            if all_received:
                sync = all(self.tt-1 == self.received_data[j][0][0] for j in self.neigh) # True if all True
            
            if sync:
                DeltaT = self.communication_time/10
                self.x_i = formation_update(DeltaT, self.x_i, self.neigh, self.received_data, self.kp, self.kv, self.agent_id)

                # publish the updated message
                msg.data = [float(self.tt)]
                [msg.data.append(float(element)) for element in self.x_i]
                self.publisher_.publish(msg)

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