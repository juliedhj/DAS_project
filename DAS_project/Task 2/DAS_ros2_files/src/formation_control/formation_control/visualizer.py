import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker 
from geometry_msgs.msg import Pose 
from std_msgs.msg import Float32MultiArray as msg_float

class Visualizer(Node):
    def __init__(self):
        super().__init__('visualizer', 
                    allow_undeclared_parameters=True,
                    automatically_declare_parameters_from_overrides=True)

        self.agent_id = self.get_parameter('agent_id').value 
        self.communication_time = self.get_parameter('communication_time').value

        #Subscription to the topics for visualization 
        self.subscription = self.create_subscription(msg_float,
                                        '/topic_{}'.format(self.agent_id),
                                        self.listener_callback, 10)

        #Create the publisher that will communicate with Rviz 
        self.timer = self.create_timer(self.communication_time, self.publish_data)
        self.publisher = self.create_publisher(Marker, '/visualization_topic', 1)

        #Initialize til current_pose method 
        self.current_pose = Pose()

    def listener_callback(self, msg):
        self.current_pose.position.x = msg.data[1]
        self.current_pose.position.y = msg.data[2]

    def publish_data(self):
        if self.current_pose.position is not None: 
            marker = Marker()

            #need a reference fram in order to make the visualization
            marker.header.frame_id = 'my_frame'
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.type = Marker.SPHERE

            marker.pose.position.x = self.current_pose.position.x
            marker.pose.position.y = self.current_pose.position.y
            marker.pose.position.z = self.current_pose.position.z

            #Select the action and namespace of the marker 
            marker.action = Marker.ADD
            marker.ns = 'agents'

            marker.id = self.agent_id

            scale = 0.2 
            marker.scale.x = marker.scale.y = marker.scale.z = scale 

            #Specify the color of the marker 
            color = [1.0, 0.0, 0.0, 1.0]
            if self.agent_id % 2: 
                color = [0.0, 0.5, 0.0, 0.5]
            
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            self.publisher.publish(marker)
    
def main():
    rclpy.init()
    visualizer = Visualizer()

    try: 
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("----- Visualizer stopped cleanly -----")
    finally: 
        rclpy.shutdown()

if __name__ == '__main__':
    main()
