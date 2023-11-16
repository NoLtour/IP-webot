import asyncio
import os
from math import atan2

from controller import Robot
from zeroros import Subscriber, Publisher
from zeroros.messages import geometry_msgs, sensor_msgs, nav_msgs
from zeroros.message_broker import MessageBroker


# Check if the platform is windows
if os.name == 'nt':
    # Set the event loop policy to avoid the following warning:
    # [...]\site-packages\zmq\_future.py:681: RuntimeWarning:
    # Proactor event loop does not implement add_reader family of methods required for
    # zmq. Registering an additional selector thread for add_reader support via tornado.
    #  Use `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` to avoid
    # this warning.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class WebotsController(Robot):
    def __init__(self):
        super(WebotsController, self).__init__()
        self.ip = "127.0.0.1"
        self.port = 5600

        self.wheel_distance = 0.135 * 2

        self.broker = MessageBroker()
        self.laserscan_pub = Publisher("/lidar", sensor_msgs.LaserScan)
        self.odom_pub = Publisher("/odom", nav_msgs.Odometry)
        self.cmd_vel_sub = Subscriber(
            "/cmd_vel", geometry_msgs.Twist, self.cmd_vel_callback
        )

        timestep = int(self.getBasicTimeStep())
        self.timeStep = timestep * 10
        print("Timestep:", timestep, "Setting controller timestep: ", self.timeStep)

        self.num_lidar_msgs = 0
        self.odom_msg = nav_msgs.Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "base_link"
        self.odom_msg.header.seq = 0

        self.lidar = self.getDevice("lidar")
        self.lidar.enable(self.timeStep)
        self.lidar.enablePointCloud()

        self.gps = self.getDevice("gps")
        self.gps.enable(self.timeStep)

        self.compass = self.getDevice("compass")
        self.compass.enable(self.timeStep)

        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.timeStep)

        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def cmd_vel_callback(self, message):
        print("Received speed: ", message.linear.x, " and angular: ", message.angular.z)
        left_speed = message.linear.x - message.angular.z * self.wheel_distance / 2.0
        right_speed = message.linear.x + message.angular.z * self.wheel_distance / 2.0
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def run(self):
        while self.step(self.timeStep) != -1:
            self.infinite_loop()

    def infinite_loop(self):
        range_image = self.lidar.getRangeImage()
        msg = sensor_msgs.LaserScan(range_image)
        msg.header.stamp = self.getTime()
        msg.header.frame_id = "lidar"
        msg.header.seq = self.num_lidar_msgs
        msg.angle_min = -self.lidar.getFov() / 2.0
        msg.angle_max = self.lidar.getFov() / 2.0
        msg.angle_increment = self.lidar.getFov() / self.lidar.getHorizontalResolution()
        msg.time_increment = self.lidar.getSamplingPeriod() / (
            1000.0 * self.lidar.getHorizontalResolution()
        )
        msg.scan_time = self.lidar.getSamplingPeriod() / 1000.0
        msg.range_min = self.lidar.getMinRange()
        msg.range_max = self.lidar.getMaxRange()
        self.laserscan_pub.publish(msg)
        self.num_lidar_msgs += 1

        pose_val = self.gps.getValues()
        speed_vector_values = self.gps.getSpeedVector()
        north = self.compass.getValues()
        angle = atan2(north[1], north[0])
        quaternion = geometry_msgs.Quaternion()
        quaternion.from_euler(0, 0, angle)
        rot_speed = self.gyro.getValues()

        # Publish odometry
        self.odom_msg.header.stamp = self.getTime()
        self.odom_msg.header.seq = self.odom_msg.header.seq + 1
        self.odom_msg.pose.pose.position.x = pose_val[0]
        self.odom_msg.pose.pose.position.y = pose_val[1]
        self.odom_msg.pose.pose.position.z = pose_val[2]
        self.odom_msg.pose.pose.orientation.x = quaternion.x
        self.odom_msg.pose.pose.orientation.y = quaternion.y
        self.odom_msg.pose.pose.orientation.z = quaternion.z
        self.odom_msg.pose.pose.orientation.w = quaternion.w
        self.odom_msg.twist.twist.linear.x = speed_vector_values[0]
        self.odom_msg.twist.twist.linear.y = speed_vector_values[1]
        self.odom_msg.twist.twist.linear.z = speed_vector_values[2]
        self.odom_msg.twist.twist.angular.x = rot_speed[0]
        self.odom_msg.twist.twist.angular.y = rot_speed[1]
        self.odom_msg.twist.twist.angular.z = rot_speed[2]
        self.odom_pub.publish(self.odom_msg)


wc = WebotsController()
wc.run()
