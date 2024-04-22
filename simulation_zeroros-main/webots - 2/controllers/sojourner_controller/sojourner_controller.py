import asyncio
import os
from enum import Enum
from math import atan2

from controller import Robot
from zeroros import Subscriber, Publisher
from zeroros.messages import geometry_msgs, sensor_msgs, nav_msgs
from zeroros.message_broker import MessageBroker

VELOCITY = 0.6

back_left_bogie = 0
front_left_bogie = 1
front_left_arm = 2
back_left_arm = 3
front_left_wheel = 4
middle_left_wheel = 5
back_left_wheel = 6
back_right_bogie = 7
front_right_bogie = 8
front_right_arm = 9
back_right_arm = 10
front_right_wheel = 11
middle_right_wheel = 12
back_right_wheel = 13
JOINTS_MAX = 14


# Check if the platform is windows
if os.name == 'nt':
    # Set the event loop policy to avoid the following warning:
    # [...]\site-packages\zmq\_future.py:681: RuntimeWarning:
    # Proactor event loop does not implement add_reader family of methods required for
    # zmq. Registering an additional selector thread for add_reader support via tornado.
    #  Use `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` to avoid
    # this warning.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class SojournerController(Robot):
    def __init__(self):
        super(SojournerController, self).__init__()
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
        self.timeStep =timestep * 10
        print("Setting controller timestep: ", self.timeStep)

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

        self.joints = [None]*JOINTS_MAX

        self.joints[back_left_bogie] = self.getDevice("BackLeftBogie");
        self.joints[front_left_bogie] = self.getDevice("FrontLeftBogie");
        self.joints[front_left_arm] = self.getDevice("FrontLeftArm");
        self.joints[back_left_arm] = self.getDevice("BackLeftArm");
        self.joints[front_left_wheel] = self.getDevice("FrontLeftWheel");
        self.joints[middle_left_wheel] = self.getDevice("MiddleLeftWheel");
        self.joints[back_left_wheel] = self.getDevice("BackLeftWheel");
        self.joints[back_right_bogie] = self.getDevice("BackRightBogie");
        self.joints[front_right_bogie] = self.getDevice("FrontRightBogie");
        self.joints[front_right_arm] = self.getDevice("FrontRightArm");
        self.joints[back_right_arm] = self.getDevice("BackRightArm");
        self.joints[front_right_wheel] = self.getDevice("FrontRightWheel");
        self.joints[middle_right_wheel] = self.getDevice("MiddleRightWheel");
        self.joints[back_right_wheel] = self.getDevice("BackRightWheel");

        self.joints[front_left_wheel].setPosition(float("inf"))
        self.joints[middle_left_wheel].setPosition(float("inf"))
        self.joints[back_left_wheel].setPosition(float("inf"))
        self.joints[front_right_wheel].setPosition(float("inf"))
        self.joints[middle_right_wheel].setPosition(float("inf"))
        self.joints[back_right_wheel].setPosition(float("inf"))

        self.joints[back_left_bogie].setPosition(-0.2)
        self.joints[front_left_bogie].setPosition(-0.2)
        self.joints[front_left_arm].setPosition(0.0)
        self.joints[back_left_arm].setPosition(0.0)


    def move_wheels(self, v):
        self.joints[front_left_wheel].setVelocity(v * VELOCITY)
        self.joints[middle_left_wheel].setVelocity(v * VELOCITY)
        self.joints[back_left_wheel].setVelocity(v * VELOCITY)
        self.joints[front_right_wheel].setVelocity(v * VELOCITY)
        self.joints[middle_right_wheel].setVelocity(v * VELOCITY)
        self.joints[back_right_wheel].setVelocity(v * VELOCITY)

    def move_4_wheels(self, v):
        self.joints[middle_right_wheel].setAvailableTorque(0.0)
        self.joints[middle_left_wheel].setAvailableTorque(0.0)
        self.move_wheels(v)

    def move_6_wheels(self, v):
        self.joints[middle_right_wheel].setAvailableTorque(2.0)
        self.joints[middle_left_wheel].setAvailableTorque(2.0)
        self.move_wheels(v)

    def turn_wheels_right(self):
        self.joints[front_left_arm].setPosition(0.4)
        self.joints[front_right_arm].setPosition(0.227)
        self.joints[back_right_arm].setPosition(-0.227)
        self.joints[back_left_arm].setPosition(-0.4)

    def turn_wheels_left(self):
        self.joints[front_left_arm].setPosition(-0.227)
        self.joints[front_right_arm].setPosition(-0.4)
        self.joints[back_right_arm].setPosition(0.4)
        self.joints[back_left_arm].setPosition(0.227)

    def turn_wheels(self, angle):
        if angle > 0:
            self.turn_wheels_right()
        else:
            self.turn_wheels_left()

    def wheels_straight(self):
        self.joints[front_left_arm].setPosition(0.0)
        self.joints[front_right_arm].setPosition(0.0)
        self.joints[back_right_arm].setPosition(0.0)
        self.joints[back_left_arm].setPosition(0.0)

    def cmd_vel_callback(self, message):
        print("Received speed: ", message.linear.x, " and angular: ", message.angular.z)
        if message.linear.x == 0.0 and message.angular.z == 0.0:
            self.move_6_wheels(0.0)
            self.wheels_straight()
        elif message.linear.x == 0.0:
            self.turn_wheels(message.angular.z)
            self.move_4_wheels(0.0)
        elif message.angular.z == 0.0:
            self.wheels_straight()
            self.move_6_wheels(message.linear.x)
        else:
            self.wheels_straight()
            self.turn_wheels(message.angular.z)
            self.move_4_wheels(message.linear.x)

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
        
        # TODO remove this if using ZeroROS messages!
        self.move_6_wheels(0.6)



wc = SojournerController()
wc.run()
