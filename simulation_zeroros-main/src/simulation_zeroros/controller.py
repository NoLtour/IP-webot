import sys
print(sys.path)

import numpy as np
import datetime
from pathlib import Path

from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Twist, Odometry
from zeroros.datalogger import DataLogger

from simulation_zeroros.console import Console

import testing

class RobotController:
    def __init__(self):
        self.wheel_distance = 0.135 * 2
        self.datalog = DataLogger()

        self.laserscan_sub = Subscriber("/lidar", LaserScan, self.laserscan_callback)
        self.odom_sub = Subscriber("/odom", Odometry, self.odometry_callback)
        self.cmd_vel_pub = Publisher("/cmd_vel", Twist)

    def run(self):
        try:
            while True:
                self.infinite_loop()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
        finally:
            self.laserscan_sub.stop()

    def infinite_loop(self):
        twist_msg = Twist()
        twist_msg.linear.x = 4.0
        twist_msg.angular.z = -1.5
        self.cmd_vel_pub.publish(twist_msg)
        self.datalog.log(twist_msg)

    def odometry_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        qw = msg.pose.pose.orientation.w
        qz = msg.pose.pose.orientation.z
        self.yaw = np.arctan2(2.0 * (qw * qz), 1.0)
        self.datalog.log(msg)

    def laserscan_callback(self, msg):
        """This is a callback function that is called whenever a message is received

        The message is of type LaserScan and these are the fields:
        - header: Header object that contains (stamp, seq, frame_id)
        - angle_min: float32 - start angle of the scan [rad]
        - angle_max: float32 - end angle of the scan [rad]
        - angle_increment: float32 - angular distance between measurements [rad]
        - time_increment: float32 - time between measurements [seconds]
        - scan_time: float32 - time between scans [seconds]
        - range_min: float32 - minimum range value [m]
        - range_max: float32 - maximum range value [m]
        - ranges: float32[] - range data [m]
        - intensities: float32[] - intensity data ## NOT USED ##
        """
        # print("Received message: ", msg)
        self.laserscan = msg
        print("Received Lidar ranges: ", msg.ranges)
        self.datalog.log(msg)

def main():
    Console.info("Starting simulation_zeroros...")

    # Get YYYYMMDD_HHMMSS timestamp
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path.cwd() / (str(stamp) + "_simulation_zeroros.log")
    Console.set_logging_file(p)
    Console.info("Logging to " + str(p))

    controller = RobotController()
    controller.run()

    print("")
    Console.info("Console log available at: " + str(p))
    Console.info("Data log available at: " + str(controller.datalog.log_file))

if __name__ == "__main__":
    main()