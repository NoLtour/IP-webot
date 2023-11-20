import sys
print(sys.path)

import numpy as np
import datetime
from pathlib import Path

from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Twist, Odometry, Vector3
from zeroros.datalogger import DataLogger

from simulation_zeroros.console import Console

import testing as ControllerMaths
import time
 
_start_millis = -1

def millis():
    """millis since first robot controller initialisation started"""
    return (round(time.time() * 1000)) - _start_millis

class RobotController:
    def __init__(this):
        this.wheel_distance = 0.135 * 2
        this.wheel_seperation = 0.075*2
        #this.datalog = DataLogger()
        
        this.navigator = ControllerMaths.Navigator( this.wheel_seperation, this.wheel_distance )
        this.navigator.addTarget(0.5, 0.2)

        this.laserscan_sub = Subscriber("/lidar", LaserScan, this.laserscan_callback)
        this.odom_sub = Subscriber("/odom", Odometry, this.odometry_callback)
        
        this.cmd_wh_vel_pub = Publisher("/cmd_wh_vel", Vector3  )
        this.get_wh_rot_sub = Subscriber("/wh_rot", Vector3, this.encoder_callback)

    def run(this):
        try:
            while True:
                this.infinite_loop()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
        finally:
            this.laserscan_sub.stop()

    def infinite_loop(this):
        """  """
        
        this.navigator.checkTarget()
        
        lVel, rVel = this.navigator.desiredTargetMVels()
        
        #this.setWheelVelocity( lVel, rVel )
        this.setWheelVelocity( 0.5, 1 )
        
    def setWheelVelocity(this, leftVel, rightVel):
        wheelRot_msg = Vector3()
        wheelRot_msg.x = leftVel
        wheelRot_msg.y = rightVel
        
        this.cmd_wh_vel_pub.publish(wheelRot_msg)
        #this.datalog.log(wheelRot_msg)  
        
        #print("setting speed:", leftVel, rightVel)

    def encoder_callback(this, msg):
        leftRotation = msg.x
        rightRotation = msg.y 
        this.navigator.updatePosition( leftRotation, rightRotation )
        print("updated positon: ", this.navigator.posTracker.worldPose.x, this.navigator.posTracker.worldPose.y, this.navigator.posTracker.worldPose.yaw)
        

    def odometry_callback(this, msg):
        this.x = msg.pose.pose.position.x
        this.y = msg.pose.pose.position.y
        qw = msg.pose.pose.orientation.w
        qz = msg.pose.pose.orientation.z
        this.yaw = np.arctan2(2.0 * (qw * qz), 1.0)
        #this.datalog.log(msg)

    def laserscan_callback(this, msg):
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
        this.laserscan = msg
        #print("Received Lidar ranges: ", msg.ranges)
        #this.datalog.log(msg)
 

def main():
    Console.info("Starting simulation_zeroros...")

    # Get YYYYMMDD_HHMMSS timestamp
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path.cwd() / (str(stamp) + "_simulation_zeroros.log")
    
    #Console.set_logging_file(p)
    #Console.info("Logging to " + str(p))
    
    _start_millis = (round(time.time() * 1000))

    controller = RobotController()
    controller.run()

    print("")
    Console.info("Console log available at: " + str(p))
    Console.info("Data log available at: " + str(controller.datalog.log_file))
    
    

if __name__ == "__main__":
    main()