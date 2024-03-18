import sys 

import numpy as np
import datetime
from pathlib import Path

from scipy.signal import convolve2d
from zeroros import Subscriber, Publisher
from zeroros.messages import LaserScan, Twist, Odometry, Vector3
from zeroros.datalogger import DataLogger
from scipy.signal import convolve2d
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom

from simulation_zeroros.console import Console

from Navigator import Navigator, CartesianPose 
from Mapper import Mapper
from ProbabilityGrid import exportScanFrames, importScanFrames
from ImageProcessor import ImageProcessor 

import importlib.util 
# Absolute path to the Python file
file_path = "C:\\IP-webot\\IP_2D_Imp_2\\RawScanFrame.py" 
# Load the module from the file
spec = importlib.util.spec_from_file_location("module_name", file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module) 

import livePlotter as lp
import time 

class MAP_PROP:
    X_MIN = -4
    X_MAX = 6
    Y_MIN = -6
    Y_MAX = 4
    
    PROB_GRID_RES = 25
 

lpWindow = lp.PlotWindow(5, 15)
lpRoboDisplay = lp.RobotDisplay(0,0,5,5,lpWindow, MAP_PROP.X_MIN, MAP_PROP.X_MAX, MAP_PROP.Y_MIN, MAP_PROP.Y_MAX)
gridDisp      = lp.LabelledGridGraphDisplay(5,0,5,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
gridDisp2     = lp.LabelledGridGraphDisplay(10,0,15,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
 
_start_millis = -1

def gaussian_kernel(size, sigma=1):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

 
G_Kernal5 = gaussian_kernel( 5, 2 )

def millis():
    """millis since first robot controller initialisation started"""
    return (round(time.time() * 1000)) - _start_millis

class RobotController:
    def __init__(this):
        this.wheel_seperation = 0.135
        this.wheel_diameter = 0.20454
        #this.datalog = DataLogger()
        
        """this.wheel_seperation = 0.135
        this.wheel_diameter = 0.135"""
        
        this.navigator = Navigator( this.wheel_diameter, this.wheel_seperation )
        #this.mapper = Mapper( this.navigator )
        this.navigator.addTarget(2, 0)
        this.navigator.addTarget(2, -2)
        this.navigator.addTarget(0, -2)
        this.navigator.addTarget(0, 0)
        
        this.navigatorWithError = Navigator( this.wheel_diameter, this.wheel_seperation )
        
        this.allRawScans = []
        
        """this.navigator.addTarget(1, 0)
        this.navigator.addTarget(1, 1)
        this.navigator.addTarget(0, 1)
        this.navigator.addTarget(0, 0)"""

        this.laserscan_sub = Subscriber("/lidar", LaserScan, this.laserscan_callback)
        this.odom_sub = Subscriber("/odom", Odometry, this.odometry_callback)
        
        this.cmd_wh_vel_pub = Publisher("/cmd_wh_vel", Vector3  )
        this.get_wh_rot_sub = Subscriber("/wh_rot", Vector3, this.encoder_callback)
        
        this.realPose = CartesianPose.zero()
        this.initRealPose = 0
         
        #this.gridMapper = GridMapper( this.navigator, MAP_PROP.X_MIN, MAP_PROP.X_MAX, MAP_PROP.Y_MIN, MAP_PROP.Y_MAX, MAP_PROP.PROB_GRID_RES )

    def run(this):
        try:
            while True:
                this.infinite_loop()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
            raise Exception( e )
        finally:
            this.laserscan_sub.stop()

    def infinite_loop(this):
        """  """ 
        
        lVel, rVel = this.navigator.desiredTargetMVels()
        this.setWheelVelocity( lVel, rVel )
        
        if (not this.navigator.hasTarget()):
            RawScanFrame.exportScanFrames( this.allRawScans, "cleanData" )
        
        return
        lpRoboDisplay.parseData( this.navigator.currentPose, this.realPose, [[0],[0]] ) 
        if ( len(this.mapper.allScans) != 0 ):
            #gridDisp.parseData( this.mapper.allScans[-1].gridData )

            #l1, l2, rv = guassianCornerDist( this.mapper.allScans[-1].gridData )

            scan = this.mapper.allScans[-1]
            
            if ( np.max( scan.positiveData ) != 0 ):
                """pMap = ImageProcessor.estimateFeatures( scan, 0.2 ) - scan.negativeData/2
                
                lambda_1, lambda_2, Rval = ImageProcessor.guassianCornerDist( pMap )
                #rend = lambda_1/(lambda_2+0.00000000000001)
                
                maxPos, vals = ImageProcessor.findMaxima( Rval, 5 )"""

                #if ( not this.navigator.hasTarget() ): exportScanFrames( this.mapper.allRawScans, "testRunData" ) 
                
                """gridDisp.parseData( pMap, maxPos[:,1], maxPos[:,0]  )
                gridDisp2.parseData( Rval/np.max(Rval), maxPos[:,1], maxPos[:,0] )"""
                #gridDisp2.parseData( this.mapper.allScans[-1].positiveData )
            else:
                gridDisp.parseData( scan.negativeData )
        
        #foundInterceptGrid, pointCloudNP = this.gridMapper.extractNearbyPoints( this.gridMapper.lastScanCloud, 0.3 )
         
        #yPoints, xPoints = np.where( foundInterceptGrid ) 
        #pointCloudIG = np.array( [xPoints, yPoints, np.zeros(yPoints.shape)] ) 
        
        #this.gridMapper.simpleICPSolve( foundInterceptGrid, pointCloudNP )
         
        lpWindow.render()
         
        #this.setWheelVelocity( 0.5, 1 )
        
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
        
        this.navigatorWithError.updatePosition( leftRotation, rightRotation )
        print("updated positon: ", this.navigatorWithError.currentPose.x, this.navigatorWithError.currentPose.y, this.navigatorWithError.currentPose.yaw)
         
         
        

    def odometry_callback(this, msg): 
        xPos = -msg.pose.pose.position.y
        yPos = msg.pose.pose.position.x
        
        qw = msg.pose.pose.orientation.w
        qz = msg.pose.pose.orientation.z 
        
        t0 = 2*(msg.pose.pose.orientation.w*msg.pose.pose.orientation.z+msg.pose.pose.orientation.x*msg.pose.pose.orientation.y)
        t1 = 1 - 2*(msg.pose.pose.orientation.y**2+msg.pose.pose.orientation.z**2)
        
        #yaw = 2.0 * np.arctan2((qw * qz), 1.0)
        yaw = -np.arctan2(t0, t1)
        
        if ( this.initRealPose == 0 ):
            this.initRealPose = CartesianPose( xPos, yPos, 0, 0, 0, yaw )  #np.arctan2(2.0 * (qw * qz), 1.0)
            this.realPose = CartesianPose.zero()

        else:
            this.realPose = CartesianPose( xPos - this.initRealPose.x
                                          , yPos-this.initRealPose.y, 0, 0, 0, yaw-this.initRealPose.yaw ) 
            
        this.navigator.currentPose = this.realPose
        
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
        
        this.allRawScans.append( RawScanFrame(
            np.array(msg.ranges),
            -msg.angle_min - np.arange( 0, len(msg.ranges) )*msg.angle_increment,
            this.navigatorWithError.currentPose,
            this.realPose
        ) ) 
        #this.mapper.pushLidarScan( msg )
        
        #this.gridMapper.pushScanCloud( ControllerMaths.ProtoLIDARInterp.calculateAbsolutePositions( this.navigator.posTracker.worldPose, msg ) )
        
        ""
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
    #Console.info("Data log available at: " + str(controller.datalog.log_file))
    
    

if __name__ == "__main__":
    main()