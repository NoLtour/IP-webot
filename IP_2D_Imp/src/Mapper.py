

import numpy as np
from ProbabilityGrid import ProbabilityGrid, ScanFrame
from Navigator import Navigator

def acuteAngle( a1, a2 ):
    """Returns the acute angle between a1 and a2 in radians"""
    absD = abs(a1-a2)%(np.pi*2)
    return min( absD, 2*np.pi - absD )

class Mapper:
    navigator: Navigator
    allScans: list[ProbabilityGrid]
    scanBuffer: list[ScanFrame]
    
    gridResolution: float
    
    def __init__(this, navigator:Navigator, gridResolution=80):
        this.navigator = navigator
        this.allScans  = []
        this.scanBuffer = []
        this.gridResolution = gridResolution
    
    def pushLidarScan( this, lidarOup, maxBuffer = 5, maxAngleChange = np.deg2rad( 30 ) ): 
        """ Pushed the scan frame onto the scan buffer and merges to make a completed image if conditions met """
        
        angles  = -lidarOup.angle_min - np.arange( 0, len(lidarOup.ranges) )*lidarOup.angle_increment
        lengths = np.array(lidarOup.ranges)
        cPose   = this.navigator.currentPose.copy()
        
        newScan = ScanFrame( scanAngles=angles, scanDistances=lengths, pose=cPose )
        
        # Under target conditions the scan buffer gets merged to make a new frame
        if ( len(this.scanBuffer) >= maxBuffer or ( len(this.scanBuffer) > 0 and acuteAngle( newScan.pose.yaw, this.scanBuffer[0].pose.yaw )  > maxAngleChange) ):
            newProbGrid = ProbabilityGrid.initFromScanFramesPoly( this.gridResolution, this.scanBuffer, 1, -0.2, 8 )
            this.allScans.append( newProbGrid )
            
            this.scanBuffer = [ newScan ]
        else:
            this.scanBuffer.append( newScan )
        
            
        
        
        
        
        
        
        
        
        







