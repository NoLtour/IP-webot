from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from CartesianPose import CartesianPose
from scipy.signal import convolve2d
import jsonpickle

 
@dataclass
class RawScanFrame:
    """Dataclass that stores a set of points representing the scan, and a raw pose from which the scans thought to be taken""" 
    scanDistances: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    scanAngles: np.ndarray = field(default_factory=lambda: np.array([], dtype=float)) 
    pose: CartesianPose = field(default_factory=lambda: CartesianPose.zero() ) 
    truePose: CartesianPose = field(default_factory=lambda: CartesianPose.zero() ) 
    
    index:int = None

    def copy(this):
        return RawScanFrame( scanDistances=this.scanDistances, scanAngles=this.scanAngles, pose=this.pose.copy(), truePose=this.truePose, index=this.index )
    
    @staticmethod
    def exportScanFrames( scanStack: list[RawScanFrame], fileName:str ):  
        rawExport = jsonpickle.encode( scanStack )
        with open( fileName, "w" ) as targFile:
            targFile.write( rawExport )
            targFile.close()

    @staticmethod
    def importScanFrames( fileName:str ):
        rawData = ""
        with open( fileName, "r" ) as targFile:
            rawData = targFile.read()
            targFile.close() 

        return jsonpickle.decode( rawData ) 