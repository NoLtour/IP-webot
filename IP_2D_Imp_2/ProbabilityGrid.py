 
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import convolve2d
from skimage.draw import polygon2mask, line
import jsonpickle
from RawScanFrame import RawScanFrame
from CartesianPose import CartesianPose

from CommonLib import gaussian_kernel

class ProbabilityGrid:
    width: int
    height: int 
    
    negativeData: np.ndarray
    positiveData: np.ndarray
    
    xMin:float
    xMax:float
    yMin:float
    yMax:float
    cellRes:float
    
    xAMin:int
    xAMax:int
    yAMin:int
    yAMax:int
    
    def __init__(this, xMin:float, xMax:float, yMin:float, yMax:float, cellRes:float):
        this.xMin = xMin
        this.xMax = xMax
        this.yMin = yMin
        this.yMax = yMax
        this.cellRes = cellRes
        
        this.xAMin = int(xMin*cellRes)
        this.xAMax = int(xMax*cellRes)
        this.yAMin = int(yMin*cellRes)
        this.yAMax = int(yMax*cellRes)
        
        this.width = int((xMax - xMin)*cellRes)
        this.height = int((yMax - yMin)*cellRes) 
        
        this.negativeData = np.zeros( (this.height, this.width) )
        this.positiveData = np.zeros( (this.height, this.width) )
    
    @staticmethod 
    def initFromScanFramesPoly( cellRes:float, scanStack:list[RawScanFrame], midScanIndex, maxScanDistance:float, offsetPose:CartesianPose=CartesianPose.zero() ):
        """
            Constructs a scene by creating an image of negative space consisiting of area covered within linecasts, then constructs positive space
            consisting of line segmants connecting nearby hits
            The image's x and y coordinates are given about the centre frame (which acts as the origin), but this can be offset using the optional inputs
        """
        
        xMin = 10000000000
        xMax = -10000000000
        yMin = 10000000000
        yMax = -10000000000

        middleScan = scanStack[midScanIndex] 
        
        zeroPose = middleScan.pose.copy()
        zeroPose.subtractPose( offsetPose )
        
        scanTerminations = []
        scanNoHits       = []
        
        # Calculates the endpoints of the scans as well as the limits of the final probability grid
        for targScan in scanStack:
            isInf = np.isinf( targScan.scanDistances )
            
            # A version of scan distances where values: inf -> maxLength
            adjustedDistances = np.where( isInf, maxScanDistance, targScan.scanDistances ) 
            xPoints = targScan.pose.x - offsetPose.x + adjustedDistances * np.cos( targScan.scanAngles + targScan.pose.yaw - offsetPose.yaw )
            yPoints = targScan.pose.y - offsetPose.y + adjustedDistances * np.sin( targScan.scanAngles + targScan.pose.yaw - offsetPose.yaw )
            
            scanTerminations.append( [ xPoints, yPoints ] )
            scanNoHits.append( isInf )
            
            xMax = max( targScan.pose.x - zeroPose.x, xMax, np.max( xPoints ) )
            xMin = min( targScan.pose.x - zeroPose.x, xMin, np.min( xPoints ) )
            yMax = max( targScan.pose.y - zeroPose.y, yMax, np.max( yPoints ) )
            yMin = min( targScan.pose.y - zeroPose.y, yMin, np.min( yPoints ) )
        
        
        finalGrid = ProbabilityGrid( xMin-0.01, xMax+0.01, yMin-0.01, yMax+0.01, cellRes )
        
        for targScan, terminations, noHits in zip(targScan, scanTerminations, scanNoHits):
            nProbGrid = ProbabilityGrid( xMin-0.01, xMax+0.01, yMin-0.01, yMax+0.01, cellRes )
            nProbGrid.addPolyLines( targScan.pose.x - zeroPose.x, targScan.pose.y - zeroPose.y, 
                               terminations[0], terminations[1], noHits  )
            
            midPointSeperation = np.sqrt( (targScan.pose.x - middleScan.pose.x)**2 + (targScan.pose.y - middleScan.pose.y)**2  )
            
            if (  midPointSeperation != 0 ):
                # TODO implement proper uncertainty model!
                sigma = midPointSeperation*cellRes*3
                
                finalGrid.negativeData += convolve2d( nProbGrid.negativeData, gaussian_kernel( 1+int(sigma*2), sigma ), mode="same" )
                finalGrid.positiveData += convolve2d( nProbGrid.positiveData, gaussian_kernel( 1+int(sigma*2), sigma ), mode="same" )
            else:
                finalGrid.negativeData += nProbGrid.negativeData
                finalGrid.positiveData += nProbGrid.positiveData
        
        finalGrid.negativeData /= len(scanStack)
        finalGrid.positiveData /= len(scanStack) 
        
        finalGrid.clipData()
                
        return finalGrid 

    def addPolyLines(this, originX: float, originY: float, lineTerminalX: np.ndarray, lineTerminalY: np.ndarray, isInf:np.ndarray, maxSep=0.1 ):
        # Get them into integer pixel coordinates
        originXCell = int((originX- this.xMin)*this.cellRes)
        originYCell = int((originY - this.yMin)*this.cellRes)
        termXCell = ((lineTerminalX- this.xMin)*this.cellRes).astype(int)
        termYCell = ((lineTerminalY - this.yMin)*this.cellRes).astype(int)
        
        scanPoints = np.array([termYCell, termXCell]).transpose()

        polyPoints = np.append( np.array([[originYCell, originXCell]]), scanPoints, axis=0 )

        observationRegion  = polygon2mask( this.negativeData.shape, polyPoints )
        interceptionRegion = np.zeros( this.negativeData.shape  )
        
        # scanpoints with inf values filtered out 
        realPoints = scanPoints[isInf==False]
        
        if ( realPoints.size != 0 ):
            lastPoint = realPoints[0] 
            for point in realPoints[1:]: 
                # Check if points are above a certain seperation 
                if ( maxSep*this.cellRes < np.sqrt( np.square(lastPoint[0]-point[0]) + np.square(lastPoint[1]-point[1]) ) ):
                    interceptionRegion[lastPoint[0],lastPoint[1]] = 1
                else:
                    interceptionRegion[ line( point[0], point[1], lastPoint[0], lastPoint[1] ) ] = 1

                lastPoint = point
                
        
        this.positiveData += interceptionRegion
        #                    observation region masked by where objects don't exist
        this.negativeData += observationRegion * (interceptionRegion==0)




