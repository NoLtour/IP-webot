 
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import convolve2d
from skimage.draw import polygon2mask, line
import jsonpickle
from RawScanFrame import RawScanFrame
from CartesianPose import CartesianPose
from scipy.ndimage import rotate

import matplotlib.pyplot as plt

from CommonLib import gaussianKernel, solidCircle,  fancyPlot

class ProbabilityGrid:
    width: int
    height: int 
    
    negativeData: np.ndarray
    positiveData: np.ndarray
    mapEstimate: np.ndarray
    
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
            xPoints = targScan.pose.x - zeroPose.x + adjustedDistances * np.cos( targScan.scanAngles + targScan.pose.yaw - zeroPose.yaw )
            yPoints = targScan.pose.y - zeroPose.y + adjustedDistances * np.sin( targScan.scanAngles + targScan.pose.yaw - zeroPose.yaw )
            
            scanTerminations.append( [ xPoints, yPoints ] )
            scanNoHits.append( isInf )
            
            xMax = max( targScan.pose.x - zeroPose.x, xMax, np.max( xPoints ) )
            xMin = min( targScan.pose.x - zeroPose.x, xMin, np.min( xPoints ) )
            yMax = max( targScan.pose.y - zeroPose.y, yMax, np.max( yPoints ) )
            yMin = min( targScan.pose.y - zeroPose.y, yMin, np.min( yPoints ) )
        
        
        finalGrid = ProbabilityGrid( xMin-0.01, xMax+0.01, yMin-0.01, yMax+0.01, cellRes )
         
        for targScan, terminations, noHits in zip(scanStack, scanTerminations, scanNoHits): 
            nProbGrid = ProbabilityGrid( xMin-0.01, xMax+0.01, yMin-0.01, yMax+0.01, cellRes )
            nProbGrid.addPolyLines( targScan.pose.x - zeroPose.x, targScan.pose.y - zeroPose.y, 
                               terminations[0], terminations[1], noHits  )
            
            midPointSeperation = 0*np.sqrt( (targScan.pose.x - middleScan.pose.x)**2 + (targScan.pose.y - middleScan.pose.y)**2  )
            
            if (  midPointSeperation != 0 ):
                # TODO implement proper uncertainty model!
                sigma = midPointSeperation*cellRes*1
                
                finalGrid.negativeData += convolve2d( nProbGrid.negativeData, gaussianKernel( sigma ), mode="same" )
                finalGrid.positiveData += convolve2d( nProbGrid.positiveData, gaussianKernel( sigma ), mode="same" )
            else:
                finalGrid.negativeData += nProbGrid.negativeData
                finalGrid.positiveData += nProbGrid.positiveData
        
        finalGrid.negativeData /= len(scanStack)
        finalGrid.positiveData /= len(scanStack) 
        
        finalGrid.negativeData = finalGrid.negativeData.clip( -1, 1 )
        finalGrid.positiveData = finalGrid.positiveData.clip( -1, 1 )
                
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

    def copyRotated( this, angle, onlyRotateEstimate=False ):
        """ returns a new probability grid which is this but rotated by angle """

        AXCentre = (this.xAMin+this.xAMax)*0.5
        AYCentre = (this.yAMin+this.yAMax)*0.5

        nNegative = None
        nPositive = None
        nEstimate = rotate( this.mapEstimate, np.rad2deg( angle ) ) 

        if ( not onlyRotateEstimate ):
            nNegative = rotate( this.negativeData, np.rad2deg( angle ) )
            nPositive = rotate( this.positiveData, np.rad2deg( angle ) ) 
            nNegative = np.where( np.abs(nNegative)<0.001, 0, nNegative )
            nPositive = np.where( np.abs(nPositive)<0.001, 0, nPositive )
 
        # Rotations leave near zero errors, so I filter them out here
        nEstimate = np.where( np.abs(nEstimate)<0.001, 0, nEstimate )

        nAWidth  = nEstimate.shape[1]
        nAHeight = nEstimate.shape[0]

        nAXMin = int(AXCentre-nAWidth*0.5)
        nAXMax = nAXMin+nAWidth

        nAYMin = int(AYCentre-nAHeight*0.5)
        nAYMax = nAYMin+nAHeight

        res = this.cellRes
        
        # TODO finish implementing this optimisation
        """# Now it trims the output to remove waste
        nonZeroXLines = np.any(nEstimate != 0, axis=1)
        nonZeroYLines = np.any(nEstimate != 0, axis=0) 

        minY_OC = np.argmax(nonZeroXLines)
        maxY_OC = nEstimate.shape[0] - np.argmax(nonZeroXLines[::-1]) - 1
        minX_OC = np.argmax(nonZeroYLines)
        maxX_OC = nEstimate.shape[1] - np.argmax(nonZeroYLines[::-1]) - 1
        
        subtract these from the abs vals!

        nEstimate = nEstimate[minY_OC:maxY_OC, minX_OC:maxX_OC]"""

        nGrid = ProbabilityGrid( nAXMin/res, nAXMax/res, nAYMin/res, nAYMax/res, res )
        nGrid.negativeData = nNegative
        nGrid.positiveData = nPositive
        nGrid.mapEstimate  = nEstimate

        plt.figure( 69 )
        plt.imshow( nEstimate )
        plt.show( block=False )

        return nGrid

    def estimateFeatures( this, estimatedWidth, sharpnessMult ):
        """ Function to fill in object to be more realistic representation of environment """

        if ( np.all( this.positiveData == 0 ) ):
            this.mapEstimate = - this.negativeData/2  

        """ Uses a model of the environment to partially fill in missing data """
        pixelWidth = estimatedWidth*this.cellRes 
        """kern = gaussianKernel( pixelWidth, 0.1 )
        kern /= np.max(kern)
        
        oup = np.maximum(convolve2d( this.positiveData, kern, mode="same" ) - this.negativeData*pixelWidth*sharpnessMult, 0)
        
        this.mapEstimate = np.minimum( oup/np.max(oup)+this.positiveData, 1 ) - this.negativeData/2
        return
        
        
        oup = np.minimum(1, convolve2d( this.positiveData, solidCircle( int(pixelWidth) ), mode="same" ) )
        oup = convolve2d( oup, gaussianKernel( pixelWidth/4 ), mode="same" )"""
        
        """plt.figure(2)
        fancyPlot( this.positiveData )
        plt.figure(3)
        fancyPlot( this.negativeData )
        plt.figure(5)""" 
        
        oup = convolve2d( this.positiveData, solidCircle( int(pixelWidth) ), mode="same" ) 
        
        """fancyPlot( oup )
        plt.figure(6)"""
        
        oup = convolve2d( oup, gaussianKernel( pixelWidth/3, 0.1 ), mode="same" )
        """fancyPlot( oup )
        plt.figure(7)"""
        
        oup = np.where( this.negativeData*sharpnessMult  > oup, 0, oup )
        """fancyPlot( oup )
        plt.figure(8)
         
        fancyPlot( np.maximum( this.positiveData, oup ) - this.negativeData )
        
        plt.show()""" 

        this.mapEstimate = np.minimum( np.maximum( this.positiveData, oup ), 1  ) - this.negativeData