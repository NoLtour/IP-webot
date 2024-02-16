 
import numpy as np
from dataclasses import dataclass, field
from Navigator import CartesianPose
from scipy.signal import convolve2d

from skimage.draw import polygon2mask, line

def draw_line(mat, x0, y0, x1, y1, termVal, lineVal ):
    if not (0 <= x0 < mat.shape[1] and 0 <= x1 < mat.shape[1] and
            0 <= y0 < mat.shape[0] and 0 <= y1 < mat.shape[0]):
        raise ValueError('Invalid coordinates.')
     
    if (x0, y0) == (x1, y1):
        mat[y0, x0] = termVal
        return 
    # Swap axes if Y slope is smaller than X slope
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1
    
    mat[y1, x1] += termVal
    # Swap line direction to go left-to-right if necessary
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0 
         
    
    # Compute intermediate coordinates using line equation
    x = np.arange(x0 + 1, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
    # Write intermediate coordinates
    mat[y, x] += lineVal 
    
    if ( transpose ):
        mat = mat.T

def gaussian_kernel(size, sigma=1):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

@dataclass
class ScanFrame:
    """Dataclass that stores a set of points representing the scan, and a pose from which the scans thought to be taken"""
    #pointXs: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    #pointYs: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    scanDistances: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    scanAngles: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    
    pose: CartesianPose = field(default_factory=lambda: CartesianPose.zero() )

    calculatedTerminations: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    calculatedInfTerminations: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def calculateTerminations( this, maxLength ):
        isInf = np.isinf( this.scanDistances )
        
        # A version of scan distances where values: inf -> maxLength
        adjustedDistances = np.where( isInf, maxLength, this.scanDistances )
        
        xPoints = this.pose.x + adjustedDistances * np.cos( this.scanAngles + this.pose.yaw )
        yPoints = this.pose.y + adjustedDistances * np.sin( this.scanAngles + this.pose.yaw )
    
        this.calculatedTerminations = np.array([xPoints, yPoints])
        this.calculatedInfTerminations = isInf

class ProbabilityGrid: 
    width: int
    height: int
    gridData: np.ndarray
    
    xMin:int
    xMax:int
    yMin:int
    yMax:int
    cellRes:float
    
    @staticmethod 
    def initFromLinecasts( cellRes:float, originX: float, originY: float, lineTerminalX: np.ndarray, lineTerminalY: np.ndarray, terminatorWeight:float, interceptWeight:float ):
        minX = np.min( lineTerminalX )
        maxX = np.max( lineTerminalX )
        minY = np.min( lineTerminalY )
        maxY = np.max( lineTerminalY )
        
        nProbGrid = ProbabilityGrid( minX, maxX, minY, maxY, cellRes )
        nProbGrid.addLines( originX, originY, lineTerminalX, lineTerminalY, terminatorWeight, interceptWeight )
        
        return nProbGrid
    
    
    @staticmethod 
    def initFromScanFrames( cellRes:float, scanStack:list[ScanFrame], terminatorWeight:float, interceptWeight:float, maxScanDistance:float ):
        xMin = 10000000000
        xMax = -10000000000
        yMin = 10000000000
        yMax = -10000000000

        middleScan = scanStack[int( len(scanStack)/2 )]
        
        for targScan in scanStack:
            targScan.calculateTerminations( maxScanDistance )
            xMax = max( targScan.pose.x, xMax, np.max( targScan.calculatedTerminations[0] ) )
            xMin = min( targScan.pose.x, xMin, np.min( targScan.calculatedTerminations[0] ) )
            yMax = max( targScan.pose.y, yMax, np.max( targScan.calculatedTerminations[1] ) )
            yMin = min( targScan.pose.y, yMin, np.min( targScan.calculatedTerminations[1] ) )
        
        finalGrid = ProbabilityGrid( xMin-0.5, xMax+0.5, yMin-0.5, yMax+0.5, cellRes )
        #seps = []
        for targScan in scanStack:
            nProbGrid = ProbabilityGrid( xMin-0.5, xMax+0.5, yMin-0.5, yMax+0.5, cellRes )
            nProbGrid.addLines( targScan.pose.x, targScan.pose.y, 
                               targScan.calculatedTerminations[0], targScan.calculatedTerminations[1], targScan.calculatedInfTerminations, 
                               terminatorWeight, interceptWeight )
            
            midPointSeperation = np.sqrt( (targScan.pose.x - middleScan.pose.x)**2 + (targScan.pose.y - middleScan.pose.y)**2  )
            
            if ( midPointSeperation != 0 ):
                # TODO implement proper uncertainty model!
                sigma = midPointSeperation*cellRes*3
                
                finalGrid.gridData += convolve2d( nProbGrid.gridData, gaussian_kernel( 1+int(sigma*2), sigma ), mode="same" )
            else:
                finalGrid.gridData += nProbGrid.gridData
        
        finalGrid.gridData /= len(scanStack)
        #print(seps)
        
        finalGrid.clipData()
                
        return finalGrid
    
    
    @staticmethod 
    def initFromScanFramesPoly( cellRes:float, scanStack:list[ScanFrame], terminatorWeight:float, interceptWeight:float, maxScanDistance:float ):
        """Connects nearby points in the point cloud as straight lines"""
        
        xMin = 10000000000
        xMax = -10000000000
        yMin = 10000000000
        yMax = -10000000000

        middleScan = scanStack[int( len(scanStack)/2 )]
        
        for targScan in scanStack:
            targScan.calculateTerminations( maxScanDistance )
            xMax = max( targScan.pose.x, xMax, np.max( targScan.calculatedTerminations[0] ) )
            xMin = min( targScan.pose.x, xMin, np.min( targScan.calculatedTerminations[0] ) )
            yMax = max( targScan.pose.y, yMax, np.max( targScan.calculatedTerminations[1] ) )
            yMin = min( targScan.pose.y, yMin, np.min( targScan.calculatedTerminations[1] ) )
        
        finalGrid = ProbabilityGrid( xMin-0.5, xMax+0.5, yMin-0.5, yMax+0.5, cellRes )
        #seps = []
        for targScan in scanStack:
            nProbGrid = ProbabilityGrid( xMin-0.5, xMax+0.5, yMin-0.5, yMax+0.5, cellRes )
            nProbGrid.addPolyLines( targScan.pose.x, targScan.pose.y, 
                               targScan.calculatedTerminations[0], targScan.calculatedTerminations[1], targScan.calculatedInfTerminations, 
                               terminatorWeight, interceptWeight )
            
            midPointSeperation = np.sqrt( (targScan.pose.x - middleScan.pose.x)**2 + (targScan.pose.y - middleScan.pose.y)**2  )
            
            if (  midPointSeperation != 0 ):
                # TODO implement proper uncertainty model!
                sigma = midPointSeperation*cellRes*3
                
                finalGrid.gridData += convolve2d( nProbGrid.gridData, gaussian_kernel( 1+int(sigma*2), sigma ), mode="same" )
            else:
                finalGrid.gridData += nProbGrid.gridData
        
        finalGrid.gridData /= len(scanStack)
        #print(seps)
        
        finalGrid.clipData()
                
        return finalGrid
    
    def __init__(this, xMin:int, xMax:int, yMin:int, yMax:int, cellRes:float):
        this.xMin = xMin
        this.xMax = xMax
        this.yMin = yMin
        this.yMax = yMax
        this.cellRes = cellRes
        
        this.width = int((xMax - xMin)*cellRes)
        this.height = int((yMax - yMin)*cellRes)
        this.gridData = np.zeros( (this.height, this.width) )
    
    # Returns the closest cell to the provided x and y coordinates
    def getCellAt( this, x:float, y:float ):
        return  int((x- this.xMin)*this.cellRes), int((y - this.yMin)*this.cellRes)  
    
    def randomise(this):
        this.gridData = np.random.random( (this.width, this.height) )
        
    def clipData(this):
        this.gridData = this.gridData.clip( -1, 1 )
    
    def addLines(this, originX: float, originY: float, lineTerminalX: np.ndarray, lineTerminalY: np.ndarray, isInf:np.ndarray, terminatorWeight:float, interceptWeight:float ):
        originXCell = int((originX- this.xMin)*this.cellRes)
        originYCell = int((originY - this.yMin)*this.cellRes)
        
        for i in range( 0, (lineTerminalX.size) ):
            termXCell = int((lineTerminalX[i]- this.xMin)*this.cellRes)
            termYCell = int((lineTerminalY[i] - this.yMin)*this.cellRes)
            
            draw_line( this.gridData, originXCell, originYCell, termXCell, termYCell, 
                      interceptWeight if isInf[i] else terminatorWeight, 
                      interceptWeight )
    
    def addPolyLines(this, originX: float, originY: float, lineTerminalX: np.ndarray, lineTerminalY: np.ndarray, isInf:np.ndarray, terminatorWeight:float, interceptWeight:float, maxSep=0.1 ):
        # Get them into integer pixel coordinates
        originXCell = int((originX- this.xMin)*this.cellRes)
        originYCell = int((originY - this.yMin)*this.cellRes)
        termXCell = ((lineTerminalX- this.xMin)*this.cellRes).astype(int)
        termYCell = ((lineTerminalY - this.yMin)*this.cellRes).astype(int)
        
        scanPoints = np.array([termYCell, termXCell]).transpose()

        polyPoints = np.append( np.array([[originYCell, originXCell]]), scanPoints, axis=0 )

        newGridData = interceptWeight * polygon2mask( this.gridData.shape, polyPoints )
        
        lastPoint = scanPoints[0] 
        for point in scanPoints[1:]:
            # Check if points are above a certain seperation 
            if ( maxSep*this.cellRes < np.sqrt( np.square(lastPoint[0]-point[0]) + np.square(lastPoint[1]-point[1]) ) ):
                newGridData[lastPoint[0],lastPoint[1]] = terminatorWeight
            else:
                newGridData[ line( point[0], point[1], lastPoint[0], lastPoint[1] ) ] = terminatorWeight
 
            lastPoint = point

        this.gridData += newGridData
        
    def debugLinecast(this, x0, y0, absAngle, length ):
        this.addLines(
            x0, y0, [x0 + length*np.cos(absAngle) ], [y0 + length*np.sin(absAngle) ], 0.1, 0.01
        )
    
    def addValue(this, x, y, v): 
        this.gridData[y, x] += v


