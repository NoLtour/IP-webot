 
import numpy as np

def draw_line(mat, x0, y0, x1, y1, termVal, lineVal ):
    if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
            0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
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

class ProbabiltyGrid: 
    width: int
    height: int
    gridData: np.ndarray
    
    xMin:int
    xMax:int
    yMin:int
    yMax:int
    cellRes:float
    
    @staticmethod 
    def initFromLinecasts( cellRes:float, originX: float, originY: float, lineTerminalX: np.ndarray, lineTerminalY: np.ndarray, terminatorWeight:float, interceptWeight:float, disableClip=False ):
        minX = np.min( lineTerminalX )
        maxX = np.max( lineTerminalX )
        minY = np.min( lineTerminalY )
        maxY = np.max( lineTerminalY )
        
        nProbGrid = ProbabiltyGrid( minX, maxX, minY, maxY, cellRes )
        nProbGrid.addLines( originX, originY, lineTerminalX, lineTerminalY, terminatorWeight, interceptWeight, disableClip )
        
        return nProbGrid
    
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
    
    def addLines(this, originX: float, originY: float, lineTerminalX: np.ndarray, lineTerminalY: np.ndarray, terminatorWeight:float, interceptWeight:float, disableClip=False ):
        originXCell = int((originX- this.xMin)*this.cellRes)
        originYCell = int((originY - this.yMin)*this.cellRes)
        
        for i in range( 0, (lineTerminalX.size) ):
            termXCell = int((lineTerminalX[i]- this.xMin)*this.cellRes)
            termYCell = int((lineTerminalY[i] - this.yMin)*this.cellRes)
            
            bounded = True
            
            if ( termXCell >= this.width ):
                termXCell = this.width-1
                bounded = False
            elif( termXCell < 0 ):
                termXCell = 0
                bounded = False
            if ( termYCell >= this.height ):
                termYCell = this.height-1
                bounded = False
            elif( termYCell < 0 ):
                termYCell = 0
                bounded = False
            
            draw_line( this.gridData, originXCell, originYCell, termXCell, termYCell, terminatorWeight if bounded else interceptWeight, interceptWeight )
        
        if ( not disableClip ):
            this.gridData.clip(0, 1)
        
    def debugLinecast(this, x0, y0, absAngle, length ):
        this.addLines(
            x0, y0, [x0 + length*np.cos(absAngle) ], [y0 + length*np.sin(absAngle) ], 0.1, 0.01
        )
         
    
    def addValue(this, x, y, v): 
        this.gridData[y, x] += v
        
        
 

