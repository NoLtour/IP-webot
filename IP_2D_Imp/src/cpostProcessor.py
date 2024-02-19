import numpy as np
import datetime
from pathlib import Path

from scipy.signal import convolve2d   
from scipy.ndimage import laplace, maximum_filter, minimum_filter, zoom
from matplotlib import pyplot as plt 
 

from Navigator import Navigator, CartesianPose
from Mapper import Mapper, MapperConfig
from ProbabilityGrid import exportScanFrames, importScanFrames
from ImageProcessor import ImageProcessor 

from time import sleep
import livePlotter as lp
 
class MAP_PROP:
    X_MIN = -4
    X_MAX = 6
    Y_MIN = -6
    Y_MAX = 4
    PROB_GRID_RES = 25
 
lpWindow = lp.PlotWindow(5, 15)
gridDisp      = lp.LabelledGridGraphDisplay(5,0,5,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
gridDisp2     = lp.LabelledGridGraphDisplay(10,0,15,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
 

mConfig = MapperConfig()
 
 
def test1( inpArray:np.ndarray ):
    plt.figure(20)

    pos = [ 122, 166 ] # frame len()==3 [ 53, 238 ] [ 55, 182 ]

    plt.imshow( inpArray[ pos[1]-7:pos[1]+8, pos[0]-7:pos[0]+8 ] )#
    #plt.plot( pos[0], pos[1], "rx" )

    outputs = ImageProcessor.extractOrientations( inpArray, np.array([pos[0]]), np.array([pos[1]]), 7, 12 )

    plt.figure(4)
    plt.plot( outputs[0] )

    outputs = ImageProcessor.extractOrientations( inpArray, np.array([pos[0]]), np.array([pos[1]]), 7, 50 )

    plt.figure(5)
    plt.plot( outputs[0] )

    outputs = ImageProcessor.extractOrientations( inpArray, np.array([pos[0]]), np.array([pos[1]]), 7, 200 )

    plt.figure(6)
    plt.plot( outputs[0] )

    plt.show()

print("importing test data...")
allScansRaw = importScanFrames( "testRunData" )
print("imported test data")
mapper = Mapper( None, mConfig )

prevScan = 0

for cRawScan in allScansRaw:
    mapper.pushScanFrame( cRawScan )
    
    scan = mapper.analyseRecentScan()

    if ( scan != None ): 
        if ( np.max( scan.constructedProbGrid.positiveData ) != 0 ): 
            fPos = scan.featurePositions
            
            gridDisp.parseData( scan.estimatedMap, fPos[:,1], fPos[:,0]  )
            #gridDisp2.parseData( Rval*1000, maxPos[:,1], maxPos[:,0]  )

            prevScan = scan

            lpWindow.render()
 
            #sleep(0.1)


""