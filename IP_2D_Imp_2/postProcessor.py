from RawScanFrame import RawScanFrame

import matplotlib.pyplot as plt
import livePlotter as lp
from IPConfig import IPConfig

import numpy as np
from scipy.ndimage import rotate

from Chunk import Chunk
 
from scipy.interpolate import RBFInterpolator 
from scipy.optimize import minimize

from CommonLib import fancyPlot

print("importing...")
allScanData = RawScanFrame.importScanFrames( "cleanDataBackup" )
print("imported")

# Noise step
for cScan in allScanData:
    cScan.scanDistances = cScan.scanDistances + 0.01*(np.random.random( cScan.scanDistances.size )-0.5)
    #cScan.pose = cScan.truePose # TODO remove

config = IPConfig()

class MAP_PROP:
    X_MIN = -4
    X_MAX = 6
    Y_MIN = -6
    Y_MAX = 4
    PROB_GRID_RES = config.GRID_RESOLUTION

lpWindow = lp.PlotWindow(5, 15)
gridDisp      = lp.LabelledGridGraphDisplay(5,0,5,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
gridDisp2     = lp.LabelledGridGraphDisplay(10,0,15,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
 

rawStack = []
chunkStack = []

def featurelessAlignerTest( testingArray:np.ndarray, inpChunk  ): 
    resultsArray = []

    for cTestData in testingArray: 
        resultsArray.append( inpChunk.determineErrorFeatureless( inpChunk, cTestData, False ) ) 
    
    resultsArray = np.array( resultsArray ) 

    signs = np.where(testingArray<0, -1, 1)

    #underShoot = signs*(  testingArray-resultsArray  )

    return resultsArray

def featurelessFullTest( inpChunk ):
    xInpErrors   = np.arange(-0.2,0.2,0.014)
    xTestDataSet = np.zeros( (xInpErrors.size, 3) )
    xTestDataSet[:,0] = xInpErrors 
    results = featurelessAlignerTest( xTestDataSet, inpChunk )
    
    plt.figure(35)
    plt.plot( xInpErrors, results[:,0], "r--", label="x" )
    plt.plot( xInpErrors, xInpErrors, "r-", label="x real" )
    plt.plot( xInpErrors, results[:,1], "b:", label="y" )
    plt.plot( xInpErrors, results[:,2], "g.", label="a (rad)" )
    plt.legend()
    plt.title("Response to pure x error")
    plt.xlabel("input error")
    plt.ylabel("compensation undershoot")

    
    yInpErrors   = np.arange(-0.2,0.2,0.014)
    yTestDataSet = np.zeros( (yInpErrors.size, 3) )
    yTestDataSet[:,1] = yInpErrors 
    results = featurelessAlignerTest( yTestDataSet, inpChunk )
    
    plt.figure(37)
    plt.plot( yInpErrors, results[:,0], "r--", label="x" )
    plt.plot( yInpErrors, yInpErrors, "b-", label="y real" )
    plt.plot( yInpErrors, results[:,1], "b:", label="y" )
    plt.plot( yInpErrors, results[:,2], "g.", label="a (rad)" )
    plt.plot( yInpErrors, results[:,2]-np.arctan( results[:,1],results[:,0] )*0.5, "m-" ) 
    plt.title("Response to pure y error")
    plt.legend()
    plt.xlabel("input error")
    plt.ylabel("compensation undershoot")

    
    aInpErrors   = np.deg2rad(np.arange(-20,20,1.5))
    aTestDataSet = np.zeros( (aInpErrors.size, 3) )
    aTestDataSet[:,2] = aInpErrors 
    results = featurelessAlignerTest( aTestDataSet, inpChunk )
    
    plt.figure(38)
    plt.plot( aInpErrors, results[:,0], "r--", label="x" )
    plt.plot( aInpErrors, results[:,1], "b:" , label="y") 
    plt.plot( aInpErrors, aInpErrors, "g-", label="angle real" )
    plt.plot( aInpErrors, results[:,2], "g.", label="angle (rad)" )
    plt.plot( aInpErrors, -np.arctan( results[:,1],results[:,0] )*0.5, "m-" ) 
    plt.title("Response to pure angle error")
    plt.legend()
    plt.xlabel("input error (rad)")
    plt.ylabel("compensation undershoot")
    
    plt.show()
    """plt.figure(36)
    plt.plot( yInErrors, yInErrors-yPrErrors )
    plt.xlabel("input error")
    plt.ylabel("compensation overshoot")"""

    plt.show()

def findDifference( inpChunk1, inpChunk2, initOffset, maxIterations ):
    trueOffset = initOffset

    offsets = []
    errors  = []
    
    errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )

    for i in range(0, maxIterations):
        xError, yError, angleError = inpChunk1.determineErrorFeatureless( inpChunk2, trueOffset, False )
        rf = min( max( errorScore/8, 0.1 ), 0.8 )
        #rf = 0.7
        trueOffset = ( trueOffset[0]-xError*rf, trueOffset[1]-yError*rf, trueOffset[2]-angleError*rf ) 
        errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )

        offsets.append( np.array(trueOffset) )
        errors.append( errorScore )

        print("e:",errorScore,"   x:", trueOffset[0],"   y:", trueOffset[1],"   a:", trueOffset[2])
    
    offsets = np.array( offsets )
    errors  = np.array( errors )
    cMinIndex = np.argmin( errors )
    """interpFunc = RBFInterpolator( offsets, errors  )
    cMinIndex = np.argmin( errors )
    nm = minimize( interpFunc, offsets[cMinIndex] )
 
    trueOffset = ( nm.x[0], nm.x[1], nm.x[2] ) 
    errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )
    print("!e:",errorScore,"   x:", trueOffset[0],"   y:", trueOffset[1],"   a:", trueOffset[2])"""

    return offsets[cMinIndex], errors[cMinIndex]

def findDifference2( inpChunk1, inpChunk2, initOffset, maxIterations ):
    trueOffset = initOffset

    offsets = []
    errors  = []
    
    errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )


    def interpFunc( offsets ):
        return inpChunk1.determineDirectDifference( inpChunk2, offsets )[0]

    calcOffset = np.array(inpChunk1.determineErrorFeatureless( inpChunk2, trueOffset, False )) 

    nm = minimize( interpFunc, np.array(trueOffset),  method="COBYLA", options={"rhobeg":calcOffset} )
 
    trueOffset = ( nm.x[0], nm.x[1], nm.x[2] ) 
    errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset ) 

    return trueOffset, errorScore

for i in range( 0, len(allScanData) ):
    cRawScan:RawScanFrame = allScanData[i]

    rawStack.append( cRawScan )
    
    midScan = rawStack[int(len(rawStack)/2)]
    if ( abs(cRawScan.pose.yaw - midScan.pose.yaw) > config.MAX_INTER_FRAME_ANGLE or len(rawStack) > config.MAX_FRAMES_MERGE ):
        # Frame merge 
        nChunk = Chunk.initFromRawScans( rawStack[0:-1], config, 0 )
        rawStack = [ rawStack[-1] ] 

        if ( len( chunkStack ) == 12 or len( chunkStack ) == 32 ):
            nChunk.constructProbabilityGrid() 
 
            gridDisp2.parseData( nChunk.cachedProbabilityGrid.mapEstimate ) 
        
        #hardEstimate = np.where( nChunk.cachedProbabilityGrid.mapEstimate<-0.3, -1, 0 ) + np.where( nChunk.cachedProbabilityGrid.mapEstimate>0.3, 1, 0 )
       
        #gridDisp.parseData(hardEstimate  ) 

        #gridDisp.parseData( nChunk.cachedProbabilityGrid.copyRotated( np.deg2rad( 20 ) ).mapEstimate  ) 
            lpWindow.render()  

        chunkStack.append( nChunk )
    
    if ( len( chunkStack ) > 37 ):
        """fancyPlot( chunkStack[12].cachedProbabilityGrid.mapEstimate )
        fancyPlot( chunkStack[32].cachedProbabilityGrid.mapEstimate )
        plt.show()"""

        parentChunk = Chunk.initEmpty( config )

        parentChunk.addChunks( chunkStack )

        #xError, yError, angleError = chunkStack[32].determineErrorFeatureless( chunkStack[32], (0,0,np.deg2rad(10)), False )

        offset, error = findDifference( chunkStack[32],chunkStack[32], (0.06, 0.06, np.deg2rad(-8)), 60 ) 
        featurelessFullTest( chunkStack[32] )

        chunkStack = []






""









