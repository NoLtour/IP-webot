from RawScanFrame import RawScanFrame

import matplotlib.pyplot as plt
import livePlotter as lp
from IPConfig import IPConfig

import numpy as np
from scipy.ndimage import rotate
from itertools import product

from Chunk import Chunk
 
from scipy.interpolate import RBFInterpolator 
from scipy.optimize import minimize

from CommonLib import fancyPlot

print("importing...")
allScanData = RawScanFrame.importScanFrames( "cleanDataBackup" )
print("imported")

# Noise step
np.random.seed(3115)
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

def genTestingSets( xSettings, ySettings, aSettings, numb ):
    xVals = np.arange( xSettings[0], xSettings[1], xSettings[2] )
    yVals = np.arange( ySettings[0], ySettings[1], ySettings[2] )
    aVals = np.arange( aSettings[0], aSettings[1], aSettings[2] )

    allCombos = list(product( xVals, yVals, aVals ))
    sampleStep = int(len(allCombos)/numb)
    if ( sampleStep < 1 ):return allCombos
    
    return [allCombos[i] for i in range(0, len(allCombos), sampleStep)]
 

def featurelessAlignerTest( testingArray:np.ndarray, inpChunk  ): 
    resultsArray = []

    for cTestData in testingArray: 
        resultsArray.append( inpChunk.determineErrorFeatureless2( inpChunk, cTestData, False ) ) 
    
    resultsArray = np.array( resultsArray ) 

    signs = np.where(testingArray<0, -1, 1)

    #underShoot = signs*(  testingArray-resultsArray  )

    return resultsArray

def featurelessFullTest( inpChunk ):
    xInpErrors   = np.arange(-0.2,0.2,0.01)
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

    
    yInpErrors   = np.arange(-0.2,0.2,0.01)
    yTestDataSet = np.zeros( (yInpErrors.size, 3) )
    yTestDataSet[:,1] = yInpErrors 
    results = featurelessAlignerTest( yTestDataSet, inpChunk )
    
    plt.figure(37)
    plt.plot( yInpErrors, results[:,0], "r--", label="x" )
    plt.plot( yInpErrors, yInpErrors, "b-", label="y real" )
    plt.plot( yInpErrors, results[:,1], "b:", label="y" )
    plt.plot( yInpErrors, results[:,2], "g.", label="a (rad)" )
    #plt.plot( yInpErrors, results[:,2]-np.arctan( results[:,1],results[:,0] )*0.5, "m-" ) 
    plt.title("Response to pure y error")
    plt.legend()
    plt.xlabel("input error")
    plt.ylabel("compensation undershoot")

    
    aInpErrors   = np.deg2rad(np.arange(-18,18,0.6))
    aTestDataSet = np.zeros( (aInpErrors.size, 3) )
    aTestDataSet[:,2] = aInpErrors 
    results = featurelessAlignerTest( aTestDataSet, inpChunk )
    
    plt.figure(38)
    plt.plot( aInpErrors, results[:,0], "r--", label="x" )
    plt.plot( aInpErrors, results[:,1], "b:" , label="y") 
    plt.plot( aInpErrors, aInpErrors, "g-", label="angle real" )
    plt.plot( aInpErrors, results[:,2], "g.", label="angle (rad)" )
    #plt.plot( aInpErrors, -np.arctan( results[:,1],results[:,0] )*0.5, "m-" ) 
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

def featurelessAutoTune( inpChunk, tuneXOffset=0.14, tuneYOffset=0.14, tuneAOffset=np.deg2rad(10) ):
    inpChunk.config.FEATURELESS_X_ERROR_SCALE = 1
    inpChunk.config.FEATURELESS_Y_ERROR_SCALE = 1
    inpChunk.config.FEATURELESS_A_ERROR_SCALE = 0
    xError, yError, angleError = inpChunk.determineErrorFeatureless2( inpChunk, (tuneXOffset,0,0), False )
    inpChunk.config.FEATURELESS_X_ERROR_SCALE = tuneXOffset/xError
     
    xError, yError, angleError = inpChunk.determineErrorFeatureless2( inpChunk, (0,tuneYOffset,0), False )
    inpChunk.config.FEATURELESS_Y_ERROR_SCALE = tuneYOffset/yError
     
    inpChunk.config.FEATURELESS_A_ERROR_SCALE = 1
    xError, yError, angleError = inpChunk.determineErrorFeatureless2( inpChunk, (0,0,tuneAOffset), False )
    inpChunk.config.FEATURELESS_A_ERROR_SCALE = tuneAOffset/angleError
    
    """# fix 2
    xError, yError, angleError = inpChunk.determineErrorFeatureless2( inpChunk, (tuneXOffset,0,0), False )
    inpChunk.config.FEATURELESS_X_ERROR_SCALE = tuneXOffset/xError
     
    xError, yError, angleError = inpChunk.determineErrorFeatureless2( inpChunk, (0,tuneYOffset,0), False )
    inpChunk.config.FEATURELESS_Y_ERROR_SCALE = tuneYOffset/yError
     
    xError, yError, angleError = inpChunk.determineErrorFeatureless2( inpChunk, (0,0,tuneAOffset), False )
    inpChunk.config.FEATURELESS_A_ERROR_SCALE = tuneAOffset/angleError"""

    print_rounded("autoTuned:",inpChunk.config.FEATURELESS_X_ERROR_SCALE,inpChunk.config.FEATURELESS_Y_ERROR_SCALE,inpChunk.config.FEATURELESS_A_ERROR_SCALE)
  


def print_rounded(*args, sf=5):
    """Prints the arguments rounded to specified number of significant figures."""
    rounded_args = [round(arg, sf - int(np.floor(np.log10(abs(arg)))) - 1) if (isinstance(arg, float) and not np.isnan(arg)) else arg for arg in args]
    print(*rounded_args)

def findDifference( inpChunk1, inpChunk2, initOffset, maxIterations, showGraph=False ):
    trueOffset = initOffset

    offsets = []
    errors  = []

    allResults = []
    
    errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )
    pErrorScore = errorScore
    initES = errorScore

    offsets.append( np.array(trueOffset) )
    errors.append( errorScore )
    
    for i in range(0, maxIterations):

        xError, yError, angleError = inpChunk1.determineErrorFeatureless2( inpChunk2, trueOffset, False )
        if ( xError+yError+angleError == 0 or abs(angleError)>np.pi/1.6 ):
            break
        #rf = min( max( errorScore/8, 0.1 ), 0.8 )
        rf = 0.6
        trueOffset = ( trueOffset[0]-xError*rf, trueOffset[1]-yError*rf, trueOffset[2]-angleError*rf ) 
        #if ( np.isnan(trueOffset[2]) ): return initOffset, initES
        pErrorScore = errorScore
        errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )

        offsets.append( np.array(trueOffset) )
        errors.append( errorScore )
        allResults.append( [errorScore,trueOffset[0],trueOffset[1],trueOffset[2] ] )

        #print_rounded("e:",errorScore,"   x:", trueOffset[0],"   y:", trueOffset[1],"   a:", trueOffset[2])
    
    offsets = np.array( offsets )
    errors  = np.array( errors )
    cMinIndex = np.argmin( errors )

    if ( showGraph ):
        allResults = np.array(allResults)
        plt.figure(353)
        plt.plot( allResults[:,0], "m-", label="error" )
        plt.legend()
        plt.figure(354)
        plt.plot( np.abs(allResults[:,1]), "r--", label="x" )
        plt.plot( np.abs(allResults[:,2]), "b:" , label="y")
        plt.plot( np.abs(allResults[:,3]), "g-" , label="a")
        plt.legend()
        plt.show()

    """interpFunc = RBFInterpolator( offsets, errors  )
    cMinIndex = np.argmin( errors )
    nm = minimize( interpFunc, offsets[cMinIndex] )
 
    trueOffset = ( nm.x[0], nm.x[1], nm.x[2] ) 
    errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )
    print("!e:",errorScore,"   x:", trueOffset[0],"   y:", trueOffset[1],"   a:", trueOffset[2])"""

    return offsets[errors.size-1], errors[errors.size-1], errors.size-1

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

def superTuner( inpChunk ):
    featurelessAutoTune( inpChunk )
    
    conf = inpChunk.config
    #conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.ANGLE_OVERWIRTE_THRESHOLD = 0.01040241, 0.01857013, 0.0087708,  0.06650539

    xScale = inpChunk.config.FEATURELESS_X_ERROR_SCALE 
    yScale = inpChunk.config.FEATURELESS_Y_ERROR_SCALE 
    aScale = inpChunk.config.FEATURELESS_A_ERROR_SCALE 
    aThr   = inpChunk.config.ANGLE_OVERWIRTE_THRESHOLD

    testInputs = genTestingSets( (-0.25,0.25,0.025),(-0.25,0.25,0.025),(np.deg2rad(-16),np.deg2rad(16),np.deg2rad(0.7)), 15 )
    testInputs.append((0.01,0.01,0.01)) 
    
    def testAll( settings ):
        inpChunk.config.FEATURELESS_X_ERROR_SCALE = settings[0]
        inpChunk.config.FEATURELESS_Y_ERROR_SCALE = settings[1]
        inpChunk.config.FEATURELESS_A_ERROR_SCALE = settings[2]
        inpChunk.config.ANGLE_OVERWIRTE_THRESHOLD = settings[3]

        erVals = []
        fails  = 0
        for testVals in testInputs:
            foundOffset, minError, successes = findDifference( inpChunk, inpChunk, testVals, 9 )
            if ( np.isnan(minError) ):break
            erVals.append( (13-successes)*((minError)**2) )
            fails += (10-successes)

        if ( len(erVals)== 0 ):
            print("fail:" ," Set:",settings)
            return 1000
        
        avrg = np.sqrt(np.average(np.array(erVals)))
        print("avg:",avrg,"fails:",fails," Set:",settings)
        
        return avrg  

    startingScore = testAll( [xScale, yScale, aScale, aThr] )

    #findDifference( inpChunk, inpChunk, testInputs[0] )
    result = minimize( testAll, np.array([xScale, yScale, aScale, aThr]),  method="COBYLA", options={"rhobeg":np.array([xScale, yScale, aScale, aThr])*0.9,'maxiter':160} )

    conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.ANGLE_OVERWIRTE_THRESHOLD = result.x[0], result.x[1], result.x[2], result.x[3]

    print(result.x)

    featurelessFullTest( inpChunk )

    ""

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

        #superTuner( chunkStack[32] )
        #featurelessAutoTune( chunkStack[32]  )
        
        conf = chunkStack[32].config
        conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.ANGLE_OVERWIRTE_THRESHOLD = 0.01247666 ,0.01801388 ,0.00037403 ,0.06981317

        offset, error, val = findDifference( chunkStack[32],chunkStack[32], (0.1, 0.17, np.deg2rad(8)), 40, True ) 
        #featurelessFullTest( chunkStack[32] )

        chunkStack = []






""









