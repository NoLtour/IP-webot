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
from scipy.optimize import differential_evolution

from CommonLib import fancyPlot

print("importing...")
allScanData = RawScanFrame.importScanFrames( "cleanData" )
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
     
    """plt.figure(36)
    plt.plot( yInErrors, yInErrors-yPrErrors )
    plt.xlabel("input error")
    plt.ylabel("compensation overshoot")"""

    plt.show()

def featurelessAutoTune( inpChunk, tuneXOffset=0.14, tuneYOffset=0.14, tuneAOffset=np.deg2rad(7) ):
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

    return offsets[cMinIndex], errors[cMinIndex], errors.size-1

def findDifference2( inpChunk1, inpChunk2, initOffset, maxIterations ):
    trueOffset = initOffset

    offsets = []
    errors  = []
    
    errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )


    def interpFunc( offsets ):
        error, area = inpChunk1.determineDirectDifference( inpChunk2, offsets )

        if (np.isnan(error)):
            return 1000
        
        return error

    #calcOffset = np.array(inpChunk1.determineErrorFeatureless2( inpChunk2, trueOffset, False )) 

    nm = minimize( interpFunc, np.array(trueOffset),  method="COBYLA", options={ 'maxiter': maxIterations} )
 
    trueOffset = ( nm.x[0], nm.x[1], nm.x[2] ) 
    errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset ) 
    #inpChunk1.plotDifference( inpChunk2, trueOffset ) 

    return trueOffset, errorScore, 1

def superTuner( inpChunk, scenarioCount=40, searchCount=5 ):
    featurelessAutoTune( inpChunk )
    
    conf = inpChunk.config
    #conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.CONFLICT_MULT_GAIN = 0.01194487, 0.00555347, 0.00045558, 0.07988497
    #conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.CONFLICT_MULT_GAIN = 0.01897726, 0.00751112 ,0.00136641 ,0.11150678
    #avg: 19.87302725899315 fails: 0  Set: [0.03100083 0.00930756 0.00110901 0.20546921]
    #avg: 18.42025281767809 fails: 0  Set: [0.03089128 0.00952647 0.0011058  0.20442749]
    
    xScale = inpChunk.config.FEATURELESS_X_ERROR_SCALE 
    yScale = inpChunk.config.FEATURELESS_Y_ERROR_SCALE 
    aScale = inpChunk.config.FEATURELESS_A_ERROR_SCALE 
    aThr   = inpChunk.config.CONFLICT_MULT_GAIN

    testInputs = genTestingSets( (-0.25,0.25,0.025),(-0.25,0.25,0.025),(np.deg2rad(-16),np.deg2rad(16),np.deg2rad(0.7)), scenarioCount )
    testInputs.append((0.01,0.01,0.01)) 
    
    def testAll( settings ):
        inpChunk.config.FEATURELESS_X_ERROR_SCALE = settings[0]
        inpChunk.config.FEATURELESS_Y_ERROR_SCALE = settings[1]
        inpChunk.config.FEATURELESS_A_ERROR_SCALE = settings[2]
        inpChunk.config.CONFLICT_MULT_GAIN = settings[3]

        try:
            erVals = []
            fails  = 0
            for testVals in testInputs:
                foundOffset, minError, successes = findDifference( inpChunk, inpChunk, testVals, searchCount )
                minError = (foundOffset[0]**2+foundOffset[1]**2+foundOffset[2]**2)
                if ( np.isnan(minError) ):break
                erVals.append( ((minError)**2) )
                fails += (searchCount -successes)

            if ( len(erVals)== 0 ):
                print("fail:" ," Set:",settings)
                return 1000
            
            avrg = np.sqrt(np.average(np.array(erVals)))
            print("avg:",avrg,"fails:",fails," Set:",settings)
            
            return avrg  
        except:
            print("critical error")
            return 10000

    #startingScore = testAll( [xScale, yScale, aScale, aThr] )

    #findDifference( inpChunk, inpChunk, testInputs[0] )
    #result = minimize( testAll, np.array([xScale, yScale, aScale, aThr]),  method="COBYLA", options={"rhobeg":np.array([xScale, yScale, aScale, aThr])*0.9,'maxiter':30} )
    result = differential_evolution( testAll, [(xScale*0.1,xScale*3), (yScale*0.1,yScale*3), (aScale*0.1,aScale*3), (aThr*0.1,aThr*3)], x0=np.array([xScale, yScale, aScale, aThr]) )

    conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.CONFLICT_MULT_GAIN = result.x[0], result.x[1], result.x[2], result.x[3]

    print(result.x)

    featurelessFullTest( inpChunk )

    ""

def errorTester( inpChunk ):   
    xOffsets = np.arange( -0.35, 0.35, 0.025 )
    yOffsets = np.arange( -0.35, 0.35, 0.025 )

    yO, xO = np.meshgrid( xOffsets, yOffsets )
    rO = np.zeros( xO.shape )

    offsetTests = np.column_stack( (xO.flatten(), yO.flatten(), rO.flatten()) )
    offsetErrors = []

    rotationTests  = np.deg2rad(np.arange( -60, 60, 0.25 ))
    rotationErrors = []

    for offsetTest in offsetTests:
        errorScore, overlapArea = inpChunk.determineDirectDifference( inpChunk, offsetTest )
        offsetErrors.append( errorScore )

    for rotationTest in rotationTests:
        errorScore, overlapArea = inpChunk.determineDirectDifference( inpChunk, np.array( (0,0,rotationTest) ) )
        rotationErrors.append( errorScore )

    plt.figure(1512)
    """plt.xticks(np.arange( -0.3, 0.3, 0.1 ))
    plt.yticks(np.arange( -0.3, 0.3, 0.1 ))"""
    bar = plt.imshow(np.array( offsetErrors ).reshape( xO.shape ), extent=[np.min(xOffsets), np.max(xOffsets), np.min(yOffsets), np.max(yOffsets)]) 
    plt.colorbar( bar )
    plt.title("error function output for known displacement error")
    plt.xlabel("x displacement error")
    plt.ylabel("y displacement error")

    plt.figure(1514)
    plt.plot( np.rad2deg(rotationTests), np.array(rotationErrors) )
    plt.title("error function output for known rotational error")
    plt.xlabel("rotation error (deg)")
    plt.ylabel("error score")
    plt.show()

    "4"
  
def method1Tester( inpChunk ):   
    xOffsets = np.arange( -0.35, 0.35, 0.04 )
    yOffsets = np.arange( -0.35, 0.35, 0.04 )

    yO, xO = np.meshgrid( xOffsets, yOffsets )
    rO = np.zeros( xO.shape )

    offsetTests = np.column_stack( (xO.flatten(), yO.flatten(), rO.flatten()) )
    offsetErrors = []

    rotationTests  = np.deg2rad(np.arange( -60, 60, 0.5 ))
    rotationErrors = []

    for offsetTest in offsetTests:
        initialError, overlapArea = inpChunk.determineDirectDifference( inpChunk, offsetTest )
        predError, newError, v = findDifference( inpChunk, inpChunk, offsetTest, 6 )
        offsetErrors.append( -100*(newError-initialError)/initialError ) 

    for rotationTest in rotationTests: 
        initialError, overlapArea = inpChunk.determineDirectDifference( inpChunk, np.array( (0,0,rotationTest) ) )
        predError, newError, v = findDifference( inpChunk, inpChunk, np.array( (0,0,rotationTest) ), 6 )
        rotationErrors.append( -100*(newError-initialError)/initialError ) 

    plt.figure(1512)
    """plt.xticks(np.arange( -0.3, 0.3, 0.1 ))
    plt.yticks(np.arange( -0.3, 0.3, 0.1 ))"""
    bar = plt.imshow(np.array( offsetErrors ).reshape( xO.shape ), extent=[np.min(xOffsets), np.max(xOffsets), np.min(yOffsets), np.max(yOffsets)]) 
    plt.colorbar( bar )
    plt.title("Percentage error reduction")
    plt.xlabel("x displacement error")
    plt.ylabel("y displacement error")

    plt.figure(1514)
    plt.plot( np.rad2deg(rotationTests), np.array(rotationErrors) )
    plt.title("Percentage error reduction")
    plt.xlabel("rotation error (deg)")
    plt.ylabel("percent error reduction")
    plt.show()

    "4"     

def method2Tester( inpChunk ):   
    xOffsets = np.arange( -0.35, 0.35, 0.04 )
    yOffsets = np.arange( -0.35, 0.35, 0.04 )

    yO, xO = np.meshgrid( xOffsets, yOffsets )
    rO = np.zeros( xO.shape )

    offsetTests = np.column_stack( (xO.flatten(), yO.flatten(), rO.flatten()) )
    offsetErrors = []

    rotationTests  = np.deg2rad(np.arange( -60, 60, 0.5 ))
    rotationErrors = []

    for offsetTest in offsetTests:
        initialError, overlapArea = inpChunk.determineDirectDifference( inpChunk, offsetTest )
        predError, newError, v = findDifference2( inpChunk, inpChunk, offsetTest, 16 )
        offsetErrors.append( -100*(newError-initialError)/initialError ) 

    for rotationTest in rotationTests: 
        initialError, overlapArea = inpChunk.determineDirectDifference( inpChunk, np.array( (0,0,rotationTest) ) )
        predError, newError, v = findDifference2( inpChunk, inpChunk, np.array( (0,0,rotationTest) ), 16 )
        rotationErrors.append( -100*(newError-initialError)/initialError ) 

    plt.figure(1512)
    """plt.xticks(np.arange( -0.3, 0.3, 0.1 ))
    plt.yticks(np.arange( -0.3, 0.3, 0.1 ))"""
    bar = plt.imshow(np.array( offsetErrors ).reshape( xO.shape ), extent=[np.min(xOffsets), np.max(xOffsets), np.min(yOffsets), np.max(yOffsets)]) 
    plt.colorbar( bar )
    plt.title("Percentage error reduction")
    plt.xlabel("x displacement error")
    plt.ylabel("y displacement error")

    plt.figure(1514)
    plt.plot( np.rad2deg(rotationTests), np.array(rotationErrors) )
    plt.title("Percentage error reduction")
    plt.xlabel("rotation error (deg)")
    plt.ylabel("percent error reduction")
    plt.show()

    "4"      

def mapMergeTest( frameCount ):
    rawStack = []
    chunkStack = []

    for i in range( 0, len(allScanData) ):
        cRawScan:RawScanFrame = allScanData[i]

        rawStack.append( cRawScan )
        
        midScan = rawStack[int(len(rawStack)/2)]
        if ( abs(cRawScan.pose.yaw - midScan.pose.yaw) > config.MAX_INTER_FRAME_ANGLE or len(rawStack) > config.MAX_FRAMES_MERGE ):
            # Frame merge 
            nChunk = Chunk.initFromRawScans( rawStack[0:-1], config, 0 )
            rawStack = [ rawStack[-1] ] 

            nChunk.constructProbabilityGrid() 

            gridDisp2.parseData( nChunk.cachedProbabilityGrid.mapEstimate ) 
            
            lpWindow.render()  

            chunkStack.append( nChunk )
        
        if ( len( chunkStack ) > frameCount or len(allScanData)-1==i ):
            """fancyPlot( chunkStack[12].cachedProbabilityGrid.mapEstimate )
            fancyPlot( chunkStack[32].cachedProbabilityGrid.mapEstimate )
            plt.show()"""

            parentChunk = Chunk.initEmpty( config )

            parentChunk.addChunks( chunkStack )

            #parentChunk.centredFeaturelessErrorReduction( True )
 
            fancyPlot(
                parentChunk.constructProbabilityGrid().mapEstimate
            )
            plt.show()

            return
            



mapMergeTest( 20 )

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

        #errorTester( chunkStack[32] )
        superTuner( chunkStack[32] )
        #featurelessAutoTune( chunkStack[32]  )
        
        conf = chunkStack[32].config
        #conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.CONFLICT_MULT_GAIN = 0.00147138 ,0.00365946 ,0.00283191 ,0.400286
        
        #method1Tester( chunkStack[32] )

        #offset, error, val = findDifference2( chunkStack[32],chunkStack[32], (0.1, -0.1, np.deg2rad(10)), 30 ) 
        #offset, error, val = findDifference( chunkStack[32],chunkStack[32], (0.1, -0.1, np.deg2rad(10)), 30, True ) 
        #featurelessFullTest( chunkStack[32] )

        chunkStack = []






""









