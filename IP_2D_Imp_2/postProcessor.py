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
from scipy import stats
import random

from CommonLib import fancyPlot

print("importing...")
allScanData:list[RawScanFrame] = RawScanFrame.importScanFrames( "cleanDataWideNew4" )
print("imported")

# Noise step
np.random.seed(3115)
for cScan in allScanData:
    #cScan.scanDistances = cScan.scanDistances + 0.01*(np.random.random( cScan.scanDistances.size )-0.5)
    #cScan.pose = cScan.truePose # TODO remove
    ""

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

def add_trend_line(x, y, labelOffset=0.9, colour="r--", labelName="Trend line" ):
    # Fit a linear regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    xEqualSpaceing = np.array((np.min(x), np.max(x)))
    line = slope * xEqualSpaceing + intercept

    # Calculate R squared
    r_squared = r_value ** 2

    # Plot the regression line
    plt.plot(xEqualSpaceing, line, colour, label=labelName)

    # Add the R squared value as a label
    plt.text(0.2, labelOffset, f'{labelName} R^2 = {r_squared:.2f}', transform=plt.gca().transAxes) 

def genTestingSets( xSettings, ySettings, aSettings, numb ):
    xVals = np.arange( xSettings[0], xSettings[1], xSettings[2] )
    yVals = np.arange( ySettings[0], ySettings[1], ySettings[2] )
    aVals = np.arange( aSettings[0], aSettings[1], aSettings[2] )

    allCombos = list(product( xVals, yVals, aVals ))
    sampleStep = int(len(allCombos)/numb)
    
    random.Random(numb).shuffle(allCombos)
    
    if ( sampleStep < 1 ):return np.array(allCombos)
    
    oups = []
    for i in range(0, len(allCombos), sampleStep):
        oups.append( allCombos[i] )
    return np.array(oups)
    
    return [allCombos[i] for i in range(0, len(allCombos), sampleStep)]
 

def featurelessAlignerTest( testingArray:np.ndarray, inpChunk  ): 
    resultsArray = []

    for cTestData in testingArray: 
        resultsArray.append( inpChunk.determineErrorFeatureless3( inpChunk, cTestData, False ) ) 
    
    resultsArray = np.array( resultsArray ) 

    signs = np.where(testingArray<0, -1, 1)

    #underShoot = signs*(  testingArray-resultsArray  )

    return resultsArray

def featurelessFullTest( inpChunk ):
    xInpErrors   = np.arange(-0.35,0.35,0.015)
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

    
    yInpErrors   = np.arange(-0.35,0.35,0.015)
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

    
    aInpErrors   = np.deg2rad(np.arange(-22,22,0.8))
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

    plt.show( block=False )

def featurelessAutoTune( inpChunk, tuneXOffset=0.14, tuneYOffset=0.14, tuneAOffset=np.deg2rad(6) ):
    inpChunk.config.FEATURELESS_X_ERROR_SCALE = 1
    inpChunk.config.FEATURELESS_Y_ERROR_SCALE = 1
    inpChunk.config.FEATURELESS_A_ERROR_SCALE = 0
    xError, yError, angleError = inpChunk.determineErrorFeatureless3( inpChunk, (tuneXOffset,0,0), False )
    inpChunk.config.FEATURELESS_X_ERROR_SCALE = tuneXOffset/xError
     
    xError, yError, angleError = inpChunk.determineErrorFeatureless3( inpChunk, (0,tuneYOffset,0), False )
    inpChunk.config.FEATURELESS_Y_ERROR_SCALE = tuneYOffset/yError
     
    inpChunk.config.FEATURELESS_A_ERROR_SCALE = 1
    xError, yError, angleError = inpChunk.determineErrorFeatureless3( inpChunk, (0,0,tuneAOffset), False )
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

        xError, yError, angleError = inpChunk1.determineErrorFeatureless3( inpChunk2, trueOffset, False )
        if ( xError+yError+angleError == 0 or abs(angleError)>np.pi/1.6 ):
            break
        #rf = min( max( errorScore/8, 0.1 ), 0.8 )
        rf = 0.7
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
        plt.plot( errors, "m-", label="error" )
        plt.legend()
        plt.figure(354)
        plt.plot( np.abs(offsets[:,0]), "r--", label="x" )
        plt.plot( np.abs(offsets[:,1]), "b:" , label="y")
        plt.plot( np.abs(offsets[:,2]), "g-" , label="a")
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

def superTuner( inpChunk, scenarioCount=20, searchCount=4 ):
    featurelessAutoTune( inpChunk )
    
    conf = inpChunk.config
    #conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.CONFLICT_MULT_GAIN = 0.01194487, 0.00555347, 0.00045558, 0.07988497
    #conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.CONFLICT_MULT_GAIN = 1.29735511e-03 ,1.79969875e-03 ,5.99011616e-04, 6.84882795e-01
    #avg: 19.87302725899315 fails: 0  Set: [0.03100083 0.00930756 0.00110901 0.20546921]
    #avg: 18.42025281767809 fails: 0  Set: [0.03089128 0.00952647 0.0011058  0.20442749]
    
    xScale = inpChunk.config.FEATURELESS_X_ERROR_SCALE 
    yScale = inpChunk.config.FEATURELESS_Y_ERROR_SCALE 
    aScale = inpChunk.config.FEATURELESS_A_ERROR_SCALE 
    aThr   = inpChunk.config.CONFLICT_MULT_GAIN

    testInputs = genTestingSets( (-0.35,0.35,0.01),(-0.35,0.35,0.01),(np.deg2rad(-25),np.deg2rad(25),np.deg2rad(0.5)), scenarioCount ) 
    
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

def errorTesterAlt( inpChunk:Chunk ):    
    yOffsets = np.arange( -1.1, 1.1, 0.05 )
    aOffsets = np.deg2rad(np.arange( -30, 30, 1 ))

    yO, aO = np.meshgrid( aOffsets, yOffsets )
    xO = np.zeros( yO.shape )

    offsetTests = np.column_stack( (xO.flatten(), yO.flatten(), aO.flatten()) )
    offsetErrors = [] 

    for offsetTest in offsetTests:
        errorScore, overlapArea = inpChunk.determineDirectDifference( inpChunk, offsetTest )
        offsetErrors.append( errorScore ) 

    plt.figure(1512)
    """plt.xticks(np.arange( -0.3, 0.3, 0.1 ))
    plt.yticks(np.arange( -0.3, 0.3, 0.1 ))"""
    bar = plt.imshow(np.array( offsetErrors ).reshape( xO.shape ), extent=[np.min(aOffsets), np.max(aOffsets), np.min(yOffsets), np.max(yOffsets)]) 
    plt.colorbar( bar )
    plt.title("error function output for known displacement error")
    plt.xlabel("angle error (rad)")
    plt.ylabel("y displacement error") 

    "4"
    
def determineErrorCoupling( inpChunk:Chunk, scenarioCount=100 ):
    distanceSampleInterval = (-0.2,0.2,0.005)
    angleSampleInterval    = (np.deg2rad(-30),np.deg2rad(30),np.deg2rad(1))
    staticInterval         = ( -0.00001, 0.00001, 0.00002 ) 
    
    testInputsX = genTestingSets( distanceSampleInterval,staticInterval,angleSampleInterval, scenarioCount )
    testInputsY = genTestingSets( staticInterval,distanceSampleInterval,angleSampleInterval, scenarioCount )
    testInputsB = genTestingSets( distanceSampleInterval,distanceSampleInterval,angleSampleInterval, scenarioCount )
    
    plt.figure(562)
    plt.plot( testInputsB[:,0], testInputsB[:,1], "bx" )
    plt.plot( testInputsB[:,0], testInputsB[:,2], "rx" )
    
    angularDeviationsX = []
    displacementRatiosX = []
    
    angularDeviationsY = []
    displacementRatiosY = []
    
    angularDeviationsB = []
    displacementRatiosB = []
    
    for testInput in testInputsX:
        xError, yError, angleError = inpChunk.determineErrorFeatureless3( inpChunk, testInput, False )
        
        if ( abs(testInput[2]) > np.deg2rad( 2 ) ):
            angularDeviationsX.append( abs( ( testInput[2] - angleError )/testInput[2] ) )
            displacementRatiosX.append( np.sqrt( (testInput[0]**2+testInput[1]**2)/testInput[2]**2 ) )
    
    for testInput in testInputsY:
        xError, yError, angleError = inpChunk.determineErrorFeatureless3( inpChunk, testInput, False )
        
        if ( abs(testInput[2]) > np.deg2rad( 2 ) ):
            angularDeviationsY.append( abs( ( testInput[2] - angleError )/testInput[2] ) )
            displacementRatiosY.append( np.sqrt( (testInput[0]**2+testInput[1]**2)/testInput[2]**2 ) )
    
    for testInput in testInputsB:
        xError, yError, angleError = inpChunk.determineErrorFeatureless3( inpChunk, testInput, False )
        
        if ( abs(testInput[2]) > np.deg2rad( 2 ) ):
            angularDeviationsB.append( abs( ( testInput[2] - angleError )/testInput[2] ) )
            displacementRatiosB.append( np.sqrt( (testInput[0]**2+testInput[1]**2)/testInput[2]**2 ) )
        
    displacementRatiosX = np.array( displacementRatiosX )
    angularDeviationsX = np.array( angularDeviationsX )
    displacementRatiosY = np.array( displacementRatiosY )
    angularDeviationsY = np.array( angularDeviationsY )
    displacementRatiosB = np.array( displacementRatiosB )
    angularDeviationsB = np.array( angularDeviationsB )
        
    plt.figure(1234)
    plt.plot( displacementRatiosX, angularDeviationsX, "r.", label="" )
    plt.plot( displacementRatiosY, angularDeviationsY, "b.", label="" )
    plt.plot( displacementRatiosB, angularDeviationsB, "m.", label="" )
    plt.xlabel("introduced translational to rotational error ratio")
    plt.ylabel("angular error estimation inaccuracy")
    add_trend_line( displacementRatiosX, angularDeviationsX, 0.8, "r--", "x " )
    add_trend_line( displacementRatiosY, angularDeviationsY, 0.75, "b--", "y "  )
    add_trend_line( displacementRatiosB, angularDeviationsB, 0.7, "m--", "x+y "  )
    plt.legend()
    plt.show() 

def errorTester( inpChunk:Chunk ):   
    xOffsets = np.arange( -1.35, 1.35, 0.025 )
    yOffsets = np.arange( -1.35, 1.35, 0.025 )

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
    xOffsets = np.arange( -0.5, 0.51, 0.015 )
    yOffsets = np.arange( -0.5, 0.51, 0.015 ) 

    yO, xO = np.meshgrid( xOffsets, yOffsets )
    rO = np.zeros( xO.shape )

    offsetTests = np.column_stack( (xO.flatten(), yO.flatten(), rO.flatten()) )
    offsetErrors = []

    rotationTests  = np.deg2rad(np.arange( -50, 50, 0.2 ))
    rotationErrors = []

    for offsetTest in offsetTests:
        initialError, overlapArea = inpChunk.determineDirectDifference( inpChunk, offsetTest )
        predError, newError, v = findDifference( inpChunk, inpChunk, offsetTest, 8 )
        offsetErrors.append( -100*(newError-initialError)/initialError ) 

    for rotationTest in rotationTests: 
        initialError, overlapArea = inpChunk.determineDirectDifference( inpChunk, np.array( (0,0,rotationTest) ) )
        predError, newError, v = findDifference( inpChunk, inpChunk, np.array( (0,0,rotationTest) ), 8 )
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

def method2Tester( inpChunk:Chunk ):   
    xOffsets = np.arange( -0.4, 0.4, 0.05 )
    yOffsets = np.arange( -0.4, 0.4, 0.05 ) 
    xOffsets = np.arange( -0.2, 0.2, 0.05 )
    yOffsets = np.arange( -0.2, 0.2, 0.05 ) 
    yO, xO = np.meshgrid( xOffsets, yOffsets )
    rO = np.zeros( xO.shape )

    offsetTests = np.column_stack( (xO.flatten(), yO.flatten(), rO.flatten()) )
    offsetErrors = []
    offsetActualE = []

    rotationTests  = np.deg2rad(np.arange( -60, 60, 0.5 )) 
    rotationTests  = np.deg2rad(np.arange( -10, 10, 0.5 )) 
    rotationErrors = [] 
    rotOutputs = []

    for offsetTest in offsetTests:
        initialError, overlapArea = inpChunk.determineDirectDifference( inpChunk, offsetTest )
        predError, newError, v = inpChunk.determineErrorFeaturelessMinimum( inpChunk, offsetTest, False ) 
        offsetErrors.append( -100*(newError-initialError)/initialError )
        offsetActualE.append( 100-100*np.sqrt(np.sum(np.square(offsetTest-predError)))/np.sqrt(np.sum(np.square(offsetTest))) )  

    for rotationTest in rotationTests: 
        initialError, overlapArea = inpChunk.determineDirectDifference( inpChunk, np.array( (0,0,rotationTest) ) )
        predError, newError, v = inpChunk.determineErrorFeaturelessMinimum( inpChunk, np.array( (0,0,rotationTest) ), False ) 
        rotationErrors.append( -100*(newError-initialError)/initialError ) 
        rotOutputs.append( np.array((0,0,rotationTest))-predError )  

    rotOutputs = np.array(rotOutputs)
    
    plt.figure(1512)
    """plt.xticks(np.arange( -0.3, 0.3, 0.1 ))
    plt.yticks(np.arange( -0.3, 0.3, 0.1 ))"""
    bar = plt.imshow(np.array( offsetErrors ).reshape( xO.shape ), extent=[np.min(xOffsets), np.max(xOffsets), np.min(yOffsets), np.max(yOffsets)]) 
    plt.colorbar( bar )
    plt.title("Percentage error reduction score")
    plt.xlabel("x displacement error")
    plt.ylabel("y displacement error")
    plt.figure(1513)
    """plt.xticks(np.arange( -0.3, 0.3, 0.1 ))
    plt.yticks(np.arange( -0.3, 0.3, 0.1 ))"""
    bar = plt.imshow(np.array( offsetActualE ).reshape( xO.shape ), extent=[np.min(xOffsets), np.max(xOffsets), np.min(yOffsets), np.max(yOffsets)]) 
    plt.colorbar( bar )
    plt.title("Percentage error reduction euclidian")
    plt.xlabel("x displacement error")
    plt.ylabel("y displacement error")

    plt.figure(1514)
    plt.plot( np.rad2deg(rotationTests), np.array(rotationErrors) ) 
    plt.title("Percentage error reduction") 
    plt.xlabel("rotation error (deg)")
    plt.ylabel("percent error reduction") 

    plt.figure(1515)
    plt.plot( np.rad2deg(rotationTests), rotOutputs[:,0], label="x" ) 
    plt.plot( np.rad2deg(rotationTests), rotOutputs[:,1], label="y" ) 
    plt.plot( np.rad2deg(rotationTests), rotOutputs[:,2], label="a" ) 
    plt.title("Remaining error")
    plt.legend()
    plt.xlabel("rotation error (deg)")
    plt.ylabel("percent error reduction")
    plt.show()

    "4"      

def mapMergeTest( index0, frameCount, skipping=0 ):
    rawStack = []
    chunkStack = []
    
    skipdex = 0
    
    for i in range( 0, len(allScanData) ):
        cRawScan:RawScanFrame = allScanData[i]

        rawStack.append( cRawScan )
        
        midScan = rawStack[int(len(rawStack)/2)]
        if ( abs(cRawScan.pose.yaw - midScan.pose.yaw) > config.MAX_INTER_FRAME_ANGLE or len(rawStack) > config.MAX_FRAMES_MERGE ):
            if ( index0<=0 and skipdex<=0 ):
                # Frame merge 
                nChunk = Chunk.initFromRawScans( rawStack[0:-1], config, 0 )
                rawStack = [ rawStack[-1] ] 

                nChunk.constructProbabilityGrid() 

                gridDisp2.parseData( nChunk.cachedProbabilityGrid.mapEstimate ) 
                
                lpWindow.render()  

                chunkStack.append( nChunk )
                
                skipdex = skipping
            else:
                index0 -= 1
                skipdex -= 1
                
                rawStack = []
                    
                if ( index0 > 0 ):
                    chunkStack = []
        
        if ( len( chunkStack ) > frameCount or len(allScanData)-1==i ):
            """fancyPlot( chunkStack[12].cachedProbabilityGrid.mapEstimate )
            fancyPlot( chunkStack[32].cachedProbabilityGrid.mapEstimate )
            plt.show()"""

            parentChunk = Chunk.initEmpty( config )

            parentChunk.addChunks( chunkStack )

            fancyPlot(
                parentChunk.constructProbabilityGrid().mapEstimate
            )
            plt.show(block=False)
            parentChunk.linearFeaturelessErrorReduction( 1 )
            parentChunk.linearFeaturelessErrorReduction( 2 ) 
            
 
            fancyPlot(
                parentChunk.constructProbabilityGrid().mapEstimate
            )
            plt.show(block=False)
            
            parentChunk.linearPrune( 1, 1.8 ) 
            parentChunk.linearFeaturelessErrorReduction( 1 )
 
            fancyPlot(
                parentChunk.constructProbabilityGrid().mapEstimate
            )
            plt.show(block=False)
            
            parentChunk.linearFeaturelessErrorReduction( 3 )
            parentChunk.linearPrune( 3, 1.2 ) 
            parentChunk.centredFeaturelessErrorReduction( False ) 
            fancyPlot(
                parentChunk.constructProbabilityGrid().mapEstimate
            )
            plt.show(block=False)

            return
 
         
def getChunk( index ):
    rawStack = []
    chunkStack = []

    for i in range( 0, len(allScanData) ):
        cRawScan:RawScanFrame = allScanData[i]

        rawStack.append( cRawScan )
        
        midScan = rawStack[int(len(rawStack)/2)]
        if ( abs(cRawScan.pose.yaw - midScan.pose.yaw) > config.MAX_INTER_FRAME_ANGLE or len(rawStack) > config.MAX_FRAMES_MERGE ):
            # Frame merge 
            nChunk = Chunk.initFromRawScans( rawStack[0:-1], config, 0 )

            if ( len( chunkStack ) == index  ):
                nChunk.constructProbabilityGrid() 
    
                return nChunk 

            rawStack = [ rawStack[-1] ] 
            chunkStack.append( nChunk )
        

def twoFramesTest( index1, index2, index3 ):
    chunks = [ getChunk(index1), getChunk(index2), getChunk(index3) ]
    
    for c in chunks:
        c.constructProbabilityGrid()
        #fancyPlot( c.cachedProbabilityGrid.mapEstimate )
    
    parentChunk = Chunk.initEmpty( config )
    
    
    parentChunk.addChunks( chunks )
    
    #parentChunk.subChunks[2].offsetFromParent = parentChunk.subChunks[2].offsetFromParent + np.array((0.1,0.1,0.1))
    
    chunks[0].plotDifference( chunks[2] )
    chunks[0].determineErrorFeaturelessDirect( chunks[2], 15, np.zeros(3), True )
    
    chunks[0].plotDifference( chunks[2] )
    
    plt.show( block=False )
    
    ""

def mergeFrameRecursive( frameCount, batchSize ):
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
            break
    
    
    
    parentChunk = Chunk.initEmpty( config )

    parentChunk.addChunks( chunkStack )

    parentChunk.centredFeaturelessErrorReduction( False )

    fancyPlot(
        parentChunk.constructProbabilityGrid().mapEstimate
    )
    plt.show()

    return
    

def singleMinTest( testingChunk:Chunk ): 
    setOffset = np.array(( -0.18, 0.18, np.deg2rad(20) ))
    previousError, aaaaaaaa  = testingChunk.determineDirectDifference( testingChunk, setOffset, False )
    
    testingChunk.plotDifference( testingChunk, setOffset )
    #predError, newError, v = testingChunk.determineErrorFeaturelessMinimum( testingChunk, setOffset, False )
    #predError, newError, v = testingChunk.determineErrorFeatureless2( testingChunk, setOffset, False )
    #predError = np.array(testingChunk.determineErrorFeatureless3( testingChunk, setOffset, False ))
    predError = setOffset-findDifference( testingChunk, testingChunk, setOffset, 10 )[0]
     
    newError, aaaa  = testingChunk.determineDirectDifference( testingChunk, setOffset-predError, False )
    
    
    testingChunk.plotDifference( testingChunk, setOffset-predError )
    plt.show( block=False )
    
    ""
      
def showRotateTest( inChunk:Chunk, angle=np.deg2rad( 10 ) ): 
    # Overlap is extracted
    thisWindow, transWindow = inChunk.copyOverlaps( inChunk, angle, (0,0) )

    fancyPlot( transWindow )
    fancyPlot( transWindow*(transWindow>0) )

    plt.show(block=False)
    ""

def plotPathError():
    estPos = []
    realPos = []

    for cScan in allScanData:
        realPos.append( [cScan.truePose.x, cScan.truePose.y] )
        estPos.append( [cScan.pose.x, cScan.pose.y] )

    estPos = np.array(estPos)
    realPos = np.array(realPos)

    plt.figure(4982)
    plt.plot( estPos[:,0], estPos[:,1], "r--" )
    plt.plot( realPos[:,0], realPos[:,1], "b-" )
    plt.show()

def descriptorTest( inpChunk1:Chunk, inpChunk2:Chunk ):
    
    #fancyPlot( inpChunk2.cachedProbabilityGrid.mapEstimate )
    #plt.show()

    inpChunk1.determineErrorKeypoints( inpChunk2 )


#plotPathError()

testingChunk = getChunk( 2 )

descriptorTest( getChunk( 0 ), getChunk( 223 ) )

#showRotateTest( testingChunk )

featurelessAutoTune( testingChunk )

mapMergeTest( 35, 30, 3 )

conf = testingChunk.config
#conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE = 0.063324, 0.061625, 0.055572
twoFramesTest( 44, 60, 47 )

#mergeFrameRecursive( 20, 5 )

#mapMergeTest( 0, 240 )
#errorTesterAlt(testingChunk)

#featurelessFullTest(testingChunk)
#singleMinTest( testingChunk )
#superTuner( testingChunk )
#method2Tester( testingChunk )

#findDifference( testingChunk, testingChunk, np.array((-0.1,0.24,np.deg2rad(22))), 100, True )


#featurelessFullTest(testingChunk)
#method1Tester( testingChunk )
#determineErrorCoupling( testingChunk, 300 )

#fancyPlot( testingChunk.cachedProbabilityGrid.mapEstimate )
plt.show()



 
if ( len( chunkStack ) > 36 ):
    """fancyPlot( chunkStack[12].cachedProbabilityGrid.mapEstimate )
    fancyPlot( chunkStack[32].cachedProbabilityGrid.mapEstimate )
    plt.show()"""

    parentChunk = Chunk.initEmpty( config )

    parentChunk.addChunks( chunkStack )

    errorTester( chunkStack[32] )
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









