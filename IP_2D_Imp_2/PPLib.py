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

from CommonLib import fancyPlot, rotationMatrix, acuteAngle



class MAP_PROP:
    X_MIN = -4
    X_MAX = 6
    Y_MIN = -6
    Y_MAX = 4
    PROB_GRID_RES = IPConfig.GRID_RESOLUTION

lpWindow = lp.PlotWindow(5, 15)
gridDisp      = lp.LabelledGridGraphDisplay(5,0,5,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
gridDisp2     = lp.LabelledGridGraphDisplay(10,0,15,5, lpWindow, (MAP_PROP.X_MAX-MAP_PROP.X_MIN)*MAP_PROP.PROB_GRID_RES, (MAP_PROP.Y_MAX-MAP_PROP.Y_MIN)*MAP_PROP.PROB_GRID_RES) 
config = IPConfig()
 
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

def featurelessAutoTune( inpChunk, tuneXOffset=0.14, tuneYOffset=0.14, tuneAOffset=np.deg2rad(5) ):
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

def findDifference( inpChunk1:Chunk, inpChunk2:Chunk, initOffset, maxIterations, showGraph=False ):
    trueOffset = initOffset

    offsets = []
    errors  = []

    allResults = []
    
    errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )
    pErrorScore = errorScore
    initES = errorScore

    offsets.append( np.array(trueOffset) )
    errors.append( errorScore )
    
    inpChunk1.plotDifference( inpChunk2, forcedOffset=np.zeros(3) )
    
    for i in range(0, maxIterations):

        xError, yError, angleError = inpChunk1.determineErrorFeatureless3( inpChunk2, trueOffset, False )
        if ( xError+yError+angleError == 0 or abs(angleError)>np.pi/1.6 ):
            break
        #rf = min( max( errorScore/8, 0.1 ), 0.8 )
        rf = 0.5
        trueOffset = ( trueOffset[0]-xError*rf, trueOffset[1]-yError*rf, trueOffset[2]-angleError*rf ) 
        #if ( np.isnan(trueOffset[2]) ): return initOffset, initES
        pErrorScore = errorScore
        errorScore, overlapArea = inpChunk1.determineDirectDifference( inpChunk2, trueOffset )

        offsets.append( np.array(trueOffset) )
        errors.append( errorScore )
        allResults.append( [errorScore,trueOffset[0],trueOffset[1],trueOffset[2] ] )

        #print_rounded("e:",errorScore,"   x:", trueOffset[0],"   y:", trueOffset[1],"   a:", trueOffset[2])
    
    inpChunk1.plotDifference( inpChunk2, trueOffset )
    
    offsets = np.array( offsets )
    errors  = np.array( errors )
    cMinIndex = np.argmin( errors )

    if ( showGraph ):
        fancyPlot( inpChunk1.cachedProbabilityGrid.mapEstimate )
        fancyPlot( inpChunk2.cachedProbabilityGrid.mapEstimate )
        
        
        allResults = np.array(allResults)
        plt.figure(353)
        plt.plot( errors, "m-", label="error" )
        plt.legend()
        plt.figure(354)
        plt.plot( (offsets[:,0]), "r--", label="x" )
        plt.plot( (offsets[:,1]), "b:" , label="y")
        plt.plot( (offsets[:,2]), "g-" , label="a")
        plt.legend()
        plt.show( block=False )

        print("minError:", min(errors), "  at: ",cMinIndex)
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
    xOffsets = np.arange( -0.5, 0.51, 0.01 )
    yOffsets = np.arange( -0.5, 0.51, 0.01 ) 

    yO, xO = np.meshgrid( xOffsets, yOffsets )
    rO = np.zeros( xO.shape )

    offsetTests = np.column_stack( (xO.flatten(), yO.flatten(), rO.flatten()) )
    offsetErrors = []
 
    rotationTests  = np.deg2rad(np.arange( -60, 60, 0.1 )) 
    rotationErrors = []

    for offsetTest in offsetTests:
        errorScore, overlapArea = inpChunk.determineDirectDifference( inpChunk, offsetTest )
        offsetErrors.append( errorScore )

    for rotationTest in rotationTests:
        errorScore, overlapArea = inpChunk.determineDirectDifference( inpChunk, np.array( (0,0,rotationTest) ) )
        rotationErrors.append( errorScore )

    normVal = np.max(np.array( offsetErrors ))

    plt.figure(1512)
    """plt.xticks(np.arange( -0.3, 0.3, 0.1 ))
    plt.yticks(np.arange( -0.3, 0.3, 0.1 ))"""
    bar = plt.imshow(np.array( offsetErrors ).reshape( xO.shape )/normVal, extent=[np.min(xOffsets), np.max(xOffsets), np.min(yOffsets), np.max(yOffsets)]) 
    plt.colorbar( bar ).set_label("error score/max error")
    plt.title("error function output for known displacement error")
    plt.xlabel("x displacement error")
    plt.ylabel("y displacement error")

    normVal = np.max(np.array( rotationErrors ))
    plt.figure(1514)
    plt.plot( np.rad2deg(rotationTests), np.array(rotationErrors) /normVal)
    plt.title("error function output for known rotational error")
    plt.xlabel("rotation error (deg)")
    plt.ylabel("error score/max error")
    plt.show()

    "4"
  
def method1Tester( inpChunk ):   
    xOffsets = np.arange( -0.5, 0.51, 0.065 )
    yOffsets = np.arange( -0.5, 0.51, 0.065 ) 
    

    yO, xO = np.meshgrid( xOffsets, yOffsets )
    rO = np.zeros( xO.shape )

    offsetTests = np.column_stack( (xO.flatten(), yO.flatten(), rO.flatten()) )
    offsetErrors = []

    rotationTests  = np.deg2rad(np.arange( -50, 50, 2.2 ))
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

def mapMergeTest( allScanData, index0, frameCount, skipping=0 ):
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

def mapMergeTestKeypoints( allScanData, index0, frameCount, skipping=0 ):
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

            parentChunk.graphSLAM.plot()
            
            fancyPlot(
                parentChunk.constructProbabilityGrid().mapEstimate
            )
            plt.show(block=False)
            parentChunk.linearHybridErrorReduction( 3 ) 
            parentChunk.linearHybridErrorReduction( 7 ) 
            parentChunk.linearHybridErrorReduction( 8 ) 
            #parentChunk.linearHybridErrorReduction( 5 ) 
        
            # parentChunk.graphSLAM.plot()
            # fancyPlot(
            #     parentChunk.constructProbabilityGrid().mapEstimate
            # )
            # plt.show(block=False)
            
            parentChunk.linearHybridErrorReduction( 5 ) 
            parentChunk.linearFeaturelessErrorReduction( 4 ) 
            parentChunk.linearHybridErrorReduction( 9 ) 
            
            # parentChunk.graphSLAM.plot()
            # fancyPlot(
            #     parentChunk.constructProbabilityGrid().mapEstimate
            # )
            # plt.show(block=False)
             
            parentChunk.linearPrune( 3, 1.2 )
            
            parentChunk.graphSLAM.plot()
            fancyPlot(
                parentChunk.constructProbabilityGrid().mapEstimate
            )
            plt.show(block=False)
            
            parentChunk.randomHybridErrorReduction()
            parentChunk.randomHybridErrorReduction()
            parentChunk.randomHybridErrorReduction()
            parentChunk.randomHybridErrorReduction()
            parentChunk.linearHybridErrorReduction( 10 )   
            parentChunk.randomHybridErrorReduction()
            
            parentChunk.graphSLAM.plot()
            fancyPlot(
                parentChunk.constructProbabilityGrid().mapEstimate
            )
            plt.show(block=False)

            
            return

def getBaseChunks( allScanData, index0, skipping=1, indexEnd=99999, chunkLength=1, paddingSize=0 ): 
    chunkStack = []
    
    skipdex = 0
    
    tmpScanData = []
    tillPadding = chunkLength
    for i in range( 0, len(allScanData) ):
        tillPadding -= 1
        
        cRawScan:RawScanFrame = allScanData[i]
        
        tmpScanData.append(cRawScan)
        
        if ( tillPadding <= 0 ):
            for I in range(i, i+paddingSize):
                if ( I+1 < len(allScanData) ):
                    tmpScanData.append(allScanData[I+1])
            tillPadding = chunkLength
    
    for i in range( 0, len(tmpScanData) ):
        cRawScan:RawScanFrame = tmpScanData[i]

        if ( i > indexEnd ):
            return chunkStack
        
        if ( index0<=0 and skipdex<=0 ):
            # Frame merge 
            nChunk = Chunk.initFromRawScans( [cRawScan], config, 0 ) 

            nChunk.constructProbabilityGrid() 

            gridDisp2.parseData( nChunk.cachedProbabilityGrid.mapEstimate ) 
            
            lpWindow.render()  

            chunkStack.append( nChunk )
            
            skipdex = skipping
        else:
            index0 -= 1
            skipdex -= 1  
    
    print("finished processing basic scans")
    return chunkStack
         
        
def mapMergeTestRec( inputData:list[Chunk], chunkSize, mergeMethod:list[int], minFrameError=85, discriminate=-1 ):
    rawStack = []
    chunkStack:list[Chunk] = []
    
    for nChunk in inputData:
        rawStack.append( nChunk )
         
        if ( len(rawStack) >= chunkSize or (nChunk == inputData[-1] and len(rawStack)>0) ): 
            
            parentChunk = Chunk.initEmpty( config )

            parentChunk.addChunks( rawStack )

            # parentChunk.subChunks[0].determineErrorKeypoints( parentChunk.subChunks[1], np.zeros(3), True )
            # parentChunk.subChunks[0].determineErrorKeypoints( parentChunk.subChunks[1], np.zeros(3), True )
            # parentChunk.subChunks[0].determineErrorKeypoints( parentChunk.subChunks[5], np.zeros(3), True )

            #parentChunk.graphSLAM.plot()
            
            # fancyPlot(
            #     parentChunk.constructProbabilityGrid().mapEstimate
            # )
            # plt.show(block=False)
            initEstimate = parentChunk.constructProbabilityGrid().mapEstimate
            
            parentChunk.repeatingHybridPrune( minFrameError, mergeMethod, maxIterations=1 ) 
            # for cM in mergeMethod:
            #     if ( len(rawStack) < abs(cM)-1 ):
            #         print("would fail to error check, len: ",len(rawStack), " to min len ", abs(cM)-1)
            #     else:
            #         if ( cM < 0 ):
            #             parentChunk.linearFeaturelessErrorReduction( -cM ) 
            #         else:
            #             parentChunk.linearHybridErrorReduction( cM ) 
            
            # parentChunk.linearHybridErrorReduction( 3 ) 
            # parentChunk.linearHybridErrorReduction( 7 ) 
            # parentChunk.linearHybridErrorReduction( 8 )  
            
            # parentChunk.linearHybridErrorReduction( 5 ) 
            # parentChunk.linearFeaturelessErrorReduction( 4 ) 
            # parentChunk.linearHybridErrorReduction( 9 )  
            
            chunkStack.append( parentChunk ) 
            
            if ( discriminate != -1 ):
                parentChunk.centredPrune( discriminate, 2 )
                parentChunk.repeatingHybridPrune( minFrameError, mergeMethod, maxIterations=1 ) 
            
            
            gridDisp.parseData( initEstimate )  
            lpWindow.render()  
            
            parentChunk.constructProbabilityGrid()  
            gridDisp2.parseData( parentChunk.cachedProbabilityGrid.mapEstimate )  
            lpWindow.render()  
            #parentChunk.graphSLAM.plot()
            
            rawStack = []
             
    return chunkStack

def getChunk( allScanData, index ):
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

def meanFeaturelessAutoTune( allData:list[RawScanFrame], sampleCount ):
    xSc = []
    ySc = []
    aSc = []
    Ind = []
    
    for i in range(0, len(allData)-2, int( (len(allData)-2)/sampleCount )):
        chunk = getChunk( allData, i )
        
        featurelessAutoTune( chunk )
        
        Ind.append( chunk.getRawCentre().index )
        xSc.append( chunk.config.FEATURELESS_X_ERROR_SCALE )
        ySc.append( chunk.config.FEATURELESS_Y_ERROR_SCALE )
        aSc.append( chunk.config.FEATURELESS_A_ERROR_SCALE )
    
    xSc = np.array(xSc)
    ySc = np.array(ySc)
    aSc = np.array(aSc)
    
    chunk.config.FEATURELESS_X_ERROR_SCALE = np.median( xSc )
    chunk.config.FEATURELESS_Y_ERROR_SCALE = np.median( ySc )
    chunk.config.FEATURELESS_A_ERROR_SCALE = np.median( aSc )
    
    print_rounded("median Tuned:",chunk.config.FEATURELESS_X_ERROR_SCALE,chunk.config.FEATURELESS_Y_ERROR_SCALE,chunk.config.FEATURELESS_A_ERROR_SCALE)

def minimiserEffectivenessTest( parent:Chunk, sampleCount, compDistance ):
    
    initErrors = []
    newErrors = []
    
    leeen = len(parent.subChunks)-(compDistance+1)
    skipDis = max(1, int((leeen)/sampleCount))
    
    for i in range( 0, leeen, skipDis ):
        rootCh = parent.subChunks[i]
        targCh = parent.subChunks[i+compDistance]
        
        offsetAdjustment, newErrorScore = rootCh.determineErrorFeaturelessDirect( targCh, 10, scoreRequired=260, forcedOffset=np.zeros(3) )
        initVector = rootCh.getLocalOffsetFromTarget( targCh )
        newVector = initVector + offsetAdjustment
        
        trueVector = rootCh.getTRUEOffsetLocal( targCh )
        
        initErrors.append( (np.abs( initVector[0] - trueVector[0] ), np.abs( initVector[1] - trueVector[1] ), acuteAngle( initVector[2], trueVector[2] )) )
 
        newErrors.append( (np.abs( newVector[0] - trueVector[0] ), np.abs( newVector[1] - trueVector[1] ), acuteAngle( newVector[2], trueVector[2] )) )
        
        #print( "vecs:", trueVector, newVector, np.abs( newVector - trueVector ) )
        ""
    
    initErrors = np.array( initErrors )
    newErrors  = np.array( newErrors )
    
    np.isnan
    
    notNull = (1!=np.isnan(newErrors[:,0]))
    print( "init score: ", np.mean( np.sqrt(initErrors[:,0][notNull]**2+initErrors[:,1][notNull]**2) ) )
    print( "final score: ", np.mean( np.sqrt(newErrors[:,0][notNull]**2+newErrors[:,1][notNull]**2) ) )
    print( "Ainit score: ", np.mean( initErrors[:,2][notNull] ) )
    print( "Afinal score: ", np.mean( newErrors[:,2][notNull] ) )
    
    plt.figure(2409)
    plt.title( "Translation Error" )
    plt.plot( np.sqrt(initErrors[:,0]**2+initErrors[:,1]**2), "r-", label="init error" )
    plt.plot( np.sqrt(newErrors[:,0]**2+newErrors[:,1]**2), "b--", label="final error" )
    plt.legend()
     
    plt.figure(2109)
    plt.plot( (initErrors[:,2] ), "r-", label="init error" )
    plt.plot(  (newErrors[:,2] ), "b--", label="final error" )
    plt.legend()
    plt.title( "Angle Error" )
    plt.show( block=False )
    
    ""
    

def autoTunePlot( allData:list[RawScanFrame], interval=0 ):
    xSc = []
    ySc = []
    aSc = []
    Ind = []
    
    meanFeaturelessAutoTune( allData, 10 )
    mXSC = config.FEATURELESS_X_ERROR_SCALE
    mYSC = config.FEATURELESS_Y_ERROR_SCALE
    mASC = config.FEATURELESS_A_ERROR_SCALE
    
    for i in range(0, len(allData)-8, interval):
        chunk = getChunk( allData, i )
        
        featurelessAutoTune( chunk )
        
        Ind.append( chunk.getRawCentre().index )
        xSc.append( config.FEATURELESS_X_ERROR_SCALE )
        ySc.append( config.FEATURELESS_Y_ERROR_SCALE )
        aSc.append( config.FEATURELESS_A_ERROR_SCALE )
    
    xSc = np.array(xSc)
    ySc = np.array(ySc)
    aSc = np.array(aSc)
    
    plt.figure(4092)
    plt.plot( Ind, np.clip(100*(xSc-mXSC)/mXSC, -100, 200), "r-", label="x length" )
    plt.plot( Ind, np.clip(100*(ySc-mYSC)/mYSC, -100, 200), "b--", label="y length" )
    plt.plot( Ind, np.clip(100*(aSc-mASC)/mASC, -100, 200), "g:", label="angular" )
    plt.xlabel("Frame index")
    plt.legend()
    plt.ylabel("Percentage tuning parameter change from initial")
    plt.title("Featureless alignment method, tuning values")
    plt.show(block=False)
    
    ""

def twoFramesTest( chunk1, chunk2 ):
    chunks:list[Chunk] = [ chunk1,  chunk2 ]
    
    for c in chunks:
        c.constructProbabilityGrid()
        #fancyPlot( c.cachedProbabilityGrid.mapEstimate )
    
    parentChunk = Chunk.initEmpty( config )
    
    
    parentChunk.addChunks( chunks )
    
    #parentChunk.subChunks[2].offsetFromParent = parentChunk.subChunks[2].offsetFromParent + np.array((0.1,0.1,0.1))
    
    chunks[0].plotDifference( chunks[1], np.zeros(3) )
    #offsetAdjustment, newErrorScore = chunks[0].determineErrorFeaturelessDirect( chunks[1], 15, np.array((0.0,0,0)), True )
    offsetAdjustment, newErrorScore = chunks[0].determineErrorFeaturelessMinimum( chunks[1], 200, forcedOffset=np.array((0.1,0,0)) )
    
    chunks[0].plotDifference( chunks[1], np.zeros(3) )
    
    plt.show( block=False )
    
    ""

def mergeFrameRecursive( allScanData, frameCount, batchSize ):
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
    setOffset = np.array(( -0.06, 0.06, np.deg2rad(4) ))
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

def plotPathError( allScanData ):
    estPos = []
    realPos = []

    for cScan in allScanData:
        realPos.append( [cScan.truePose.x, cScan.truePose.y] )
        estPos.append( [cScan.pose.x, cScan.pose.y, cScan.pose.yaw] )

    estPos = np.array(estPos)
    realPos = np.array(realPos)
    
    rM = rotationMatrix( estPos[0][2] )
    
    estPos = estPos - estPos[0]
    estPos = np.dot( estPos[:,0:2], rM )

    plt.figure(4982+int(np.random.random()*150))
    plt.plot( estPos[:,0], estPos[:,1], "r--" )
    plt.plot( realPos[:,0], realPos[:,1], "b-" )
    plt.show(block=False)

def descriptorTest( inpChunk1:Chunk, inpChunk2:Chunk ):
    
    #fancyPlot( inpChunk2.cachedProbabilityGrid.mapEstimate )
    #plt.show()
    parent:Chunk = Chunk.initEmpty( config )
    parent.addChunks([ inpChunk1, inpChunk2 ])

    inpChunk1.determineErrorKeypoints( inpChunk2, np.array((0,0,0)), True )
    
    ""
def descriptorTestExt( inpChunk :Chunk ):
    
    #fancyPlot( inpChunk2.cachedProbabilityGrid.mapEstimate )
    #plt.show()
    
    inpChunk.constructProbabilityGrid().extractDescriptors( plotThingy=True )
    
    ""

from ImageProcessor import ImageProcessor
from CommonLib import gaussianKernel

def frameTestMaxs( chunk:Chunk ): 
    
    x1,x2,y1,y2 = 5,150  , 250,395
    
    plt.figure(124890)
    plt.imshow( chunk.constructProbabilityGrid().mapEstimate   , origin="lower" )
    plt.axis('off')
    #plt.xlim(x1, x2)
    #plt.ylim(y1, y2)

    
    plt.figure(124891)
    lambda_1, lambda_2, Rval = ImageProcessor.guassianCornerDist( chunk.constructProbabilityGrid().mapEstimate, gaussianKernel(4) )
    
    maxPosX, maxPosY, maxInt = ImageProcessor.findMaxima( Rval, 5, 0.01 )
    
    
    plt.imshow( Rval , origin="lower" )
    plt.plot( maxPosX, maxPosY, "rx" )
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.axis('off')
    
    plt.show( block=False )
    
    ""
 
def massInterframeTesting( parent:Chunk, compDistances:list[int] = [ 1,2,4,6,8,14,24,40 ] ):
    
    def MIT_internal( parent:Chunk, sampleCount, compDistance ):
        initErrors = []
        newErrors = []
        initErrorScores = []
        newErrorScores = []
        
        leeen = len(parent.subChunks)-(compDistance+1)
        skipDis = max(1, int((leeen)/sampleCount))
        
        for i in range( 0, leeen, skipDis ):
            rootCh = parent.subChunks[i]
            targCh = parent.subChunks[i+compDistance]
            
            initErrorScore, area = rootCh.determineDirectDifference( targCh, forcedOffset=np.zeros(3) )
            initErrorScores.append( initErrorScore )
            
            # MODIFY
            #offsetAdjustment, newErrorScore = rootCh.determineErrorFeaturelessDirect( targCh, 10, scoreRequired=260, forcedOffset=np.zeros(3) )
            #offsetAdjustment, newErrorScore = rootCh.determineErrorFeaturelessMinimum( targCh, 250, np.zeros(3), scoreRequired=260 )
            #offsetAdjustment, newErrorScore = rootCh.determineOffsetKeypoints( targCh, np.zeros(3), scoreRequired=300, returnOnPoorScore=True )
            offsetAdjustment, newErrorScore = rootCh.determineHybridErrorReduction( targCh, np.zeros(3), scoreRequired=300 )
            
            newErrorScores.append( newErrorScore )
            
            initVector = rootCh.getLocalOffsetFromTarget( targCh )
            newVector = initVector + offsetAdjustment
            
            trueVector = rootCh.getTRUEOffsetLocal( targCh )
            
            initErrors.append( (np.abs( initVector[0] - trueVector[0] ), np.abs( initVector[1] - trueVector[1] ), acuteAngle( initVector[2], trueVector[2] )) )
    
            newErrors.append( (np.abs( newVector[0] - trueVector[0] ), np.abs( newVector[1] - trueVector[1] ), acuteAngle( newVector[2], trueVector[2] )) )
            
            #print( "vecs:", trueVector, newVector, np.abs( newVector - trueVector ) )
            ""
        
        initErrors = np.array( initErrors )
        newErrors  = np.array( newErrors )
        initErrorScores  = np.array( initErrorScores )
        newErrorScores  = np.array( newErrorScores ) 
        
        notNull = (1!=np.isnan(newErrors[:,0]))
        
        print( "init score: ", np.mean( np.sqrt(initErrors[:,0][notNull]**2+initErrors[:,1][notNull]**2) ) )
        print( "final score: ", np.mean( np.sqrt(newErrors[:,0][notNull]**2+newErrors[:,1][notNull]**2) ) )
        print( "Ainit score: ", np.mean( initErrors[:,2][notNull] ) )
        print( "Afinal score: ", np.mean( newErrors[:,2][notNull] ) )
        
        plt.figure(2409)
        plt.title( "Translation Error" )
        plt.plot( np.sqrt(initErrors[:,0]**2+initErrors[:,1]**2), "r-", label="init error" )
        plt.plot( np.sqrt(newErrors[:,0]**2+newErrors[:,1]**2), "b--", label="final error" )
        plt.legend()
        
        plt.figure(2109)
        plt.plot( (initErrors[:,2] ), "r-", label="init error" )
        plt.plot(  (newErrors[:,2] ), "b--", label="final error" )
        plt.legend()
        plt.title( "Angle Error" )
        plt.show( block=False )
        
        return initErrors, newErrors, notNull, initErrorScores, newErrorScores
    
    allResults = []
    
    for compDist in compDistances:
        print("starting on ", compDist)
        initErrors, newErrors, notNull, initErrorScores, newErrorScores = MIT_internal( parent, 999999999, compDist )
        
        allResults.append( [initErrors, newErrors, notNull, initErrorScores, newErrorScores] )
    
    ""
    







