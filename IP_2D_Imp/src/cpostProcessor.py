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
import json
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

def matchingTest( frame1Index, frame2Index ):
    def custom_serializer(obj):
        if isinstance(obj, np.ndarray):
            return "..."
        # Add additional custom serialization logic if needed
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    scan1 = mapper.allScans[frame1Index]
    scan2 = mapper.allScans[frame2Index]

    mapper.compareScans( scan1, scan2 )
    
    pointCol = ["rx", "bx", "gx", "yx", "mx", "cx", "wx", "ro", "bo", "go", "yo", "mo", "co", "ko", "wo", "r+", "b+", "g+", "y+", "m+", "c+", "k+", "w+"]
    pointColMap = {}

    scans = mapper.allScans[frame1Index:frame2Index]

    scanGroups = []
    allGroups = []

    for i in range( 0, len(scans) ):
        scanGroups.append( list(scans[i].featureDict.keys()) )
        allGroups += ( scanGroups[-1] )

    allGroups = list( set( allGroups ) )
    numm = 0
    for group in allGroups:
        pointColMap[group] = pointCol[numm]
        numm+=1


    for i in range( 0, len(scans) ):  
        tScan = scans[i]

        plt.figure(21+i)
        plt.xlim(0, 180)
        plt.ylim(300, 120)
        plt.imshow( tScan.estimatedMap, cmap='gray' )

        for group in scanGroups[i]:
            for feat, I in zip(tScan.featureDict[group], range(0, len(tScan.featureDict[group]))):
                plt.plot( feat[0][1], feat[0][0], pointColMap[group], markersize=12 )

    """plt.figure(21)
    plt.imshow( scan1.estimatedMap )
    plt.plot( scan1.featurePositions[:,1], scan1.featurePositions[:,0], "rx" )
    
    plt.figure(22)
    plt.imshow( scan2.estimatedMap )
    plt.plot( scan2.featurePositions[:,1], scan2.featurePositions[:,0], "rx" )"""

    """print("scan 1")
    print( json.dumps( scan1.featureDict, indent=3, default=custom_serializer ) )
    print("scan 2")
    print( json.dumps( scan2.featureDict, indent=3, default=custom_serializer ) )"""

    plt.show()
    ""

def matchingTest2( frame1Index, frame2Index ):
    scan1 = mapper.allScans[frame1Index]
    scan2 = mapper.allScans[frame2Index]
 
    
    pointCol = ["rx", "bx", "gx", "yx", "mx", "cx", "wx", "ro", "bo", "go", "yo", "mo", "co", "ko", "wo", "r+", "b+", "g+", "y+", "m+", "c+", "k+", "w+"] 

    """mapper.compareScans2( scan1, scan2, 1402 )
    plt.figure(23)
    plt.imshow( scan1.estimatedMap )
    plt.plot( scan1.featurePositions[:,1], scan1.featurePositions[:,0], "rx" )
    
    plt.figure(24)
    plt.imshow( scan2.estimatedMap )
    plt.plot( scan2.featurePositions[:,1], scan2.featurePositions[:,0], "rx" )
    plt.show() """
    
    matchValues, matchIndecies = mapper.computeAllFeatureMatches( scan1, scan2, 3 )
    

    frameXChange = scan2.scanPose.x - scan1.scanPose.x 
    frameYChange = scan2.scanPose.y - scan1.scanPose.y
    frameYawChange = scan2.scanPose.yaw - scan1.scanPose.yaw
    
    xError, yError   = 0, 0
    xOffset, yOffset, yawOffset = frameXChange-0, frameYChange+0, frameYawChange+np.deg2rad( 0 )
    print( "initOffsets:", xOffset, yOffset, np.rad2deg(yawOffset) )
    
    errorScore, mArea = mapper.determineImageMatchSuccess( scan2, scan1, yawOffset, [ xOffset, yOffset ] ) 
    
    #errorScore, areaOverlap =  mapper.determineImageTranslation( scan1, scan2, frameYawChange, [frameXChange, frameYChange] )

    for i in range(0, 5):
        xError, yError, yawError = mapper.determineImageTranslation( scan2, scan1, yawOffset, [ xOffset, yOffset ], True )
        errorScore, mArea = mapper.determineImageMatchSuccess( scan2, scan1, yawOffset, [ xOffset, yOffset ] )
        print("error:", errorScore,"\n-------")
        
        dampener = 0.8

        xOffset -= xError * dampener
        yOffset -= yError * dampener
        yawOffset -= yawError * dampener
        
        print( "newOffsets:", xOffset, yOffset, np.rad2deg(yawOffset) )

    mapper.determineImageTranslation( scan2, scan1, yawOffset, [ xOffset, yOffset ], True )
        
    
    """plt.figure(21)
    plt.imshow( nSc1, origin='lower' ) 
    
    plt.figure(22)
    plt.imshow( nSc2, origin='lower' ) """


    """plt.figure(23)
    plt.plot( matchValues ) """

    plt.show() 

for cRawScan in allScansRaw:
    mapper.pushScanFrame( cRawScan )
    
    scan = mapper.analyseRecentScan()

    if ( scan != None ): 
        if ( np.max( scan.constructedProbGrid.positiveData ) != 0 ): 
            fPos = scan.featurePositions
            
            #gridDisp.parseData( scan.estimatedMap, fPos[:,1], fPos[:,0]  )
            #gridDisp2.parseData( Rval*1000, maxPos[:,1], maxPos[:,0]  )
            
            #if ( len(mapper.allScans) > 6 ): 
            #    matchingTest2( 2, 3 )
            if ( len(mapper.allScans) > 23 ): 
                matchingTest2( 22, 23 )

            prevScan = scan
            print("i: ",len(mapper.allScans))

            #lpWindow.render()
 
            #sleep(0.1)


""