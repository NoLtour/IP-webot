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

from CartesianPose import CartesianPose

import PPLib as pl

print("importing...")
#allScanData:list[RawScanFrame] = RawScanFrame.importScanFrames( "TEST2_rawInpData-2" )
allScanData:list[RawScanFrame] = RawScanFrame.importScanFrames( "TEST1_LinearMatchedData-3" )
print("imported")


#pl.plotPathError( allScanData )
# Noise step
np.random.seed(3115)

if (True):
    allPoseChanges = []
    for i in range(0,len(allScanData)): 
        cScan:RawScanFrame = allScanData[i]
        
        if (i!=0):
            pScan:RawScanFrame = allScanData[i-1]
            
            interFrameChange = cScan.pose.copy()
            interFrameChange.subtractPose( pScan.pose )
            interFrameChange.divide( 0.9  + 0.1*(np.random.random()*2-1))
            size = (abs(interFrameChange.x)+abs(interFrameChange.y))*3
            interFrameChange.addPose( CartesianPose( (np.random.random()-0.5)*size, (np.random.random()-0.5)*size, 0, 0, 0, -size*0.5 ) )
            
            allPoseChanges.append(interFrameChange)
            
            ""
        
        cScan.index = i 
    for i in range(0,len(allScanData)): 
        cScan:RawScanFrame = allScanData[i]
        
        if (i!=0):
            pScan:RawScanFrame = allScanData[i-1]
            interFrameChange = allPoseChanges[i-1]
            
            cScan.pose = pScan.pose.copy()
            cScan.pose.addPose( interFrameChange )
            
            ""
        
        cScan.index = i 
    
    
#pl.plotPathError( allScanData )

config = IPConfig()


class MAP_PROP:
    X_MIN = -4
    X_MAX = 6
    Y_MIN = -6
    Y_MAX = 4
    PROB_GRID_RES = config.GRID_RESOLUTION
 
lpWindow = pl.lpWindow
gridDisp      = pl.gridDisp
gridDisp2     = pl.gridDisp2

def testtt():
    
    target1 = pl.getChunk( allScanData, 700 )
    target2 = pl.getChunk( allScanData, 704 )
    
    #pl.featurelessAutoTune( pl.getChunk( allScanData, 0 ) ) 
    #pl.findDifference( target1, target2, np.array((0.2,0.0,0.0)), 100, True  )
    pl.meanFeaturelessAutoTune( allScanData, 8 ) 
    
    #pl.twoFramesTest( target1, target2 )
    mmm = pl.getChunk( allScanData, 0 )
    fancyPlot( mmm.constructProbabilityGrid().mapEstimate )
    pl.featurelessFullTest( mmm )
    
    # pl.twoFramesTest( target1, target2 )
    procScans = pl.getBaseChunks(allScanData, 0, 10, 9999999 )
    merged1 = pl.mapMergeTestRec( procScans, 99999999, [], minFrameError=70 )#  9,8,7,6,5,4,3,2,1,15,14,12,10,8,6,3,2,1,3,2,1 
    parent = merged1[0]
    
    pl.minimiserEffectivenessTest( parent, 55550, 1 )
    
    ""
    
testtt()
 
def test1ChunkExport(): 
    pl.meanFeaturelessAutoTune( allScanData, 6 )
    
    #pl.config.FEATURELESS_X_ERROR_SCALE,pl.config.FEATURELESS_Y_ERROR_SCALE,pl.config.FEATURELESS_A_ERROR_SCALE = 2, 2, 0.1 
 
    procScans = pl.getBaseChunks(allScanData, 0, 30, 99999999 )
    merged1 = pl.mapMergeTestRec( procScans, 99999999, [ 1, 2 ], minFrameError=70 )#  9,8,7,6,5,4,3,2,1,15,14,12,10,8,6,3,2,1,3,2,1 
    merged1[0].graphSLAM.plot() 
    
    merged1[0].randomHybridErrorReduction( 90 )
    merged1[0].randomHybridErrorReduction( 90 )
    merged1[0].randomHybridErrorReduction( 90 )
    merged1[0].randomHybridErrorReduction( 90 )
    merged1[0].graphSLAM.plot() 
    #merged1 = pl.mapMergeTestRec( procScans, 12, [ 1, 5, 4, 3, 2 ] )


    asRaws = merged1[0].exportAsRaws()

    #RawScanFrame.exportScanFrames( asRaws, "TEST2_LinearMatchedData-2.2" )

    asChunkz = Chunk.initFromProcessedScanStack( asRaws, 30, config )

    for asChunk in asChunkz: 

        gridDisp.parseData( asChunk.constructProbabilityGrid().mapEstimate )  
        lpWindow.render()  
        # plt.show( block=False )
        
    ""

 
test1ChunkExport()

# merged2 = mapMergeTestRec( asChunkz, 5, [ 2, 4, -3, 1 ] )

# merged3 = mapMergeTestRec( merged2, 5, [ 2, 4, -3 ] )


# procScans = getBaseChunks( 60, 4, 60+4*30*7 )
# merged1 = mapMergeTestRec( procScans, 30, [ 10, 5, 2, -4, 20, 23, 26 ] )

# featurelessAutoTune( merged1[0] ) 

# merged2 = mapMergeTestRec( merged1, 7, [ 6, 5, 4, -3, 2, 1 ] ) 

# featurelessAutoTune( merged2[0] ) 

# merged3 = mapMergeTestRec( merged2, 5, [ 2, 4, -3 ] )



# #featurelessFullTest(testingChunk)
# #method1Tester( testingChunk )

# mapMergeTestKeypoints( 26, 600, 3  ) 
# mapMergeTest( 0, 30, 3 )

# conf = testingChunk.config
# #conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE = 0.063324, 0.061625, 0.055572
# twoFramesTest( 44, 60, 47 )

# #mergeFrameRecursive( 20, 5 )

# #mapMergeTest( 0, 240 )
# #errorTesterAlt(testingChunk)

# #featurelessFullTest(testingChunk)
# #singleMinTest( testingChunk )
# #superTuner( testingChunk )
# #method2Tester( testingChunk )

# #findDifference( testingChunk, testingChunk, np.array((-0.1,0.24,np.deg2rad(22))), 100, True )


# #featurelessFullTest(testingChunk)
# #method1Tester( testingChunk )
# #determineErrorCoupling( testingChunk, 300 )

# #fancyPlot( testingChunk.cachedProbabilityGrid.mapEstimate )
# plt.show()



 
# if ( len( chunkStack ) > 36 ):
#     """fancyPlot( chunkStack[12].cachedProbabilityGrid.mapEstimate )
#     fancyPlot( chunkStack[32].cachedProbabilityGrid.mapEstimate )
#     plt.show()"""

#     parentChunk = Chunk.initEmpty( config )

#     parentChunk.addChunks( chunkStack )

#     errorTester( chunkStack[32] )
#     superTuner( chunkStack[32] )
#     #featurelessAutoTune( chunkStack[32]  )
    
#     conf = chunkStack[32].config
#     #conf.FEATURELESS_X_ERROR_SCALE,conf.FEATURELESS_Y_ERROR_SCALE,conf.FEATURELESS_A_ERROR_SCALE,conf.CONFLICT_MULT_GAIN = 0.00147138 ,0.00365946 ,0.00283191 ,0.400286
    
#     #method1Tester( chunkStack[32] )

#     #offset, error, val = findDifference2( chunkStack[32],chunkStack[32], (0.1, -0.1, np.deg2rad(10)), 30 ) 
#     #offset, error, val = findDifference( chunkStack[32],chunkStack[32], (0.1, -0.1, np.deg2rad(10)), 30, True ) 
#     #featurelessFullTest( chunkStack[32] )

#     chunkStack = []





""









