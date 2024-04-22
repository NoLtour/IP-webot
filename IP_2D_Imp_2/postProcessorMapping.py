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

import PPLib as pl

lpWindow = pl.lpWindow
gridDisp      = pl.gridDisp
gridDisp2     = pl.gridDisp2
config = pl.config

print("importing...")
allScanData:list[RawScanFrame] = RawScanFrame.importScanFrames( "TEST2_LinearMatchedData-2.4" )
print("imported")

# Noise step
np.random.seed(3115)
for i in range(0,len(allScanData)):
    cScan = allScanData[i]
    
    cScan.index = i 
    ""

config = IPConfig() 

 
"""
    processedData3 -> chunks test1 high quality
    
"""

def test1MapCreation(): 
    testingChunk = pl.getChunk( allScanData, 2 )  
    pl.meanFeaturelessAutoTune( allScanData, 16 ) 
    
    chunkLength = 10
    paddingSize = 6
    
    procScans = pl.getBaseChunks( allScanData, 0, 1, 100, chunkLength, paddingSize )
    merged1 = pl.mapMergeTestRec( procScans, (chunkLength+paddingSize), [ 1,2,5 ], minFrameError=100, discriminate=1.2 ) # [ 5, 4, 3, 2, 1, -3, -2 ]
    
    #pl.featurelessAutoTune( merged1[0] ) 
    
    
    mergedMapped = Chunk.initEmpty( config )
    mergedMapped.addChunks( merged1 )
    fancyPlot( mergedMapped.constructProbabilityGrid().mapEstimate )
 
    mergedMapped.subChunks[0].determineErrorKeypoints( mergedMapped.subChunks[1], np.array((0,0,0)), True )
 
    for asChunk in merged1: 
        gridDisp.parseData( asChunk.constructProbabilityGrid().mapEstimate )  
        lpWindow.render()  
        
        ""
     
    mergedMapped.graphSLAM.plot()
    mergedMapped.randomHybridErrorReduction( 100, 100 )
    mergedMapped.randomHybridErrorReduction( 100, 100 )
    mergedMapped.randomHybridErrorReduction( 100, 100 )
    mergedMapped.repeatingHybridPrune( 100, [13,12,11,10,9,8,7,6,5,4,3,2,1], maxIterations=2, errorCompSep=1 )
    mergedMapped.randomHybridErrorReduction( 100, 100 )
    mergedMapped.randomHybridErrorReduction( 100, 100 )
    mergedMapped.graphSLAM.plot()
    fancyPlot( mergedMapped.constructProbabilityGrid().mapEstimate )
    
    plt.show()
    
    asRaws = mergedMapped.exportAsRaws()
    RawScanFrame.exportScanFrames( asRaws, "TEST2_MapExport-2.2" )
    
    ""
        
test1MapCreation()

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









